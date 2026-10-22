"""The drift guard between the chart's JS indicators and the strategies' own.

`Dependencies/dashboard_assets/dashboard_indicators.js` is a SECOND
implementation of algebra this repository already has in
`Signal Generators/misc_strategy_common.py`. It exists because the dashboard's
history pages are rendered once at startup into immutable bytes, so a
server-side indicator could only ever cover the live window and would vanish
the moment the operator scrolled back (see ADR-0020).

A second implementation is a liability unless something keeps it honest. That
is this file: it runs every `compute` in the asset under node against a shared
fixture and compares the result with the strategies' pinned TA-Lib helpers,
element by element, leading warm-up included.

The warm-up matters as much as the values. TA-Lib populates each output only
from the index where it is fully determined -- `period - 1` for an SMA or EMA,
`period` for RSI and ATR, `2 * period - 1` for ADX -- and an off-by-one there
shifts every number that follows while still looking like a plausible line on
screen. So the comparison asserts where each series STARTS before it asserts
what it contains.

Skipped rather than failed where node is absent: node is not a dependency of
this project, only something CI runners happen to have.
"""

from __future__ import annotations

import json
import math
import shutil

# Both are needed to run the page's own JavaScript, which is the entire point
# of this file -- a source assertion cannot catch a seeding bug.
import subprocess  # nosec B404 - a fixed argv, no shell; see `_node`
import tempfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "Dependencies" / "dashboard_assets" / "dashboard_indicators.js"

#: Element-wise tolerance. These are the same arithmetic in two languages, both
#: in IEEE-754 doubles, so the gap should be floating-point noise and nothing
#: else. A failure here is a semantic difference, not a rounding one -- raising
#: this number would hide exactly the bug the file exists to catch.
TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# The strategies' own helpers
# ---------------------------------------------------------------------------
def _misc_strategy_common():
    """The module the strategies trade on, loaded the way the runner loads it.

    Its folder has a space in its name, so it cannot be imported normally --
    ADR-0009. TA-Lib backs every helper in it and is an optional extra, so a
    box without it skips rather than fails.
    """

    import importlib.util

    path = ROOT / "Signal Generators" / "misc_strategy_common.py"
    spec = importlib.util.spec_from_file_location("misc_strategy_common", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:  # pragma: no cover - depends on the box
        pytest.skip(f"the strategies' indicator helpers are unavailable here: {exc}")
    return module


# ---------------------------------------------------------------------------
# A shared fixture, used by both sides
# ---------------------------------------------------------------------------
def _bars(count: int = 400) -> list[dict]:
    """Deterministic OHLC with a real shape: trend, swings and chop.

    Not random, and not monotonic. A ramp pins RSI at 100 and leaves ADX with
    nothing to measure, so a broken implementation would agree with a correct
    one; the two sine terms put turning points at different scales so every
    branch below -- Wilder's decay, a Supertrend flip, a Bollinger squeeze --
    is actually exercised.
    """

    rows = []
    for index in range(count):
        close = (
            24500.0
            + 120.0 * math.sin(index / 41.0)
            + 30.0 * math.sin(index / 7.0)
            + 6.0 * math.sin(index / 2.0)
        )
        spread = 4.0 + 2.0 * abs(math.sin(index / 13.0))
        rows.append(
            {
                "time": 1_700_000_000 + index * 60,
                "open": round(close - spread / 2, 4),
                "high": round(close + spread, 4),
                "low": round(close - spread, 4),
                "close": round(close, 4),
            }
        )
    return rows


def _frame(bars: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(bar["time"], unit="s") for bar in bars],
            "open": [bar["open"] for bar in bars],
            "high": [bar["high"] for bar in bars],
            "low": [bar["low"] for bar in bars],
            "close": [bar["close"] for bar in bars],
        }
    )


# ---------------------------------------------------------------------------
# Running the asset
# ---------------------------------------------------------------------------
def _run_js(bars: list[dict], requests: list[dict]) -> dict[str, list[list]]:
    """Run the registry's `compute` under node and return each line's values.

    The whole asset is loaded rather than lifted function by function: it is a
    self-contained IIFE with a CommonJS export for exactly this, so the test
    exercises the file the server actually serves.
    """

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the chart's JS cannot be executed here")

    harness = (
        f"const api = require({json.dumps(str(ASSET))});\n"
        f"const bars = {json.dumps(bars)};\n"
        f"const requests = {json.dumps(requests)};\n"
        "const out = {};\n"
        "for (const request of requests) {\n"
        "  const entry = api.byId(request.id);\n"
        "  const params = api.normalizeParams(entry, request.params);\n"
        "  out[request.key] = entry.compute(bars, params).map((line) => line.values);\n"
        "}\n"
        "console.log(JSON.stringify(out));\n"
    )
    return json.loads(_node(node, harness))


def _node(node: str, script: str) -> str:
    """Run a script through node, by FILE rather than `-e`.

    A few hundred bars of JSON inlined into `node -e` exceeds Windows'
    command-line limit and fails as "the filename or extension is too long",
    which reads like a path problem and is not one.
    """

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "harness.js"
        path.write_text(script, encoding="utf-8")
        # A fixed argv and no shell. The script is built here from this file's
        # own literals plus the repository's own asset path -- nothing
        # user-supplied ever reaches it.
        finished = subprocess.run(  # nosec B603
            [node, str(path)], capture_output=True, text=True, timeout=120, check=False
        )
    assert finished.returncode == 0, finished.stderr
    return finished.stdout


def _assert_matches(name: str, actual: list, expected: pd.Series) -> None:
    """Element-wise agreement, warm-up and all.

    `expected` is a pandas Series whose warm-up is NaN; `actual` is a JS array
    whose warm-up is null. Both are compared as "no value here", and the FIRST
    index carrying a value is asserted before any number is, because that is
    where a seeding bug shows up first and most clearly.
    """

    expected_values = [None if pd.isna(value) else float(value) for value in expected]
    assert len(actual) == len(expected_values), f"{name}: length"

    def first_value(values):
        return next((i for i, value in enumerate(values) if value is not None), None)

    assert first_value(actual) == first_value(expected_values), (
        f"{name}: the series starts at a different bar -- "
        f"js={first_value(actual)} talib={first_value(expected_values)}. "
        "That is a warm-up/seeding difference, not rounding."
    )

    for index, (mine, theirs) in enumerate(zip(actual, expected_values, strict=True)):
        if theirs is None:
            assert mine is None, f"{name}[{index}]: js has {mine} where TA-Lib has none"
            continue
        assert mine is not None, f"{name}[{index}]: js has none where TA-Lib has {theirs}"
        assert abs(mine - theirs) <= TOLERANCE, (
            f"{name}[{index}]: js={mine} talib={theirs} (diff {abs(mine - theirs):g})"
        )


# ---------------------------------------------------------------------------
# The guard itself
# ---------------------------------------------------------------------------
def test_every_chart_indicator_matches_the_strategies_own_helper():
    """One test, every indicator: the drift guard the second implementation needs."""

    misc = _misc_strategy_common()
    bars = _bars()
    frame = _frame(bars)
    close = frame["close"]

    requests = [
        {"key": "ema", "id": "ema", "params": {"period": 21}},
        {"key": "sma", "id": "sma", "params": {"period": 50}},
        {"key": "bbands", "id": "bbands", "params": {"period": 20, "stddev": 2}},
        {"key": "supertrend", "id": "supertrend", "params": {"period": 10, "multiplier": 3}},
        {"key": "rsi", "id": "rsi", "params": {"period": 14}},
        {"key": "macd", "id": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}},
        {"key": "atr", "id": "atr", "params": {"period": 14}},
        {"key": "adx", "id": "adx", "params": {"period": 14}},
    ]
    js = _run_js(bars, requests)

    _assert_matches("EMA 21", js["ema"][0], misc.ema(close, 21))
    _assert_matches("SMA 50", js["sma"][0], misc.sma(close, 50))

    upper, middle, lower = misc.bollinger_bands(close, 20, 2.0)
    _assert_matches("BB upper", js["bbands"][0], upper)
    _assert_matches("BB middle", js["bbands"][1], middle)
    _assert_matches("BB lower", js["bbands"][2], lower)

    line, _direction = misc.supertrend(frame, 10, 3.0)
    _assert_matches("Supertrend", js["supertrend"][0], line)

    _assert_matches("RSI 14", js["rsi"][0], misc.rsi(close, 14))

    macd_line, signal_line, _hist = misc.macd(close, 12, 26, 9)
    _assert_matches("MACD line", js["macd"][0], macd_line)
    _assert_matches("MACD signal", js["macd"][1], signal_line)

    _assert_matches("ATR 14", js["atr"][0], misc.atr(frame, 14))
    _assert_matches("ADX 14", js["adx"][0], misc.adx(frame, 14))


@pytest.mark.parametrize(
    ("indicator", "params"),
    [
        ("ema", {"period": 5}),
        ("ema", {"period": 200}),
        ("rsi", {"period": 2}),
        ("atr", {"period": 50}),
        ("adx", {"period": 7}),
        ("sma", {"period": 3}),
    ],
)
def test_the_match_holds_at_other_periods_too(indicator, params):
    """One period could agree by luck; a short one and a long one could not.

    Period 2 and 200 are the interesting ends: the first has almost no warm-up
    to get wrong, and the second leaves most of a 400-bar fixture empty, so an
    off-by-one in either direction is unmissable.
    """

    misc = _misc_strategy_common()
    bars = _bars()
    frame = _frame(bars)
    close = frame["close"]
    period = params["period"]

    js = _run_js(bars, [{"key": "only", "id": indicator, "params": params}])
    expected = {
        "ema": lambda: misc.ema(close, period),
        "sma": lambda: misc.sma(close, period),
        "rsi": lambda: misc.rsi(close, period),
        "atr": lambda: misc.atr(frame, period),
        "adx": lambda: misc.adx(frame, period),
    }[indicator]()

    _assert_matches(f"{indicator} {period}", js["only"][0], expected)


def test_a_series_shorter_than_the_warm_up_is_all_nulls_rather_than_a_crash():
    """The chart draws from bar one, long before any indicator is determined.

    An empty pane is the honest answer there. Throwing would take the whole
    page down, because the draw path has no guard of its own.
    """

    bars = _bars(5)
    js = _run_js(
        bars,
        [
            {"key": "ema", "id": "ema", "params": {"period": 50}},
            {"key": "adx", "id": "adx", "params": {"period": 14}},
            {"key": "macd", "id": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}},
        ],
    )

    for key in ("ema", "adx", "macd"):
        for values in js[key]:
            assert len(values) == len(bars)
            assert all(value is None for value in values), key


def test_a_parameter_the_operator_typed_is_clamped_before_it_reaches_the_maths():
    """The menu takes free text, so `compute` must be handed something sane.

    Garbage falls back to the parameter's default rather than producing a NaN
    series, and a period is rounded because it is a bar COUNT -- a fractional
    one would floor inside the loops and disagree with the label drawn beside
    the line.
    """

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the chart's JS cannot be executed here")

    harness = (
        f"const api = require({json.dumps(str(ASSET))});\n"
        "const ema = api.byId('ema');\n"
        "const bb = api.byId('bbands');\n"
        "console.log(JSON.stringify({\n"
        "  empty: api.normalizeParams(ema, {}),\n"
        "  text: api.normalizeParams(ema, { period: 'twenty' }),\n"
        "  tooSmall: api.normalizeParams(ema, { period: -5 }),\n"
        "  tooBig: api.normalizeParams(ema, { period: 9999 }),\n"
        "  fractional: api.normalizeParams(ema, { period: 20.6 }),\n"
        "  fractionalAllowed: api.normalizeParams(bb, { period: 20, stddev: 2.5 }),\n"
        "}));\n"
    )
    result = json.loads(_node(node, harness))

    assert result["empty"]["period"] == 21, "a missing value falls back to the default"
    assert result["text"]["period"] == 21, "unparseable text must not become NaN"
    assert result["tooSmall"]["period"] == 2, "clamped to the minimum"
    assert result["tooBig"]["period"] == 400, "clamped to the maximum"
    assert result["fractional"]["period"] == 21, "a bar count is a whole number"
    assert result["fractionalAllowed"]["stddev"] == 2.5, (
        "a parameter declaring a step keeps its fraction"
    )
