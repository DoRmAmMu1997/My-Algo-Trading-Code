"""
================================================================================
NIFTY Opening Strike PCR VWAP ATR Signal Generator
================================================================================

WHAT THIS FILE DOES
-------------------
This file is a standalone, import-able signal generator. You give it:

1. NIFTY OHLC candles.
2. Option-chain OI-change rows.
3. Optional config values.

It returns one structured decision for the latest candle:

    * BUY_CALL
    * BUY_PUT
    * NO_SIGNAL

IMPORTANT BOUNDARY
------------------
This module generates signals only. It does NOT:

    * place orders,
    * connect to broker APIs,
    * fetch live data,
    * resolve real option-contract security IDs,
    * execute stop-loss or trailing-stop logic.

That work belongs to the front-test, backtest, paper-trade, or execution layer.

BEGINNER MENTAL MODEL
---------------------
The strategy has four big ideas:

1. At market open, freeze one "starting strike" based on the NIFTY opening
   price. This strike stays fixed for the trading day.
2. Around that fixed strike, read a small window of option-chain OI-change
   data and calculate a PCR-change number.
3. On the latest candle, check whether price is near VWAP using an ATR-based
   buffer.
4. If PCR, candle color, VWAP, and optional RSI all agree, return a buy signal
   for the ATM option of the current NIFTY close.

QUICK START
-----------
    engine = NiftyOpeningStrikePCRVWAPATRSignalGenerator()
    decision = engine.evaluate(ohlc_df, option_chain_oi_change_df)

Keep the same `engine` instance alive during one front-test day so it can keep
the opening strike fixed and block duplicate entries when configured to do so.
Create a new engine, or call `reset_trading_day()`, for a new trading day.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

try:
    # TA-Lib is preferred when available because its ATR/RSI implementations
    # match the standard indicators many charting platforms use.
    import talib  # type: ignore

    _TALIB_AVAILABLE = True
except ImportError:  # pragma: no cover - only used on machines without TA-Lib
    talib = None  # type: ignore
    _TALIB_AVAILABLE = False


# =============================================================================
# CONFIG AND DECISION OBJECTS
# =============================================================================
@dataclass(frozen=True)
class NiftyOpeningStrikePCRVWAPATRConfig:
    """All tunable strategy parameters live in one small config object."""

    strike_step: int = 50
    strike_window_n: int = 3
    pcr_bullish_threshold: float = 1.2
    pcr_bearish_threshold: float = 0.8
    option_moneyness: str = "ATM"
    atr_period: int = 14
    vwap_near_atr_multiplier: float = 0.5
    enable_rsi_filter: bool = False
    rsi_period: int = 14
    buy_rsi_min: float = 50.0
    sell_rsi_max: float = 50.0
    initial_sl_points: float = 20.0
    expiry_selection: str = "NEXT_WEEKLY"
    allow_multiple_entries: bool = False

    def __post_init__(self) -> None:
        # Validate values that would break calculations if they were zero or
        # negative. Unsupported option moneyness is handled as a NO_SIGNAL
        # decision later, because that is a strategy choice rather than a math
        # failure.
        positive_int_fields = {
            "strike_step": self.strike_step,
            "atr_period": self.atr_period,
            "rsi_period": self.rsi_period,
        }
        invalid_positive = [name for name, value in positive_int_fields.items() if int(value) <= 0]
        if invalid_positive:
            raise ValueError(f"These config values must be positive: {', '.join(invalid_positive)}")

        if int(self.strike_window_n) < 0:
            raise ValueError("`strike_window_n` cannot be negative.")
        if float(self.vwap_near_atr_multiplier) < 0.0:
            raise ValueError("`vwap_near_atr_multiplier` cannot be negative.")
        if float(self.initial_sl_points) < 0.0:
            raise ValueError("`initial_sl_points` cannot be negative.")


@dataclass(slots=True)
class NiftyOpeningStrikePCRVWAPATRDecision:
    """
    One structured answer from the signal generator.

    `action` is always one of:
    - BUY_CALL
    - BUY_PUT
    - NO_SIGNAL
    """

    action: str = "NO_SIGNAL"
    signal_triggered: bool = False
    option_type: str = ""
    expiry: str = ""
    selected_call_strike: int = 0
    selected_put_strike: int = 0
    selected_strike: int = 0
    entry_underlying: float = 0.0
    rejection_reasons: list[str] = field(default_factory=list)
    risk_metadata: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# SMALL INPUT AND INDICATOR HELPERS
# =============================================================================
def _find_first_col(frame: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    """
    Find a column by comparing lower-case, trimmed column names.

    This lets callers pass columns like "Open", "OPEN", "datetime", or
    "call oi change" without renaming everything before calling the generator.
    """

    normalized_targets = {str(name).strip().lower() for name in names}
    for column in frame.columns:
        if str(column).strip().lower() in normalized_targets:
            return column
    return None


def _round_to_nearest_strike(price: float, strike_step: int) -> int:
    """Round a NIFTY price to the nearest valid listed strike."""

    # NIFTY prices are positive, so this half-up formula avoids Python's
    # banker-rounding behavior on exact .5 cases.
    return int(math.floor((float(price) / int(strike_step)) + 0.5) * int(strike_step))


def _is_finite_number(value: Any) -> bool:
    """Return True only for real finite numeric values."""

    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _normalize_ohlc_frame(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize OHLC input into predictable lowercase columns.

    Required after normalization:
    - timestamp
    - open
    - high
    - low
    - close

    Optional:
    - volume
    - vwap
    """

    if data is None or not isinstance(data, pd.DataFrame):
        raise ValueError("OHLC input must be a pandas DataFrame.")

    frame = data.copy()
    column_map = {
        "timestamp": _find_first_col(frame, ["timestamp", "datetime", "date", "time"]),
        "open": _find_first_col(frame, ["open"]),
        "high": _find_first_col(frame, ["high"]),
        "low": _find_first_col(frame, ["low"]),
        "close": _find_first_col(frame, ["close"]),
        "volume": _find_first_col(frame, ["volume", "vol"]),
        "vwap": _find_first_col(frame, ["vwap"]),
    }

    missing = [name for name in ("open", "high", "low", "close") if column_map[name] is None]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {', '.join(missing)}")

    normalized = pd.DataFrame(index=frame.index)
    timestamp_col = column_map["timestamp"]
    if timestamp_col is not None:
        normalized["timestamp"] = pd.to_datetime(frame[timestamp_col], errors="coerce")
    else:
        # A datetime-like index is accepted as a fallback because several
        # pandas workflows naturally keep candle time in the index.
        normalized["timestamp"] = pd.to_datetime(pd.Series(frame.index, index=frame.index), errors="coerce")

    if normalized["timestamp"].isna().all():
        raise ValueError("A valid 'timestamp'/'datetime' column or datetime-like index is required.")

    for name in ("open", "high", "low", "close"):
        normalized[name] = pd.to_numeric(frame[column_map[name]], errors="coerce")

    if column_map["volume"] is not None:
        normalized["volume"] = pd.to_numeric(frame[column_map["volume"]], errors="coerce")
    if column_map["vwap"] is not None:
        normalized["vwap"] = pd.to_numeric(frame[column_map["vwap"]], errors="coerce")

    normalized = normalized.sort_values("timestamp").drop_duplicates("timestamp")
    normalized = normalized.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return normalized


def _normalize_option_chain_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize option-chain OI-change data into strike/call/put columns."""

    if data is None or not isinstance(data, pd.DataFrame):
        raise ValueError("Option-chain OI-change input must be a pandas DataFrame.")

    frame = data.copy()
    column_map = {
        "strike": _find_first_col(frame, ["strike", "strike_price", "strike price"]),
        "call_oi_change": _find_first_col(
            frame,
            ["call_oi_change", "call oi change", "ce_oi_change", "call_change_oi"],
        ),
        "put_oi_change": _find_first_col(
            frame,
            ["put_oi_change", "put oi change", "pe_oi_change", "put_change_oi"],
        ),
    }

    missing = [name for name, column in column_map.items() if column is None]
    if missing:
        raise ValueError(f"Missing required option-chain columns: {', '.join(missing)}")

    normalized = pd.DataFrame()
    normalized["strike"] = pd.to_numeric(frame[column_map["strike"]], errors="coerce")
    normalized["call_oi_change"] = pd.to_numeric(frame[column_map["call_oi_change"]], errors="coerce")
    normalized["put_oi_change"] = pd.to_numeric(frame[column_map["put_oi_change"]], errors="coerce")
    normalized = normalized.dropna(subset=["strike", "call_oi_change", "put_oi_change"])

    # Multiple rows for the same strike are grouped because some option-chain
    # exports split rows by expiry/right. This module only needs the supplied
    # OI-change sums for each strike.
    normalized["strike"] = normalized["strike"].round().astype(int)
    grouped = normalized.groupby("strike", as_index=False)[["call_oi_change", "put_oi_change"]].sum()
    return grouped


def _attach_vwap(frame: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str]]:
    """Use an existing VWAP column or calculate VWAP from OHLCV."""

    result = frame.copy()
    if "vwap" in result.columns:
        result["vwap"] = pd.to_numeric(result["vwap"], errors="coerce")
        return result, None

    if "volume" not in result.columns:
        return result, "VWAP column is missing and volume is required to calculate VWAP from OHLCV."

    result["volume"] = pd.to_numeric(result["volume"], errors="coerce")
    typical_price = (result["high"] + result["low"] + result["close"]) / 3.0
    cumulative_volume = result["volume"].cumsum()
    cumulative_value = (typical_price * result["volume"]).cumsum()
    result["vwap"] = cumulative_value / cumulative_volume.replace(0.0, np.nan)
    return result, None


def _fallback_atr(frame: pd.DataFrame, period: int) -> pd.Series:
    """Pandas Wilder-style ATR fallback used when TA-Lib is unavailable."""

    prev_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            (frame["high"] - frame["low"]).abs(),
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / int(period), adjust=False, min_periods=int(period)).mean()


def _calculate_atr(frame: pd.DataFrame, period: int) -> pd.Series:
    """Calculate ATR with TA-Lib first, then a pandas fallback."""

    if _TALIB_AVAILABLE:
        atr_values = talib.ATR(  # type: ignore[union-attr]
            frame["high"].to_numpy(dtype=float),
            frame["low"].to_numpy(dtype=float),
            frame["close"].to_numpy(dtype=float),
            timeperiod=int(period),
        )
        return pd.Series(atr_values, index=frame.index)
    return _fallback_atr(frame, period)


def _fallback_rsi(close: pd.Series, period: int) -> pd.Series:
    """Pandas Wilder-style RSI fallback used when TA-Lib is unavailable."""

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / int(period), adjust=False, min_periods=int(period)).mean()
    avg_loss = loss.ewm(alpha=1.0 / int(period), adjust=False, min_periods=int(period)).mean()
    relative_strength = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))

    # When there are gains but no losses, RSI is conventionally 100. When both
    # are zero, price is flat; 50 is a neutral, easy-to-read fallback.
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain == 0.0), 50.0)
    return rsi


def _calculate_rsi(close: pd.Series, period: int) -> pd.Series:
    """Calculate RSI with TA-Lib first, then a pandas fallback."""

    if _TALIB_AVAILABLE:
        rsi_values = talib.RSI(close.to_numpy(dtype=float), timeperiod=int(period))  # type: ignore[union-attr]
        return pd.Series(rsi_values, index=close.index)
    return _fallback_rsi(close, period)


def _risk_metadata(config: NiftyOpeningStrikePCRVWAPATRConfig) -> dict[str, Any]:
    """Return risk instructions for downstream execution systems."""

    return {
        "initial_sl_points": float(config.initial_sl_points),
        "initial_sl_formula": "option_entry_price - initial_sl_points",
        "initial_sl_note": "Execution layer should apply this to the option entry price.",
        "rr_trailing_rule": [
            {"at_reward_r": 2, "trail_sl_to_r": 1},
            {"at_reward_r": 3, "trail_sl_to_r": 2},
            {"at_reward_r": 4, "trail_sl_to_r": 3},
        ],
        "rr_trailing_pattern": "For reward level N >= 2, trail SL to N-1 R.",
        "signal_generator_note": "No stop-loss or trailing order is executed by this module.",
    }


def _base_debug(config: NiftyOpeningStrikePCRVWAPATRConfig) -> dict[str, Any]:
    """Create a complete debug dictionary with stable keys."""

    return {
        "starting_strike": None,
        "starting_strike_source": "",
        "selected_pcr_strikes": [],
        "missing_selected_pcr_strikes": [],
        "put_oi_change_sum": None,
        "call_oi_change_sum": None,
        "change_pcr": None,
        "vwap": None,
        "atr": None,
        "near_vwap_threshold": None,
        "near_vwap_by_close": False,
        "near_vwap_by_range": False,
        "near_vwap": False,
        "candle_is_green": False,
        "candle_is_red": False,
        "bullish_vwap_close_condition": False,
        "bearish_vwap_close_condition": False,
        "rsi_enabled": bool(config.enable_rsi_filter),
        "rsi": None,
        "buy_rsi_condition": None,
        "sell_rsi_condition": None,
        "current_atm_strike": None,
        "option_moneyness": str(config.option_moneyness),
        "allow_multiple_entries": bool(config.allow_multiple_entries),
        "duplicate_entry_blocked": False,
        "raw_signal_action": "",
        "final_decision": "NO_SIGNAL",
    }


# =============================================================================
# PUBLIC SIGNAL GENERATOR
# =============================================================================
class NiftyOpeningStrikePCRVWAPATRSignalGenerator:
    """
    Stateful signal engine for the opening-strike PCR/VWAP/ATR strategy.

    State kept across evaluations:
    - `starting_strike`: fixed after the first evaluation of the day.
    - `_entry_signal_sent`: used to block duplicate entries when configured.
    """

    def __init__(self, config: Optional[NiftyOpeningStrikePCRVWAPATRConfig] = None):
        self.config = config or NiftyOpeningStrikePCRVWAPATRConfig()
        self.starting_strike: Optional[int] = None
        self._starting_strike_source = ""
        self._entry_signal_sent = False

    def reset_trading_day(self) -> None:
        """Clear day-level state before evaluating a new trading day."""

        self.starting_strike = None
        self._starting_strike_source = ""
        self._entry_signal_sent = False

    def _no_signal(
        self,
        rejection_reasons: list[str],
        debug: dict[str, Any],
    ) -> NiftyOpeningStrikePCRVWAPATRDecision:
        """Build a stable NO_SIGNAL response."""

        debug["final_decision"] = "NO_SIGNAL"
        return NiftyOpeningStrikePCRVWAPATRDecision(
            action="NO_SIGNAL",
            signal_triggered=False,
            rejection_reasons=rejection_reasons,
            risk_metadata=_risk_metadata(self.config),
            debug=debug,
        )

    def _selected_pcr_strikes(self) -> list[int]:
        """Return the inclusive PCR strike window around the fixed start strike."""

        if self.starting_strike is None:
            return []
        start = self.starting_strike - int(self.config.strike_window_n) * int(self.config.strike_step)
        end = self.starting_strike + int(self.config.strike_window_n) * int(self.config.strike_step)
        return list(range(start, end + int(self.config.strike_step), int(self.config.strike_step)))

    def _ensure_starting_strike(
        self,
        ohlc: pd.DataFrame,
        opening_price: Optional[float],
        debug: dict[str, Any],
    ) -> Optional[str]:
        """Set starting strike once, then keep it fixed."""

        if self.starting_strike is not None:
            debug["starting_strike"] = self.starting_strike
            debug["starting_strike_source"] = self._starting_strike_source
            return None

        if opening_price is not None:
            source_price = opening_price
            source_label = "explicit_opening_price"
        else:
            source_price = ohlc.iloc[0]["open"] if not ohlc.empty else np.nan
            source_label = "first_ohlc_open"

        if not _is_finite_number(source_price):
            return "Could not derive starting_strike because opening price is missing or invalid."

        self.starting_strike = _round_to_nearest_strike(float(source_price), self.config.strike_step)
        self._starting_strike_source = source_label
        debug["starting_strike"] = self.starting_strike
        debug["starting_strike_source"] = self._starting_strike_source
        return None

    def _calculate_pcr_debug(
        self,
        option_chain: pd.DataFrame,
        debug: dict[str, Any],
    ) -> Optional[str]:
        """Calculate PCR-change inputs and store them in debug."""

        selected_strikes = self._selected_pcr_strikes()
        debug["selected_pcr_strikes"] = selected_strikes

        available_strikes = set(int(strike) for strike in option_chain["strike"].tolist())
        missing_strikes = [strike for strike in selected_strikes if strike not in available_strikes]
        debug["missing_selected_pcr_strikes"] = missing_strikes
        if missing_strikes:
            return f"Missing selected PCR strikes: {missing_strikes}"

        selected = option_chain[option_chain["strike"].isin(selected_strikes)]
        put_sum = float(selected["put_oi_change"].sum())
        call_sum = float(selected["call_oi_change"].sum())
        debug["put_oi_change_sum"] = put_sum
        debug["call_oi_change_sum"] = call_sum

        if not np.isfinite(call_sum) or call_sum <= 0.0:
            return f"Invalid call OI change sum for PCR calculation: {call_sum}"
        if not np.isfinite(put_sum):
            return f"Invalid put OI change sum for PCR calculation: {put_sum}"

        debug["change_pcr"] = put_sum / call_sum
        return None

    def _decision_for_entry(
        self,
        *,
        action: str,
        option_type: str,
        selected_strike: int,
        entry_underlying: float,
        debug: dict[str, Any],
    ) -> NiftyOpeningStrikePCRVWAPATRDecision:
        """Build a BUY_CALL or BUY_PUT response and remember the entry."""

        debug["raw_signal_action"] = action
        if not self.config.allow_multiple_entries and self._entry_signal_sent:
            debug["duplicate_entry_blocked"] = True
            return self._no_signal(
                [f"Duplicate entry blocked because allow_multiple_entries=False and {action} was already emitted."],
                debug,
            )

        self._entry_signal_sent = True
        debug["final_decision"] = action
        common_payload = {
            "action": action,
            "signal_triggered": True,
            "option_type": option_type,
            "expiry": str(self.config.expiry_selection),
            "selected_strike": selected_strike,
            "entry_underlying": float(entry_underlying),
            "rejection_reasons": [],
            "risk_metadata": _risk_metadata(self.config),
            "debug": debug,
        }

        if action == "BUY_CALL":
            return NiftyOpeningStrikePCRVWAPATRDecision(
                selected_call_strike=selected_strike,
                **common_payload,
            )

        return NiftyOpeningStrikePCRVWAPATRDecision(
            selected_put_strike=selected_strike,
            **common_payload,
        )

    def evaluate(
        self,
        ohlc: pd.DataFrame,
        option_chain_oi_change: pd.DataFrame,
        opening_price: Optional[float] = None,
    ) -> NiftyOpeningStrikePCRVWAPATRDecision:
        """
        Evaluate the latest supplied OHLC candle and return one decision.

        The caller controls timeframe. The input may be 1-minute, 3-minute, or
        5-minute candles; this function simply evaluates the newest row it is
        given.
        """

        debug = _base_debug(self.config)

        if str(self.config.option_moneyness).strip().upper() != "ATM":
            return self._no_signal(
                [f"Unsupported option_moneyness={self.config.option_moneyness!r}; only 'ATM' is supported."],
                debug,
            )

        try:
            candles = _normalize_ohlc_frame(ohlc)
            option_chain = _normalize_option_chain_frame(option_chain_oi_change)
        except ValueError as exc:
            return self._no_signal([str(exc)], debug)

        if candles.empty:
            return self._no_signal(["OHLC data has no usable candles after normalization."], debug)
        if option_chain.empty:
            return self._no_signal(["Option-chain OI-change data has no usable rows after normalization."], debug)

        starting_strike_error = self._ensure_starting_strike(candles, opening_price, debug)
        if starting_strike_error:
            return self._no_signal([starting_strike_error], debug)

        pcr_error = self._calculate_pcr_debug(option_chain, debug)
        if pcr_error:
            return self._no_signal([pcr_error], debug)

        candles, vwap_error = _attach_vwap(candles)
        if vwap_error:
            return self._no_signal([vwap_error], debug)

        candles["atr"] = _calculate_atr(candles, self.config.atr_period)
        if self.config.enable_rsi_filter:
            candles["rsi"] = _calculate_rsi(candles["close"], self.config.rsi_period)

        current = candles.iloc[-1]
        open_price = float(current["open"])
        high_price = float(current["high"])
        low_price = float(current["low"])
        close_price = float(current["close"])
        vwap = float(current["vwap"]) if _is_finite_number(current["vwap"]) else np.nan
        atr = float(current["atr"]) if _is_finite_number(current["atr"]) else np.nan
        debug["vwap"] = vwap if np.isfinite(vwap) else None
        debug["atr"] = atr if np.isfinite(atr) else None

        if not np.isfinite(vwap):
            return self._no_signal(["Latest VWAP is missing or invalid."], debug)
        if not np.isfinite(atr) or atr <= 0.0:
            return self._no_signal(
                [f"ATR could not be calculated yet; need enough candles for atr_period={self.config.atr_period}."],
                debug,
            )

        near_vwap_threshold = float(self.config.vwap_near_atr_multiplier) * atr
        near_by_close = abs(close_price - vwap) <= near_vwap_threshold
        near_by_range = low_price <= (vwap + near_vwap_threshold) and high_price >= (vwap - near_vwap_threshold)
        near_vwap = bool(near_by_close or near_by_range)

        candle_is_green = close_price > open_price
        candle_is_red = close_price < open_price
        bullish_vwap_close = close_price > vwap
        bearish_vwap_close = close_price < vwap
        current_atm_strike = _round_to_nearest_strike(close_price, self.config.strike_step)

        debug.update(
            {
                "near_vwap_threshold": near_vwap_threshold,
                "near_vwap_by_close": bool(near_by_close),
                "near_vwap_by_range": bool(near_by_range),
                "near_vwap": near_vwap,
                "candle_is_green": bool(candle_is_green),
                "candle_is_red": bool(candle_is_red),
                "bullish_vwap_close_condition": bool(bullish_vwap_close),
                "bearish_vwap_close_condition": bool(bearish_vwap_close),
                "current_atm_strike": current_atm_strike,
            }
        )

        buy_rsi_condition = True
        sell_rsi_condition = True
        if self.config.enable_rsi_filter:
            rsi = float(current["rsi"]) if _is_finite_number(current["rsi"]) else np.nan
            debug["rsi"] = rsi if np.isfinite(rsi) else None
            if not np.isfinite(rsi):
                return self._no_signal(
                    [f"RSI could not be calculated yet; need enough candles for rsi_period={self.config.rsi_period}."],
                    debug,
                )
            buy_rsi_condition = rsi > float(self.config.buy_rsi_min)
            sell_rsi_condition = rsi < float(self.config.sell_rsi_max)
            debug["buy_rsi_condition"] = bool(buy_rsi_condition)
            debug["sell_rsi_condition"] = bool(sell_rsi_condition)
        else:
            debug["buy_rsi_condition"] = "DISABLED"
            debug["sell_rsi_condition"] = "DISABLED"

        change_pcr = float(debug["change_pcr"])
        bullish_pcr = change_pcr > float(self.config.pcr_bullish_threshold)
        bearish_pcr = change_pcr < float(self.config.pcr_bearish_threshold)

        buy_call_valid = all(
            [
                bullish_pcr,
                candle_is_green,
                near_vwap,
                bullish_vwap_close,
                buy_rsi_condition,
            ]
        )
        buy_put_valid = all(
            [
                bearish_pcr,
                candle_is_red,
                near_vwap,
                bearish_vwap_close,
                sell_rsi_condition,
            ]
        )

        if buy_call_valid:
            return self._decision_for_entry(
                action="BUY_CALL",
                option_type="CE",
                selected_strike=current_atm_strike,
                entry_underlying=close_price,
                debug=debug,
            )

        if buy_put_valid:
            return self._decision_for_entry(
                action="BUY_PUT",
                option_type="PE",
                selected_strike=current_atm_strike,
                entry_underlying=close_price,
                debug=debug,
            )

        rejection_reasons: list[str] = []
        if not bullish_pcr and not bearish_pcr:
            rejection_reasons.append(
                "PCR threshold not met: "
                f"change_pcr={change_pcr:.4f}, bullish>{self.config.pcr_bullish_threshold}, "
                f"bearish<{self.config.pcr_bearish_threshold}."
            )

        if bullish_pcr:
            if not candle_is_green:
                rejection_reasons.append("Bullish candle condition failed: close is not greater than open.")
            if not near_vwap:
                rejection_reasons.append("Bullish near-VWAP condition failed: candle is not near VWAP.")
            if not bullish_vwap_close:
                rejection_reasons.append("Bullish VWAP close condition failed: close is not above VWAP.")
            if not buy_rsi_condition:
                rejection_reasons.append(
                    f"Bullish RSI condition failed: RSI is not greater than {self.config.buy_rsi_min}."
                )

        if bearish_pcr:
            if not candle_is_red:
                rejection_reasons.append("Bearish candle condition failed: close is not less than open.")
            if not near_vwap:
                rejection_reasons.append("Bearish near-VWAP condition failed: candle is not near VWAP.")
            if not bearish_vwap_close:
                rejection_reasons.append("Bearish VWAP close condition failed: close is not below VWAP.")
            if not sell_rsi_condition:
                rejection_reasons.append(
                    f"Bearish RSI condition failed: RSI is not less than {self.config.sell_rsi_max}."
                )

        if not rejection_reasons:
            rejection_reasons.append("No complete BUY_CALL or BUY_PUT setup on the latest candle.")

        return self._no_signal(rejection_reasons, debug)


# =============================================================================
# FUNCTIONAL WRAPPER
# =============================================================================
def get_latest_nifty_opening_strike_pcr_vwap_atr_signal(
    ohlc: pd.DataFrame,
    option_chain_oi_change: pd.DataFrame,
    config: Optional[NiftyOpeningStrikePCRVWAPATRConfig] = None,
    opening_price: Optional[float] = None,
) -> NiftyOpeningStrikePCRVWAPATRDecision:
    """
    One-shot helper for callers who do not want to instantiate the engine.

    For front-testing across multiple candles, prefer keeping a
    `NiftyOpeningStrikePCRVWAPATRSignalGenerator` instance alive so the opening
    strike and duplicate-entry state are preserved.
    """

    engine = NiftyOpeningStrikePCRVWAPATRSignalGenerator(config=config)
    return engine.evaluate(ohlc, option_chain_oi_change, opening_price=opening_price)


__all__ = [
    "NiftyOpeningStrikePCRVWAPATRConfig",
    "NiftyOpeningStrikePCRVWAPATRDecision",
    "NiftyOpeningStrikePCRVWAPATRSignalGenerator",
    "get_latest_nifty_opening_strike_pcr_vwap_atr_signal",
]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    # Small, fake candle set for demonstration only. Real callers should pass
    # their own front-test/backtest candles into `evaluate(...)`.
    sample_candles = []
    for index in range(20):
        base = 22000.0 + index
        sample_candles.append(
            {
                "timestamp": pd.Timestamp("2026-05-11 09:15") + pd.Timedelta(minutes=index),
                "open": base,
                "high": base + 18.0,
                "low": base - 12.0,
                "close": base + 8.0,
                "volume": 1000 + index,
                "vwap": base + 3.0,
            }
        )

    # Make the latest candle clearly green, near VWAP, and closing above VWAP.
    sample_candles[-1].update(
        {
            "open": 22020.0,
            "high": 22045.0,
            "low": 22010.0,
            "close": 22035.0,
            "vwap": 22030.0,
        }
    )
    sample_ohlc = pd.DataFrame(sample_candles)

    sample_option_chain = pd.DataFrame(
        {
            "strike": [21850, 21900, 21950, 22000, 22050, 22100, 22150],
            "call_oi_change": [100.0] * 7,
            "put_oi_change": [150.0] * 7,
        }
    )

    generator = NiftyOpeningStrikePCRVWAPATRSignalGenerator()
    sample_decision = generator.evaluate(sample_ohlc, sample_option_chain)
    print(asdict(sample_decision))
