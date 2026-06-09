"""
NIFTY ML Ensemble Signal Generator.

A machine-learning strategy ported from the public TradingBot reference repo. It
is the only Misc generator that is not a hand-written rule - instead a model
learns the rule from data.

The idea (plain English)
------------------------
1. For each past candle we compute a handful of "features" (RSI, MACD, Bollinger
   width, ATR, recent returns, volume ratio, candle range - see FEATURE_COLUMNS).
2. We label each past candle with the answer we wish we'd known: "was price higher
   `forward_bars` candles later? yes/no".
3. A RandomForest (an ensemble of decision trees) learns the link between the
   features and that yes/no outcome.
4. On the latest candle we feed today's features in and the model returns a
   probability that price will rise. High probability -> go long; low -> go short;
   in between -> stay flat.

The model is RE-trained every `retrain_every` candles on a rolling window, and -
crucially - only on candles whose forward outcome is already known, so there is
NO look-ahead/peeking at the future (see ``_maybe_train``).

Signals
-------
- ENTER_LONG  when the predicted P(up) > buy_threshold.
- ENTER_SHORT when P(up) < sell_threshold.
- EXIT        on a fixed % stop or % target.

Dependency: this is the only Misc generator that requires scikit-learn. If it is
not installed, constructing the engine raises a clear ImportError; the other
twelve generators are unaffected. It also needs a long warm-up (training window +
forward window) before it can trade.

This module does not resample - feed it already-prepared OHLC candles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from misc_strategy_common import (
    atr,
    bollinger_bands,
    finite,
    macd,
    normalize_ohlc_frame,
    require_columns,
    rsi,
)

# The inputs the model learns from. Each is one number per candle describing a
# different facet of recent behaviour: momentum (RSI/MACD), volatility (Bollinger
# width, normalised ATR), recent returns over 1/5/10 candles, relative volume, and
# the candle's range. build_ml_ensemble_with_indicators() fills these columns in.
FEATURE_COLUMNS = [
    "feat_rsi_fast",    # short-lookback RSI (fast momentum)
    "feat_rsi_slow",    # long-lookback RSI (slower momentum)
    "feat_macd",        # MACD line value
    "feat_macd_hist",   # MACD histogram (momentum acceleration)
    "feat_bb_width",    # Bollinger band width / mid -> how volatile right now
    "feat_atr_norm",    # ATR as a fraction of price -> volatility, scale-free
    "feat_ret_1",       # 1-candle return
    "feat_ret_5",       # 5-candle return
    "feat_ret_10",      # 10-candle return
    "feat_vol_ratio",   # volume vs its recent average (1.0 if no real volume)
    "feat_range_norm",  # (high - low) / close -> this candle's range, scale-free
]


@dataclass(frozen=True)
class MLEnsembleConfig:
    """Tunable settings for the ML ensemble strategy."""

    rsi_fast: int = 7          # fast RSI feature lookback
    rsi_slow: int = 14         # slow RSI feature lookback
    macd_fast: int = 12        # MACD fast EMA (feature)
    macd_slow: int = 26        # MACD slow EMA (feature)
    macd_signal: int = 9       # MACD signal EMA (feature)
    bb_period: int = 20        # Bollinger length for the band-width feature
    bb_std: float = 2.0        # Bollinger width for the band-width feature
    atr_period: int = 14       # ATR length for the normalised-volatility feature
    vol_window: int = 20       # window for the volume-ratio feature
    training_window: int = 200 # how many recent labelled candles to train on
    retrain_every: int = 20    # re-fit the model every this many new candles
    forward_bars: int = 5      # label horizon: "is price higher N candles ahead?"
    buy_threshold: float = 0.6 # go long when predicted P(up) is above this
    sell_threshold: float = 0.4# go short when predicted P(up) is below this
    n_estimators: int = 100    # number of trees in the RandomForest
    max_depth: int = 5         # max depth per tree (caps overfitting)
    min_training_rows: int = 50# refuse to train on fewer clean rows than this
    random_state: int = 42     # fixed seed -> reproducible model/predictions
    stop_loss_pct: float = 0.025  # protective stop, 2.5% from entry
    target_pct: float = 0.05      # profit target, 5% from entry

    def __post_init__(self) -> None:
        positive_ints = {
            "rsi_fast": self.rsi_fast,
            "rsi_slow": self.rsi_slow,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "bb_period": self.bb_period,
            "atr_period": self.atr_period,
            "vol_window": self.vol_window,
            "training_window": self.training_window,
            "retrain_every": self.retrain_every,
            "forward_bars": self.forward_bars,
            "n_estimators": self.n_estimators,
            "min_training_rows": self.min_training_rows,
        }
        invalid = [name for name, value in positive_ints.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"Config values must be positive: {', '.join(invalid)}")
        if not (0.0 < float(self.sell_threshold) < float(self.buy_threshold) < 1.0):
            raise ValueError("Require 0 < sell_threshold < buy_threshold < 1.")
        if float(self.stop_loss_pct) <= 0.0 or float(self.target_pct) <= 0.0:
            raise ValueError("stop_loss_pct and target_pct must be greater than zero.")


@dataclass
class MLEnsemblePositionContext:
    direction: str
    entry_underlying: float
    stop_underlying: float
    target_underlying: float


@dataclass
class MLEnsembleDecision:
    action: str = "HOLD"            # HOLD, ENTER_LONG, ENTER_SHORT, or EXIT
    entry_underlying: float = 0.0   # underlying price to enter at (ENTER_* actions)
    stop_underlying: float = 0.0    # protective stop price for the trade
    target_underlying: float = 0.0  # profit-target price (0.0 when not used)
    exit_underlying: float = 0.0    # price the exit triggered at (EXIT action)
    exit_reason: str = ""           # why we exited: e.g. STOP, TARGET
    reason: str = ""                # plain-English reason we entered (e.g. ML_PROB_UP)
    signal_triggered: bool = False  # True if a valid signal fired (even if no order placed)
    debug: dict[str, Any] = field(default_factory=dict)  # optional extras, e.g. prob_up


def build_ml_ensemble_with_indicators(
    ohlc: pd.DataFrame,
    config: Optional[MLEnsembleConfig] = None,
) -> pd.DataFrame:
    """Enrich OHLC candles with the model feature set, the training label, and risk levels."""
    config = config or MLEnsembleConfig()
    frame = normalize_ohlc_frame(ohlc)
    if frame.empty:
        return frame

    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float)

    frame["feat_rsi_fast"] = rsi(close, config.rsi_fast)
    frame["feat_rsi_slow"] = rsi(close, config.rsi_slow)
    macd_line, _, macd_hist = macd(close, config.macd_fast, config.macd_slow, config.macd_signal)
    frame["feat_macd"] = macd_line
    frame["feat_macd_hist"] = macd_hist
    bb_upper, bb_mid, bb_lower = bollinger_bands(close, config.bb_period, config.bb_std)
    frame["feat_bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0.0, np.nan)
    frame["feat_atr_norm"] = atr(frame, config.atr_period) / close.replace(0.0, np.nan)
    frame["feat_ret_1"] = close.pct_change(1)
    frame["feat_ret_5"] = close.pct_change(5)
    frame["feat_ret_10"] = close.pct_change(10)

    vol_mean = volume.rolling(window=config.vol_window, min_periods=config.vol_window).mean()
    vol_ratio = volume / vol_mean.replace(0.0, np.nan)
    # Many index/underlying feeds carry no real volume; fall back to a neutral 1.
    frame["feat_vol_ratio"] = vol_ratio.fillna(1.0)
    frame["feat_range_norm"] = (high - low) / close.replace(0.0, np.nan)

    # Training label: was price higher `forward_bars` candles later? The last
    # `forward_bars` rows are intentionally left NaN (their outcome is unknown).
    forward_return = close.shift(-config.forward_bars) / close - 1.0
    frame["ml_target"] = (forward_return > 0.0).astype("float64")
    frame.loc[forward_return.isna(), "ml_target"] = np.nan

    frame["long_entry_price"] = close
    frame["short_entry_price"] = close
    frame["long_stop_from_setup"] = close * (1.0 - config.stop_loss_pct)
    frame["long_target_from_setup"] = close * (1.0 + config.target_pct)
    frame["short_stop_from_setup"] = close * (1.0 + config.stop_loss_pct)
    frame["short_target_from_setup"] = close * (1.0 - config.target_pct)
    return frame


def _load_random_forest():
    """Import RandomForestClassifier lazily so the dependency is optional."""
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:  # pragma: no cover - exercised only without sklearn
        raise ImportError(
            "The ML Ensemble signal generator requires scikit-learn. "
            "Install it with `pip install scikit-learn` to use this strategy."
        ) from exc
    return RandomForestClassifier


class MLEnsembleSignalEngine:
    """Stateful decision engine: trains/caches a RandomForest and predicts entries."""

    def __init__(self, config: Optional[MLEnsembleConfig] = None) -> None:
        self.config = config or MLEnsembleConfig()
        self._rf_class = _load_random_forest()
        self.model = None
        self._trained_len = -1

    def minimum_history_bars(self) -> int:
        warmup = max(
            self.config.macd_slow + self.config.macd_signal,
            self.config.bb_period,
            self.config.atr_period,
            self.config.rsi_slow,
            10,
            self.config.vol_window,
        )
        return warmup + int(self.config.training_window) + int(self.config.forward_bars) + 5

    @staticmethod
    def _hold(reason: str = "", *, signal_triggered: bool = False) -> MLEnsembleDecision:
        return MLEnsembleDecision(
            action="HOLD",
            signal_triggered=signal_triggered,
            debug={"reason": reason} if reason else {},
        )

    @staticmethod
    def _direction(direction: str) -> str:
        return str(direction).strip().upper()

    def _maybe_train(self, frame: pd.DataFrame) -> None:
        """
        Re-fit the RandomForest on recent history, but only when it is "due".

        Two beginner points:
        - We only retrain every `retrain_every` candles (training is expensive),
          and the very first time the model is needed.
        - Leak-free labels: a candle's label needs `forward_bars` FUTURE candles to
          know the outcome. So we drop the most recent `forward_bars` rows from the
          training set - their future hasn't happened yet from this candle's point
          of view. Training on them would be "cheating" (look-ahead bias).
        """
        n = len(frame)
        need_train = self.model is None or (n - self._trained_len) >= int(self.config.retrain_every)
        if not need_train:
            return
        self._trained_len = n

        # Only rows whose forward outcome is already known by the latest bar are
        # eligible for training -> drop the trailing `forward_bars` rows.
        trainable = frame.iloc[: max(0, n - int(self.config.forward_bars))]
        trainable = trainable.dropna(subset=FEATURE_COLUMNS + ["ml_target"])
        if len(trainable) < int(self.config.min_training_rows):
            return
        trainable = trainable.iloc[-int(self.config.training_window):]

        x = trainable[FEATURE_COLUMNS].to_numpy(dtype="float64")
        y = trainable["ml_target"].to_numpy(dtype="int64")
        if np.unique(y).size < 2:
            # A single-class window cannot train a useful classifier.
            return

        model = self._rf_class(
            n_estimators=int(self.config.n_estimators),
            max_depth=int(self.config.max_depth),
            random_state=int(self.config.random_state),
            n_jobs=1,
        )
        model.fit(x, y)
        self.model = model

    def _predict_up_probability(self, current: pd.Series) -> Optional[float]:
        if self.model is None:
            return None
        features = [current.get(col) for col in FEATURE_COLUMNS]
        if not all(finite(value) for value in features):
            return None
        x = np.asarray(features, dtype="float64").reshape(1, -1)
        proba = self.model.predict_proba(x)[0]
        classes = list(self.model.classes_)
        if 1 in classes:
            return float(proba[classes.index(1)])
        # Model only ever saw the "down" class -> probability of up is ~0.
        return 0.0

    def _evaluate_exit(self, current: pd.Series, position: MLEnsemblePositionContext) -> MLEnsembleDecision:
        direction = self._direction(position.direction)
        high = float(current["high"])
        low = float(current["low"])
        stop = float(position.stop_underlying)
        target = float(position.target_underlying)

        if direction == "LONG":
            if low <= stop:
                return MLEnsembleDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if high >= target:
                return MLEnsembleDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            return self._hold()
        if direction == "SHORT":
            if high >= stop:
                return MLEnsembleDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if low <= target:
                return MLEnsembleDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            return self._hold()
        return self._hold("Unknown position direction.")

    def evaluate_candle(
        self,
        candles_with_indicators: pd.DataFrame,
        position: Optional[MLEnsemblePositionContext] = None,
    ) -> MLEnsembleDecision:
        if candles_with_indicators is None or candles_with_indicators.empty:
            return self._hold("No candles supplied.")
        require_columns(candles_with_indicators, ["open", "high", "low", "close"])
        current = candles_with_indicators.iloc[-1]

        if position is not None:
            return self._evaluate_exit(current, position)

        require_columns(candles_with_indicators, FEATURE_COLUMNS + ["ml_target"])
        if len(candles_with_indicators) < self.minimum_history_bars():
            return self._hold("Insufficient history.")

        self._maybe_train(candles_with_indicators)
        prob_up = self._predict_up_probability(current)
        if prob_up is None:
            return self._hold("Model unavailable or features incomplete.")

        if prob_up > self.config.buy_threshold:
            entry = float(current["long_entry_price"])
            stop = float(current["long_stop_from_setup"])
            target = float(current["long_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and stop < entry < target:
                return MLEnsembleDecision(
                    action="ENTER_LONG",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="ML_PROB_UP",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "prob_up": prob_up, "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid long entry prices.", signal_triggered=True)

        if prob_up < self.config.sell_threshold:
            entry = float(current["short_entry_price"])
            stop = float(current["short_stop_from_setup"])
            target = float(current["short_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and target < entry < stop:
                return MLEnsembleDecision(
                    action="ENTER_SHORT",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="ML_PROB_DOWN",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "prob_up": prob_up, "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid short entry prices.", signal_triggered=True)

        return self._hold("Model probability in neutral band.")


class MLEnsembleSignalGenerator:
    """Convenience wrapper for full-history and latest-candle ML ensemble signals."""

    def __init__(self, config: Optional[MLEnsembleConfig] = None) -> None:
        self.config = config or MLEnsembleConfig()
        self.engine = MLEnsembleSignalEngine(self.config)

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = build_ml_ensemble_with_indicators(data, self.config)
        if frame.empty:
            return frame

        actions: list[str] = []
        entries: list[float] = []
        stops: list[float] = []
        targets: list[float] = []
        stream: list[int] = []
        position: Optional[MLEnsemblePositionContext] = None

        for index in range(len(frame)):
            decision = self.engine.evaluate_candle(frame.iloc[: index + 1], position=position)
            actions.append(decision.action)
            entries.append(decision.entry_underlying)
            stops.append(decision.stop_underlying)
            targets.append(decision.target_underlying)

            if decision.action == "ENTER_LONG":
                stream.append(1)
                position = MLEnsemblePositionContext(
                    "LONG", decision.entry_underlying, decision.stop_underlying, decision.target_underlying
                )
            elif decision.action == "ENTER_SHORT":
                stream.append(-1)
                position = MLEnsemblePositionContext(
                    "SHORT", decision.entry_underlying, decision.stop_underlying, decision.target_underlying
                )
            elif decision.action == "EXIT":
                stream.append(2)
                position = None
            else:
                stream.append(0)

        result = frame.copy()
        result["signalAction"] = actions
        result["entryUnderlying"] = entries
        result["stopUnderlying"] = stops
        result["targetUnderlying"] = targets
        result["backtestStream"] = stream
        return result

    def latest_signal(
        self,
        data: pd.DataFrame,
        position: Optional[MLEnsemblePositionContext] = None,
    ) -> MLEnsembleDecision:
        frame = build_ml_ensemble_with_indicators(data, self.config)
        return self.engine.evaluate_candle(frame, position=position)


def generate_ml_ensemble_signals(
    data: pd.DataFrame,
    config: Optional[MLEnsembleConfig] = None,
) -> pd.DataFrame:
    return MLEnsembleSignalGenerator(config=config).generate(data)


def get_latest_ml_ensemble_signal(
    data: pd.DataFrame,
    config: Optional[MLEnsembleConfig] = None,
    position: Optional[MLEnsemblePositionContext] = None,
) -> MLEnsembleDecision:
    return MLEnsembleSignalGenerator(config=config).latest_signal(data, position=position)


NiftyMLEnsembleSignalGenerator = MLEnsembleSignalGenerator
generate_nifty_ml_ensemble_signals = generate_ml_ensemble_signals
get_latest_nifty_ml_ensemble_signal = get_latest_ml_ensemble_signal


__all__ = [
    "FEATURE_COLUMNS",
    "MLEnsembleConfig",
    "MLEnsemblePositionContext",
    "MLEnsembleDecision",
    "build_ml_ensemble_with_indicators",
    "MLEnsembleSignalEngine",
    "MLEnsembleSignalGenerator",
    "generate_ml_ensemble_signals",
    "get_latest_ml_ensemble_signal",
    "NiftyMLEnsembleSignalGenerator",
    "generate_nifty_ml_ensemble_signals",
    "get_latest_nifty_ml_ensemble_signal",
]
