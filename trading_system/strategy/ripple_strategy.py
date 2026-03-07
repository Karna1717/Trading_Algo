"""Ripple EMA crossover strategy with candle filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from trading_system.config import SESSION_END, SESSION_START
from trading_system.strategy.indicators import add_ema_indicators

SignalType = Literal["LONG", "SHORT", "NONE"]


@dataclass(slots=True)
class StrategySignal:
    """Signal payload produced by the strategy engine."""

    signal: SignalType
    signal_candle_time: pd.Timestamp
    debug: dict[str, bool]


class RippleStrategy:
    """Generate entry signals for the Ripple EMA strategy."""

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators and all debug rule columns."""
        enriched = add_ema_indicators(df)

        prev_ema5 = enriched["ema5"].shift(1)
        prev_ema13 = enriched["ema13"].shift(1)
        prev_ema26 = enriched["ema26"].shift(1)

        long_cross = (
            ((prev_ema5 <= prev_ema13) & (enriched["ema5"] > enriched["ema13"]))
            | ((prev_ema5 <= prev_ema26) & (enriched["ema5"] > enriched["ema26"]))
            | ((prev_ema13 <= prev_ema26) & (enriched["ema13"] > enriched["ema26"]))
        )
        short_cross = (
            ((prev_ema5 >= prev_ema13) & (enriched["ema5"] < enriched["ema13"]))
            | ((prev_ema5 >= prev_ema26) & (enriched["ema5"] < enriched["ema26"]))
            | ((prev_ema13 >= prev_ema26) & (enriched["ema13"] < enriched["ema26"]))
        )

        candle_range = enriched["high"] - enriched["low"]
        candle_body = (enriched["close"] - enriched["open"]).abs()
        safe_range = candle_range.replace(0, np.nan)
        strong_body = (candle_body >= 0.5 * safe_range) & (candle_body < 25)
        strong_body = strong_body.fillna(False)

        bullish = enriched["close"] > enriched["open"]
        bearish = enriched["close"] < enriched["open"]

        long_wick = (enriched["close"] - enriched["low"]) < 25
        short_wick = (enriched["high"] - enriched["close"]) < 25

        times = pd.Series(enriched.index.strftime("%H:%M"), index=enriched.index)
        session_condition = (times >= SESSION_START) & (times <= SESSION_END)

        enriched["ema_cross"] = long_cross | short_cross
        enriched["strong_body"] = strong_body
        enriched["session_condition"] = session_condition

        long_setup = long_cross & strong_body & bullish & session_condition
        short_setup = short_cross & strong_body & bearish & session_condition

        long_signal = long_setup & long_wick
        short_signal = short_setup & short_wick

        enriched["wick_condition"] = np.select(
            [long_cross, short_cross],
            [long_wick, short_wick],
            default=False,
        )
        enriched["signal"] = np.select(
            [long_signal, short_signal],
            ["LONG", "SHORT"],
            default="NONE",
        )
        return enriched

    def generate_signal_for_candle(self, df: pd.DataFrame, candle_index: int) -> StrategySignal:
        """Return the strategy decision for a specific candle index."""
        candle = df.iloc[candle_index]
        debug = {
            "ema_cross": bool(candle["ema_cross"]),
            "strong_body": bool(candle["strong_body"]),
            "wick_condition": bool(candle["wick_condition"]),
            "session_condition": bool(candle["session_condition"]),
        }
        return StrategySignal(
            signal=candle["signal"],
            signal_candle_time=df.index[candle_index],
            debug=debug,
        )
