"""Helpers for keeping trade logs backward compatible across schema changes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_system.config import TRADE_LOG_PATH

TRADE_LOG_COLUMNS = [
    "signal_time",
    "signal_type",
    "execution_status",
    "rejection_reason",
    "entry_time",
    "exit_time",
    "trade_type",
    "entry_price",
    "exit_price",
    "profit_points",
    "exit_reason",
    "ema_cross",
    "strong_body",
    "wick_condition",
    "session_condition",
]


def normalize_trade_log_schema(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Map legacy trade logs into the current schema and ensure all columns exist."""
    normalized = trades_df.copy()

    if "signal_time" not in normalized.columns:
        normalized["signal_time"] = pd.NaT
    if "signal_type" not in normalized.columns and "trade_type" in normalized.columns:
        normalized["signal_type"] = normalized["trade_type"]
    if "signal_type" not in normalized.columns:
        normalized["signal_type"] = pd.NA
    if "execution_status" not in normalized.columns:
        normalized["execution_status"] = "EXECUTED"
    if "rejection_reason" not in normalized.columns:
        normalized["rejection_reason"] = pd.NA

    for column in TRADE_LOG_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA

    for column in ["entry_time", "exit_time", "signal_time"]:
        normalized[column] = pd.to_datetime(normalized[column], errors="coerce")

    return normalized[TRADE_LOG_COLUMNS].copy()


def read_trade_log(log_path: Path = TRADE_LOG_PATH) -> pd.DataFrame:
    """Read a trade log and normalize its schema."""
    if not log_path.exists():
        return pd.DataFrame(columns=TRADE_LOG_COLUMNS)
    raw_df = pd.read_csv(log_path)
    return normalize_trade_log_schema(raw_df)

