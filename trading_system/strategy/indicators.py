"""Indicator calculation utilities."""

from __future__ import annotations

import pandas as pd


def add_ema_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add Ripple strategy EMA indicators using vectorized pandas operations."""
    enriched = df.copy()
    enriched["ema5"] = enriched["close"].ewm(span=5, adjust=False).mean()
    enriched["ema13"] = enriched["close"].ewm(span=13, adjust=False).mean()
    enriched["ema26"] = enriched["close"].ewm(span=26, adjust=False).mean()
    return enriched

