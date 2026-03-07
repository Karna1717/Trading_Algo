"""Position management and risk-control layer for signal approval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from trading_system.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class PositionManagerConfig:
    """Risk controls applied before a signal becomes an executed trade."""

    max_trades_per_day: int = 5
    max_daily_loss: float = 500.0
    allow_only_single_position: bool = True


@dataclass(slots=True)
class PositionApproval:
    """Result of evaluating whether a signal can become a trade."""

    approved: bool
    rejection_reason: Optional[str] = None


@dataclass(slots=True)
class PositionSnapshot:
    """State of the currently open position."""

    trade_type: str
    entry_price: float
    entry_time: pd.Timestamp


class PositionManager:
    """Approve or reject trades based on position state and daily risk limits."""

    def __init__(self, config: Optional[PositionManagerConfig] = None) -> None:
        self.config = config or PositionManagerConfig()
        self.current_position: Optional[PositionSnapshot] = None
        self.entry_price: Optional[float] = None
        self.entry_time: Optional[pd.Timestamp] = None
        self.daily_loss: float = 0.0
        self.trade_count: int = 0
        self._active_day: Optional[pd.Timestamp] = None

    def can_open_trade(self, signal_time: pd.Timestamp) -> PositionApproval:
        """Return whether a new trade can be opened on the given trading day."""
        self._roll_day_if_needed(signal_time)

        if self.config.allow_only_single_position and self.current_position is not None:
            return PositionApproval(False, "open_position_exists")

        if self.daily_loss >= self.config.max_daily_loss:
            return PositionApproval(False, "max_daily_loss_reached")

        if self.trade_count >= self.config.max_trades_per_day:
            return PositionApproval(False, "max_trades_per_day_reached")

        return PositionApproval(True)

    def register_trade_open(
        self,
        trade_type: str,
        entry_price: float,
        entry_time: pd.Timestamp,
    ) -> None:
        """Mark a trade as opened and increment the daily trade counter."""
        self._roll_day_if_needed(entry_time)
        self.current_position = PositionSnapshot(
            trade_type=trade_type,
            entry_price=entry_price,
            entry_time=entry_time,
        )
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.trade_count += 1

    def register_trade_close(self, profit_points: float, exit_time: pd.Timestamp) -> None:
        """Release the position and accumulate realized daily losses."""
        self._roll_day_if_needed(exit_time)
        if profit_points < 0:
            self.daily_loss += abs(profit_points)
        self.current_position = None
        self.entry_price = None
        self.entry_time = None

    def state(self) -> dict[str, object]:
        """Expose current state for dashboards and future multi-asset expansion."""
        return {
            "current_position": None if self.current_position is None else self.current_position.trade_type,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "daily_loss": round(self.daily_loss, 2),
            "trade_count": self.trade_count,
        }

    def _roll_day_if_needed(self, timestamp: pd.Timestamp) -> None:
        trading_day = timestamp.normalize()
        if self._active_day is None or trading_day != self._active_day:
            self._active_day = trading_day
            self.daily_loss = 0.0
            self.trade_count = 0
            logger.info("Position manager reset for trading day %s", trading_day.date())

