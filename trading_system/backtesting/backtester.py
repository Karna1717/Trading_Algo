"""Backtesting runtime for the Ripple EMA trading system."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from trading_system.analytics.performance import (
    PerformanceSummary,
    compute_performance_summary,
)
from trading_system.config import DEFAULT_INTERVAL, TRADE_LOG_PATH
from trading_system.data.yfinance_loader import MarketDataConfig, YFinanceLoader
from trading_system.engine.position_manager import PositionManager, PositionManagerConfig
from trading_system.engine.event_engine import EventEngine, trades_to_dataframe
from trading_system.logging_utils import get_logger
from trading_system.runtime_status import write_runtime_status
from trading_system.strategy.ripple_strategy import RippleStrategy

logger = get_logger(__name__)


class Backtester:
    """Run historical backtests and persist trade logs."""

    def __init__(
        self,
        symbol: str,
        interval: str = DEFAULT_INTERVAL,
        position_manager_config: Optional[PositionManagerConfig] = None,
        log_path: Path = TRADE_LOG_PATH,
    ) -> None:
        self.symbol = symbol
        self.loader = YFinanceLoader(MarketDataConfig(symbol=symbol, interval=interval))
        self.strategy = RippleStrategy()
        self.position_manager_config = position_manager_config or PositionManagerConfig()
        self.event_engine = EventEngine(
            strategy=self.strategy,
            position_manager=PositionManager(self.position_manager_config),
        )
        self.log_path = log_path

    def run(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: Optional[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, PerformanceSummary]:
        """Run a full historical simulation from data download to analytics."""
        raw_df = self.loader.load_historical_data(start=start, end=end, period=period)
        prepared_df = self.strategy.prepare_dataframe(raw_df)
        annotated_df, trades = self.event_engine.run(prepared_df)
        trades_df = trades_to_dataframe(trades)
        self._write_trade_log(trades_df)
        summary = compute_performance_summary(trades_df)
        executed_count = int((trades_df["execution_status"] == "EXECUTED").sum()) if not trades_df.empty else 0
        rejected_count = int((trades_df["execution_status"] == "REJECTED").sum()) if not trades_df.empty else 0
        write_runtime_status(
            {
                "mode": "backtest",
                "symbol": self.symbol,
                "latest_signal": str(prepared_df.iloc[-1]["signal"]) if not prepared_df.empty else "NONE",
                "open_trade": None,
                "total_profit": summary.total_profit,
                "last_trade_count": executed_count,
                "rejected_signal_count": rejected_count,
            }
        )
        logger.info(
            "Backtest finished for %s with %s trades and total profit %.2f",
            self.symbol,
            len(trades_df),
            summary.total_profit,
        )
        return annotated_df, trades_df, summary

    def _write_trade_log(self, trades_df: pd.DataFrame) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(self.log_path, index=False)
