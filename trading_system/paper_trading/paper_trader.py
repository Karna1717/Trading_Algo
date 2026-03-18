"""Paper trading runtime using live yfinance candles."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from trading_system.analytics.performance import PerformanceSummary, compute_performance_summary
from trading_system.config import DEFAULT_INTERVAL, TRADE_LOG_PATH
from trading_system.data.yfinance_loader import MarketDataConfig, YFinanceLoader
from trading_system.engine.event_engine import EventEngine, trades_to_dataframe
from trading_system.logging_utils import get_logger
from trading_system.runtime_status import write_runtime_status
from trading_system.strategy.ripple_strategy import RippleStrategy
from trading_system.trade_log_utils import TRADE_LOG_COLUMNS, normalize_trade_log_schema, read_trade_log

logger = get_logger(__name__)


@dataclass(slots=True)
class PaperTraderState:
    """State snapshot for paper trading."""

    latest_signal: str = "NONE"
    open_position: Optional[str] = None


class PaperTrader:
    """Run a polling loop that simulates live trading with delayed candle completion."""

    def __init__(
        self,
        symbol: str,
        interval: str = DEFAULT_INTERVAL,
        polling_seconds: int = 60,
        lookback: str = "5d",
        log_path: Path = TRADE_LOG_PATH,
    ) -> None:
        self.symbol = symbol
        self.polling_seconds = polling_seconds
        self.lookback = lookback
        self.loader = YFinanceLoader(MarketDataConfig(symbol=symbol, interval=interval))
        self.strategy = RippleStrategy()
        self.event_engine = EventEngine(strategy=self.strategy)
        self.log_path = log_path
        self.state = PaperTraderState()

    def run_once(self) -> tuple[pd.DataFrame, pd.DataFrame, PerformanceSummary]:
        """Process the latest completed candles once and append any new trades."""
        self._ensure_trade_log_schema()
        latest_df = self.loader.fetch_latest_candles(lookback=self.lookback)
        prepared_df = self.strategy.prepare_dataframe(latest_df)
        annotated_df, trades = self.event_engine.run(prepared_df)
        trades_df = normalize_trade_log_schema(trades_to_dataframe(trades))

        if not trades_df.empty:
            trades_df["trade_key"] = (
                trades_df["signal_time"].astype(str)
                + "|"
                + trades_df["execution_status"].astype(str)
                + "|"
                + trades_df["signal_type"].astype(str)
                + "|"
                + trades_df["rejection_reason"].fillna("")
            )

        new_trades_df = self._filter_new_trades(trades_df)
        if not new_trades_df.empty:
            self._append_trade_log(new_trades_df.drop(columns=["trade_key"]))

        latest_signal_row = prepared_df.iloc[-1]
        self.state.latest_signal = str(latest_signal_row["signal"])
        self.state.open_position = self._infer_open_position(annotated_df)

        persisted_trades = self._read_trade_log()
        summary = compute_performance_summary(persisted_trades)
        executed_count = int((persisted_trades["execution_status"] == "EXECUTED").sum()) if not persisted_trades.empty else 0
        rejected_count = int((persisted_trades["execution_status"] == "REJECTED").sum()) if not persisted_trades.empty else 0
        write_runtime_status(
            {
                "mode": "paper",
                "symbol": self.symbol,
                "latest_signal": self.state.latest_signal,
                "open_trade": self.state.open_position,
                "total_profit": summary.total_profit,
                "last_trade_count": executed_count,
                "rejected_signal_count": rejected_count,
            }
        )
        return annotated_df, new_trades_df.drop(columns=["trade_key"], errors="ignore"), summary

    def run_forever(self) -> None:
        """Poll continuously for new completed candles."""
        logger.info("Starting paper trader for %s", self.symbol)
        while True:
            try:
                _, new_trades_df, summary = self.run_once()
                logger.info(
                    "Paper trading heartbeat | symbol=%s | new_trades=%s | total_profit=%.2f",
                    self.symbol,
                    len(new_trades_df),
                    summary.total_profit,
                )
            except Exception:
                logger.exception("Paper trader iteration failed for %s", self.symbol)
            time.sleep(self.polling_seconds)

    def _filter_new_trades(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        if trades_df.empty:
            return trades_df
        existing = self._read_trade_log()
        if existing.empty:
            return trades_df.copy()
        existing["trade_key"] = (
            existing["signal_time"].astype(str)
            + "|"
            + existing["execution_status"].astype(str)
            + "|"
            + existing["signal_type"].astype(str)
            + "|"
            + existing["rejection_reason"].fillna("")
        )
        known_keys = set(existing["trade_key"])
        return trades_df[~trades_df["trade_key"].isin(known_keys)].copy()

    def _append_trade_log(self, trades_df: pd.DataFrame) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        header = not self.log_path.exists()
        normalized = normalize_trade_log_schema(trades_df)
        normalized.to_csv(self.log_path, mode="a", index=False, header=header)

    def _read_trade_log(self) -> pd.DataFrame:
        return read_trade_log(self.log_path)

    def _ensure_trade_log_schema(self) -> None:
        """Upgrade any legacy trade log to the current schema before appending."""
        if not self.log_path.exists():
            return
        normalized = read_trade_log(self.log_path)
        normalized.to_csv(self.log_path, index=False, columns=TRADE_LOG_COLUMNS)

    @staticmethod
    def _infer_open_position(annotated_df: pd.DataFrame) -> Optional[str]:
        if annotated_df.empty:
            return None
        last_state = str(annotated_df.iloc[-1]["position_state"])
        return None if last_state == "FLAT" else last_state
