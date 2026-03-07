"""Candle-by-candle event engine for backtesting and paper trading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional

import pandas as pd

from trading_system.config import STOP_LOSS_POINTS, TAKE_PROFIT_POINTS
from trading_system.engine.position_manager import PositionManager, PositionManagerConfig
from trading_system.logging_utils import get_logger
from trading_system.strategy.ripple_strategy import RippleStrategy

logger = get_logger(__name__)

TradeType = Literal["LONG", "SHORT"]
ExecutionStatus = Literal["EXECUTED", "REJECTED"]


@dataclass(slots=True)
class PendingEntry:
    """Signal accepted on one candle to be entered at the next candle open."""

    trade_type: TradeType
    signal_time: pd.Timestamp
    debug: dict[str, bool]


@dataclass(slots=True)
class PendingExit:
    """Exit condition detected on one candle to be executed at the next candle open."""

    exit_reason: str
    detected_time: pd.Timestamp


@dataclass(slots=True)
class Position:
    """Open paper or backtest position."""

    trade_type: TradeType
    entry_time: pd.Timestamp
    entry_price: float
    tp_price: float
    sl_price: float
    signal_time: pd.Timestamp
    debug: dict[str, bool]


@dataclass(slots=True)
class TradeRecord:
    """Normalized execution or rejection record for logging and analytics."""

    signal_time: pd.Timestamp
    signal_type: TradeType
    execution_status: ExecutionStatus
    rejection_reason: Optional[str]
    entry_time: Optional[pd.Timestamp]
    exit_time: Optional[pd.Timestamp]
    trade_type: TradeType
    entry_price: Optional[float]
    exit_price: Optional[float]
    profit_points: float
    exit_reason: Optional[str]
    ema_cross: bool
    strong_body: bool
    wick_condition: bool
    session_condition: bool


class EventEngine:
    """Sequentially process candles and simulate order lifecycle without look-ahead."""

    def __init__(
        self,
        strategy: Optional[RippleStrategy] = None,
        position_manager: Optional[PositionManager] = None,
        take_profit_points: float = TAKE_PROFIT_POINTS,
        stop_loss_points: float = STOP_LOSS_POINTS,
    ) -> None:
        self.strategy = strategy or RippleStrategy()
        self.position_manager_config = (
            position_manager.config if position_manager is not None else PositionManagerConfig()
        )
        self.take_profit_points = take_profit_points
        self.stop_loss_points = stop_loss_points

    def run(self, prepared_df: pd.DataFrame) -> tuple[pd.DataFrame, list[TradeRecord]]:
        """Process candles sequentially and return annotated candles plus trade records."""
        position_manager = PositionManager(self.position_manager_config)
        df = prepared_df.copy()
        df["position_state"] = "FLAT"
        df["entry_marker"] = pd.NA
        df["exit_marker"] = pd.NA
        df["event_note"] = ""

        trades: list[TradeRecord] = []
        position: Optional[Position] = None
        pending_entry: Optional[PendingEntry] = None
        pending_exit: Optional[PendingExit] = None
        seen_signal_times: set[pd.Timestamp] = set()

        for idx in range(len(df)):
            candle = df.iloc[idx]
            timestamp = df.index[idx]

            if pending_entry is not None and position is None:
                position = Position(
                    trade_type=pending_entry.trade_type,
                    entry_time=timestamp,
                    entry_price=float(candle["open"]),
                    tp_price=self._calculate_take_profit(
                        pending_entry.trade_type,
                        float(candle["open"]),
                    ),
                    sl_price=self._calculate_stop_loss(
                        pending_entry.trade_type,
                        float(candle["open"]),
                    ),
                    signal_time=pending_entry.signal_time,
                    debug=pending_entry.debug,
                )
                position_manager.register_trade_open(
                    trade_type=pending_entry.trade_type,
                    entry_price=float(candle["open"]),
                    entry_time=timestamp,
                )
                df.at[timestamp, "position_state"] = pending_entry.trade_type
                df.at[timestamp, "entry_marker"] = float(candle["open"])
                df.at[timestamp, "event_note"] = (
                    f"Entered on next candle open | TP={position.tp_price:.2f} | SL={position.sl_price:.2f}"
                )
                pending_entry = None

            if position is not None:
                df.at[timestamp, "position_state"] = position.trade_type
                if pending_exit is not None:
                    exit_price = float(candle["open"])
                    record = self._close_position(
                        position,
                        timestamp,
                        exit_price,
                        pending_exit.exit_reason,
                    )
                    position_manager.register_trade_close(record.profit_points, timestamp)
                    trades.append(record)
                    df.at[timestamp, "exit_marker"] = exit_price
                    df.at[timestamp, "event_note"] = (
                        f"{pending_exit.exit_reason} executed on next candle open"
                    )
                    position = None
                    pending_exit = None
                    continue

                exit_reason = self._check_exit(df, idx, position)
                if exit_reason is not None:
                    if idx < len(df) - 1:
                        pending_exit = PendingExit(
                            exit_reason=exit_reason,
                            detected_time=timestamp,
                        )
                        df.at[timestamp, "event_note"] = (
                            f"{exit_reason} detected; exit queued for next candle open"
                        )
                    else:
                        last_close = float(candle["close"])
                        record = self._close_position(position, timestamp, last_close, "END_OF_DATA")
                        position_manager.register_trade_close(record.profit_points, timestamp)
                        trades.append(
                            record
                        )
                        df.at[timestamp, "exit_marker"] = last_close
                        df.at[timestamp, "event_note"] = "end_of_data"
                        position = None
                    continue

            if position is None and pending_entry is None and pending_exit is None and idx < len(df) - 1:
                signal_payload = self.strategy.generate_signal_for_candle(df, idx)
                if signal_payload.signal in {"LONG", "SHORT"}:
                    if signal_payload.signal_candle_time in seen_signal_times:
                        logger.warning("Duplicate signal ignored at %s", signal_payload.signal_candle_time)
                    else:
                        seen_signal_times.add(signal_payload.signal_candle_time)
                        approval = position_manager.can_open_trade(
                            signal_time=signal_payload.signal_candle_time
                        )
                        if approval.approved:
                            pending_entry = PendingEntry(
                                trade_type=signal_payload.signal,
                                signal_time=signal_payload.signal_candle_time,
                                debug=signal_payload.debug,
                            )
                            df.at[timestamp, "event_note"] = "Signal approved"
                        else:
                            trades.append(
                                self._build_rejected_trade_record(
                                    signal_time=signal_payload.signal_candle_time,
                                    signal_type=signal_payload.signal,
                                    rejection_reason=approval.rejection_reason or "rejected",
                                    debug=signal_payload.debug,
                                )
                            )
                            df.at[timestamp, "event_note"] = (
                                f"Signal rejected: {approval.rejection_reason}"
                            )

        if position is not None:
            last_timestamp = df.index[-1]
            last_close = float(df.iloc[-1]["close"])
            record = self._close_position(position, last_timestamp, last_close, "END_OF_DATA")
            position_manager.register_trade_close(record.profit_points, last_timestamp)
            trades.append(record)
            df.at[last_timestamp, "exit_marker"] = last_close
            df.at[last_timestamp, "event_note"] = "end_of_data"

        return df, trades

    def _check_exit(
        self,
        df: pd.DataFrame,
        idx: int,
        position: Position,
    ) -> Optional[str]:
        candle = df.iloc[idx]

        if position.trade_type == "LONG":
            stop_hit = float(candle["low"]) <= position.sl_price
            target_hit = float(candle["high"]) >= position.tp_price
            if stop_hit and target_hit:
                return "SL"
            if stop_hit:
                return "SL"
            if target_hit:
                return "TP"
        else:
            target_hit = float(candle["low"]) <= position.tp_price
            stop_hit = float(candle["high"]) >= position.sl_price
            if stop_hit and target_hit:
                return "SL"
            if stop_hit:
                return "SL"
            if target_hit:
                return "TP"

        return None

    def _close_position(
        self,
        position: Position,
        exit_time: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
    ) -> TradeRecord:
        profit = (
            exit_price - position.entry_price
            if position.trade_type == "LONG"
            else position.entry_price - exit_price
        )
        return TradeRecord(
            signal_time=position.signal_time,
            signal_type=position.trade_type,
            execution_status="EXECUTED",
            rejection_reason=None,
            entry_time=position.entry_time,
            exit_time=exit_time,
            trade_type=position.trade_type,
            entry_price=round(position.entry_price, 2),
            exit_price=round(exit_price, 2),
            profit_points=round(profit, 2),
            exit_reason=exit_reason,
            ema_cross=position.debug["ema_cross"],
            strong_body=position.debug["strong_body"],
            wick_condition=position.debug["wick_condition"],
            session_condition=position.debug["session_condition"],
        )

    def _calculate_take_profit(self, trade_type: TradeType, entry_price: float) -> float:
        if trade_type == "LONG":
            return entry_price + self.take_profit_points
        return entry_price - self.take_profit_points

    def _calculate_stop_loss(self, trade_type: TradeType, entry_price: float) -> float:
        if trade_type == "LONG":
            return entry_price - self.stop_loss_points
        return entry_price + self.stop_loss_points

    def _build_rejected_trade_record(
        self,
        signal_time: pd.Timestamp,
        signal_type: TradeType,
        rejection_reason: str,
        debug: dict[str, bool],
    ) -> TradeRecord:
        return TradeRecord(
            signal_time=signal_time,
            signal_type=signal_type,
            execution_status="REJECTED",
            rejection_reason=rejection_reason,
            entry_time=None,
            exit_time=None,
            trade_type=signal_type,
            entry_price=None,
            exit_price=None,
            profit_points=0.0,
            exit_reason=None,
            ema_cross=debug["ema_cross"],
            strong_body=debug["strong_body"],
            wick_condition=debug["wick_condition"],
            session_condition=debug["session_condition"],
        )


def trades_to_dataframe(trades: list[TradeRecord]) -> pd.DataFrame:
    """Convert trade records into a DataFrame for CSV logging and analytics."""
    if not trades:
        return pd.DataFrame(
            columns=[
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
        )
    return pd.DataFrame(asdict(trade) for trade in trades)
