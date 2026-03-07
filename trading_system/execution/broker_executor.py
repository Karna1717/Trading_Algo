"""Broker execution interfaces for future live trading integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

TradeSide = Literal["BUY", "SELL"]


@dataclass(slots=True)
class OrderRequest:
    """Broker-agnostic order payload."""

    symbol: str
    side: TradeSide
    quantity: int
    order_type: str = "MARKET"
    price: Optional[float] = None


class BaseBrokerExecutor(ABC):
    """Abstract broker adapter contract."""

    broker_name: str

    @abstractmethod
    def connect(self) -> None:
        """Initialize broker session."""

    @abstractmethod
    def place_order(self, order: OrderRequest) -> str:
        """Place an order and return a broker order id."""

    @abstractmethod
    def close_position(self, symbol: str) -> None:
        """Close any open position for the symbol."""


class ZerodhaKiteExecutor(BaseBrokerExecutor):
    broker_name = "Zerodha Kite"

    def connect(self) -> None:
        raise NotImplementedError("Zerodha Kite integration will be added in a future release.")

    def place_order(self, order: OrderRequest) -> str:
        raise NotImplementedError("Zerodha Kite integration will be added in a future release.")

    def close_position(self, symbol: str) -> None:
        raise NotImplementedError("Zerodha Kite integration will be added in a future release.")


class AngelOneExecutor(BaseBrokerExecutor):
    broker_name = "Angel One SmartAPI"

    def connect(self) -> None:
        raise NotImplementedError("Angel One SmartAPI integration will be added in a future release.")

    def place_order(self, order: OrderRequest) -> str:
        raise NotImplementedError("Angel One SmartAPI integration will be added in a future release.")

    def close_position(self, symbol: str) -> None:
        raise NotImplementedError("Angel One SmartAPI integration will be added in a future release.")


class UpstoxExecutor(BaseBrokerExecutor):
    broker_name = "Upstox"

    def connect(self) -> None:
        raise NotImplementedError("Upstox integration will be added in a future release.")

    def place_order(self, order: OrderRequest) -> str:
        raise NotImplementedError("Upstox integration will be added in a future release.")

    def close_position(self, symbol: str) -> None:
        raise NotImplementedError("Upstox integration will be added in a future release.")

