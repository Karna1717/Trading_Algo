"""Shared configuration values for the trading system."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
TRADE_LOG_PATH = LOGS_DIR / "trade_log.csv"
APP_LOG_PATH = LOGS_DIR / "system.log"
STATUS_PATH = LOGS_DIR / "runtime_status.json"

DEFAULT_INTERVAL = "5m"
DEFAULT_PERIOD = "60d"
MARKET_TIMEZONE = "Asia/Kolkata"

SESSION_START = "09:18"
SESSION_END = "15:21"

TAKE_PROFIT_POINTS = 30.0
STOP_LOSS_POINTS = 12.0
