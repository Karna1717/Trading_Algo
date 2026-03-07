"""Logging helpers for the trading system."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from trading_system.config import APP_LOG_PATH, LOGS_DIR


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with console and file handlers."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = RotatingFileHandler(APP_LOG_PATH, maxBytes=1_000_000, backupCount=3)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger

