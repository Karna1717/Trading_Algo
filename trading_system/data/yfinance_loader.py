"""Market data loader built on top of yfinance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf

from trading_system.config import DEFAULT_INTERVAL, DEFAULT_PERIOD, MARKET_TIMEZONE
from trading_system.logging_utils import get_logger

logger = get_logger(__name__)

INTRADAY_LOOKBACK_LIMIT_DAYS = {
    "1m": 7,
    "2m": 60,
    "3m": 7,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "90m": 60,
}


@dataclass(slots=True)
class MarketDataConfig:
    """Configuration for yfinance market data downloads."""

    symbol: str
    interval: str = DEFAULT_INTERVAL
    period: str = DEFAULT_PERIOD
    auto_adjust: bool = False
    prepost: bool = False


class YFinanceLoader:
    """Load historical and latest OHLCV candles from Yahoo Finance."""

    def __init__(self, config: MarketDataConfig) -> None:
        self.config = config

    def load_historical_data(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: Optional[str] = None,
    ) -> pd.DataFrame:
        """Download historical candles and drop the most recent incomplete candle."""
        self._validate_request_window(start=start, end=end, period=period)
        download_interval = self._effective_yfinance_interval()
        try:
            df = yf.download(
                tickers=self.config.symbol,
                start=start,
                end=end,
                period=period or (None if start or end else self.config.period),
                interval=download_interval,
                auto_adjust=self.config.auto_adjust,
                prepost=self.config.prepost,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            logger.exception("Historical data download failed for %s", self.config.symbol)
            raise RuntimeError("Failed to load historical data") from exc

        cleaned = self._normalize_dataframe(df)
        cleaned = self._resample_if_needed(cleaned)
        return self._drop_incomplete_candle(cleaned)

    def fetch_latest_candles(self, lookback: str = "2d") -> pd.DataFrame:
        """Fetch recent candles for paper trading and exclude the unfinished candle."""
        download_interval = self._effective_yfinance_interval()
        try:
            df = yf.download(
                tickers=self.config.symbol,
                period=lookback,
                interval=download_interval,
                auto_adjust=self.config.auto_adjust,
                prepost=self.config.prepost,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            logger.exception("Latest candle fetch failed for %s", self.config.symbol)
            raise RuntimeError("Failed to fetch latest candles") from exc

        cleaned = self._normalize_dataframe(df)
        cleaned = self._resample_if_needed(cleaned)
        return self._drop_incomplete_candle(cleaned)

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            logger.warning("No data returned for %s", self.config.symbol)
            raise ValueError(f"No data returned for {self.config.symbol}")

        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(-1, axis=1)

        normalized = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )[["open", "high", "low", "close", "volume"]].copy()

        if isinstance(normalized.index, pd.DatetimeIndex):
            if normalized.index.tz is None:
                normalized.index = normalized.index.tz_localize("UTC")
            normalized.index = normalized.index.tz_convert(MARKET_TIMEZONE)

        normalized = normalized[~normalized.index.duplicated(keep="last")]
        normalized = normalized.dropna(subset=["open", "high", "low", "close"])
        normalized.sort_index(inplace=True)
        return normalized

    def _drop_incomplete_candle(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        last_ts = df.index[-1]
        if not isinstance(last_ts, pd.Timestamp):
            return df.iloc[:-1]

        now = pd.Timestamp.now(tz=MARKET_TIMEZONE)
        interval_minutes = self._parse_interval_minutes(self.config.interval)
        candle_end = last_ts + pd.Timedelta(minutes=interval_minutes)

        if candle_end > now:
            logger.info("Dropping unfinished candle at %s for %s", last_ts, self.config.symbol)
            return df.iloc[:-1].copy()

        return df.copy()

    @staticmethod
    def _parse_interval_minutes(interval: str) -> int:
        if not interval.endswith("m"):
            raise ValueError(f"Unsupported interval: {interval}")
        return int(interval[:-1])

    def _effective_yfinance_interval(self) -> str:
        """Map unsupported internal intervals to the nearest yfinance source interval."""
        if self.config.interval == "3m":
            return "1m"
        return self.config.interval

    def _resample_if_needed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample source candles when the requested interval is not natively supported."""
        if self.config.interval != "3m":
            return df

        resampled = (
            df.resample("3min", label="left", closed="left")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna(subset=["open", "high", "low", "close"])
        )
        return resampled

    def _validate_request_window(
        self,
        start: Optional[str],
        end: Optional[str],
        period: Optional[str],
    ) -> None:
        if period or not self.config.interval.endswith("m") or (start is None and end is None):
            return

        now = pd.Timestamp.now(tz=MARKET_TIMEZONE)
        end_ts = pd.Timestamp(end, tz=MARKET_TIMEZONE) if end else now
        lookback_limit_days = self._intraday_lookback_limit_days()
        start_ts = (
            pd.Timestamp(start, tz=MARKET_TIMEZONE)
            if start
            else end_ts - pd.Timedelta(days=lookback_limit_days)
        )

        lookback_days = (now - start_ts).total_seconds() / 86400
        requested_span_days = (end_ts - start_ts).total_seconds() / 86400

        if lookback_days > lookback_limit_days or requested_span_days > lookback_limit_days:
            raise ValueError(
                f"Yahoo Finance only provides data for {self.config.interval} requests over roughly the last "
                f"{lookback_limit_days} days. Requested range {start_ts.date()} to "
                f"{end_ts.date()} is not available. Use a recent window or pass --period 5d."
            )

    def _intraday_lookback_limit_days(self) -> int:
        source_interval = self._effective_yfinance_interval()
        return INTRADAY_LOOKBACK_LIMIT_DAYS.get(source_interval, 60)
