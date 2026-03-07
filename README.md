# Trading System

Modular algorithmic trading platform in Python with:

- Historical backtesting using `yfinance`
- Live paper trading using latest completed 3 minute candles
- Broker execution adapter placeholders
- CSV trade logging for Excel or Power BI
- Streamlit dashboard for monitoring performance

## Folder Structure

```text
trading_system/
    analytics/
    backtesting/
    dashboard/
    data/
    engine/
    execution/
    logs/
    paper_trading/
    strategy/
    config.py
    logging_utils.py
    main.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Run A Backtest

```bash
python -m trading_system.main backtest --symbol ^NSEI --period 5d
```

For the internal `3m` timeframe, the system downloads `1m` Yahoo Finance data and resamples it to 3-minute candles. That also means history is limited to the recent `1m` retention window.

If you want explicit dates, keep them recent:

```bash
python -m trading_system.main backtest --symbol ^NSEI --start 2026-03-01 --end 2026-03-06
```

Trade logs are written to `trading_system/logs/trade_log.csv`.

## Run Paper Trading

```bash
python -m trading_system.main paper --symbol ^NSEI --polling-seconds 60
```

## Run Dashboard

```bash
streamlit run trading_system/dashboard/streamlit_app.py
```

## Notes

- The system drops the most recent incomplete 3 minute candle before signal generation.
- Entries happen on the next candle open to avoid look-ahead bias.
- Broker execution modules are placeholders for Zerodha, Angel One, and Upstox integration.
