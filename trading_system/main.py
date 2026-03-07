"""Convenience CLI entry points for backtesting and paper trading."""

from __future__ import annotations

import argparse
import sys

from trading_system.backtesting.backtester import Backtester
from trading_system.paper_trading.paper_trader import PaperTrader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trading system runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest = subparsers.add_parser("backtest", help="Run a historical backtest")
    backtest.add_argument("--symbol", required=True, help="Yahoo Finance ticker symbol")
    backtest.add_argument("--start", help="Start date in YYYY-MM-DD")
    backtest.add_argument("--end", help="End date in YYYY-MM-DD")
    backtest.add_argument(
        "--period",
        help="Recent Yahoo Finance period such as 5d, 30d, or 60d. Recommended for intraday data.",
    )

    paper = subparsers.add_parser("paper", help="Run paper trading loop")
    paper.add_argument("--symbol", required=True, help="Yahoo Finance ticker symbol")
    paper.add_argument("--polling-seconds", type=int, default=60)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "backtest":
        if args.period and (args.start or args.end):
            parser.error("Use either --period or --start/--end for backtests, not both.")

        backtester = Backtester(symbol=args.symbol)
        try:
            _, trades_df, summary = backtester.run(
                start=args.start,
                end=args.end,
                period=args.period,
            )
        except ValueError as exc:
            print(f"Backtest request invalid: {exc}", file=sys.stderr)
            sys.exit(2)

        print(trades_df.tail())
        print(summary)
    elif args.command == "paper":
        trader = PaperTrader(symbol=args.symbol, polling_seconds=args.polling_seconds)
        trader.run_forever()


if __name__ == "__main__":
    main()
