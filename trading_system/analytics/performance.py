"""Performance analytics for backtests and paper trading."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(slots=True)
class PerformanceSummary:
    """Aggregated trading performance metrics."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    bullish_profit: float
    bearish_profit: float
    maximum_drawdown: float
    profit_factor: float


def compute_equity_curve(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Create an equity curve DataFrame from trade records."""
    if "execution_status" in trades_df.columns:
        trades_df = trades_df[trades_df["execution_status"] == "EXECUTED"].copy()

    if trades_df.empty:
        return pd.DataFrame(columns=["exit_time", "equity", "drawdown"])

    curve = trades_df.copy().sort_values("exit_time")
    curve["equity"] = curve["profit_points"].cumsum()
    curve["peak_equity"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] - curve["peak_equity"]
    return curve[["exit_time", "equity", "drawdown"]]


def compute_performance_summary(trades_df: pd.DataFrame) -> PerformanceSummary:
    """Compute the requested trade performance metrics."""
    if "execution_status" in trades_df.columns:
        trades_df = trades_df[trades_df["execution_status"] == "EXECUTED"].copy()

    if trades_df.empty:
        return PerformanceSummary(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    winning = trades_df[trades_df["profit_points"] > 0]
    losing = trades_df[trades_df["profit_points"] <= 0]

    gross_profit = winning["profit_points"].sum()
    gross_loss = losing["profit_points"].abs().sum()
    equity_curve = compute_equity_curve(trades_df)

    return PerformanceSummary(
        total_trades=int(len(trades_df)),
        winning_trades=int(len(winning)),
        losing_trades=int(len(losing)),
        win_rate=round((len(winning) / len(trades_df)) * 100, 2),
        total_profit=round(float(trades_df["profit_points"].sum()), 2),
        bullish_profit=round(
            float(trades_df.loc[trades_df["trade_type"] == "LONG", "profit_points"].sum()),
            2,
        ),
        bearish_profit=round(
            float(trades_df.loc[trades_df["trade_type"] == "SHORT", "profit_points"].sum()),
            2,
        ),
        maximum_drawdown=round(float(equity_curve["drawdown"].min()), 2),
        profit_factor=round(float(gross_profit / gross_loss), 2) if gross_loss else float("inf"),
    )


def plot_equity_curve(trades_df: pd.DataFrame) -> plt.Figure:
    """Return a matplotlib figure for the equity curve."""
    curve = compute_equity_curve(trades_df)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(curve["exit_time"], curve["equity"], label="Equity Curve", linewidth=2)
    ax.fill_between(curve["exit_time"], curve["drawdown"], 0, alpha=0.2, label="Drawdown")
    ax.set_title("Equity Curve")
    ax.set_xlabel("Exit Time")
    ax.set_ylabel("Profit Points")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
