"""Streamlit dashboard for monitoring strategy outputs."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_system.analytics.performance import compute_equity_curve, compute_performance_summary
from trading_system.config import DEFAULT_INTERVAL, TRADE_LOG_PATH
from trading_system.data.yfinance_loader import MarketDataConfig, YFinanceLoader
from trading_system.runtime_status import read_runtime_status
from trading_system.trade_log_utils import read_trade_log


def load_trade_log(log_path: Path = TRADE_LOG_PATH) -> pd.DataFrame:
    """Load the persisted trade log."""
    return read_trade_log(log_path)


def filter_executed_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Return only executed trades for charts and performance metrics."""
    if trades_df.empty or "execution_status" not in trades_df.columns:
        return trades_df
    return trades_df[trades_df["execution_status"] == "EXECUTED"].copy()


@st.cache_data(ttl=60, show_spinner=False)
def load_recent_candles(symbol: str, lookback: str) -> pd.DataFrame:
    """Load recent completed candles for the dashboard symbol."""
    loader = YFinanceLoader(MarketDataConfig(symbol=symbol, interval=DEFAULT_INTERVAL))
    return loader.fetch_latest_candles(lookback=lookback)


def build_trade_scatter(trades_df: pd.DataFrame) -> go.Figure:
    """Build an entry/exit marker chart from trade history."""
    fig = go.Figure()
    if trades_df.empty:
        fig.update_layout(title="Trade History", xaxis_title="Time", yaxis_title="Price")
        return fig

    fig.add_trace(
        go.Scatter(
            x=trades_df["entry_time"],
            y=trades_df["entry_price"],
            mode="markers",
            name="Entries",
            marker=dict(symbol="triangle-up", size=11, color="#1b9e77"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trades_df["exit_time"],
            y=trades_df["exit_price"],
            mode="markers",
            name="Exits",
            marker=dict(symbol="x", size=10, color="#d95f02"),
        )
    )
    fig.update_layout(title="Trade Entries and Exits", xaxis_title="Time", yaxis_title="Price")
    return fig


def build_candlestick_chart(candles_df: pd.DataFrame, trades_df: pd.DataFrame) -> go.Figure:
    """Build a candlestick chart with trade entry and exit overlays."""
    fig = go.Figure()
    if candles_df.empty:
        fig.update_layout(title="Recent Price Action", xaxis_title="Time", yaxis_title="Price")
        return fig

    fig.add_trace(
        go.Candlestick(
            x=candles_df.index,
            open=candles_df["open"],
            high=candles_df["high"],
            low=candles_df["low"],
            close=candles_df["close"],
            name="Price",
            increasing_line_color="#1b9e77",
            decreasing_line_color="#d95f02",
        )
    )

    if not trades_df.empty:
        fig.add_trace(
            go.Scatter(
                x=trades_df["entry_time"],
                y=trades_df["entry_price"],
                mode="markers",
                name="Long/Short Entries",
                marker=dict(
                    symbol="triangle-up",
                    size=11,
                    color=trades_df["trade_type"].map({"LONG": "#2ca02c", "SHORT": "#1f77b4"}),
                    line=dict(width=1, color="#1f1f1f"),
                ),
                customdata=trades_df[["trade_type", "profit_points", "exit_reason"]],
                hovertemplate=(
                    "Entry: %{x}<br>"
                    "Price: %{y}<br>"
                    "Type: %{customdata[0]}<br>"
                    "PnL: %{customdata[1]}<br>"
                    "Exit: %{customdata[2]}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=trades_df["exit_time"],
                y=trades_df["exit_price"],
                mode="markers",
                name="Exits",
                marker=dict(symbol="x", size=10, color="#c1121f", line=dict(width=1, color="#1f1f1f")),
                customdata=trades_df[["trade_type", "profit_points", "exit_reason"]],
                hovertemplate=(
                    "Exit: %{x}<br>"
                    "Price: %{y}<br>"
                    "Type: %{customdata[0]}<br>"
                    "PnL: %{customdata[1]}<br>"
                    "Reason: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"Recent {DEFAULT_INTERVAL} Candles With Trade Markers",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_xanchor="right",
        legend_x=1,
        height=560,
    )
    return fig


def build_pnl_distribution_chart(trades_df: pd.DataFrame) -> go.Figure:
    """Build a per-trade PnL distribution chart for executed trades."""
    fig = go.Figure()
    if trades_df.empty:
        fig.update_layout(title="PnL Per Trade", xaxis_title="Trade #", yaxis_title="Profit Points")
        return fig

    distribution_df = trades_df.copy().reset_index(drop=True)
    distribution_df["trade_number"] = distribution_df.index + 1
    colors = distribution_df["profit_points"].apply(
        lambda value: "#1b9e77" if value > 0 else "#c1121f"
    )

    fig.add_trace(
        go.Bar(
            x=distribution_df["trade_number"],
            y=distribution_df["profit_points"],
            marker_color=colors,
            customdata=distribution_df[["trade_type", "exit_reason", "entry_time", "exit_time"]],
            hovertemplate=(
                "Trade: %{x}<br>"
                "PnL: %{y}<br>"
                "Type: %{customdata[0]}<br>"
                "Exit: %{customdata[1]}<br>"
                "Entry: %{customdata[2]}<br>"
                "Close: %{customdata[3]}<extra></extra>"
            ),
            name="PnL",
        )
    )
    fig.update_layout(
        title="PnL Per Trade Distribution",
        xaxis_title="Trade Number",
        yaxis_title="Profit Points",
        height=340,
        showlegend=False,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#4a4a4a")
    return fig


def filter_trades_to_visible_window(candles_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """Limit trades to the candle window currently shown on the chart."""
    if candles_df.empty or trades_df.empty:
        return trades_df

    window_start = candles_df.index.min()
    window_end = candles_df.index.max()
    return trades_df[
        ((trades_df["entry_time"] >= window_start) & (trades_df["entry_time"] <= window_end))
        | ((trades_df["exit_time"] >= window_start) & (trades_df["exit_time"] <= window_end))
    ].copy()


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(page_title="Trading System Dashboard", layout="wide")
    st.title("Modular Trading System Dashboard")

    trades_df = load_trade_log()
    executed_trades_df = filter_executed_trades(trades_df)
    runtime_status = read_runtime_status()
    summary = compute_performance_summary(executed_trades_df)
    equity_curve = compute_equity_curve(executed_trades_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", summary.total_trades)
    col2.metric("Win Rate", f"{summary.win_rate}%")
    col3.metric("Total Profit", summary.total_profit)
    col4.metric("Max Drawdown", summary.maximum_drawdown)

    symbol = runtime_status.get("symbol", "^NSEI")
    live_signal = runtime_status.get("latest_signal", "NONE")
    open_trade = runtime_status.get("open_trade", "None")

    lookback = st.selectbox("Chart Window", options=["1d", "2d", "5d"], index=1)
    candles_error = None
    candles_df = pd.DataFrame()
    try:
        candles_df = load_recent_candles(symbol=symbol, lookback=lookback)
    except Exception as exc:
        candles_error = str(exc)

    visible_trades_df = filter_trades_to_visible_window(candles_df, executed_trades_df)
    rejected_signals = 0
    if not trades_df.empty and "execution_status" in trades_df.columns:
        rejected_signals = int((trades_df["execution_status"] == "REJECTED").sum())

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Live Candle Chart")
        if candles_error is not None:
            st.warning(f"Could not load recent candles for {symbol}: {candles_error}")
        else:
            st.plotly_chart(
                build_candlestick_chart(candles_df, visible_trades_df),
                use_container_width=True,
            )

        st.subheader("Equity Curve")
        if equity_curve.empty:
            st.info("No trades logged yet.")
        else:
            st.line_chart(equity_curve.set_index("exit_time")["equity"])

        st.subheader("Entry / Exit Markers")
        st.plotly_chart(build_trade_scatter(visible_trades_df), use_container_width=True)

        st.subheader("PnL Distribution")
        st.plotly_chart(build_pnl_distribution_chart(executed_trades_df), use_container_width=True)

    with right:
        st.subheader("Runtime Status")
        st.write(
            {
                "mode": runtime_status.get("mode", "unknown"),
                "symbol": symbol,
                "live_signal": live_signal,
                "open_trade": open_trade,
                "chart_trades": len(visible_trades_df),
                "rejected_signals": rejected_signals,
                "log_path": str(TRADE_LOG_PATH),
            }
        )
        st.subheader("Profit Breakdown")
        avg_win = round(float(executed_trades_df.loc[executed_trades_df["profit_points"] > 0, "profit_points"].mean()), 2) if not executed_trades_df.empty and (executed_trades_df["profit_points"] > 0).any() else 0.0
        avg_loss = round(float(executed_trades_df.loc[executed_trades_df["profit_points"] <= 0, "profit_points"].mean()), 2) if not executed_trades_df.empty and (executed_trades_df["profit_points"] <= 0).any() else 0.0
        st.write(
            {
                "bullish_profit": summary.bullish_profit,
                "bearish_profit": summary.bearish_profit,
                "profit_factor": summary.profit_factor,
                "average_win": avg_win,
                "average_loss": avg_loss,
            }
        )

    st.subheader("Trade History")
    st.dataframe(trades_df, use_container_width=True)


if __name__ == "__main__":
    main()
