"""Shared price indicators: single source of truth for app.py and bot.py.

SMA20 / SMA50 trend, Wilder-style 14-period RSI and the crossover + RSI
signal rules. Pure pandas functions, no Streamlit, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SMA_SHORT = 20
SMA_LONG = 50
RSI_PERIOD = 14
RSI_BUY_MAX = 70.0
RSI_SELL_MIN = 80.0


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """14-period RSI in Wilder style using rolling means (0-100)."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    no_loss = avg_loss == 0
    has_gain = avg_gain > 0
    rsi = rsi.where(~no_loss, np.where(has_gain, 100.0, 0.0))
    return rsi


def compute_indicators(
    df: pd.DataFrame,
    sma_short: int = SMA_SHORT,
    sma_long: int = SMA_LONG,
    rsi_period: int = RSI_PERIOD,
) -> pd.DataFrame:
    """Add SMA20, SMA50 and RSI columns (copy, never mutates the input)."""
    out = df.copy()
    out["SMA20"] = out["Close"].rolling(window=sma_short, min_periods=1).mean()
    out["SMA50"] = out["Close"].rolling(window=sma_long, min_periods=1).mean()
    out["RSI"] = compute_rsi(out["Close"], period=rsi_period)
    return out


def detect_signals(
    df: pd.DataFrame,
    rsi_buy_max: float = RSI_BUY_MAX,
    rsi_sell_min: float = RSI_SELL_MIN,
) -> pd.DataFrame:
    """Flag Buy (+1) / Sell (-1) / Hold (0) rows.

    Buy: SMA20 crosses above SMA50 AND RSI < rsi_buy_max.
    Sell: SMA20 crosses below SMA50 OR RSI > rsi_sell_min.
    """
    out = df.copy()
    prev_short = out["SMA20"].shift(1)
    prev_long = out["SMA50"].shift(1)

    cross_up = (prev_short <= prev_long) & (out["SMA20"] > out["SMA50"])
    cross_down = (prev_short >= prev_long) & (out["SMA20"] < out["SMA50"])

    buy = cross_up & (out["RSI"] < rsi_buy_max)
    sell = cross_down | (out["RSI"] > rsi_sell_min)

    out["Signal"] = 0
    out.loc[buy.fillna(False), "Signal"] = 1
    out.loc[sell.fillna(False), "Signal"] = -1
    return out


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience: indicators + signals with default windows (CLI/backtest use)."""
    return detect_signals(compute_indicators(df))
