"""Indicators: SMA/RSI/signal math on synthetic series with known answers."""

import numpy as np
import pandas as pd

from indicators import add_indicators, compute_indicators, compute_rsi, detect_signals


def _flat(n=60, price=100.0):
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame({"Close": np.full(n, price)}, index=idx)


def test_rsi_flat_market_is_zero():
    rsi = compute_rsi(_flat()["Close"])
    assert (rsi.dropna() == 0.0).all()


def test_rsi_steady_climb_is_overbought():
    idx = pd.date_range("2025-01-01", periods=40, freq="D")
    close = pd.Series(100 + np.arange(40, dtype=float), index=idx)
    assert float(compute_rsi(close).iloc[-1]) == 100.0


def test_sma_values_are_exact():
    df = compute_indicators(_flat(60, 100.0))
    assert float(df["SMA20"].iloc[-1]) == 100.0
    assert float(df["SMA50"].iloc[-1]) == 100.0


def test_buy_signal_on_up_cross_with_cool_rsi():
    idx = pd.date_range("2025-01-01", periods=60, freq="D")
    df = pd.DataFrame(
        {"Close": 100.0, "SMA20": 100.0, "SMA50": 100.0, "RSI": 60.0},
        index=idx,
    )
    df.loc[idx[30:], "SMA20"] = 101.0  # cross above on day 30
    out = detect_signals(df)
    assert out["Signal"].iloc[30] == 1
    assert (out["Signal"].iloc[:30] == 0).all()
    assert (out["Signal"].iloc[31:] == 0).all()


def test_overbought_rsi_forces_sell():
    idx = pd.date_range("2025-01-01", periods=40, freq="D")
    close = pd.Series(100 + np.arange(40, dtype=float) * 5, index=idx)
    df = add_indicators(pd.DataFrame({"Close": close}))
    assert df["Signal"].iloc[-1] == -1  # RSI 100 > 80


def test_no_lookahead_sma_uses_only_past():
    idx = pd.date_range("2025-01-01", periods=30, freq="D")
    close = pd.Series([100.0] * 29 + [1000.0], index=idx)
    df = compute_indicators(pd.DataFrame({"Close": close}))
    # SMA20 at row 28 must ignore the spike on row 29.
    assert float(df["SMA20"].iloc[28]) == 100.0
