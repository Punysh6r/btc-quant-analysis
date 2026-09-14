"""Threshold bot: hysteresis, validation and causal behavior."""

import numpy as np
import pandas as pd
import pytest

from bot import backtest, compute_bull_scores
from indicators import add_indicators


def _window(n=160, amplitude=0.15, period=40.0):
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    wave = 1 + amplitude * np.sin(2 * np.pi * np.arange(n) / period)
    close = pd.Series(60000 * wave, index=idx)
    return add_indicators(pd.DataFrame({"Close": close}))


def test_crossed_thresholds_raise():
    df = _window()
    with pytest.raises(ValueError):
        backtest(df, 30, 70)
    with pytest.raises(ValueError):
        backtest(df, 50, 50)


def test_hysteresis_buys_high_sells_low():
    df = _window()
    _, stats, trades = backtest(df, 70, 30, 10000)
    sides = trades["side"].tolist()
    assert sides[0] == "BUY"
    # Buys and sells must strictly alternate.
    closed = [s for s in sides if s in ("BUY", "SELL")]
    for a, b in zip(closed, closed[1:]):
        assert a != b


def test_never_trades_flat_market():
    idx = pd.date_range("2025-01-01", periods=100, freq="D")
    df = add_indicators(pd.DataFrame({"Close": np.full(100, 50000.0)}, index=idx))
    _, stats, trades = backtest(df, 70, 30, 10000)
    assert stats["n_buys"] == 0 and stats["final_value"] == 10000.0


def test_scores_stay_in_range_and_span_both_thresholds():
    scores = compute_bull_scores(_window())
    assert ((scores >= 5) & (scores <= 95)).all()
    assert float(scores.max()) >= 70 and float(scores.min()) <= 30
