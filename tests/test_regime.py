"""Bull/bear regime: extremes, clamping, coherence and no-lookahead."""

import numpy as np
import pandas as pd

from bull_bear import get_bull_bear_regime, regime_blocks, simplify_regime
from indicators import add_indicators


def _trend_df(n=100, drift=1.0, rsi=65.0, signal=0):
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    close = pd.Series(100 + np.arange(n, dtype=float) * drift, index=idx)
    df = add_indicators(pd.DataFrame({"Close": close}))
    df["RSI"] = rsi
    df["Signal"] = signal
    return df


def test_steady_uptrend_is_bullrun():
    df = _trend_df()
    regime, bull, bear, _, _ = get_bull_bear_regime(df.iloc[-1], df)
    assert regime == "Short-term bullrun" and bull >= 70 and bull + bear == 100


def test_steady_downtrend_is_bear():
    df = _trend_df(drift=-1.0, rsi=25.0, signal=-1)
    regime, bull, bear, _, _ = get_bull_bear_regime(df.iloc[-1], df)
    assert regime == "Short-term bear market" and bull <= 30 and bull + bear == 100


def test_score_is_clamped():
    df = _trend_df(n=300, drift=10.0, rsi=65.0, signal=1)
    _, bull, _, _, details = get_bull_bear_regime(df.iloc[-1], df)
    assert 5 <= bull <= 95
    assert details["bull_score"] == bull


def test_simplify_bands_match_labels():
    assert simplify_regime(80) == "bull"
    assert simplify_regime(50) == "sideways"
    assert simplify_regime(20) == "bear"


def test_regime_blocks_merge_consecutive_days():
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    df = pd.DataFrame({"Close": [1, 2, 3, 4, 5]}, index=idx)
    blocks = regime_blocks(df, pd.Series([80, 82, 20, 25, 60], index=idx))
    assert [(s, e, r) for s, e, r in blocks] == [
        (idx[0], idx[1], "bull"),
        (idx[2], idx[3], "bear"),
        (idx[4], idx[4], "bull"),
    ]
