"""Plain-language guide for every indicator and acronym in the app.

Pure categorization helpers (no Streamlit) + one render function.
All bands match the rules used in app.py / bull_bear.py / bot.py.
Academic explanations, not investment advice.
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Pure helpers: value -> (reading, tone) where tone is good/neutral/caution/bad
# ---------------------------------------------------------------------------


def rsi_zone(rsi: float) -> tuple[str, str, str]:
    """Categorize an RSI-14 value. Returns (label, tone, tip)."""
    if pd.isna(rsi):
        return ("Not available", "neutral", "Needs 14 days of data.")
    if rsi < 30:
        return (
            "Oversold",
            "neutral",
            "Historically cheap zone, but a bounce is not a bullrun — check the trend first.",
        )
    if rsi < 50:
        return (
            "Healthy pullback",
            "good",
            "Constructive zone for bulls; e.g. RSI 46.9 means momentum cooled without breaking.",
        )
    if rsi < 70:
        return (
            "Bullish",
            "neutral",
            "Upside momentum confirmed, but extended — avoid chasing, prefer a dip.",
        )
    if rsi <= 80:
        return (
            "Hot",
            "caution",
            "Overheated — correction risk rises, not a moment to buy.",
        )
    return (
        "Overbought",
        "bad",
        "Statistically the worst zone to chase; high pullback risk.",
    )


def trend_zone(sma20: float, sma50: float) -> tuple[str, str, str, float]:
    """Categorize the SMA20 vs SMA50 trend. Returns (label, tone, tip, gap_pct)."""
    gap = ((sma20 - sma50) / sma50 * 100.0) if sma50 != 0 else 0.0
    if gap >= 5:
        return ("Strong uptrend", "good", "Bull structure firmly in place.", gap)
    if gap >= 1:
        return ("Uptrend", "good", "SMA20 above SMA50 — bulls in control.", gap)
    if gap > -1:
        return ("Flat / cross zone", "neutral", "Trend undecided — crossovers happen here.", gap)
    if gap > -5:
        return ("Downtrend", "caution", "SMA20 below SMA50 — buying is riskier.", gap)
    return ("Strong downtrend", "bad", "Bear structure dominates.", gap)


def momentum_zone(close: float, sma20: float) -> tuple[str, str, str, float]:
    """Categorize price vs SMA20. Returns (label, tone, tip, pct)."""
    pct = ((close - sma20) / sma20 * 100.0) if sma20 != 0 else 0.0
    if pct >= 5:
        return ("Extended above average", "caution", "Strong but stretched — pullbacks are normal.", pct)
    if pct >= 1:
        return ("Healthy above average", "good", "Short-term momentum supports the bull case.", pct)
    if pct > -1:
        return ("Riding the average", "neutral", "Price hugs SMA20 — no momentum edge.", pct)
    if pct > -5:
        return ("Weak below average", "caution", "Short-term momentum supports the bear case.", pct)
    return ("Deep below average", "bad", "Sellers firmly in control short-term.", pct)


def drawdown_zone(close: float, high_90: float) -> tuple[str, str, str, float]:
    """Categorize distance to the 90-day high. Returns (label, tone, tip, pct)."""
    dd = ((close - high_90) / high_90 * 100.0) if high_90 != 0 else 0.0
    if dd >= -3:
        return ("Near highs", "good", "Strength — bullrun zone.", dd)
    if dd >= -10:
        return ("Normal digestion", "neutral", "Healthy pause after a run.", dd)
    if dd >= -20:
        return ("Correction", "caution", "Bear risk rising.", dd)
    return ("Technical bear zone", "bad", "More than 20% below recent highs.", dd)


def regime_label(bull_score: float) -> tuple[str, str]:
    """Map the 0-100 bull score to (label, tone)."""
    if bull_score >= 70:
        return ("Short-term bullrun", "good")
    if bull_score >= 55:
        return ("Moderately bullish", "good")
    if bull_score >= 45:
        return ("Sideways / undecided", "neutral")
    if bull_score >= 31:
        return ("Correction / bearish risk", "caution")
    return ("Short-term bear market", "bad")


def signal_reading(signal: int) -> tuple[str, str, str]:
    """Explain today's crossover signal. Returns (label, tone, tip)."""
    if signal == 1:
        return ("BUY (+1)", "good", "SMA20 just crossed above SMA50 with RSI < 70.")
    if signal == -1:
        return ("SELL (-1)", "bad", "SMA20 just crossed below SMA50, or RSI > 80.")
    return ("HOLD (0)", "neutral", "No fresh crossover — follow trend and momentum.")


def f1_reading(score: float) -> tuple[str, str]:
    """Categorize a macro-F1 / accuracy score for the news classifier."""
    if score >= 0.7:
        return ("Strong", "good")
    if score >= 0.5:
        return ("Weak", "caution")
    return ("No signal", "bad")


_TONE_DOT = {"good": "🟢", "neutral": "⚪", "caution": "🟡", "bad": "🔴"}


def render_indicator_guide(df: pd.DataFrame, last_row: pd.Series) -> None:
    """Render the 'What do these numbers mean?' guide with live readings."""
    import streamlit as st

    from bull_bear import get_bull_bear_regime

    close = float(last_row["Close"])
    sma20 = float(last_row["SMA20"])
    sma50 = float(last_row["SMA50"])
    rsi = float(last_row["RSI"]) if pd.notna(last_row["RSI"]) else float("nan")
    signal = int(last_row["Signal"])

    t_label, t_tone, t_tip, gap = trend_zone(sma20, sma50)
    m_label, m_tone, m_tip, mpct = momentum_zone(close, sma20)
    r_label, r_tone, r_tip = rsi_zone(rsi)
    s_label, s_tone, s_tip = signal_reading(signal)
    regime, bull, bear, _, _ = get_bull_bear_regime(last_row, df)
    g_tone = regime_label(bull)[1]
    window = df.tail(90) if len(df) >= 90 else df
    high_90 = float(window["Close"].max()) if len(window) else close
    d_label, d_tone, d_tip, dd = drawdown_zone(close, high_90)

    st.subheader("What do these numbers mean?")
    st.caption(
        "Your current values translated to plain language. Academic guide, not investment advice."
    )
    guide_rows = [
        {
            "Indicator": "SMA trend (20 vs 50 days)",
            "Now": f"SMA20 {'above' if gap >= 0 else 'below'} SMA50 ({gap:+.1f}%)",
            "Reading": f"{_TONE_DOT[t_tone]} {t_label} — {t_tip}",
        },
        {
            "Indicator": "Momentum (price vs SMA20)",
            "Now": f"{mpct:+.1f}% vs SMA20",
            "Reading": f"{_TONE_DOT[m_tone]} {m_label} — {m_tip}",
        },
        {
            "Indicator": "RSI-14 (momentum 0-100)",
            "Now": f"{rsi:.1f}" if pd.notna(rsi) else "n/a",
            "Reading": f"{_TONE_DOT[r_tone]} {r_label} — {r_tip}",
        },
        {
            "Indicator": "Signal (+1 / -1 / 0)",
            "Now": s_label,
            "Reading": f"{_TONE_DOT[s_tone]} {s_tip}",
        },
        {
            "Indicator": "Bull score (0-100)",
            "Now": f"Bull {bull}% / Bear {bear}%",
            "Reading": f"{_TONE_DOT[g_tone]} {regime}",
        },
        {
            "Indicator": "Distance to 90-day high",
            "Now": f"{dd:.1f}%",
            "Reading": f"{_TONE_DOT[d_tone]} {d_label} — {d_tip}",
        },
    ]
    st.table(guide_rows)

    with st.expander("SMA — Simple Moving Average (trend)"):
        st.write(
            "SMA20 is the average close of the last 20 days (fast line); "
            "SMA50 of the last 50 days (slow line). When the fast line is above the slow one, "
            "the trend is up. Bands: +5% or more strong uptrend (🟢), +1% to +5% uptrend (🟢), "
            "-1% to +1% flat (⚪), -5% to -1% downtrend (🟡), below -5% strong downtrend (🔴)."
        )
    with st.expander("RSI — Relative Strength Index (momentum 0-100)"):
        st.write(
            "RSI-14 compares recent gains vs losses. Below 30 oversold (⚪, cheap but unconfirmed); "
            "30-50 healthy pullback (🟢, e.g. RSI 46.9 is constructive); 50-70 bullish (⚪, extended); "
            "70-80 hot (🟡, wait); above 80 overbought (🔴, do not chase). "
            "Buy signals require RSI < 70; any RSI > 80 is an automatic sell signal in this app."
        )
    with st.expander("Signal — crossover verdict (+1 / -1 / 0)"):
        st.write(
            "BUY (+1, 🟢): SMA20 crossed above SMA50 with RSI < 70. "
            "SELL (-1, 🔴): SMA20 crossed below SMA50, or RSI > 80. "
            "HOLD (0, ⚪): anything else — most days are HOLD, which is why the app also gives "
            "a verdict, a regime and a bot instead of relying on sparse signals."
        )
    with st.expander("Bull score and regime (0-100)"):
        st.write(
            "Starts at 50 and adds trend (±30), momentum (±20), RSI (+15/-20), fresh signals (±15) "
            "and distance to the 90-day high (+10/-20). 70%+ bullrun (🟢), 55-69% moderately bullish (🟢), "
            "45-54% sideways (⚪), 31-44% correction (🟡), 30% or less bear market (🔴). "
            "Bear % is always 100 minus bull %."
        )
    with st.expander("Bot backtest words"):
        st.write(
            "Buy threshold (high): bull score that triggers buying. Sell threshold (low): score that "
            "triggers selling; it must stay below the buy threshold (hysteresis). "
            "Win rate: share of closed sells with profit — needs dozens of trades to mean anything. "
            "Max DD (drawdown): worst peak-to-trough fall of the bot's equity curve. "
            "Buy & hold: what the same capital would be worth just holding BTC — the benchmark to beat."
        )
    with st.expander("News model words (TF-IDF, F1, BUY/SELL/HOLD)"):
        st.write(
            "TF-IDF turns headlines into word statistics; LogisticRegression learns which words preceded "
            "rises or falls. Labels come from the 24h forward return: BUY above +2%, SELL below -2%, "
            "else HOLD (demo used ±0.4% to get both classes from 16 rows). "
            "Accuracy: share of test headlines guessed right. Macro-F1: average quality across classes — "
            "0.7+ strong (🟢), 0.5-0.7 weak (🟡), below 0.5 no signal (🔴). "
            "Confusion matrix: rows are the truth, columns the prediction — diagonal is correct."
        )
