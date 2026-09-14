"""Bullrun vs Bear short-term regime (technical only, no ML, no lookahead).

Simple 0-100 score for a TFM, easy to explain:
  start at 50 (neutral), then add/subtract:
  - SMA20 vs SMA50 trend gap (up to +/-30)
  - Close vs SMA20 momentum (up to +/-20)
  - RSI-14 zone (+15 to -20)
  - Fresh Buy/Sell crossover (+/-15, or +/-5 if within 5 days)
  - Distance to 90-day high (+10 to -20)

Bull % + Bear % = 100%. Clamped to 5-95 so it never says 0% or 100%.
Academic demo, not investment advice.
"""

from __future__ import annotations

import pandas as pd


def get_bull_bear_regime(last_row: pd.Series, df: pd.DataFrame):
    """Return (regime_label, bull_prob, bear_prob, reasons, details)."""
    reasons: list[str] = []
    details: dict = {}
    close = float(last_row["Close"])
    sma20 = float(last_row["SMA20"])
    sma50 = float(last_row["SMA50"])
    rsi = float(last_row["RSI"]) if pd.notna(last_row["RSI"]) else float("nan")
    signal = int(last_row["Signal"])
    score = 50.0

    # 1. Trend: SMA20 vs SMA50.
    trend_gap_pct = ((sma20 - sma50) / sma50 * 100.0) if sma50 != 0 else 0.0
    if sma20 > sma50:
        trend_pts = 20.0 + min(max(trend_gap_pct, 0.0) * 5.0, 10.0)
        reasons.append(
            f"Uptrend: SMA20 above SMA50 (gap {trend_gap_pct:+.1f}%) - bullrun structure holds."
        )
    else:
        trend_pts = -20.0 - min(abs(trend_gap_pct) * 5.0, 10.0)
        reasons.append(
            f"Downtrend: SMA20 below SMA50 (gap {trend_gap_pct:+.1f}%) - bear structure dominates."
        )
    score += trend_pts
    details["trend_gap_pct"] = round(trend_gap_pct, 2)
    details["trend_pts"] = round(trend_pts, 1)

    # 2. Momentum: Close vs SMA20.
    mom_pct = ((close - sma20) / sma20 * 100.0) if sma20 != 0 else 0.0
    if close > sma20:
        mom_pts = 10.0 + min(max(mom_pct, 0.0) * 2.0, 10.0)
        reasons.append(
            f"Price {mom_pct:+.1f}% above SMA20 - short-term momentum supports the bull case."
        )
    else:
        mom_pts = -10.0 - min(abs(mom_pct) * 2.0, 10.0)
        reasons.append(
            f"Price {mom_pct:+.1f}% below SMA20 - short-term momentum supports the bear case."
        )
    score += mom_pts
    details["momentum_pct"] = round(mom_pct, 2)
    details["momentum_pts"] = round(mom_pts, 1)

    # 3. RSI zone.
    if pd.isna(rsi):
        rsi_pts = 0.0
        reasons.append("RSI not available yet (need 14 days of data).")
    elif rsi < 30:
        rsi_pts = 5.0
        reasons.append(
            f"RSI {rsi:.1f} oversold (< 30): bounce possible, but not a bullrun yet."
        )
    elif rsi < 50:
        rsi_pts = 10.0
        reasons.append(
            f"RSI {rsi:.1f} in healthy pullback zone (30-50): constructive for bulls."
        )
    elif rsi < 70:
        rsi_pts = 15.0
        reasons.append(f"RSI {rsi:.1f} bullish (50-70): momentum confirms the upside.")
    elif rsi <= 80:
        rsi_pts = -10.0
        reasons.append(
            f"RSI {rsi:.1f} hot (70-80): overheated, bear/correction risk rises."
        )
    else:
        rsi_pts = -20.0
        reasons.append(
            f"RSI {rsi:.1f} overbought (> 80): statistically the worst zone to chase."
        )
    score += rsi_pts
    details["rsi"] = round(rsi, 1) if pd.notna(rsi) else None
    details["rsi_pts"] = round(rsi_pts, 1)

    # 4. Fresh signals.
    sig_pts = 0.0
    if signal == 1:
        sig_pts = 15.0
        reasons.append("Fresh BUY crossover today - classic bullrun trigger.")
    elif signal == -1:
        sig_pts = -15.0
        reasons.append("Fresh SELL signal today - classic bear/correction trigger.")
    else:
        buys_idx = df.index[df["Signal"] == 1]
        sells_idx = df.index[df["Signal"] == -1]
        if len(buys_idx):
            days_buy = (df.index[-1] - buys_idx[-1]).days
            if days_buy <= 5:
                sig_pts += 5.0
                reasons.append(
                    f"Buy signal {days_buy} day(s) ago - bullish impulse still fresh."
                )
        if len(sells_idx):
            days_sell = (df.index[-1] - sells_idx[-1]).days
            if days_sell <= 5:
                sig_pts -= 5.0
                reasons.append(
                    f"Sell signal {days_sell} day(s) ago - bearish pressure still fresh."
                )
        if sig_pts == 0.0:
            reasons.append(
                "No fresh crossover - regime depends on trend and momentum only."
            )
    score += sig_pts
    details["signal_pts"] = round(sig_pts, 1)

    # 5. Distance to 90-day high.
    window = df.tail(90) if len(df) >= 90 else df
    high_90 = float(window["Close"].max()) if len(window) else close
    dd_pct = ((close - high_90) / high_90 * 100.0) if high_90 != 0 else 0.0
    if dd_pct >= -3.0:
        dd_pts = 10.0
        reasons.append(f"Price {dd_pct:.1f}% from 90-day high - strength, bullrun zone.")
    elif dd_pct >= -10.0:
        dd_pts = 0.0
        reasons.append(
            f"Price {dd_pct:.1f}% from 90-day high - normal digestion, no regime edge."
        )
    elif dd_pct >= -20.0:
        dd_pts = -10.0
        reasons.append(
            f"Price {dd_pct:.1f}% from 90-day high - correction, bear risk rising."
        )
    else:
        dd_pts = -20.0
        reasons.append(
            f"Price {dd_pct:.1f}% from 90-day high - technical bear zone (-20% rule)."
        )
    score += dd_pts
    details["dist_90d_high_pct"] = round(dd_pct, 2)
    details["dist_pts"] = round(dd_pts, 1)

    score = max(5.0, min(95.0, score))
    bull_prob = int(round(score))
    bear_prob = 100 - bull_prob
    details["bull_score"] = bull_prob

    if bull_prob >= 70:
        regime = "Short-term bullrun"
    elif bull_prob >= 55:
        regime = "Moderately bullish"
    elif bull_prob >= 45:
        regime = "Sideways / undecided"
    elif bull_prob >= 31:
        regime = "Correction / bearish risk"
    else:
        regime = "Short-term bear market"
    return regime, bull_prob, bear_prob, reasons, details


def render_bull_bear_regime(df: pd.DataFrame, last_row: pd.Series) -> None:
    """Render the Bullrun vs Bear panel in Streamlit (short-term, technical)."""
    import streamlit as st

    st.subheader("Bullrun vs Bear Regime (short-term)")
    st.caption(
        "Technical regime only: SMA20/SMA50 trend + price vs SMA20 + RSI-14 + "
        "fresh crossovers + distance to 90-day high. Academic demo, not investment advice."
    )
    regime, bull_prob, bear_prob, reasons, details = get_bull_bear_regime(
        last_row, df
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Regime", regime)
    c2.metric("Bullrun chance", f"{bull_prob}%")
    c3.metric("Bear chance", f"{bear_prob}%")
    try:
        st.progress(
            bull_prob / 100.0, text=f"Bull pressure {bull_prob}% / Bear {bear_prob}%"
        )
    except Exception:
        st.progress(bull_prob / 100.0)
    for reason in reasons:
        st.write(f"- {reason}")
    with st.expander("How is this score built?"):
        st.write(
            f"Trend {details.get('trend_pts', 0):+.1f} pts "
            f"(gap {details.get('trend_gap_pct', 0):+.2f}%), "
            f"momentum {details.get('momentum_pts', 0):+.1f} pts "
            f"({details.get('momentum_pct', 0):+.2f}%), "
            f"RSI {details.get('rsi_pts', 0):+.1f} pts, "
            f"signals {details.get('signal_pts', 0):+.1f} pts, "
            f"90-day high {details.get('dist_pts', 0):+.1f} pts "
            f"({details.get('dist_90d_high_pct', 0):+.2f}%). Starts at 50, clamped 5-95."
        )
        st.caption(
            "Bull % + Bear % = 100%. 70%+ is bullrun, 30%- is bear, 45-54% is sideways."
        )


REGIME_COLORS = {
    "bull": "rgba(22, 163, 74, 0.10)",
    "sideways": "rgba(148, 163, 184, 0.10)",
    "bear": "rgba(220, 38, 38, 0.10)",
}
REGIME_COLORS_MPL = {
    "bull": (0.13, 0.64, 0.29, 0.10),
    "sideways": (0.58, 0.64, 0.70, 0.10),
    "bear": (0.86, 0.15, 0.15, 0.10),
}


def simplify_regime(score):
    if score >= 55:
        return "bull"
    if score < 45:
        return "bear"
    return "sideways"


def regime_blocks(df, scores):
    blocks = []
    if len(df) == 0 or len(scores) == 0:
        return blocks
    regs = [simplify_regime(float(s)) for s in scores]
    start = df.index[0]
    cur = regs[0]
    for i in range(1, len(df)):
        if regs[i] != cur:
            blocks.append((start, df.index[i - 1], cur))
            start = df.index[i]
            cur = regs[i]
    blocks.append((start, df.index[-1], cur))
    return blocks


def add_regime_bands_plotly(fig, df, scores):
    try:
        for start, end, reg in regime_blocks(df, scores):
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=REGIME_COLORS.get(reg, REGIME_COLORS["sideways"]),
                line_width=0,
                layer="below",
            )
    except Exception:
        pass


def add_regime_bands_matplotlib(ax, df, scores):
    try:
        import matplotlib.dates as mdates

        for start, end, reg in regime_blocks(df, scores):
            ax.axvspan(
                mdates.date2num(start),
                mdates.date2num(end),
                color=REGIME_COLORS_MPL.get(reg, REGIME_COLORS_MPL["sideways"]),
                linewidth=0,
                zorder=0,
            )
    except Exception:
        pass
