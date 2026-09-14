"""Bitcoin Quantitative Analysis (BTC-USD / BTC-EUR) — Interactive Streamlit App.

Academic project: download historical Bitcoin prices (BTC-USD or BTC-EUR,
selected in the sidebar) with yfinance, compute SMA20 / SMA50 trend
indicators and a 14-period RSI, detect simple SMA-crossover +
RSI-filtered signals, and display interactive price/indicator charts,
metrics, and data tables.

Run locally with:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:  # Fallback to Matplotlib when Plotly is unavailable.
    HAS_PLOTLY = False

import matplotlib.pyplot as plt

from bull_bear import add_regime_bands_matplotlib, add_regime_bands_plotly, render_bull_bear_regime

from bot import compute_bull_scores, render_bot_backtest

from indicators import compute_indicators, detect_signals

from glossary import render_indicator_guide

try:
    import yfinance as yf

    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

SUPPORTED_CURRENCIES = {
    "USD": {"ticker": "BTC-USD", "symbol": "$", "label": "US Dollar"},
    "EUR": {"ticker": "BTC-EUR", "symbol": "€", "label": "Euro"},
}
DEFAULT_CURRENCY = "USD"
# Indicator windows live in indicators.py (single source of truth).


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Downloading Bitcoin data...")
def load_btc_data(start: dt.date, end: dt.date, ticker: str) -> pd.DataFrame:
    """Download historical Bitcoin OHLCV data with yfinance.

    Args:
        start: Inclusive start date.
        end: Exclusive end date (yfinance convention).
        ticker: Yahoo Finance ticker, e.g. ``BTC-USD`` or ``BTC-EUR``.

    Returns:
        DataFrame indexed by date with at least a ``Close`` column.

    Raises:
        RuntimeError: If yfinance is not installed or the download fails.
        ValueError: If the download succeeds but returns no rows.
    """
    if not HAS_YFINANCE:
        raise RuntimeError(
            "The 'yfinance' package is not installed. "
            "Install it with: pip install -r requirements.txt"
        )
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as exc:  # Network errors, invalid dates, etc.
        raise RuntimeError(f"Download failed for {ticker}: {exc}") from exc

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for {ticker} between {start} and {end}. "
            "Try a wider date range (BTC data starts in 2014)."
        )

    # Newer yfinance versions return MultiIndex columns; flatten them.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # Normalize the index to plain dates and sort chronologically.
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------


def get_entry_exit_points(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split signal rows into buy (entry) and sell (exit) dataframes.

    Args:
        df: DataFrame with ``Close`` and ``Signal`` columns.

    Returns:
        Tuple ``(buys, sells)`` containing only the signal rows.
    """
    buys = df[df["Signal"] == 1]
    sells = df[df["Signal"] == -1]
    return buys, sells


def current_signal_label(signal_value: int) -> str:
    """Map the last-row signal value to a human-readable state label."""
    if signal_value == 1:
        return "Buy"
    if signal_value == -1:
        return "Sell"
    return "Hold / Neutral"


def get_market_verdict(
    last_row: pd.Series, df: pd.DataFrame, currency: str = "USD"
) -> tuple[str, str, list[str]]:
    """Build a plain-language verdict: is now a good moment to buy or not.

    Combines trend (SMA20 vs SMA50), momentum (RSI zones), and distance
    to the last signals so most days still get guidance instead of
    a sparse Hold / Neutral.

    Returns:
        Tuple (verdict_title, verdict_level, reasons) where level is one
        of "good", "caution", "bad" to drive Streamlit banner styling.
    """
    cur_sym = SUPPORTED_CURRENCIES.get(currency, SUPPORTED_CURRENCIES["USD"])["symbol"]
    reasons: list[str] = []
    close = float(last_row["Close"])
    sma20 = float(last_row["SMA20"])
    sma50 = float(last_row["SMA50"])
    rsi = float(last_row["RSI"]) if pd.notna(last_row["RSI"]) else float("nan")
    signal = int(last_row["Signal"])

    bullish_trend = sma20 > sma50
    trend_gap_pct = ((sma20 - sma50) / sma50 * 100.0) if sma50 != 0 else 0.0
    above_short = close > sma20

    if bullish_trend:
        reasons.append(
            f"Uptrend: SMA20 ({cur_sym}{sma20:,.0f}) is above SMA50 ({cur_sym}{sma50:,.0f}, gap {trend_gap_pct:+.1f}%)."
        )
    else:
        reasons.append(
            f"Downtrend: SMA20 ({cur_sym}{sma20:,.0f}) is below SMA50 ({cur_sym}{sma50:,.0f}, gap {trend_gap_pct:+.1f}%). "
            "Buying into a downtrend is riskier."
        )

    if above_short:
        reasons.append(f"Price ({cur_sym}{close:,.0f}) is above SMA20 — short-term momentum is positive.")
    else:
        reasons.append(f"Price ({cur_sym}{close:,.0f}) is below SMA20 — short-term momentum is weak.")

    rsi_state = "unknown"
    if pd.isna(rsi):
        reasons.append("RSI is not available yet (need 14 days of data).")
    elif rsi < 30:
        rsi_state = "oversold"
        reasons.append(f"RSI {rsi:.1f} is oversold (< 30): historically cheap zone, but confirm the trend first.")
    elif rsi < 50:
        rsi_state = "healthy"
        reasons.append(f"RSI {rsi:.1f} is in a healthy pullback zone (< 50): better risk/reward for buyers.")
    elif rsi < 70:
        rsi_state = "bullish"
        reasons.append(f"RSI {rsi:.1f} is bullish (50-70): trend is extended, avoid chasing, prefer a dip.")
    elif rsi <= 80:
        rsi_state = "caution"
        reasons.append(f"RSI {rsi:.1f} is hot (70-80): not a good moment to buy, wait for a cooldown.")
    else:
        rsi_state = "overbought"
        reasons.append(f"RSI {rsi:.1f} is overbought (> 80): bad moment to buy, high pullback risk.")

    if signal == 1:
        reasons.append("Fresh BUY signal today (SMA20 crossed above SMA50 with RSI < 70).")
    elif signal == -1:
        reasons.append("Fresh SELL signal today (breakdown or RSI > 80) — do not buy now.")

    # Distance to last signals for context.
    buys_idx = df.index[df["Signal"] == 1]
    sells_idx = df.index[df["Signal"] == -1]
    if len(buys_idx):
        days_since_buy = (df.index[-1] - buys_idx[-1]).days
        reasons.append(f"Last buy signal was {days_since_buy} day(s) ago ({buys_idx[-1].date()}).")
    if len(sells_idx):
        days_since_sell = (df.index[-1] - sells_idx[-1]).days
        reasons.append(f"Last sell signal was {days_since_sell} day(s) ago ({sells_idx[-1].date()}).")

    if signal == -1 or rsi_state in ("overbought", "caution") or not bullish_trend:
        if signal == -1 or rsi_state == "overbought":
            title = "Not a good moment to buy — wait"
            level = "bad"
        elif not bullish_trend:
            title = "Not a good moment to buy — downtrend"
            level = "bad"
        else:
            title = "Not ideal to chase — wait for a dip"
            level = "caution"
    elif signal == 1 or (bullish_trend and rsi_state in ("healthy", "oversold")):
        title = "Good moment to consider buying"
        level = "good"
    elif bullish_trend:
        title = "Neutral to slightly positive — small size only"
        level = "caution"
    else:
        title = "Not a good moment to buy — wait"
        level = "bad"
    return title, level, reasons


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def price_figure_plotly(
    df: pd.DataFrame,
    buys: pd.DataFrame,
    sells: pd.DataFrame,
    currency: str = "USD",
) -> "go.Figure":
    """Build an interactive Plotly price chart with SMAs and markers.

    Args:
        df: Indicator DataFrame with ``Close``, ``SMA20``, ``SMA50``.
        buys: Rows flagged as buy entries (markers).
        sells: Rows flagged as sell exits (markers).
        currency: Quote currency code ("USD" or "EUR") for labels.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name=f"BTC Close ({currency})", mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", mode="lines"))
    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys.index,
                y=buys["Close"],
                name="Buy",
                mode="markers",
                marker=dict(symbol="triangle-up", size=12, color="green"),
            )
        )
    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells.index,
                y=sells["Close"],
                name="Sell",
                mode="markers",
                marker=dict(symbol="triangle-down", size=12, color="red"),
            )
        )
    fig.update_layout(
        title=f"BTC-{currency} Price with SMA20 / SMA50 and Buy-Sell Markers",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency})",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def rsi_figure_plotly(df: pd.DataFrame) -> "go.Figure":
    """Build an interactive Plotly RSI chart (currency-independent)."""
    """Build an interactive Plotly RSI chart with 30/70/80 reference lines."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI (14)", mode="lines"))
    for level, label, color in (
        (80, "Overbought (80)", "red"),
        (70, "Caution (70)", "orange"),
        (30, "Oversold (30)", "green"),
    ):
        fig.add_hline(y=level, line_dash="dash", line_color=color, annotation_text=label)
    fig.update_layout(
        title="RSI (14) with 30 / 70 / 80 Levels",
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis=dict(range=[0, 100]),
        hovermode="x unified",
    )
    return fig


def price_figure_matplotlib(
    df: pd.DataFrame,
    buys: pd.DataFrame,
    sells: pd.DataFrame,
    currency: str = "USD",
) -> plt.Figure:
    """Build a Matplotlib price chart (fallback when Plotly is missing)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["Close"], label="BTC Close")
    ax.plot(df.index, df["SMA20"], label="SMA20")
    ax.plot(df.index, df["SMA50"], label="SMA50")
    if not buys.empty:
        ax.scatter(buys.index, buys["Close"], marker="^", s=80, label="Buy", color="green")
    if not sells.empty:
        ax.scatter(sells.index, sells["Close"], marker="v", s=80, label="Sell", color="red")
    ax.set_title("BTC-USD Price with SMA20 / SMA50 and Buy-Sell Markers")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def rsi_figure_matplotlib(df: pd.DataFrame) -> plt.Figure:
    """Build a Matplotlib RSI chart (fallback when Plotly is missing)."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df.index, df["RSI"], label="RSI (14)")
    ax.axhline(80, linestyle="--", color="red", label="Overbought (80)")
    ax.axhline(70, linestyle="--", color="orange", label="Caution (70)")
    ax.axhline(30, linestyle="--", color="green", label="Oversold (30)")
    ax.set_title("RSI (14) with 30 / 70 / 80 Levels")
    ax.set_xlabel("Date")
    ax.set_ylabel("RSI")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# News sentiment model (academic ML extension, additive and read-only)
# ---------------------------------------------------------------------------

NEWS_MODEL_TRAIN_COMMANDS = (
    "python download_news_rss.py --since 2025-01-01 --max-items 2000",
    "python download_prices.py",
    "python build_dataset.py",
    "python train_model.py",
)
NEWS_MODEL_N_TITLES = 20


@st.cache_resource(show_spinner="Loading news sentiment model...")
def load_news_model_bundle():
    """Load the trained TF-IDF + LogisticRegression bundle (read-only)."""
    try:
        import joblib
    except ImportError:
        return None
    model_path = Path(__file__).resolve().parent / "artifacts" / "model.joblib"
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


@st.cache_data(show_spinner="Loading labeled news dataset...")
def load_labeled_dataset():
    """Load data/dataset.csv with forward-return labels (read-only)."""
    ds_path = Path(__file__).resolve().parent / "data" / "dataset.csv"
    if not ds_path.exists():
        return None
    try:
        df = pd.read_csv(ds_path)
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        df = df.dropna(subset=["published_at", "title"]).sort_values("published_at")
        return df
    except Exception:
        return None


def render_news_model_signal(num_titles: int = NEWS_MODEL_N_TITLES) -> None:
    """Render the News Sentiment panel (keyless RSS pipeline)."""
    st.subheader("News Sentiment (RSS, no API key)")
    st.caption(
        "Source: free RSS feeds (CoinDesk / Cointelegraph / Bitcoin Magazine) "
        "via download_news_rss.py. Labels are 24h forward returns "
        "(BUY > +2%, SELL < -2%, else HOLD). Academic demo, not investment advice."
    )
    bundle = load_news_model_bundle()
    dataset = load_labeled_dataset()
    news_path = Path(__file__).resolve().parent / "data" / "news_btc.csv"

    if bundle is not None and news_path.exists():
        try:
            vectorizer = bundle["vectorizer"]
            model = bundle["model"]
            news = pd.read_csv(news_path)
            news["published_at"] = pd.to_datetime(
                news["published_at"], utc=True, errors="coerce"
            )
            news = news.dropna(subset=["published_at", "title"]).sort_values(
                "published_at"
            )
            recent = news.tail(num_titles).copy()
            if recent.empty:
                raise ValueError("no headlines with a valid date and title")
            bodies = (
                recent["body"].fillna("").astype(str).tolist()
                if "body" in recent.columns
                else ["" for _ in range(len(recent))]
            )
            titles = recent["title"].astype(str).tolist()
            texts = [
                t if not b.strip() else t + chr(10) + b.strip()[:2000]
                for t, b in zip(titles, bodies)
            ]
            features = vectorizer.transform(texts)
            proba = model.predict_proba(features)
            classes = [str(c) for c in model.classes_]
            predicted = [str(x) for x in model.predict(features)]
            mean_proba = proba.mean(axis=0)
            winner = classes[int(mean_proba.argmax())]
            start_ts = recent["published_at"].iloc[0]
            end_ts = recent["published_at"].iloc[-1]
            st.metric("News model signal", winner)
            st.caption(
                "Mean predicted probabilities over "
                + str(len(recent))
                + " headlines ("
                + str(start_ts.date())
                + " to "
                + str(end_ts.date())
                + "): "
                + ", ".join(
                    c + " " + "{:.1%}".format(p) for c, p in zip(classes, mean_proba)
                )
                + "."
            )
            sources = (
                recent["source"].fillna("").astype(str).tolist()
                if "source" in recent.columns
                else ["" for _ in range(len(recent))]
            )
            table = pd.DataFrame(
                {
                    "date": recent["published_at"].dt.date.astype(str).tolist(),
                    "source": sources,
                    "title": titles,
                    "predicted label": predicted,
                    "top probability": ["{:.1%}".format(max(r)) for r in proba],
                }
            )
            with st.expander("Show classified headlines"):
                st.dataframe(table)
            return
        except Exception as exc:
            st.warning("Trained model unreadable (" + str(exc) + "); showing dataset labels.")

    if dataset is not None and not dataset.empty:
        dist = dataset["label"].value_counts() if "label" in dataset.columns else pd.Series(dtype=int)
        n = len(dataset)
        start_d = dataset["published_at"].iloc[0].date()
        end_d = dataset["published_at"].iloc[-1].date()
        mean_ret = float(dataset["ret_24h"].mean()) if "ret_24h" in dataset.columns else float("nan")
        winner = str(dist.idxmax()) if len(dist) else "HOLD"
        cols = st.columns(4)
        cols[0].metric("Headlines labeled", str(n))
        cols[1].metric("BUY", str(int(dist.get("BUY", 0))))
        cols[2].metric("SELL", str(int(dist.get("SELL", 0))))
        cols[3].metric("HOLD", str(int(dist.get("HOLD", 0))))
        if pd.notna(mean_ret):
            st.caption(str(start_d) + " to " + str(end_d) + " | mean 24h forward return {0:+.2%}".format(mean_ret))
        else:
            st.caption(str(start_d) + " to " + str(end_d))
        if len(dist) <= 1:
            st.info(
                "Only " + winner + " in this window (RSS covers recent weeks with low "
                "24h moves). The signal gets interesting as you accumulate more "
                "weeks of RSS history."
            )
        else:
            st.metric("Dataset signal (majority label)", winner)
        try:
            chart_df = pd.DataFrame({"count": dist}).sort_index()
            st.bar_chart(chart_df)
        except Exception:
            pass
        show_cols = [c for c in ["published_at", "title", "source", "label", "ret_24h", "url"] if c in dataset.columns]
        recent_ds = dataset.tail(num_titles).iloc[::-1].copy()
        if "ret_24h" in recent_ds.columns:
            recent_ds["ret_24h"] = (recent_ds["ret_24h"] * 100).map(lambda x: "{0:+.2f}%".format(x))
        with st.expander("Show labeled headlines (forward-return labels)"):
            st.dataframe(recent_ds[show_cols] if show_cols else recent_ds)
        if bundle is None:
            st.caption(
                "ML classifier needs 200+ rows with 2+ classes (you have " + str(n) + "). "
                "Keep running download_news_rss.py weekly to grow history, then "
                "build_dataset.py + train_model.py."
            )
        return

    if news_path.exists():
        try:
            news = pd.read_csv(news_path)
            news["published_at"] = pd.to_datetime(news["published_at"], utc=True, errors="coerce")
            news = news.dropna(subset=["published_at", "title"]).sort_values("published_at")
            if not news.empty:
                st.metric("Headlines collected (RSS)", str(len(news)))
                st.caption(str(news["published_at"].iloc[0].date()) + " to " + str(news["published_at"].iloc[-1].date()))
                show = news.tail(num_titles).iloc[::-1]
                show_cols = [c for c in ["published_at", "title", "source", "url"] if c in show.columns]
                with st.expander("Show recent headlines"):
                    st.dataframe(show[show_cols] if show_cols else show)
                st.info("Run python build_dataset.py to label them with 24h forward returns.")
                return
        except Exception as exc:
            st.warning("Could not read news CSV (" + str(exc) + ").")

    st.info(
        "No news data yet. Collect free RSS headlines with: "
        + " | ".join(NEWS_MODEL_TRAIN_COMMANDS)
    )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def render_sidebar() -> tuple[str, dt.date, dt.date]:
    """Render sidebar controls and return the selected (currency, start, end).

    The currency selector (USD / EUR) drives the ticker used for the whole
    analysis; the date range works exactly as before, with presets or custom
    dates.
    """
    st.sidebar.header("Settings")
    today = dt.date.today()

    currency = st.sidebar.radio(
        "Quote currency",
        options=list(SUPPORTED_CURRENCIES.keys()),
        captions=[cfg["label"] for cfg in SUPPORTED_CURRENCIES.values()],
        help="All prices, charts, and metrics are quoted in this currency.",
    )

    preset = st.sidebar.selectbox(
        "Period shortcut",
        options=["Custom", "Last 90 days", "Last 180 days", "Last 365 days", "Last 2 years"],
        index=3,
        help="Choose a preset range or 'Custom' to pick exact dates below.",
    )

    preset_days = {
        "Last 90 days": 90,
        "Last 180 days": 180,
        "Last 365 days": 365,
        "Last 2 years": 730,
    }

    if preset in preset_days:
        default_end = today
        default_start = today - dt.timedelta(days=preset_days[preset])
    else:
        default_end = today
        default_start = today - dt.timedelta(days=365)

    start_date = st.sidebar.date_input(
        "Start date",
        value=default_start,
        max_value=today,
        help="First day of the analysis window (inclusive).",
    )
    end_date = st.sidebar.date_input(
        "End date",
        value=default_end,
        max_value=today,
        help="Last day of the analysis window (exclusive in the download).",
    )

    if preset in preset_days:
        st.sidebar.info(
            f"Using preset '{preset}' ({default_start} to {default_end}). "
            "Switch to 'Custom' to use the date inputs freely."
        )
        return currency, default_start, default_end
    return currency, start_date, end_date


def _load_terminal_css() -> None:
    """Inject the trading-terminal stylesheet (assets/terminal.css), silently skipped if missing."""
    css_path = Path(__file__).resolve().parent / "assets" / "terminal.css"
    try:
        css = css_path.read_text(encoding="utf-8")
    except OSError:
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="Bitcoin Quantitative Analysis", layout="wide")
    _load_terminal_css()
    st.title("Bitcoin Quantitative Analysis (BTC-USD / BTC-EUR)")
    st.markdown(
        """
        **What this app does:** it downloads historical Bitcoin prices, computes
        trend (SMA20 / SMA50) and momentum (RSI-14) indicators, and flags simple
        crossover-based Buy / Sell signals.

        **How to use it:**
        1. Pick a date range in the sidebar (or use a period shortcut).
        2. Review the latest price, RSI, and signal state at the top.
        3. Explore the interactive price and RSI charts.
        4. Inspect the recent data table and the signal summary counts.

        **Signal rules:** Buy when SMA20 crosses above SMA50 while RSI < 70.
        Sell when SMA20 crosses below SMA50 **or** RSI > 80. Otherwise hold.
        This is an academic demo, not investment advice.
        """
    )

    currency, start_date, end_date = render_sidebar()
    currency_cfg = SUPPORTED_CURRENCIES.get(
        currency, SUPPORTED_CURRENCIES[DEFAULT_CURRENCY]
    )
    ticker = currency_cfg["ticker"]
    currency_symbol = currency_cfg["symbol"]

    # Validate the date range before any download.
    if start_date >= end_date:
        st.error("Start date must be before end date. Adjust the sidebar dates.")
        st.stop()
    if end_date > dt.date.today():
        st.error("End date cannot be in the future.")
        st.stop()

    # Download data with graceful error handling.
    try:
        raw = load_btc_data(start_date, end_date, ticker)
    except ValueError as exc:
        st.warning(str(exc))
        st.stop()
    except RuntimeError as exc:
        st.error(f"Could not load BTC data. {exc}")
        st.stop()
    except Exception as exc:  # Catch-all so the app never crashes cryptically.
        st.error(f"Unexpected error while loading data: {exc}")
        st.stop()

    # Compute indicators and signals.
    df = detect_signals(compute_indicators(raw))
    buys, sells = get_entry_exit_points(df)

    # Headline metrics from the last available row.
    last_row = df.iloc[-1]
    last_price = float(last_row["Close"])
    current_rsi = float(last_row["RSI"]) if pd.notna(last_row["RSI"]) else float("nan")
    state = current_signal_label(int(last_row["Signal"]))

    col1, col2, col3 = st.columns(3)
    col1.metric(
        f"Last BTC price ({currency})", f"{currency_symbol}{last_price:,.2f}"
    )
    col2.metric(
        "Current RSI (14)",
        f"{current_rsi:.1f}" if pd.notna(current_rsi) else "n/a",
    )
    col3.metric("Current signal state", state)

    verdict_title, verdict_level, verdict_reasons = get_market_verdict(last_row, df, currency)
    st.subheader("Should you buy now?")
    if verdict_level == "good":
        st.success(f"✅ {verdict_title}", icon="✅")
    elif verdict_level == "caution":
        st.warning(f"⚠️ {verdict_title}", icon="⚠️")
    else:
        st.error(f"🛑 {verdict_title}", icon="🛑")
    for reason in verdict_reasons:
        st.write(f"- {reason}")
    st.caption("Educational verdict from trend + RSI, not investment advice.")

    render_bull_bear_regime(df, last_row)

    render_bot_backtest(df)

    render_indicator_guide(df, last_row)

    render_news_model_signal()

    st.caption(
        f"Showing {len(df)} trading days from {df.index[0].date()} "
        f"to {df.index[-1].date()} for {ticker}."
    )

    # Charts: prefer Plotly, fall back to Matplotlib.
    st.subheader("Price and Signals")
    st.caption("Background bands: green = bullish regime, red = bearish regime, gray = sideways (from the bull/bear score).")
    if HAS_PLOTLY:
        price_fig = price_figure_plotly(df, buys, sells, currency)
        try:
            add_regime_bands_plotly(price_fig, df, compute_bull_scores(df))
        except Exception:
            pass
        st.plotly_chart(
            price_fig,
            use_container_width=True,
        )
    else:
        st.info("Plotly is not installed; showing a static Matplotlib chart instead.")
        mpl_fig = price_figure_matplotlib(df, buys, sells, currency)
        try:
            add_regime_bands_matplotlib(mpl_fig.axes[0], df, compute_bull_scores(df))
        except Exception:
            pass
        st.pyplot(mpl_fig)

    st.subheader("Momentum (RSI)")
    if HAS_PLOTLY:
        st.plotly_chart(rsi_figure_plotly(df), use_container_width=True)
    else:
        st.pyplot(rsi_figure_matplotlib(df))

    # Data table and signal summary.
    st.subheader("Recent Data")
    rows = st.slider(
        "Rows to show",
        min_value=5,
        max_value=60,
        value=15,
        help="Number of most recent rows displayed in the table below.",
    )
    st.dataframe(df.tail(rows))

    st.subheader("Signal Summary")
    buy_count = int((df["Signal"] == 1).sum())
    sell_count = int((df["Signal"] == -1).sum())
    hold_count = int((df["Signal"] == 0).sum())
    summary = pd.DataFrame(
        {
            "Signal": ["Buy (+1)", "Sell (-1)", "Hold (0)"],
            "Count": [buy_count, sell_count, hold_count],
        }
    )
    scol1, scol2 = st.columns([1, 2])
    scol1.dataframe(summary, hide_index=True)
    scol2.write(
        f"Detected **{buy_count}** buy signals and **{sell_count}** sell signals "
        f"({hold_count} hold days) in the selected window."
    )
    if not buys.empty:
        with st.expander("Show buy (entry) points"):
            st.dataframe(buys[["Close", "SMA20", "SMA50", "RSI"]])
    if not sells.empty:
        with st.expander("Show sell (exit) points"):
            st.dataframe(sells[["Close", "SMA20", "SMA50", "RSI"]])

    st.divider()
    st.caption(
        "Data source: Yahoo Finance via yfinance (BTC-USD / BTC-EUR, auto-adjusted). "
        "Academic project for educational purposes only — not investment advice."
    )


if __name__ == "__main__":
    main()
