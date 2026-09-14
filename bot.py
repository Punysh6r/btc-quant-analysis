"""Threshold trading bot on the bull/bear regime score (backtest only, no real money).

Strategy (hysteresis, no lookahead):
  - Compute the bull score 0-100 for every day with the same rules as
    bull_bear.get_bull_bear_regime (trend + momentum + RSI + fresh signal + 90d high).
  - If score >= buy_threshold (high) and flat -> BUY at Close (all-in).
  - If score <= sell_threshold (low) and long -> SELL at Close (exit to cash).
  - Otherwise HOLD. buy_threshold must be > sell_threshold.

Example:
  python bot.py
  python bot.py --buy 70 --sell 30 --capital 10000
  python bot.py --prices data/prices_btc_usd.csv --buy 65 --sell 40 --out data/bot_trades.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRICES = "data/prices_btc_usd.csv"
DEFAULT_OUT = "data/bot_trades.csv"
DEFAULT_BUY = 70
DEFAULT_SELL = 30
DEFAULT_CAPITAL = 10000.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest a threshold bot on the bull/bear score (no real trading)."
    )
    parser.add_argument("--prices", default=DEFAULT_PRICES)
    parser.add_argument("--buy", type=float, default=DEFAULT_BUY)
    parser.add_argument("--sell", type=float, default=DEFAULT_SELL)
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    return parser.parse_args(argv)


def compute_bull_scores(df: pd.DataFrame) -> pd.Series:
    """Recompute the bull/bear score for every row without lookahead.

    Same rules as bull_bear.get_bull_bear_regime, applied causally:
    each day only sees data up to that day (90-day high is trailing).
    Expects Close, SMA20, SMA50, RSI, Signal columns.
    Returns a Series 5-95 indexed like df.
    """
    closes = df["Close"].to_numpy(dtype=float)
    sma20 = df["SMA20"].to_numpy(dtype=float)
    sma50 = df["SMA50"].to_numpy(dtype=float)
    rsi = df["RSI"].to_numpy(dtype=float)
    sig = df["Signal"].to_numpy(dtype=int)
    idx = df.index
    n = len(df)
    scores = np.full(n, 50.0)

    # Precompute trailing 90-day high (causal: includes today).
    roll_high = df["Close"].rolling(window=90, min_periods=1).max().to_numpy(dtype=float)

    # Positions of past signals for the freshness bonus (causal loop is cheap enough
    # for a few thousand daily rows and keeps the rule identical to the live panel).
    last_buy = np.full(n, -10**9)
    last_sell = np.full(n, -10**9)
    last_b = -10**9
    last_s = -10**9
    for i in range(n):
        if sig[i] == 1:
            last_b = i
        elif sig[i] == -1:
            last_s = i
        last_buy[i] = last_b
        last_sell[i] = last_s

    # Calendar days between rows for the 5-day freshness window.
    # df index is datetime; fall back to row distance if not.
    try:
        days = idx.to_series().diff().dt.days.fillna(1).to_numpy()
        abs_day = np.cumsum(np.where(np.isnan(days), 1, days))
    except Exception:
        abs_day = np.arange(n)

    for i in range(n):
        score = 50.0
        # 1. Trend.
        if sma50[i] != 0 and not np.isnan(sma50[i]) and not np.isnan(sma20[i]):
            gap = (sma20[i] - sma50[i]) / sma50[i] * 100.0
            if sma20[i] > sma50[i]:
                score += 20.0 + min(max(gap, 0.0) * 5.0, 10.0)
            else:
                score -= 20.0 + min(abs(gap) * 5.0, 10.0)
        # 2. Momentum.
        if sma20[i] != 0 and not np.isnan(sma20[i]) and not np.isnan(closes[i]):
            mom = (closes[i] - sma20[i]) / sma20[i] * 100.0
            if closes[i] > sma20[i]:
                score += 10.0 + min(max(mom, 0.0) * 2.0, 10.0)
            else:
                score -= 10.0 + min(abs(mom) * 2.0, 10.0)
        # 3. RSI.
        r = rsi[i]
        if np.isnan(r):
            pass
        elif r < 30:
            score += 5.0
        elif r < 50:
            score += 10.0
        elif r < 70:
            score += 15.0
        elif r <= 80:
            score -= 10.0
        else:
            score -= 20.0
        # 4. Fresh signals.
        if sig[i] == 1:
            score += 15.0
        elif sig[i] == -1:
            score -= 15.0
        else:
            if last_b > -10**8 and (abs_day[i] - abs_day[int(last_b)]) <= 5:
                score += 5.0
            if last_s > -10**8 and (abs_day[i] - abs_day[int(last_s)]) <= 5:
                score -= 5.0
        # 5. Distance to trailing 90-day high.
        high = roll_high[i]
        if high != 0 and not np.isnan(high):
            dd = (closes[i] - high) / high * 100.0
            if dd >= -3.0:
                score += 10.0
            elif dd >= -10.0:
                score += 0.0
            elif dd >= -20.0:
                score -= 10.0
            else:
                score -= 20.0
        scores[i] = max(5.0, min(95.0, score))
    return pd.Series(scores, index=idx, name="bull_score")


def backtest(
    df: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
    capital: float = DEFAULT_CAPITAL,
) -> tuple[pd.DataFrame, dict]:
    """Run the hysteresis backtest. Returns (trades_df, stats dict)."""
    if not (0 < sell_threshold < buy_threshold < 100):
        raise ValueError(
            f"Need 0 < sell ({sell_threshold}) < buy ({buy_threshold}) < 100."
        )
    if capital <= 0:
        raise ValueError("Capital must be positive.")
    scores = compute_bull_scores(df)
    closes = df["Close"].astype(float)

    cash = float(capital)
    btc = 0.0
    position = 0  # 0 flat, 1 long
    entry_price = 0.0
    entry_date = None
    trades: list[dict] = []
    equity = []

    for ts, price, score in zip(df.index, closes, scores):
        price = float(price)
        if position == 0 and score >= buy_threshold:
            btc = cash / price if price > 0 else 0.0
            entry_price = price
            entry_date = ts
            cash = 0.0
            position = 1
            trades.append(
                {"date": ts, "side": "BUY", "price": price, "bull_score": float(score)}
            )
        elif position == 1 and score <= sell_threshold:
            cash = btc * price
            ret = (price - entry_price) / entry_price if entry_price else 0.0
            trades.append(
                {
                    "date": ts,
                    "side": "SELL",
                    "price": price,
                    "bull_score": float(score),
                    "entry_price": entry_price,
                    "entry_date": entry_date,
                    "trade_return": ret,
                }
            )
            btc = 0.0
            position = 0
        equity.append(cash + btc * price)

    equity_s = pd.Series(equity, index=df.index, name="equity")
    # Close open position at the end for reporting (keep trades as-is).
    final_price = float(closes.iloc[-1])
    final_value = cash + btc * final_price
    if position == 1:
        ret = (final_price - entry_price) / entry_price if entry_price else 0.0
        trades.append(
            {
                "date": df.index[-1],
                "side": "OPEN_END",
                "price": final_price,
                "bull_score": float(scores.iloc[-1]),
                "entry_price": entry_price,
                "entry_date": entry_date,
                "trade_return": ret,
            }
        )

    first = float(closes.iloc[0])
    buy_hold = float(capital) * (final_price / first) if first > 0 else float(capital)
    total_ret = (final_value - capital) / capital if capital else 0.0
    bh_ret = (buy_hold - capital) / capital if capital else 0.0

    closed = [t for t in trades if t["side"] == "SELL"]
    wins = sum(1 for t in closed if t.get("trade_return", 0) > 0)
    # Max drawdown on equity curve.
    roll_max = equity_s.cummax()
    dd = (equity_s - roll_max) / roll_max.replace(0, float("nan"))
    max_dd = float(dd.min()) if len(dd) else 0.0

    stats = {
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "capital": capital,
        "final_value": round(final_value, 2),
        "total_return": round(total_ret * 100, 2),
        "buy_hold_value": round(buy_hold, 2),
        "buy_hold_return": round(bh_ret * 100, 2),
        "n_buys": sum(1 for t in trades if t["side"] == "BUY"),
        "n_sells": len(closed),
        "win_rate": round(wins / len(closed) * 100, 1) if closed else 0.0,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "in_position_end": bool(position == 1),
        "last_bull_score": round(float(scores.iloc[-1]), 1) if len(scores) else None,
    }
    out = pd.DataFrame(
        {
            "bull_score": scores.round(1),
            "close": closes,
            "equity": equity_s.round(2),
        }
    )
    trades_df = pd.DataFrame(trades)
    return out, stats, trades_df


def load_prices(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    prices = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
    if "Close" not in prices.columns:
        raise RuntimeError(f"{path} has no 'Close' column.")
    return prices


from indicators import add_indicators  # shared with app.py


def render_bot_backtest(df: pd.DataFrame) -> None:
    """Streamlit panel: sidebar thresholds + backtest on the loaded window."""
    import streamlit as st

    st.sidebar.divider()
    st.sidebar.subheader("Trading bot (backtest)")
    buy_th = st.sidebar.slider(
        "Buy threshold (high)", 50, 95, DEFAULT_BUY,
        help="BUY when bull score >= this. Must stay above the sell threshold.",
    )
    sell_th = st.sidebar.slider(
        "Sell threshold (low)", 5, 50, DEFAULT_SELL,
        help="SELL when bull score <= this. Must stay below the buy threshold.",
    )
    capital = st.sidebar.number_input(
        "Starting capital", min_value=100.0, value=float(DEFAULT_CAPITAL), step=500.0
    )
    if sell_th >= buy_th:
        st.warning(
            f" thresholds cross: buy ({buy_th}) must be above sell ({sell_th}). "
            "Move the sliders apart."
        )
        return
    try:
        curve, stats, trades = backtest(df, buy_th, sell_th, float(capital))
    except ValueError as exc:
        st.warning(str(exc))
        return

    st.subheader("Threshold Bot (backtest only, no real money)")
    st.caption(
        f"Hysteresis on the bull score: BUY >= {buy_th}, SELL <= {sell_th}. "
        "Academic simulation on the selected window, not investment advice."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final value", f"${stats['final_value']:,.0f} ({stats['total_return']:+.1f}%)")
    m2.metric("Buy & hold", f"${stats['buy_hold_value']:,.0f} ({stats['buy_hold_return']:+.1f}%)")
    m3.metric("Trades", f"{stats['n_buys']} buys / {stats['n_sells']} sells")
    m4.metric("Win rate / Max DD", f"{stats['win_rate']}% / {stats['max_drawdown_pct']}%")
    try:
        st.line_chart(curve[["equity"]])
    except Exception:
        pass
    if stats["in_position_end"]:
        st.info(
            f"Bot ends LONG (in position) — last bull score {stats['last_bull_score']}%."
        )
    else:
        st.info(
            f"Bot ends FLAT (in cash) — last bull score {stats['last_bull_score']}%."
        )
    if not trades.empty:
        with st.expander("Show bot trades"):
            st.dataframe(trades)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not (0 < args.sell < args.buy < 100):
        print(
            f"Need 0 < sell ({args.sell}) < buy ({args.buy}) < 100.",
            file=sys.stderr,
        )
        return 1
    try:
        prices = load_prices(args.prices)
    except Exception as exc:
        print(f"Could not read prices: {exc}", file=sys.stderr)
        return 1
    df = add_indicators(prices)
    try:
        curve, stats, trades = backtest(df, args.buy, args.sell, args.capital)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        f"Thresholds buy={stats['buy_threshold']} sell={stats['sell_threshold']} "
        f"capital=${stats['capital']:,.0f}"
    )
    print(
        f"Bot: ${stats['final_value']:,.2f} ({stats['total_return']:+.2f}%) | "
        f"Buy&Hold: ${stats['buy_hold_value']:,.2f} ({stats['buy_hold_return']:+.2f}%)"
    )
    print(
        f"Trades: {stats['n_buys']} buys / {stats['n_sells']} sells, "
        f"win rate {stats['win_rate']}%, max DD {stats['max_drawdown_pct']}%."
    )
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = SCRIPT_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_path, index=False)
    print(f"Saved {len(trades)} trade rows to {out_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
