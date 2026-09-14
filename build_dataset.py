"""Label BTC news with 24h forward returns to build a training dataset.

For each news item, the reference close is the last daily Close at or
before the publication time, and the forward close is the first daily
Close strictly after publication time + 24h (next available day, so
weekends/holidays are handled). Items without a forward close are
skipped. Labels use +/-2% thresholds: BUY (> +2%), SELL (< -2%),
otherwise HOLD.

Example:
    python build_dataset.py
    python build_dataset.py --prices data/prices_btc_usd.csv --news data/news_btc.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRICES = "data/prices_btc_usd.csv"
DEFAULT_NEWS = "data/news_btc.csv"
DEFAULT_OUT = "data/dataset.csv"
THRESHOLD = 0.02
HORIZON_HOURS = 24
BODY_CHARS = 2000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Label BTC news with 24h forward returns."
    )
    parser.add_argument(
        "--prices",
        default=DEFAULT_PRICES,
        help="Prices CSV path with a Date index and a Close column, "
        "resolved relative to this script's directory (default: %(default)s).",
    )
    parser.add_argument(
        "--news",
        default=DEFAULT_NEWS,
        help="News CSV path with published_at/title/body columns, "
        "resolved relative to this script's directory (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="Output dataset CSV path, resolved relative to this script's "
        "directory (default: %(default)s).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help="Label threshold as a fraction (default: %(default)s). "
        "BUY above +threshold, SELL below -threshold, else HOLD.",
    )
    return parser.parse_args(argv)


def resolve(path_str: str) -> Path:
    """Resolve a path relative to this script's directory."""
    path = Path(path_str)
    return path if path.is_absolute() else SCRIPT_DIR / path


def label_return(ret: float, threshold: float = THRESHOLD) -> str:
    """Map a 24h forward return to a BUY / SELL / HOLD label."""
    if ret > threshold:
        return "BUY"
    if ret < -threshold:
        return "SELL"
    return "HOLD"


def label_news(
    prices: pd.DataFrame, news: pd.DataFrame, threshold: float = THRESHOLD
) -> tuple[list[dict], int, int]:
    """Join news items with reference/forward closes and label them.

    Args:
        prices: Daily prices sorted ascending with a naive datetime index
            and a ``Close`` column.
        news: News items with tz-aware ``published_at``, ``title`` and
            optional ``body``/``url``/``source`` columns, sorted ascending.

    Returns:
        Tuple ``(rows, skipped_no_ref, skipped_no_fwd)`` with one dict per
        labeled item (``published_at, title, text, url, source, close_ref,
        close_fwd, ret_24h, label``).
    """
    price_index = prices.index
    closes = prices["Close"].to_numpy()
    rows: list[dict] = []
    skipped_no_ref = 0
    skipped_no_fwd = 0
    horizon = pd.Timedelta(hours=HORIZON_HOURS)
    for _, item in news.iterrows():
        published_at = item["published_at"]
        if isinstance(published_at, pd.Timestamp) and published_at.tzinfo is not None:
            pub_naive = published_at.tz_localize(None)
        else:
            pub_naive = published_at

        ref_pos = price_index.searchsorted(pub_naive, side="right") - 1
        if ref_pos < 0:
            skipped_no_ref += 1
            continue
        fwd_pos = price_index.searchsorted(pub_naive + horizon, side="right")
        if fwd_pos >= len(price_index):
            skipped_no_fwd += 1
            continue

        close_ref = float(closes[ref_pos])
        close_fwd = float(closes[fwd_pos])
        if close_ref == 0:
            skipped_no_ref += 1
            continue
        ret = (close_fwd - close_ref) / close_ref

        title = str(item["title"])
        body = str(item.get("body", "") or "")
        text = title if not body.strip() else f"{title}\n{body.strip()[:BODY_CHARS]}"
        rows.append(
            {
                "published_at": published_at.isoformat(),
                "title": title,
                "text": text,
                "url": item.get("url", ""),
                "source": item.get("source", ""),
                "close_ref": close_ref,
                "close_fwd": close_fwd,
                "ret_24h": ret,
                "label": label_return(ret, threshold),
            }
        )
    return rows, skipped_no_ref, skipped_no_fwd


def main(argv: list[str] | None = None) -> int:
    """Join news with forward prices and write the labeled dataset."""
    args = parse_args(argv)
    prices_path = resolve(args.prices)
    news_path = resolve(args.news)
    out_path = resolve(args.out)

    for path, name in ((prices_path, "--prices"), ((news_path, "--news"))):
        if not path.exists():
            print(f"Input file for {name} not found: {path}", file=sys.stderr)
            return 1

    try:
        prices = pd.read_csv(prices_path, parse_dates=["Date"], index_col="Date")
    except Exception as exc:
        print(f"Could not read prices file {prices_path}: {exc}", file=sys.stderr)
        return 1
    if "Close" not in prices.columns:
        print(f"Prices file {prices_path} has no 'Close' column.", file=sys.stderr)
        return 1
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    try:
        news = pd.read_csv(news_path)
    except Exception as exc:
        print(f"Could not read news file {news_path}: {exc}", file=sys.stderr)
        return 1
    if "published_at" not in news.columns or "title" not in news.columns:
        print(
            f"News file {news_path} must contain 'published_at' and 'title' columns.",
            file=sys.stderr,
        )
        return 1
    news["published_at"] = pd.to_datetime(news["published_at"], utc=True, errors="coerce")
    news = news.dropna(subset=["published_at"]).sort_values("published_at")
    if news.empty:
        print(f"No usable news rows in {news_path}.", file=sys.stderr)
        return 1

    rows, skipped_no_ref, skipped_no_fwd = label_news(prices, news, args.threshold)

    if not rows:
        print(
            "No dataset rows could be built (every item lacked a reference "
            f"or forward close; no-ref={skipped_no_ref}, no-fwd={skipped_no_fwd}).",
            file=sys.stderr,
        )
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = pd.DataFrame(
        rows,
        columns=[
            "published_at",
            "title",
            "text",
            "url",
            "source",
            "close_ref",
            "close_fwd",
            "ret_24h",
            "label",
        ],
    )
    dataset.to_csv(out_path, index=False)
    dist = dataset["label"].value_counts()
    dist_str = ", ".join(
        f"{label}={dist.get(label, 0)}" for label in ("BUY", "SELL", "HOLD")
    )
    print(
        f"Saved {len(dataset)} labeled rows to {out_path} "
        f"(threshold={args.threshold:.4f}; {dist_str}; skipped no-ref={skipped_no_ref}, no-fwd={skipped_no_fwd})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
