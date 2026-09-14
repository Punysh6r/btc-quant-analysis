"""Download daily BTC-USD OHLCV prices with yfinance and save them to CSV.

The CSV uses a ``Date`` index and keeps the OHLCV columns (including
``Close``). The file is only written when the download succeeds and
returns usable rows.

Example:
    python download_prices.py
    python download_prices.py --start 2020-01-01 --out data/prices_btc_usd.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TICKER = "BTC-USD"
DEFAULT_START = "2014-09-17"
DEFAULT_OUT = "data/prices_btc_usd.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download daily BTC-USD prices with yfinance."
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help="Inclusive start date as YYYY-MM-DD (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="Output CSV path, resolved relative to this script's "
        "directory (default: %(default)s).",
    )
    return parser.parse_args(argv)


def resolve_out(path_str: str) -> Path:
    """Resolve an output path relative to this script's directory."""
    path = Path(path_str)
    return path if path.is_absolute() else SCRIPT_DIR / path


def main(argv: list[str] | None = None) -> int:
    """Download BTC-USD prices and save them to CSV."""
    args = parse_args(argv)
    try:
        start = date.fromisoformat(args.start)
    except ValueError:
        print(
            f"Invalid --start date {args.start!r}; expected YYYY-MM-DD.",
            file=sys.stderr,
        )
        return 2

    try:
        import yfinance as yf
    except ImportError:
        print(
            "The 'yfinance' package is not installed. "
            "Install it with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    try:
        df = yf.download(TICKER, start=start.isoformat(), auto_adjust=True, progress=False)
    except Exception as exc:  # Network errors, invalid dates, etc.
        print(f"Download failed for {TICKER}: {exc}", file=sys.stderr)
        return 1

    if df is None or df.empty:
        print(f"No data returned for {TICKER} since {start}.", file=sys.stderr)
        return 1

    # Newer yfinance versions return MultiIndex columns; flatten them
    # (same pattern as app.py).
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    if "Close" not in df.columns:
        print(
            f"Download for {TICKER} has no 'Close' column: {list(df.columns)}",
            file=sys.stderr,
        )
        return 1

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    out_path = resolve_out(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index_label="Date")
    print(
        f"Saved {len(df)} rows to {out_path} "
        f"({df.index[0].date()} to {df.index[-1].date()})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
