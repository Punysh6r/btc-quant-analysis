"""Download BTC-relevant English crypto news via the CryptoCompare v2 API.

The script paginates the news endpoint backwards (newest first) by
advancing ``toTs`` to the oldest seen ``published_on`` timestamp, keeps
only BTC-relevant items, dedupes by id, and overwrites the output CSV on
each run. The file is only written when the fetch succeeds.

Example:
    python download_news.py
    python download_news.py --since 2024-01-01 --max-items 500
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
API_URL = "https://min-api.cryptocompare.com/data/v2/news/"
DEFAULT_SINCE = "2023-01-01"
DEFAULT_MAX_ITEMS = 20000
DEFAULT_OUT = "data/news_btc.csv"
PAGE_SLEEP_S = 1.0
MAX_RETRIES = 3
RETRY_BACKOFF_S = 2.0

BTC_PATTERN = re.compile(r"bitcoin|\bbtc\b", re.IGNORECASE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download BTC-relevant news via the CryptoCompare v2 API."
    )
    parser.add_argument(
        "--since",
        default=DEFAULT_SINCE,
        help="Only keep news published on/after this date (YYYY-MM-DD). "
        "Pagination stops once items get older (default: %(default)s).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=DEFAULT_MAX_ITEMS,
        help="Maximum number of BTC-relevant items to keep (default: %(default)s).",
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


def _as_text(value: object) -> str:
    """Coerce a metadata field (str, list, or dict) to plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(v) for v in value)
    if isinstance(value, dict):
        return " ".join(str(v) for v in value.values())
    return str(value)


def is_btc_relevant(item: dict) -> bool:
    """Check whether a news item mentions Bitcoin/BTC.

    Looks at the categories, tags, title, and body so BTC-specific news
    is kept even when the headline alone is generic.
    """
    haystack = " ".join(
        [
            _as_text(item.get("categories")),
            _as_text(item.get("tags")),
            _as_text(item.get("title")),
            _as_text(item.get("body")),
        ]
    )
    return BTC_PATTERN.search(haystack) is not None


def fetch_page(session, to_ts: int | None) -> list:
    """Fetch one news page (newest first), retrying transient HTTP errors.

    Raises:
        RuntimeError: If the request fails after all retries.
    """
    import requests

    params = {"lang": "EN"}
    if to_ts is not None:
        params["toTs"] = int(to_ts)

    backoff = RETRY_BACKOFF_S
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = session.get(API_URL, params=params, timeout=30)
        except requests.RequestException as exc:
            if attempt >= MAX_RETRIES:
                raise RuntimeError(
                    f"CryptoCompare request failed after {MAX_RETRIES} retries: {exc}"
                ) from exc
            time.sleep(backoff)
            backoff *= 2
            continue
        if response.status_code == 200:
            try:
                return response.json().get("Data", [])
            except ValueError as exc:
                raise RuntimeError(
                    f"CryptoCompare returned non-JSON data: {response.text[:200]}"
                ) from exc
        if response.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
            time.sleep(backoff)
            backoff *= 2
            continue
        raise RuntimeError(
            f"CryptoCompare request failed: HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )
    raise RuntimeError("CryptoCompare request failed after retries.")


def to_record(item_id: str, item: dict) -> dict:
    """Convert a raw API item to a flat CSV record."""
    published_on = int(item["published_on"])
    published_at = datetime.fromtimestamp(published_on, tz=timezone.utc).isoformat()
    source = item.get("source") or _as_text(item.get("source_info")) or ""
    return {
        "id": item_id,
        "published_at": published_at,
        "title": item.get("title", ""),
        "body": item.get("body", ""),
        "url": item.get("url", ""),
        "source": source,
        "categories": _as_text(item.get("categories")),
    }


def main(argv: list[str] | None = None) -> int:
    """Paginate the news API backwards and save BTC-relevant items to CSV."""
    args = parse_args(argv)
    try:
        since_dt = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        print(
            f"Invalid --since date {args.since!r}; expected YYYY-MM-DD.",
            file=sys.stderr,
        )
        return 2
    if args.max_items <= 0:
        print("--max-items must be a positive integer.", file=sys.stderr)
        return 2

    try:
        import requests
    except ImportError:
        print(
            "The 'requests' package is not installed. "
            "Install it with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    session = requests.Session()
    api_key = os.environ.get("CRYPTOCOMPARE_API_KEY", "").strip()
    if api_key:
        session.headers.update({"authorization": f"Apikey {api_key}"})

    collected: dict[str, dict] = {}
    seen_ids: set[str] = set()
    pages = 0
    exhausted = False
    to_ts: int | None = None
    try:
        while len(collected) < args.max_items and not exhausted:
            items = fetch_page(session, to_ts)
            pages += 1
            if not items:
                break
            fresh = 0
            oldest_ts: int | None = None
            for item in items:
                raw_ts = item.get("published_on")
                if raw_ts is None:
                    continue
                ts = int(raw_ts)
                if oldest_ts is None or ts < oldest_ts:
                    oldest_ts = ts
                published_at = datetime.fromtimestamp(ts, tz=timezone.utc)
                if published_at < since_dt:
                    exhausted = True
                    break
                raw_id = item.get("id")
                item_id = str(
                    raw_id
                    if raw_id is not None
                    else item.get("guid", item.get("url", ts))
                )
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)
                fresh += 1
                if not is_btc_relevant(item):
                    continue
                collected[item_id] = to_record(item_id, item)
                if len(collected) >= args.max_items:
                    break
            if exhausted or len(collected) >= args.max_items or oldest_ts is None:
                break
            if fresh == 0:
                # Same window repeats without new items; stop instead of
                # looping forever on an inclusive cursor.
                break
            if to_ts is not None and oldest_ts >= to_ts:
                # The cursor is inclusive; nudge it back one second so the
                # next page is guaranteed to move backwards.
                to_ts = oldest_ts - 1
            else:
                to_ts = oldest_ts
            time.sleep(PAGE_SLEEP_S)
    except RuntimeError as exc:
        print(f"News download failed: {exc}", file=sys.stderr)
        return 1

    if not collected:
        print(
            "No BTC-relevant news found (nothing written). "
            "Try a wider --since window or check API availability.",
            file=sys.stderr,
        )
        return 1

    out_path = resolve_out(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        collected.values(),
        columns=["id", "published_at", "title", "body", "url", "source", "categories"],
    )
    df.to_csv(out_path, index=False)
    coverage_start = df["published_at"].min()
    coverage_end = df["published_at"].max()
    print(
        f"Saved {len(df)} BTC-relevant items to {out_path} "
        f"({pages} page(s), coverage {coverage_start} to {coverage_end})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
