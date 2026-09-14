"""Download BTC-relevant news from free RSS feeds (no API key required).

Drop-in alternative to ``download_news.py`` (CryptoCompare, now requires an
API key). Fetches CoinDesk / Cointelegraph / Bitcoin Magazine RSS, keeps
only BTC-relevant items, and writes the SAME CSV schema that
``build_dataset.py`` expects::

    id,published_at,title,body,url,source,categories

Example:
    python download_news_rss.py
    python download_news_rss.py --since 2025-01-01 --max-items 2000
    python download_news_rss.py --out data/news_btc.csv
    python download_news_rss.py --append  # weekly run: merge without losing history
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SINCE = "2023-01-01"
DEFAULT_MAX_ITEMS = 20000
DEFAULT_OUT = "data/news_btc.csv"

DEFAULT_FEEDS: list[tuple[str, str]] = [
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Cointelegraph", "https://cointelegraph.com/rss"),
    ("BitcoinMagazine", "https://bitcoinmagazine.com/.rss/full/"),
]

USER_AGENT = "Mozilla/5.0 (TFM-BTC research; Streamlit academic project)"
FETCH_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF_S = 2.0

BTC_PATTERN = re.compile(r"bitcoin|\bbtc\b", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")

CONTENT_NS = "{http://purl.org/rss/1.0/modules/content/}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download BTC news from free RSS feeds (no API key)."
    )
    parser.add_argument(
        "--since",
        default=DEFAULT_SINCE,
        help="Only keep news published on/after this date (YYYY-MM-DD) "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=DEFAULT_MAX_ITEMS,
        help="Maximum number of BTC-relevant items to keep, most recent "
        "first (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="Output CSV path, resolved relative to this script's directory "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge with the existing CSV instead of overwriting, "
        "deduping by id (use for weekly accumulation).",
    )
    return parser.parse_args(argv)


def resolve_out(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else SCRIPT_DIR / path


def strip_html(value: str) -> str:
    """Remove HTML tags and unescape entities, collapsing whitespace."""
    if not value:
        return ""
    text = TAG_RE.sub(" ", value)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_pubdate(raw: str) -> datetime | None:
    """Parse an RSS pubDate to a tz-aware UTC datetime."""
    if not raw or not raw.strip():
        return None
    try:
        dt = parsedate_to_datetime(raw.strip())
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fetch_feed(session, name: str, url: str) -> bytes:
    """Fetch one RSS feed, retrying transient HTTP errors."""
    import requests

    backoff = RETRY_BACKOFF_S
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=FETCH_TIMEOUT)
        except requests.RequestException as exc:
            if attempt >= MAX_RETRIES:
                raise RuntimeError(f"[{name}] request failed: {exc}") from exc
            time.sleep(backoff)
            backoff *= 2
            continue
        if resp.status_code == 200 and resp.content:
            return resp.content
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
            time.sleep(backoff)
            backoff *= 2
            continue
        raise RuntimeError(
            f"[{name}] HTTP {resp.status_code}: {resp.text[:200]}"
        )
    raise RuntimeError(f"[{name}] fetch failed after retries.")


def parse_feed_xml(source: str, payload: bytes) -> list[dict]:
    """Parse RSS 2.0 items into raw dicts with title/link/body/date."""
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise RuntimeError(f"[{source}] invalid XML: {exc}") from exc

    items = root.findall(".//item")
    records: list[dict] = []
    for it in items:
        def text(tag: str) -> str:
            el = it.find(tag)
            return (el.text or "").strip() if el is not None and el.text else ""

        title = text("title")
        link = text("link")
        guid = text("guid")
        pub_raw = text("pubDate") or text("date")
        desc = text("description")
        content_el = it.find(f"{CONTENT_NS}encoded")
        content_raw = (
            content_el.text.strip()
            if content_el is not None and content_el.text
            else ""
        )
        categories = [
            (c.text or "").strip()
            for c in it.findall("category")
            if c.text and c.text.strip()
        ]
        records.append(
            {
                "source": source,
                "title": html.unescape(title),
                "link": link or guid,
                "guid": guid or link,
                "pub_raw": pub_raw,
                "description": desc,
                "content": content_raw,
                "categories": categories,
            }
        )
    return records


def to_record(raw: dict) -> dict | None:
    """Normalize one raw RSS item to the shared news CSV schema."""
    pub_dt = parse_pubdate(raw["pub_raw"])
    if pub_dt is None:
        return None
    title = raw["title"].strip()
    if not title:
        return None
    body_parts = [strip_html(raw["description"]), strip_html(raw["content"])]
    # Prefer full content when available, else description.
    body = body_parts[1] if len(body_parts[1]) > len(body_parts[0]) else body_parts[0]
    categories = ", ".join(raw["categories"])
    haystack = " ".join([title, body, categories])
    if not BTC_PATTERN.search(haystack):
        return None
    item_id = (raw["guid"] or raw["link"] or f"{title}-{pub_dt.isoformat()}").strip()
    return {
        "id": item_id,
        "published_at": pub_dt.isoformat(),
        "published_dt": pub_dt,
        "title": title,
        "body": body,
        "url": raw["link"],
        "source": raw["source"],
        "categories": categories,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        since_dt = datetime.strptime(args.since, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
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
    session.headers.update({"User-Agent": USER_AGENT})

    collected: dict[str, dict] = {}
    try:
        for source, url in DEFAULT_FEEDS:
            try:
                payload = fetch_feed(session, source, url)
            except RuntimeError as exc:
                print(f"Warning: {exc}", file=sys.stderr)
                continue
            try:
                raw_items = parse_feed_xml(source, payload)
            except RuntimeError as exc:
                print(f"Warning: {exc}", file=sys.stderr)
                continue
            print(f"[{source}] {len(raw_items)} items in feed.", file=sys.stderr)
            for raw in raw_items:
                rec = to_record(raw)
                if rec is None:
                    continue
                if rec["published_dt"] < since_dt:
                    continue
                if rec["id"] not in collected:
                    collected[rec["id"]] = rec
                # Gentle pause avoided per-feed; feeds are only 3 requests.
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 1

    if not collected:
        print(
            "No BTC-relevant news found in RSS feeds (nothing written). "
            "RSS only covers recent weeks; run with --append weekly to accumulate history.",
            file=sys.stderr,
        )
        return 1

    if args.append:
        pre_path = resolve_out(args.out)
        if pre_path.exists():
            try:
                prev = __import__("pandas").read_csv(pre_path)
                for _, row in prev.iterrows():
                    rid = str(row.get("id", "") or "").strip()
                    if not rid or rid in collected:
                        continue
                    try:
                        pdt = __import__("datetime").datetime.fromisoformat(str(row.get("published_at", "")))
                        if pdt.tzinfo is None:
                            pdt = pdt.replace(tzinfo=__import__("datetime").timezone.utc)
                    except ValueError:
                        continue
                    if pdt < since_dt:
                        continue
                    collected[rid] = {
                        "id": rid,
                        "published_at": pdt.isoformat(),
                        "published_dt": pdt,
                        "title": str(row.get("title", "")),
                        "body": str(row.get("body", "")),
                        "url": str(row.get("url", "")),
                        "source": str(row.get("source", "")),
                        "categories": str(row.get("categories", "")),
                    }
                print("Merged with " + str(len(prev)) + " previous rows (--append).", file=sys.stderr)
            except Exception as exc:
                print("Warning: could not merge previous CSV (" + str(exc) + ").", file=sys.stderr)

    # Deterministic order: oldest first; keep most recent max_items.
    ordered = sorted(collected.values(), key=lambda r: r["published_dt"])
    if len(ordered) > args.max_items:
        ordered = ordered[-args.max_items :]

    out_path = resolve_out(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "id": r["id"],
                "published_at": r["published_at"],
                "title": r["title"],
                "body": r["body"],
                "url": r["url"],
                "source": r["source"],
                "categories": r["categories"],
            }
            for r in ordered
        ],
        columns=["id", "published_at", "title", "body", "url", "source", "categories"],
    )
    df.to_csv(out_path, index=False)
    print(
        f"Saved {len(df)} BTC-relevant items to {out_path} "
        f"(RSS: {', '.join(s for s, _ in DEFAULT_FEEDS)}; "
        f"coverage {df['published_at'].min()} to {df['published_at'].max()})."
    )
    print(
        "Note: RSS covers recent weeks only. "
        "Tip: run this script weekly and append results to grow a historical dataset for the TFM."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
