"""Indian market news headlines from public RSS feeds, for the app's News section and the daily email.

Only headlines, source, time and the link are shown (no article text): readers click through to the publisher.
Headlines are ranked by relevance to index options (NIFTY, Sensex, RBI, FII flows, volatility...) and recency,
and duplicates across feeds are dropped. Results are cached for ten minutes.

NEWS_FEEDS (optional setting): comma-separated "Name|url" pairs to replace the defaults.
"""
from __future__ import annotations

import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Optional

import requests

DEFAULT_FEEDS = [
    ("Economic Times", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("Mint", "https://www.livemint.com/rss/markets"),
    ("Business Standard", "https://www.business-standard.com/rss/markets-106.rss"),
    ("Google News", "https://news.google.com/rss/search?q=NIFTY%20OR%20Sensex%20OR%20RBI%20OR%20%22F%26O%22"
                    "&hl=en-IN&gl=IN&ceid=IN:en"),
]
# words that make a headline relevant to an index-options desk, with weights
KEYWORDS = {
    r"\bnifty\b": 5, r"\bsensex\b": 4, r"\bbank ?nifty\b": 4, r"\bf&o\b|\bderivative": 4, r"\boption": 3,
    r"\bvix\b|\bvolatil": 4, r"\bexpiry\b": 3, r"\brbi\b|\brepo\b|\bmonetary policy": 4, r"\bfii|\bfpi|\bdii": 3,
    r"\binflation|\bcpi\b|\bgdp\b": 3, r"\brupee\b": 2, r"\bcrude\b|\boil prices": 2, r"\bfed\b|\bus fed": 2,
    r"\bsebi\b": 2, r"\bmarket(s)?\b": 1, r"\bstock(s)?\b|\bshares\b": 1, r"\bbudget\b": 2,
}
UA = {"User-Agent": "Mozilla/5.0 (OptionPricingAgent news reader)"}
_cache: dict = {}
_lock = threading.Lock()


def _parse(xml_text: str, source: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        if not title or not link.startswith("http"):
            continue
        src = source
        if source == "Google News":  # "Headline - Publisher"
            pub = it.findtext("source")
            if pub:
                src = pub.strip()
                title = re.sub(r"\s+-\s+" + re.escape(src) + r"\s*$", "", title)
        try:
            when = parsedate_to_datetime(it.findtext("pubDate") or "")
            when = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            when = None
        out.append({"title": title, "link": link, "source": src,
                    "published": when.isoformat() if when else None, "_ts": when.timestamp() if when else 0.0})
    return out


def relevance(title: str) -> int:
    t = title.lower()
    return sum(w for pat, w in KEYWORDS.items() if re.search(pat, t))


def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower())[:70]


def headlines(limit: int = 12, get: Optional[Callable[[str, str], str]] = None, ttl: float = 600,
              fetch: Optional[Callable[[str], str]] = None, max_age_hours: float = 48) -> list[dict]:
    """The most relevant recent market headlines across the feeds (cached ``ttl`` seconds)."""
    feeds = DEFAULT_FEEDS
    if get and get("NEWS_FEEDS", ""):
        feeds = [tuple(x.split("|", 1)) for x in get("NEWS_FEEDS", "").split(",") if "|" in x]
    key = (tuple(feeds), limit)
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    fetch = fetch or (lambda url: requests.get(url, headers=UA, timeout=12).text)
    items: list[dict] = []
    for name, url in feeds:
        try:
            items += _parse(fetch(url), name)
        except Exception:  # noqa: BLE001 - one feed down must not empty the section
            continue
    now = time.time()
    seen, ranked = set(), []
    for it in sorted(items, key=lambda x: -x["_ts"]):
        k = _norm(it["title"])
        if k in seen or (it["_ts"] and now - it["_ts"] > max_age_hours * 3600):
            continue
        seen.add(k)
        it["relevance"] = relevance(it["title"])
        ranked.append(it)
    # relevant first, then the most recent; keep a few general market stories when relevance ties
    ranked.sort(key=lambda x: (-(min(x["relevance"], 6)), -x["_ts"]))
    out = [{k: v for k, v in it.items() if k != "_ts"} for it in ranked[:limit]]
    with _lock:
        _cache[key] = (time.time(), out)
    return out


def ago(iso: Optional[str]) -> str:
    if not iso:
        return ""
    secs = (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds()
    if secs < 3600:
        return f"{max(1, int(secs // 60))} min ago"
    if secs < 86400:
        return f"{int(secs // 3600)} h ago"
    return f"{int(secs // 86400)} d ago"
