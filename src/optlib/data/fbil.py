"""FBIL MIBOR-OIS and overnight MIBOR loader (public website API, no credentials).

The public endpoints currently return data with a lag of several business days
(observed 23 Oct 2026: latest published date 15 Oct). Callers get the latest fixing on
or before the requested date, and the curve records its age so the validator can flag it.
Same-day rates can be supplied manually with ``OISQuotes`` (e.g. from Bloomberg).
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from ..curves import OISQuotes, tenor_months

BASE_URL = "https://www.fbil.org.in/wasdm"


class FBILError(Exception):
    pass


def _get_json(url: str, params: dict, session=None, timeout: float = 30.0):
    if session is None:
        import requests

        session = requests
    resp = session.get(url, params=params, timeout=timeout)
    if resp.status_code != 200:
        raise FBILError(f"FBIL HTTP {resp.status_code} for {url}")
    try:
        return resp.json()
    except ValueError as exc:
        raise FBILError(f"FBIL returned non-JSON for {url}") from exc


def parse_mibor_ois(rows: list[dict]) -> dict[date, dict[str, float]]:
    out: dict[date, dict[str, float]] = defaultdict(dict)
    for r in rows:
        d = datetime.fromisoformat(r["processRunDate"][:10]).date()
        tenor = r["tenorName"].strip().upper()
        tenor_months(tenor)
        out[d][tenor] = float(r["rate"])
    return dict(out)


def parse_overnight(rows: list[dict]) -> dict[date, float]:
    return {datetime.fromisoformat(r["processRunDate"][:10]).date(): float(r["rate"]) for r in rows}


class FBILClient:
    def __init__(self, session=None, cache_dir: Optional[Path] = None,
                 fetch: Optional[Callable[[str, dict], object]] = None):
        self.session = session
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._fetch = fetch or (lambda url, params: _get_json(url, params, self.session))

    def _range(self, path: str, start: date, end: date):
        params = {"fromDate": start.isoformat(), "toDate": end.isoformat(), "authenticated": "false"}
        cache = None
        if self.cache_dir:
            cache = self.cache_dir / f"fbil_{path.replace('/', '_')}_{start}_{end}.json"
            if cache.is_file():
                return json.loads(cache.read_text())
        data = self._fetch(f"{BASE_URL}/{path}", params)
        if not isinstance(data, list):
            raise FBILError(f"unexpected FBIL response for {path}")
        if cache is not None and end < date.today():  # only cache closed date ranges
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data))
        return data

    def ois_quotes(self, on_or_before: date, lookback_days: int = 21) -> OISQuotes:
        """Latest complete MIBOR-OIS fixing (with overnight MIBOR) on or before a date."""
        start = on_or_before - timedelta(days=lookback_days)
        ois = parse_mibor_ois(self._range("miborois/fetchfiltered", start, on_or_before))
        ovn = parse_overnight(self._range("ovnmibor/fetchfiltered", start, on_or_before))
        for d in sorted(ois, reverse=True):
            if d <= on_or_before and d in ovn and len(ois[d]) >= 5:
                return OISQuotes(fixing_date=d, overnight=ovn[d], tenors=ois[d], source="FBIL")
        raise FBILError(f"no FBIL MIBOR-OIS fixing between {start} and {on_or_before}")
