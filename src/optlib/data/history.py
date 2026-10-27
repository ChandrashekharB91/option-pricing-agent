"""Daily index and India VIX history for VaR, backtests and stress tests.

Source: Upstox's public daily candles (no token needed), fetched one calendar year per request
(longer ranges are rejected) and cached as CSV. A committed sample (2015 to 23 Oct 2026) lets the
risk engine run offline.

CSV layout: date, NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, NIFTYNXT50, INDIAVIX (closing levels;
blank where an index did not exist yet).
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote

import numpy as np

from .upstox import UNDERLYINGS

VIX_KEY = "NSE_INDEX|India VIX"
SERIES = {**UNDERLYINGS, "INDIAVIX": VIX_KEY}
CANDLE_URL = "https://api.upstox.com/v2/historical-candle/{key}/day/{to}/{frm}"


@dataclass
class History:
    dates: list[date]
    closes: dict[str, np.ndarray]  # series name -> closes aligned to dates (NaN where missing)

    def series(self, name: str) -> np.ndarray:
        if name not in self.closes:
            raise KeyError(f"no history for {name}; have {', '.join(self.closes)}")
        return self.closes[name]

    def before(self, day: date) -> "History":
        """History up to and excluding ``day`` (no look-ahead when pricing intraday on ``day``)."""
        n = sum(d < day for d in self.dates)
        return History(self.dates[:n], {k: v[:n] for k, v in self.closes.items()})

    def to_csv(self, path: Path) -> None:
        names = list(self.closes)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["date", *names])
            for i, d in enumerate(self.dates):
                w.writerow([d.isoformat()] + ["" if math.isnan(self.closes[n][i]) else f"{self.closes[n][i]:.4f}"
                                              for n in names])

    @classmethod
    def from_csv(cls, path: Path) -> "History":
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        names = [k for k in rows[0] if k != "date"]
        dates = [date.fromisoformat(r["date"]) for r in rows]
        return cls(dates, {n: np.array([float(r[n]) if r[n] else np.nan for r in rows]) for n in names})


def _get_json(url: str, session=None):
    if session is None:
        import requests

        session = requests
    resp = session.get(url, timeout=60, headers={"Accept": "application/json"})
    try:
        body = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"history: non-JSON response from {url}") from exc
    if resp.status_code != 200 or body.get("status") != "success":
        raise RuntimeError(f"history: {resp.status_code} {str(body)[:200]}")
    return body["data"]["candles"]


def fetch_history(start: date, end: date, names=None, session=None,
                  fetch: Optional[Callable[[str], list]] = None) -> History:
    """Daily closes from Upstox's public candle API, one calendar year per request."""
    names = list(names or SERIES)
    fetch = fetch or (lambda url: _get_json(url, session))
    per: dict[str, dict[date, float]] = {}
    for n in names:
        per[n] = {}
        for y in range(start.year, end.year + 1):
            frm, to = max(start, date(y, 1, 1)), min(end, date(y, 12, 31))
            for c in fetch(CANDLE_URL.format(key=quote(SERIES[n], safe=""), to=to.isoformat(), frm=frm.isoformat())):
                per[n][date.fromisoformat(c[0][:10])] = float(c[4])
    dates = sorted(set().union(*[set(v) for v in per.values()]))
    return History(dates, {n: np.array([per[n].get(d, np.nan) for d in dates]) for n in names})


def load_history(cache_dir: Path, end: date, start: date = date(2015, 1, 1), session=None) -> History:
    """Cached history up to ``end`` (refetched at most once per end date)."""
    cache_dir = Path(cache_dir)
    path = cache_dir / f"index_history_{start}_{end}.csv"
    if path.is_file():
        return History.from_csv(path)
    hist = fetch_history(start, end, session=session)
    cache_dir.mkdir(parents=True, exist_ok=True)
    hist.to_csv(path)
    return hist
