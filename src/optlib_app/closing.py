"""Closing snapshot: keep the 15:20 IST market picture for use after the close.

After NSE closes (15:30 IST) Upstox keeps answering, but bids and asks thin out or vanish, so a surface built from
after-hours quotes can fail its checks. In live mode the service therefore records the raw Upstox responses it
receives; between 15:20 and 15:30 IST it saves them as a snapshot (the same format as ``scripts/upstox_smoke.py
--save``), and from 15:30 until the next open it prices from that snapshot, labelled "closing snapshot".

Without a snapshot (the app was not running at 15:20) it falls back to live after-hours quotes, as before.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
OPEN, CAPTURE_FROM, CLOSE = time(9, 15), time(15, 20), time(15, 30)
MAX_AGE_DAYS = 5  # a long weekend or holiday still uses the last close
log = logging.getLogger(__name__)


def market_open(now: datetime) -> bool:
    now = now.astimezone(IST)
    return now.weekday() < 5 and OPEN <= now.time() < CLOSE


def in_capture_window(now: datetime) -> bool:
    now = now.astimezone(IST)
    return now.weekday() < 5 and CAPTURE_FROM <= now.time() < CLOSE


class RecordingSession:
    """Wraps a requests session; keeps the latest successful JSON response for each URL and parameters."""

    def __init__(self, inner):
        self.inner = inner
        self._records: dict = {}
        self._lock = threading.Lock()

    def get(self, url, params=None, headers=None, timeout=None):
        resp = self.inner.get(url, params=params, headers=headers, timeout=timeout)
        if getattr(resp, "status_code", 0) == 200:
            try:
                body = resp.json()
            except ValueError:
                return resp
            if isinstance(body, dict) and body.get("status") == "success":
                key = (url, json.dumps(params, sort_keys=True, default=str))
                with self._lock:
                    self._records[key] = {"url": url, "params": params, "status": 200, "body": body}
        return resp

    def records(self) -> list[dict]:
        with self._lock:
            return list(self._records.values())


def instrument_records(master, underlyings) -> list[dict]:
    """The instrument-master rows the snapshot needs, in Upstox's own record format."""
    from optlib.data.upstox import UNDERLYINGS

    out = []
    for c in master.contracts:
        if c.underlying not in underlyings:
            continue
        out.append({"segment": "NSE_FO", "underlying_key": UNDERLYINGS[c.underlying],
                    "instrument_type": c.instrument_type, "instrument_key": c.instrument_key,
                    "expiry": c.expiry.isoformat(), "strike_price": c.strike, "lot_size": c.lot_size,
                    "tick_size": c.tick_size * 100.0, "weekly": c.weekly})
    return out


def folder(cache_dir: Path) -> Path:
    return Path(cache_dir) / "closing"


def save(cache_dir: Path, as_of: datetime, responses: list[dict], instruments: list[dict],
         underlyings: list[str]) -> Path:
    d = folder(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    stem = f"close_{as_of.astimezone(IST):%Y%m%d}"
    (d / f"{stem}.instruments.json").write_text(json.dumps(instruments), encoding="utf-8")
    tmp = d / f"{stem}.part"
    tmp.write_text(json.dumps({"as_of": as_of.isoformat(), "underlying": ",".join(underlyings),
                               "underlyings": underlyings, "responses": responses}, default=str), encoding="utf-8")
    path = d / f"{stem}.json"
    tmp.replace(path)
    return path


def latest(cache_dir: Path, now: datetime) -> Optional[Path]:
    """The snapshot to use now: none while the market is open; else the newest one from the last few days."""
    if market_open(now):
        return None
    d = folder(cache_dir)
    if not d.is_dir():
        return None
    today = now.astimezone(IST).date()
    best = None
    for p in sorted(d.glob("close_*.json")):
        if p.name.endswith(".instruments.json"):
            continue
        try:
            day = datetime.strptime(p.stem.split("_")[1], "%Y%m%d").date()
        except (IndexError, ValueError):
            continue
        if today - timedelta(days=MAX_AGE_DAYS) <= day <= today:
            if day == today and now.astimezone(IST).time() < CLOSE:
                continue  # today's 15:20 picture is used only once the market has closed
            best = p
    return best


def instruments_path(snapshot: Path) -> Path:
    return snapshot.with_name(snapshot.stem + ".instruments.json")


def captured_on(cache_dir: Path, day: date) -> bool:
    return (folder(cache_dir) / f"close_{day:%Y%m%d}.json").is_file()


class CloseRecorder:
    """Background thread: on trading days between 15:20 and 15:30 IST, save the closing snapshot once."""

    def __init__(self, svc, every: float = 30.0):
        self.svc, self.every = svc, every
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="close-recorder", daemon=True)

    def start(self):
        if self.svc.mode == "live":
            self.thread.start()
        return self

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(self.every):
            now = datetime.now(IST)
            if in_capture_window(now) and not captured_on(self.svc.settings.cache_dir, now.date()):
                try:
                    r = self.svc.capture_close()
                    log.info("closing snapshot saved: %s", r)
                except Exception as exc:  # noqa: BLE001 - retried on the next tick inside the window
                    log.warning("closing snapshot failed: %s", exc)
