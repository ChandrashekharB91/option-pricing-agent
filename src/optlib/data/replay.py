"""Replay saved market data through the real adapters, with no network.

Snapshots written by ``scripts/upstox_smoke.py --save`` hold the raw Upstox responses, so
any past run can be reproduced exactly: same parsing, same numbers. Used by the tests,
the notebook, and for audit ("show me the inputs behind this price").
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .fbil import FBILClient
from .upstox import InstrumentMaster, UpstoxClient, UpstoxSource


class _Response:
    def __init__(self, status, body):
        self.status_code, self._body, self.headers = status, body, {}

    def json(self):
        return self._body


class ReplaySession:
    """Answers each GET with the recorded response for the same URL and parameters."""

    def __init__(self, records: list[dict]):
        self.records = records

    def get(self, url, params=None, headers=None, timeout=None):
        for r in self.records:
            if r["url"] == url and r["params"] == params:
                return _Response(r["status"], r["body"])
        raise LookupError(f"no recorded response for {url} {params}")


def snapshot_source(snapshot_path: Path | str, instruments_path: Path | str) -> tuple[UpstoxSource, dict]:
    """An ``UpstoxSource`` that replays a saved snapshot, with its clock frozen at the snapshot time.

    ``instruments_path`` is a JSON list of instrument-master records (for expiries and futures keys).
    Returns (source, snapshot metadata)."""
    snap = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    as_of = datetime.fromisoformat(snap["as_of"])
    master = InstrumentMaster.from_records(json.loads(Path(instruments_path).read_text(encoding="utf-8")))
    client = UpstoxClient("replay-token-not-used", session=ReplaySession(snap["responses"]))
    meta = {k: v for k, v in snap.items() if k != "responses"}
    return UpstoxSource(client, instruments=master, clock=lambda: as_of), meta


def fbil_from_files(miborois_path: Path | str, ovnmibor_path: Path | str) -> FBILClient:
    """An ``FBILClient`` answering from saved FBIL responses."""
    data = {"miborois": json.loads(Path(miborois_path).read_text(encoding="utf-8")),
            "ovnmibor": json.loads(Path(ovnmibor_path).read_text(encoding="utf-8"))}
    return FBILClient(fetch=lambda url, params: data[url.rstrip("/").split("/")[-2]])
