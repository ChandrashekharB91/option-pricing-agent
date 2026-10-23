"""Shared loaders for recorded market data (no network)."""
import json
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

from optlib.curves import bootstrap_mibor_ois
from optlib.data import InstrumentMaster, UpstoxClient, UpstoxSource
from optlib.data.fbil import FBILClient

FIX = Path(__file__).parent / "fixtures"
PRICING_DATE = date(2026, 10, 23)


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body, self.headers = status, body, {}

    def json(self):
        return self._body


class ReplaySession:
    def __init__(self, records):
        self.records = records

    def get(self, url, params=None, headers=None, timeout=None):
        for r in self.records:
            if r["url"] == url and r["params"] == params:
                return _Resp(r["status"], r["body"])
        raise AssertionError(f"unexpected request {url} {params}")


def fbil_client():
    fx = {"miborois": json.loads((FIX / "fbil_miborois_20261001_20261023.json").read_text()),
          "ovnmibor": json.loads((FIX / "fbil_ovnmibor_20261001_20261023.json").read_text())}
    return FBILClient(fetch=lambda url, params: fx[url.split("/")[-2]])


@lru_cache(maxsize=None)
def fbil_curve():
    return bootstrap_mibor_ois(fbil_client().ois_quotes(PRICING_DATE), PRICING_DATE)


@lru_cache(maxsize=None)
def live_market():
    snap = json.loads((FIX / "upstox_live_nifty_20261023.json").read_text())
    as_of = datetime.fromisoformat(snap["as_of"])
    master = InstrumentMaster.from_records(json.loads((FIX / "upstox_instruments_sample.json").read_text()))
    src = UpstoxSource(UpstoxClient("replay-token-xxxxxxxx", session=ReplaySession(snap["responses"])),
                       instruments=master, clock=lambda: as_of)
    chain = src.option_chain("NIFTY", snap["expiry"])
    return chain, src.futures("NIFTY")
