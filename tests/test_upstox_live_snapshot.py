"""Regression tests on a recorded live Upstox snapshot (NIFTY, 23 Oct 2026 11:03 IST, 27 Oct expiry).

Replays the exact responses through the adapter, then checks what the pricing tools
rely on: parity-implied forward vs the future, and optlib IVs vs Upstox's IVs.
"""
import json
import math
import statistics
from datetime import date, datetime
from pathlib import Path

import pytest

from optlib import implied_vol_black76
from optlib.data import InstrumentMaster, UpstoxClient, UpstoxSource

FIX = Path(__file__).parent / "fixtures"
SNAP = json.loads((FIX / "upstox_live_nifty_20261023.json").read_text())
RATE = 0.065  # placeholder until the MIBOR-OIS curve is built


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


@pytest.fixture(scope="module")
def market():
    as_of = datetime.fromisoformat(SNAP["as_of"])
    master = InstrumentMaster.from_records(json.loads((FIX / "upstox_instruments_sample.json").read_text()))
    src = UpstoxSource(UpstoxClient("replay-token-xxxxxxxx", session=ReplaySession(SNAP["responses"])),
                       instruments=master, clock=lambda: as_of)
    chain = src.option_chain("NIFTY", SNAP["expiry"])
    return {"spot": src.spot("NIFTY"), "chain": chain, "futures": src.futures("NIFTY")}


def parity_forward(chain, n=9):
    T = chain.time_to_expiry()
    df = math.exp(-RATE * T)
    near = sorted(chain.strikes, key=lambda k: abs(k - chain.spot))[:n]
    fwds = [k + (chain.quote(k, "call").mid - chain.quote(k, "put").mid) / df for k in near]
    return statistics.median(fwds), max(fwds) - min(fwds)


def test_live_parse(market):
    chain = market["chain"]
    assert market["spot"] == 22461.1 == chain.spot
    assert chain.expiry == date(2026, 10, 27)
    assert len(chain.strikes) == 60 and len(chain.quotes) == 120
    assert all(q.two_sided for q in chain.quotes)
    assert chain.time_to_expiry() * 365 == pytest.approx(4.185, abs=1e-3)
    assert [f.expiry for f in market["futures"]] == [date(2026, 10, 27), date(2026, 11, 23), date(2026, 12, 29)]


def test_vendor_zero_iv_is_missing(market):
    q = market["chain"].quote(21500.0, "call")  # Upstox reports iv 0, delta 1 here
    assert math.isnan(q.vendor_iv) and math.isnan(q.vendor_delta)
    assert q.vendor_iv != 0


def test_parity_forward_matches_future(market):
    """Spec gate: parity-implied forward within 0.05% of the same-expiry future."""
    F, spread = parity_forward(market["chain"])
    fut = market["futures"][0]
    assert spread < 2.0  # nine strikes agree to within 2 index points
    assert abs(F - fut.mid) / fut.mid < 0.0005


def test_spot_carry_would_misprice_the_forward(market):
    """Why the forward comes from options: spot carried at the rate misses the futures basis."""
    chain = market["chain"]
    F, _ = parity_forward(chain)
    naive = chain.spot * math.exp(RATE * chain.time_to_expiry())
    assert (F - naive) / F > 0.0015  # 35.6 points (0.16%) too low on this day


def test_optlib_iv_close_to_upstox_near_the_money(market):
    chain = market["chain"]
    F, _ = parity_forward(chain)
    T = chain.time_to_expiry()
    diffs = []
    for k in chain.strikes:
        if abs(k - F) > 1000:
            continue
        side = "call" if k >= F else "put"
        q = chain.quote(k, side)
        ours = implied_vol_black76(q.mid, F, k, T, RATE, side) * 100
        diffs.append(ours - q.vendor_iv)
    assert len(diffs) == 40
    assert max(abs(d) for d in diffs) < 0.5  # vol points; Upstox uses its own forward and timing
    assert abs(statistics.mean(diffs)) < 0.1


def test_smile_shape(market):
    """Short-dated Nifty smile: both wings above ATM, put wing steeper."""
    chain = market["chain"]
    F, _ = parity_forward(chain)
    T = chain.time_to_expiry()

    def vol(k):
        side = "call" if k >= F else "put"
        return implied_vol_black76(chain.quote(k, side).mid, F, k, T, RATE, side)

    atm = vol(22500.0)
    assert 0.12 < atm < 0.22
    assert vol(21500.0) > atm and vol(23500.0) > atm
    assert vol(21500.0) - atm > vol(23500.0) - atm - 0.01
