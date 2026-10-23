"""Shared loaders for recorded market data (no network)."""
from datetime import date
from functools import lru_cache
from pathlib import Path

from optlib.curves import bootstrap_mibor_ois
from optlib.data.replay import ReplaySession, fbil_from_files, snapshot_source  # noqa: F401  (re-export)

FIX = Path(__file__).parent / "fixtures"
PRICING_DATE = date(2026, 10, 23)


def fbil_client():
    return fbil_from_files(FIX / "fbil_miborois_20261001_20261023.json", FIX / "fbil_ovnmibor_20261001_20261023.json")


@lru_cache(maxsize=None)
def fbil_curve():
    return bootstrap_mibor_ois(fbil_client().ois_quotes(PRICING_DATE), PRICING_DATE)


@lru_cache(maxsize=None)
def live_market():
    src, meta = snapshot_source(FIX / "upstox_live_nifty_20261023.json", FIX / "upstox_instruments_sample.json")
    return src.option_chain("NIFTY", meta["expiry"]), src.futures("NIFTY")
