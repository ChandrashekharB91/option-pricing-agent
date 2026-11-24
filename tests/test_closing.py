"""Closing snapshot: record live responses, save them at 15:20 IST, price from them after the close."""
import json
from datetime import datetime

from optlib.data.replay import ReplaySession, fbil_from_files
from optlib.data.upstox import InstrumentMaster, UpstoxClient, UpstoxSource
from optlib_app import closing
from optlib_app.config import FIX, Settings
from optlib_app.service import PricingService

IST = closing.IST
SNAP = FIX / "upstox_live_nifty_4exp_20261029.json"
INSTR = FIX / "upstox_instruments_20261029.json"


def _live_service(tmp_path, wall):
    """A live-mode service whose 'Upstox' answers from the 29 Oct recording, with the clock set to ``wall``."""
    s = Settings(data_mode="live")
    s.cache_dir = tmp_path
    svc = PricingService(s)
    svc._wall = lambda: wall[0]
    snap = json.loads(SNAP.read_text(encoding="utf-8"))
    rows = json.loads(INSTR.read_text(encoding="utf-8"))
    # the fixture master lists monthly expiries only; add the recorded weeklies, as the daily Upstox file does
    weekly = [r for r in rows if r.get("instrument_type") == "CE" and r.get("underlying_key") == "NSE_INDEX|Nifty 50"][:1]
    for d in ("2026-11-03", "2026-11-09", "2026-11-17"):
        rows += [dict(r, expiry=d, instrument_key=r["instrument_key"] + d) for r in weekly]
    master = InstrumentMaster.from_records(rows)

    def live_source():
        client = UpstoxClient("t", session=ReplaySession(snap["responses"]))
        src = UpstoxSource(client, instruments=master, clock=lambda: wall[0])
        svc._recorder = closing.RecordingSession(client._session)
        client._session = svc._recorder
        return src

    svc._live_source = live_source
    base = Settings()
    svc._fbil = fbil_from_files(base.fbil_ois_path, base.fbil_overnight_path)
    return svc


def test_window_rules():
    assert closing.market_open(datetime(2026, 12, 2, 10, 0, tzinfo=IST))
    assert not closing.market_open(datetime(2026, 12, 2, 17, 26, tzinfo=IST))
    assert not closing.market_open(datetime(2026, 12, 5, 11, 0, tzinfo=IST))  # Saturday
    assert closing.in_capture_window(datetime(2026, 12, 2, 15, 22, tzinfo=IST))
    assert not closing.in_capture_window(datetime(2026, 12, 2, 15, 31, tzinfo=IST))


def test_capture_then_price_from_the_snapshot_after_close(tmp_path):
    wall = [datetime(2026, 10, 29, 15, 22, tzinfo=IST)]
    svc = _live_service(tmp_path, wall)
    assert svc.data_note() == "live quotes"
    live = svc.overview("NIFTY")
    r = svc.capture_close()
    assert r["underlyings"] == ["NIFTY"] and r["responses"] >= 5
    assert closing.captured_on(tmp_path, wall[0].date())
    # still market hours: live quotes are used, not the snapshot
    assert closing.latest(tmp_path, wall[0]) is None and svc.data_note() == "live quotes"

    wall[0] = datetime(2026, 10, 29, 17, 26, tzinfo=IST)  # the audition slot
    svc._live_source = lambda: (_ for _ in ()).throw(AssertionError("live quotes must not be used after close"))
    svc._source = None
    assert svc.data_note().startswith("closing snapshot 29 Oct 15:22")
    after = svc.overview("NIFTY")
    assert after["spot"] == live["spot"] and [e["expiry"] for e in after["expiries"]] == \
        [e["expiry"] for e in live["expiries"]]
    assert abs(after["expiries"][0]["atm_vol"] - live["expiries"][0]["atm_vol"]) < 0.002
    assert svc.now().hour == 15 and svc.now().minute == 22
    oi = svc.open_interest("NIFTY")
    assert oi["pcr"] > 0

    wall[0] = datetime(2026, 10, 30, 9, 20, tzinfo=IST)  # next morning: back to live
    assert closing.latest(tmp_path, wall[0]) is None
