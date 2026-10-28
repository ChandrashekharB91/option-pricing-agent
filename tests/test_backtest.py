"""Backtest engine on recorded end-of-day data: rebuild a day, settle, and the calibration statistics."""
from datetime import date
from pathlib import Path

import pytest

from helpers import FIX
from optlib import backtest as bt
from optlib.charges import Charges
from optlib.data.bhavcopy import parse_rows
from optlib.data.history import History
from optlib.margin import MarginParams

ROOT = Path(__file__).resolve().parents[1]
CH = Charges.load(ROOT / "config" / "charges.json")
MP = MarginParams.load(ROOT / "config" / "margin.json")
HIST = History.from_csv(FIX / "index_history_20150101_20261023.csv")
DAY = date(2026, 10, 22)


@pytest.fixture(scope="module")
def rows():
    bhav = parse_rows((FIX / "nse_fo_bhavcopy_nifty_20261022.csv").read_text(), {"NIFTY"})
    spot = bt._close(HIST, DAY, "NIFTY")
    fwd = bt.day_state(bhav, DAY, spot, HIST, bt.BacktestConfig()).forward
    settle = {date(2026, 10, 27): fwd * 1.01}  # NIFTY ends 1% above the forward
    return bt.run_day(bhav, DAY, HIST, settle.get, bt.BacktestConfig(), CH, MP)


def test_day_rebuild_and_settlement(rows):
    assert len(rows) > 30
    r = rows[0]
    assert r["expiry"] == "2026-10-27" and 4.5 < r["days"] < 5.5 and r["lot_size"] == 65
    assert abs(r["forward"] / r["spot"] - 1) < 0.01 and 0.05 < r["atm_vol"] < 0.5
    calls = [x for x in rows if x["strategy"] == "Long call"]
    puts = [x for x in rows if x["strategy"] == "Long put"]
    assert all(x["realised_pnl"] < 0 for x in puts if x["breakevens"] and x["breakevens"][0] < x["settle"])
    assert any(x["won"] for x in calls)  # a 1% rally pays some calls
    for x in rows:
        assert 0 <= x["model_pop"] <= 1 and 0 <= x["market_pop"] <= 1
        assert x["model_recommended"] == (x["model_expected"] > 0 and x["model_pop"] >= 0.5)
        assert not x["market_recommended"] or x["market_expected"] > 0


def test_no_look_ahead(rows):
    """The model view's realised vol uses closes before the entry day only."""
    from optlib.recommender import ewma_vol

    expected = ewma_vol(HIST.before(DAY).series("NIFTY"))
    assert rows[0]["realised_vol"] == pytest.approx(expected)


def test_statistics():
    fake = [{"model_pop": p, "won": w, "model_expected": e, "lot_size": 65, "realised_pnl": pnl, "strategy": s,
             "family": "buy", "model_recommended": e > 0 and p >= 0.5, "day": d, "label": s, "risk": 1000.0}
            for p, w, e, pnl, s, d in [(0.9, True, 650, 650, "A", "2026-01-01"), (0.9, True, 650, 1300, "A", "2026-01-02"),
                                       (0.9, False, 650, -1300, "A", "2026-01-03"), (0.2, False, -65, -65, "B", "2026-01-01")]]
    cal = bt.calibration(fake, "model")
    top = next(c for c in cal if c["bucket"] == "90%-100%")
    assert top["n"] == 3 and top["realised"] == pytest.approx(2 / 3)
    assert bt.brier(fake, "model") == pytest.approx(((0.1 ** 2) * 2 + 0.9 ** 2 + 0.2 ** 2) / 4)
    a = next(s for s in bt.by_strategy(fake, "model") if s["strategy"] == "A")
    assert a["realised_pnl_pts"] == pytest.approx((10 + 20 - 20) / 3)
    rule = bt.rule_performance(fake, "model")
    assert rule["ideas"] == 3 and rule["top"]["n"] == 3 and rule["top"]["total_rs"] == pytest.approx(650)
    assert rule["top"]["max_drawdown_rs"] == pytest.approx(1300)
