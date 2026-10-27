"""Risk engine, P&L explain, history and bhavcopy loaders, risk endpoints and agent tools."""
import math
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from helpers import FIX, fbil_client
from optlib import black76_price
from optlib.curves import bootstrap_mibor_ois, flat_curve
from optlib.data.bhavcopy import parse_rows
from optlib.data.history import History, fetch_history
from optlib.pnl_explain import explain
from optlib.risk import (
    Leg, Limits, backtest, check_limits, christoffersen, historical_scenarios, kupiec, revalue, stress_tests,
    traffic_light, var_es,
)
from optlib_app import api as api_module
from optlib_app.agent import OptionPricingAgent
from optlib_app.config import Settings
from optlib_app.llm import LLMReply, ToolCall
from optlib_app.service import PricingService, ServiceError

EXP = "2026-10-27"
BOOK = [{"expiry": EXP, "strike": 22500, "option_type": "call", "lots": -2},
        {"expiry": EXP, "strike": 22500, "option_type": "put", "lots": -2},
        {"expiry": EXP, "strike": 22000, "option_type": "put", "lots": 2}]
HIST = History.from_csv(FIX / "index_history_20150101_20261023.csv")


def leg(strike=22500.0, kind="call", qty=65.0, F=22500.0, vol=0.15, T=30 / 365, r=0.06, und="NIFTY"):
    return Leg(f"{strike} {kind}", und, EXP, strike, kind, qty, F, r, T, vol, float(black76_price(F, strike, T, r, vol, kind)))


# ------------------------------------------------------------------ data

def test_history_fixture_and_known_days():
    n, v = HIST.series("NIFTY"), HIST.series("INDIAVIX")
    i = HIST.dates.index(date(2020, 3, 23))
    assert n[i] / n[i - 1] - 1 == pytest.approx(-0.1298, abs=5e-4)  # COVID crash
    j = HIST.dates.index(date(2024, 6, 4))
    assert n[j] / n[j - 1] - 1 == pytest.approx(-0.0593, abs=5e-4)  # election results
    assert HIST.dates[-1] == date(2026, 10, 23) and v[-1] == 16.86
    assert np.isnan(HIST.series("MIDCPNIFTY")[0])  # Midcap Select history starts in 2022
    assert HIST.before(date(2026, 10, 23)).dates[-1] == date(2026, 10, 22)


def test_fetch_history_chunks_by_year():
    urls = []

    def fake(url):
        urls.append(url)
        return [["2024-01-02T00:00:00+05:30", 1, 1, 1, 100.0, 0, 0]] if "2024" in url else \
            [["2025-01-02T00:00:00+05:30", 1, 1, 1, 110.0, 0, 0]]
    h = fetch_history(date(2024, 6, 1), date(2025, 3, 1), names=["NIFTY"], fetch=fake)
    assert len(urls) == 2 and "/day/2024-12-31/2024-06-01" in urls[0] and "/day/2025-03-01/2025-01-01" in urls[1]
    assert h.dates == [date(2024, 1, 2), date(2025, 1, 2)] and list(h.series("NIFTY")) == [100.0, 110.0]


def test_bhavcopy_parsing():
    rows = parse_rows((FIX / "nse_fo_bhavcopy_nifty_20261023.csv").read_text())
    c = [r for r in rows if r.kind == "option" and r.strike == 22500 and r.expiry == date(2026, 10, 27)]
    assert {r.option_type for r in c} == {"call", "put"}
    call = next(r for r in c if r.option_type == "call")
    assert call.close == 175.80 and call.underlying == 22600.0 and call.lot_size == 65
    assert any(r.kind == "future" for r in rows)


# ------------------------------------------------------------------ engine

def test_revalue_identity_and_direction():
    legs = [leg(), leg(kind="put", qty=-65.0)]
    assert np.allclose(revalue(legs, [0.0], [1.0]), 0.0, atol=1e-9)
    up = revalue([leg()], [math.log(1.02)], [1.0])[0, 0]
    assert up > 0 and revalue([leg()], [0.0], [1.2])[0, 0] > 0  # long call gains on rally and on vol
    assert revalue([leg()], [0.0], [1.0], dt=1 / 365)[0, 0] < 0  # and loses to time decay


def test_var_es_properties():
    legs = [leg(qty=-650.0)]
    scen = historical_scenarios(HIST, "NIFTY", 500)
    v = var_es(legs, scen)
    assert v.window == 500 and v.es >= v.var > 0
    assert sum(c["contribution"] for c in v.es_contributions) == pytest.approx(v.es)
    assert v.pnl[HIST.dates.index(date.fromisoformat(v.worst["date"])) - (len(HIST.dates) - 500)] == v.worst["pnl"]


def test_multi_underlying_scenarios_align():
    scen = historical_scenarios(HIST, ["NIFTY", "MIDCPNIFTY"], None)
    assert scen.dates[0] >= date(2022, 1, 10) and set(scen.log_ret) == {"NIFTY", "MIDCPNIFTY"}
    legs = [leg(), leg(und="MIDCPNIFTY", F=12000.0, strike=12000.0)]
    assert revalue(legs, scen.log_ret, scen.vol_mult).shape == (len(scen.dates), 2)


def test_backtest_statistics():
    lr, p = kupiec(0, 250, 0.01)
    assert lr == pytest.approx(-2 * 250 * math.log(0.99)) and 0.02 < p < 0.03
    assert kupiec(3, 250, 0.01)[1] > 0.5
    assert christoffersen(np.zeros(250, int))[0] == 0.0
    clustered = np.zeros(250, int); clustered[100:105] = 1
    assert christoffersen(clustered)[1] < 0.01
    assert [traffic_light(x) for x in (4, 5, 9, 10)] == ["green", "amber", "amber", "red"]
    bt = backtest([leg(qty=-650.0)], HIST.before(date(2026, 10, 23)))
    assert bt.days == 250 and len(bt.var) == 250 and bt.traffic_light in ("green", "amber", "red")


def test_stress_tests():
    st = stress_tests([leg(qty=-650.0)], HIST)
    names = [s["scenario"] for s in st]
    assert any("COVID crash" in n for n in names) and any(n.startswith("Gap down 10%") for n in names)
    assert [s["pnl"] for s in st] == sorted(s["pnl"] for s in st)
    covid = next(s for s in st if "COVID crash" in s["scenario"])
    assert covid["index_move"] == pytest.approx(-0.1298, abs=5e-4) and covid["kind"] == "historical"
    assert covid["pnl"] > 0  # short call gains in a crash


def test_limits():
    lim = Limits(delta_units=100, gamma=1, vega_rs=1000, theta_rs=1000, var99_rs=1000, stress_rs=1000)
    out = {l["limit"]: l for l in check_limits({"delta": -85, "gamma": 0.1, "vega": 2000, "theta": 10}, 500, -2000, lim)}
    assert out["Net delta"]["status"] == "WARN" and out["Net vega"]["status"] == "FAIL"
    assert out["Net gamma"]["status"] == "PASS" and out["Worst stress loss"]["value"] == 2000
    assert Limits.load(Path("config/risk_limits.json")).var99_rs == 500000


# ------------------------------------------------------------------ P&L explain on real NSE closes

def eod():
    r0 = parse_rows((FIX / "nse_fo_bhavcopy_nifty_20261022.csv").read_text())
    r1 = parse_rows((FIX / "nse_fo_bhavcopy_nifty_20261023.csv").read_text())
    q = fbil_client()
    c0 = bootstrap_mibor_ois(q.ois_quotes(date(2026, 10, 22)), date(2026, 10, 22))
    c1 = bootstrap_mibor_ois(q.ois_quotes(date(2026, 10, 23)), date(2026, 10, 23))
    return r0, r1, c0, c1


def test_pnl_explain_real_day():
    e = explain(BOOK, *eod())
    t = e.totals
    assert t["actual"] == pytest.approx((175.80 - 281.05) * -130 + (107.05 - 72.05) * -130 + (13.15 - 10.40) * 130)
    parts = sum(t[k] for k in ("delta", "gamma", "vega", "theta", "cross", "rate", "unexplained"))
    assert parts == pytest.approx(t["actual"])
    assert t["delta"] > 0 and t["theta"] > 0 and t["gamma"] < 0  # short straddle, market fell
    assert e.unexplained_share < 0.05 and e.status == "PASS"
    for l in e.legs:  # each leg reprices to its close by construction
        assert 0.1 < l.vol0 < 0.3 and 0.1 < l.vol1 < 0.3


def test_pnl_explain_errors():
    r0, r1, c0, c1 = eod()
    with pytest.raises(ValueError, match="not in both"):
        explain([{"expiry": EXP, "strike": 22510, "option_type": "call", "lots": 1}], r0, r1, c0, c1)
    with pytest.raises(ValueError, match="after day 0"):
        explain(BOOK, r1, r0, c1, c0)


# ------------------------------------------------------------------ service, API, agent

@pytest.fixture(scope="module")
def svc():
    return PricingService(Settings(data_mode="replay"))


def test_service_risk(svc):
    r = svc.risk(BOOK)
    assert r["history_last_date"] == "2026-10-22"  # nothing from the pricing day itself
    assert r["var"]["var"] > 0 and r["var"]["es"] >= r["var"]["var"]
    assert r["backtest"]["days"] == 250 and len(r["limits"]) == 6
    assert r["stress"][0]["pnl"] <= r["stress"][-1]["pnl"]
    assert sum(x["value"] for x in r["by_expiry"]) == pytest.approx(r["book"]["totals"]["value"])


def test_service_pnl_explain_and_report(svc):
    e = svc.pnl_explain(BOOK, "2026-10-22", "2026-10-23")
    assert e["status"] == "PASS" and e["totals"]["actual"] == pytest.approx(9490.0)
    with pytest.raises(ServiceError, match="replay data"):
        svc.pnl_explain(BOOK, "2026-10-21", "2026-10-23")
    html = svc.risk_report(BOOK, "2026-10-22", "2026-10-23")
    assert html.startswith("<!doctype html>") and "Value at Risk" in html and "P&amp;L explain" in html


def test_risk_api(svc):
    api_module._service = svc
    c = TestClient(api_module.app)
    assert c.get("/risk/eod_days").json()["days"] == ["2026-10-22", "2026-10-23"]
    r = c.post("/risk", json={"positions": BOOK, "window": 250})
    assert r.status_code == 200 and r.json()["var"]["window"] == 250
    assert c.post("/risk", json={"positions": BOOK, "window": 10}).status_code == 422
    e = c.post("/risk/pnl_explain", json={"positions": BOOK, "day0": "2026-10-22", "day1": "2026-10-23"})
    assert e.status_code == 200 and e.json()["status"] == "PASS"
    h = c.post("/risk/report", json={"positions": BOOK})
    assert h.status_code == 200 and h.headers["content-type"].startswith("text/html")


class Scripted:
    provider, model = "scripted", "m"

    def __init__(self, turns):
        self.turns = list(turns)

    def chat(self, system, messages, tools=None, max_tokens=2048):
        if tools is None:
            return LLMReply("Consistent.\nVERDICT: PASS")
        return self.turns.pop(0)


def test_agent_risk_tools(svc):
    llm = Scripted([LLMReply("", [ToolCall("a", "risk_summary", {"positions": BOOK}),
                                  ToolCall("b", "pnl_explain", {"positions": BOOK, "day0": "2026-10-22",
                                                                "day1": "2026-10-23"})]),
                    LLMReply("Done.")])
    reply = OptionPricingAgent(svc, llm=llm).ask("What is my VaR and explain yesterday's P&L")
    rs, pe = reply.tool_calls[0]["output"], reply.tool_calls[1]["output"]
    assert rs["var_99_1d"] > 0 and rs["backtest"]["traffic_light"] in ("green", "amber", "red")
    assert len(rs["worst_stress"]) == 5 and "pnl" not in rs  # compact: no 500-day series sent to the model
    assert pe["totals"]["actual"] == pytest.approx(9490.0)
