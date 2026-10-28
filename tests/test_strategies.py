"""Phase 3: charges, margin estimate, strategy library, P&L projector, recommender, hedge finder, API, agent."""
import math
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from optlib.charges import Charges
from optlib.margin import MarginParams, estimate_margin
from optlib.projector import NotQuoted, build_views, project
from optlib.recommender import DISCLAIMER, View, find_hedges, recommend
from optlib.strategies import FAMILIES, Strategy, StrategyLeg, StrikeGrid, generate_candidates
from optlib_app import api as api_module
from optlib_app.agent import OptionPricingAgent
from optlib_app.config import Settings
from optlib_app.llm import LLMReply, ToolCall
from optlib_app.service import PricingService, ServiceError

EXP = date(2026, 10, 27)
ROOT = Path(__file__).resolve().parents[1]
CH = Charges.load(ROOT / "config" / "charges.json")
MP = MarginParams.load(ROOT / "config" / "margin.json")
BOOK = [{"expiry": "2026-10-27", "strike": 22500, "option_type": "call", "lots": -2},
        {"expiry": "2026-10-27", "strike": 22500, "option_type": "put", "lots": -2},
        {"expiry": "2026-10-27", "strike": 22000, "option_type": "put", "lots": 2}]


@pytest.fixture(scope="module")
def svc(tmp_path_factory):
    s = Settings(data_mode="replay")
    s.audit_log_path = tmp_path_factory.mktemp("audit") / "rec.jsonl"
    return PricingService(s)


@pytest.fixture(scope="module")
def state(svc):
    return svc.market_state("NIFTY", EXP)


def strat(*legs, name="test"):
    return Strategy.custom(legs, name)


# ------------------------------------------------------------------ charges and margin

def test_charges_one_order():
    sell = CH.order("call", -1, 100.0, 65)  # premium value Rs 6,500
    assert sell["stt"] == pytest.approx(6500 * 0.0015) and sell["stamp"] == 0.0
    assert sell["exchange"] == pytest.approx(6500 * 0.0003503)
    assert sell["gst"] == pytest.approx(0.18 * (20 + sell["exchange"] + sell["sebi"]))
    buy = CH.order("call", 1, 100.0, 65)
    assert buy["stt"] == 0.0 and buy["stamp"] == pytest.approx(6500 * 0.00003)
    fut = CH.order("future", -1, 22500.0, 65)
    assert fut["stt"] == pytest.approx(22500 * 65 * 0.0005)


def test_margin_estimate(state):
    assert estimate_margin([StrategyLeg("call", 1, EXP, 22500.0)], 65, state, MP)["total"] == 0.0
    fut = estimate_margin([StrategyLeg("future", -1, EXP)], 65, state, MP)
    notional = state.forward * 65
    assert fut["span"] == pytest.approx(MP.price_scan_range * notional, rel=1e-6)
    assert fut["total"] == pytest.approx((MP.price_scan_range + MP.exposure_margin) * notional, rel=1e-6)
    naked = estimate_margin([StrategyLeg("call", -1, EXP, 22500.0)], 65, state, MP)["total"]
    spread = estimate_margin([StrategyLeg("call", -1, EXP, 22500.0), StrategyLeg("call", 1, EXP, 22700.0)],
                             65, state, MP)["total"]
    assert 0 < spread < naked  # the hedge earns a margin benefit


# ------------------------------------------------------------------ library

def test_leg_and_strategy_validation():
    with pytest.raises(ValueError):
        StrategyLeg("call", 1, EXP)  # option without strike
    with pytest.raises(ValueError):
        StrategyLeg("future", 1, EXP, 22500.0)
    with pytest.raises(ValueError):
        Strategy.custom([StrategyLeg("call", 1, EXP, 22500.0), StrategyLeg("call", -1, date(2026, 11, 3), 22700.0)])
    leg = StrategyLeg.from_dict({"instrument": "Put", "side": "sell", "strike": 22000, "expiry": "2026-10-27"})
    assert leg.side == -1 and leg.instrument == "put"
    assert strat(StrategyLeg("call", 1, EXP, 22500.0), StrategyLeg("call", -1, EXP, 22700.0)).family == "debit"
    assert strat(StrategyLeg("call", -1, EXP, 22500.0), StrategyLeg("call", 1, EXP, 22700.0)).family == "hedged"
    assert strat(StrategyLeg("put", -1, EXP, 22000.0)).family == "naked"
    assert strat(StrategyLeg("future", 1, EXP), StrategyLeg("put", 1, EXP, 22000.0)).family == "futures"


def test_candidates_cover_every_family(state):
    calls = sorted(k for (k, t) in state.quotes if t == "call")
    puts = sorted(k for (k, t) in state.quotes if t == "put")
    c = generate_candidates(StrikeGrid(state.forward, state.sd, calls, puts, EXP))
    assert {s.family for s in c} == set(FAMILIES)
    assert len({s.key for s in c}) == len(c)
    for s in c:
        if s.name == "Bull call spread":
            lo, hi = sorted(l.strike for l in s.legs)
            assert next(l for l in s.legs if l.strike == lo).side == 1 and hi > lo
        if s.name == "Iron condor":
            assert sorted(l.strike for l in s.legs) == [l.strike for l in s.legs]


# ------------------------------------------------------------------ projector

def test_market_view_is_martingale(state):
    for t in (state.T, state.T / 2):
        v = build_views(state, t)
        assert v["market"].mean() == pytest.approx(state.forward, rel=2e-4)
        assert v["model"].mean() == pytest.approx(state.forward, rel=1e-4)
        assert v["user"].mean() == pytest.approx(state.forward, rel=1e-4)
    v = build_views(state, state.T, target=23000.0)
    assert v["user"].mean() == pytest.approx(23000.0, rel=1e-4)


def test_long_call_payoff_and_costs(state):
    s = strat(StrategyLeg("call", 1, EXP, 22500.0))
    views = build_views(state, state.T)
    p = project(s, state, views, state.T, CH, MP, lots=1)
    ask = state.quotes[(22500.0, "call")][1]
    assert p["premium"] == pytest.approx(ask * 65)
    assert p["max_loss"] == pytest.approx(p["premium"] + p["entry_charges"], rel=1e-9)
    assert p["max_profit"] is None and not p["unlimited_loss"]
    be = p["breakevens"][0]
    assert be == pytest.approx(22500 + (p["premium"] + p["entry_charges"]) / 65, abs=3.0)  # + STT on exercise
    # under the market's own distribution the trade loses about its costs: bid-ask, charges, exercise STT
    e = p["views"]["market"]["expected"]
    cost = p["premium"] - p["fair_premium"] + p["entry_charges"]
    assert -cost - 30 < e < -cost + 5  # the rest is STT on exercise
    assert p["margin"]["total"] == 0 and p["capital_required"] == pytest.approx(p["premium"] + p["entry_charges"])


def test_spread_bounds_and_put_profit_cap(state):
    views = build_views(state, state.T)
    s = strat(StrategyLeg("call", 1, EXP, 22500.0), StrategyLeg("call", -1, EXP, 22700.0))
    p = project(s, state, views, state.T, CH, MP)
    width = 200 * 65
    assert p["max_profit"] + p["max_loss"] == pytest.approx(width, abs=60)  # minus charges and exercise STT
    put = project(strat(StrategyLeg("put", 1, EXP, 22500.0)), state, views, state.T, CH, MP)
    assert put["max_profit"] is not None and put["max_profit"] < 22500 * 65  # capped at an index of zero
    short = project(strat(StrategyLeg("call", -1, EXP, 23000.0)), state, views, state.T, CH, MP)
    assert short["unlimited_loss"] and short["max_loss"] is None and short["stress_loss"] > 0


def test_unquoted_leg_refused(state):
    with pytest.raises(NotQuoted):
        project(strat(StrategyLeg("call", 1, EXP, 22510.0)), state, build_views(state, state.T), state.T, CH, MP)


# ------------------------------------------------------------------ recommender

def test_recommend_respects_capital_and_risk(state):
    r = recommend(state, 20000, View("user", target=22900.0), CH, MP)
    assert r["disclaimer"] == DISCLAIMER and r["ideas"] and r["min_prob_profit"] == 0.5
    for i in r["ideas"]:
        assert i["capital_required"] * (1 + MP.capital_buffer) <= 20000 + 1
        risk = i["stress_loss"] if i["unlimited_loss"] else i["max_loss"]
        assert risk <= 0.5 * 20000 + 1
        assert i["views"]["user"]["expected"] > 0 and i["views"]["user"]["prob_profit"] >= 0.5
        assert i["recommended"] and not i["reasons"]
        assert i["rationale"]["basis"].startswith("Bullish") and "22,900" in i["rationale"]["fit"]
    assert any(b["family"] == "Naked selling" for b in r["blocked"])  # Rs 20k cannot carry a naked short


def test_recommendation_rule_and_near_misses(state):
    r = recommend(state, 20000, View("user", target=22900.0), CH, MP, min_prob_profit=0.5)
    assert r["near_misses"] and all(not m["recommended"] and m["reasons"] for m in r["near_misses"])
    for m in r["near_misses"]:
        v = m["views"]["user"]
        assert v["expected"] <= 0 or v["prob_profit"] < 0.5
    looser = recommend(state, 20000, View("user", target=22900.0), CH, MP, min_prob_profit=0.0)
    assert looser["recommended_count"] > r["recommended_count"]
    none = recommend(state, 20000, View("blend"), CH, MP)  # target = forward: costs dominate
    assert not none["ideas"] and none["near_misses"] and none["note"].startswith("No recommendation")
    with pytest.raises(ValueError):
        recommend(state, 20000, View(), CH, MP, min_prob_profit=1.5)


def test_rationale_reads_the_shape(state):
    r = recommend(state, 300000, View("blend", target=22500.0), CH, MP)
    condor = next(i for i in r["ideas"] + r["near_misses"] if i["strategy"]["name"] in ("Iron condor", "Short strangle"))
    assert "stays between" in condor["rationale"]["fit"] and "sells volatility" in condor["rationale"]["fit"]
    put = project(strat(StrategyLeg("put", 1, EXP, 22500.0)), state, build_views(state, state.T), state.T, CH, MP)
    assert put["max_profit"] > 10 * put["max_profit_3sd"] > 0  # the cap needs NIFTY near zero


def test_recommend_scales_with_capital(state):
    small = recommend(state, 20000, View("model"), CH, MP)
    big = recommend(state, 500000, View("model"), CH, MP)
    assert big["feasible"] > small["feasible"]
    assert any(i["strategy"]["family"] in ("naked", "hedged") for i in big["ideas"])


def test_market_view_has_no_free_lunch(state):
    r = recommend(state, 300000, View("market"), CH, MP, min_prob_profit=0.0)
    assert not r["ideas"] and "market-implied" in r["note"]


def test_recommend_validation(state):
    with pytest.raises(ValueError):
        recommend(state, 0, View(), CH, MP)
    with pytest.raises(ValueError):
        View(source="guess")


def test_hedge_finder_cuts_tail(state):
    h = find_hedges(state, BOOK, View("model"), CH, MP, budget=100000)
    assert h["hedges"] and h["disclaimer"] == DISCLAIMER and h["rank_by"] == "worst_case"
    best = h["hedges"][0]
    assert best["worst_case"] < h["before"]["worst_case"] and best["cash_needed"] <= 100000
    assert best["es95"] < h["before"]["es95"]
    assert len(best["curve_after"]) == len(h["curves"]["F"]) == len(h["curves"]["before"])
    cheap = find_hedges(state, BOOK, View("model"), CH, MP, budget=100000, rank_by="per_rupee")["hedges"][0]
    assert cheap["cash_needed"] < best["cash_needed"]
    with pytest.raises(ValueError):
        find_hedges(state, BOOK, View("model"), CH, MP, budget=1000, rank_by="vibes")


def test_hedge_finder_offers_call_and_put_together(state):
    """A naked short straddle is exposed both ways: the best hedge buys a call AND a put (an iron butterfly)."""
    h = find_hedges(state, BOOK[:2], View("model"), CH, MP, budget=50000)
    b, best = h["before"], h["hedges"][0]
    assert b["loss_if_up"] > 50000 and b["loss_if_down"] > 50000
    kinds = {l["leg"].split()[-1] for l in best["legs"]}
    assert kinds == {"call", "put"} and all(l["leg"].startswith("Buy") for l in best["legs"])
    assert best["loss_if_up"] < 0.2 * b["loss_if_up"] and best["loss_if_down"] < 0.2 * b["loss_if_down"]


# ------------------------------------------------------------------ service, API, agent

def test_service_and_audit_log(svc):
    r = svc.recommend(30000, {"source": "user", "target": 22800, "horizon": "2026-10-26"})
    assert r["view"]["horizon"] == "2026-10-26" and r["data_mode"] == "replay"
    assert svc.settings.audit_log_path.read_text().count('"kind": "recommend"') >= 1
    with pytest.raises(ServiceError, match="horizon"):
        svc.recommend(30000, {"horizon": "2026-12-31"})
    with pytest.raises(ServiceError):
        svc.project_strategy([{"instrument": "call", "side": "buy", "strike": 22510, "expiry": "2026-10-27"}])


def test_strategy_api(svc):
    api_module._service = svc
    c = TestClient(api_module.app)
    meta = c.get("/strategy/meta").json()
    assert meta["disclaimer"] == DISCLAIMER and meta["expiries"] == ["2026-10-27"]
    r = c.post("/strategy/recommend", json={"capital": 25000, "view": {"source": "blend", "target": 22800}})
    assert r.status_code == 200 and r.json()["capital"] == 25000
    assert c.post("/strategy/recommend", json={"capital": -5}).status_code == 422
    assert c.post("/strategy/recommend", json={"capital": 5000, "risk_budget": 2}).status_code == 422
    p = c.post("/strategy/project", json={"legs": [{"instrument": "put", "side": "buy", "strike": 22400,
                                                     "expiry": "2026-10-27"}], "lots": 2})
    assert p.status_code == 200 and p.json()["lots"] == 2
    assert p.json()["grid"]["times"] == ["now", "Fri 23 Oct close", "Mon 26 Oct close", "Tue 27 Oct close"]
    h = c.post("/strategy/hedge", json={"positions": BOOK, "budget": 50000, "rank_by": "es95", "top": 5})
    assert h.status_code == 200 and h.json()["rank_by"] == "es95" and len(h.json()["hedges"]) == 5
    assert c.post("/strategy/hedge", json={"positions": BOOK, "budget": 50000, "rank_by": "x"}).status_code == 422


class Scripted:
    provider, model = "scripted", "m"

    def __init__(self, turns):
        self.turns = list(turns)

    def chat(self, system, messages, tools=None, max_tokens=2048):
        if tools is None:
            return LLMReply("Consistent.\nVERDICT: PASS")
        return self.turns.pop(0)


def test_agent_strategy_tools_append_disclaimer(svc):
    llm = Scripted([LLMReply("", [ToolCall("a", "recommend_strategies",
                                           {"capital": 20000, "view": {"source": "user", "target": 22900}})]),
                    LLMReply("Here are the ideas.")])
    reply = OptionPricingAgent(svc, llm=llm).ask("I have 20000 and think NIFTY reaches 22900. What should I do?")
    out = reply.tool_calls[0]["output"]
    assert out["recommended"] and "rule" in out and out["recommended"][0]["why_this_strategy"]
    assert reply.answer.endswith(DISCLAIMER)
