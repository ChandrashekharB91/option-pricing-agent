"""Service, API and agent tests in replay mode (no network, no API key)."""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from optlib_app import api as api_module
from optlib_app.agent import VALIDATOR_PROMPT, AgentUnavailable, OptionPricingAgent
from optlib_app.config import Settings
from optlib_app.service import PricingService, ServiceError

EXP = "2026-10-27"
STRADDLE = [{"expiry": EXP, "strike": 22500, "option_type": "call", "lots": -2},
            {"expiry": EXP, "strike": 22500, "option_type": "put", "lots": -2},
            {"expiry": EXP, "strike": 22000, "option_type": "put", "lots": 2}]


@pytest.fixture(scope="module")
def svc():
    return PricingService(Settings(data_mode="replay"))


@pytest.fixture(scope="module")
def client(svc):
    api_module._service = svc
    return TestClient(api_module.app)


# ------------------------------------------------------------------ service

def test_service_overview(svc):
    o = svc.overview("nifty")
    assert o["spot"] == 22461.1 and o["lot_size"] == 65 and o["data_mode"] == "replay"
    e = o["expiries"][0]
    assert e["expiry"] == EXP and e["forward_check"]["verdict"] == "PASS"
    assert 0.16 < e["atm_vol"] < 0.165
    json.dumps(o)  # JSON-safe


def test_service_book_totals_add_up(svc):
    b = svc.book(STRADDLE)
    for g in ("value", "delta", "gamma", "vega", "theta"):
        assert b["totals"][g] == pytest.approx(sum(p[g] for p in b["positions"]))
    assert b["totals"]["gamma"] < 0 and b["totals"]["theta"] > 0  # short straddle: short gamma, earns theta
    assert b["positions"][0]["quantity"] == -130


def test_service_scenarios(svc):
    sc = svc.scenarios(STRADDLE, [-0.02, 0.0, 0.02], [0.0, 0.05])
    assert sc["pnl"][0][1] == pytest.approx(0.0, abs=1e-6)  # no shock, no P&L
    assert sc["pnl"][0][0] < 0 and sc["pnl"][0][2] < 0  # short straddle loses on big moves
    assert sc["pnl"][1][1] < sc["pnl"][0][1]  # and when vol rises


def test_service_errors(svc):
    with pytest.raises(ServiceError):
        svc.price("SENSEX", EXP, 22500, "call")
    with pytest.raises(ServiceError, match="beyond"):
        svc.price("NIFTY", "2026-11-23", 22500, "call")  # not in the recorded snapshot: no extrapolation
    with pytest.raises(ServiceError):
        svc.book([])


# ------------------------------------------------------------------ API

def test_api_endpoints(client):
    assert client.get("/health").json()["data_mode"] == "replay"
    assert client.get("/expiries/NIFTY").json()["expiries"] == [EXP]
    assert client.get("/overview/NIFTY").status_code == 200
    s = client.get(f"/smile/NIFTY/{EXP}").json()
    assert len(s["quotes"]) == 114 and len(s["svi_curve"]) == 200
    p = client.post("/price", json={"expiry": EXP, "strike": 22500, "option_type": "call", "lots": 1}).json()
    assert p["result"]["price"] == pytest.approx(163.44, abs=0.01) and p["validation"]["released"]
    b = client.post("/book", json={"positions": STRADDLE}).json()
    assert len(b["positions"]) == 3
    sc = client.post("/scenarios", json={"positions": STRADDLE}).json()
    assert len(sc["pnl"]) == 6 and len(sc["pnl"][0]) == 9


def test_api_validation_errors(client):
    assert client.post("/price", json={"expiry": EXP, "strike": -5, "option_type": "call"}).status_code == 422
    assert client.post("/price", json={"expiry": EXP, "strike": 22500, "option_type": "straddle"}).status_code == 422
    r = client.get("/overview/SENSEX")
    assert r.status_code == 400 and "unknown underlying" in r.json()["detail"]
    assert client.post("/book", json={"positions": []}).status_code == 422


def test_api_agent_without_key(client, monkeypatch):
    monkeypatch.setattr(Settings, "anthropic_api_key", staticmethod(lambda: ""))
    r = client.post("/agent/ask", json={"question": "price the ATM call"})
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]


# ------------------------------------------------------------------ agents (scripted Claude)

def text(t):
    return SimpleNamespace(type="text", text=t)


def tool(id_, name, inp):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


class ScriptedClaude:
    """Plays back orchestrator turns; answers validator calls with a fixed review."""

    def __init__(self, turns, verdict="PASS"):
        self.turns = list(turns)
        self.verdict = verdict
        self.calls = []
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kw):
        self.calls.append(kw)
        if kw.get("system") == VALIDATOR_PROMPT:
            return SimpleNamespace(stop_reason="end_turn",
                                   content=[text(f"Inputs match the question.\nVERDICT: {self.verdict}")])
        content = self.turns.pop(0)
        stop = "tool_use" if any(b.type == "tool_use" for b in content) else "end_turn"
        return SimpleNamespace(stop_reason=stop, content=content)


def tool_results(fake):
    msgs = fake.calls[1]["messages"]  # the conversation as sent on the second orchestrator turn
    return [b for m in msgs if m["role"] == "user" and isinstance(m["content"], list) for b in m["content"]]


def test_agent_prices_and_validates(svc):
    fake = ScriptedClaude([
        [text("Pricing it."), tool("t1", "price_option",
                                   {"underlying": "NIFTY", "expiry": EXP, "strike": 22500, "option_type": "call",
                                    "lots": 2})],
        [text("The 22,500 call is 163.44 per unit (tool result).")],
    ])
    reply = OptionPricingAgent(svc, client=fake).ask("Price 2 lots of the 27 Oct 22500 call")
    assert reply.released and reply.status == "WARN"  # deterministic WARNs carry through
    assert reply.validator_verdict == "PASS" and reply.deterministic_status == "WARN"
    assert reply.tool_calls[0]["output"]["price_per_unit"] == pytest.approx(163.44, abs=0.01)
    assert reply.tool_calls[0]["output"]["value_rs"] == pytest.approx(163.4418 * 130, rel=1e-4)
    # the validator saw only the question and tool calls, never the orchestrator's text
    val_call = [c for c in fake.calls if c.get("system") == VALIDATOR_PROMPT][0]
    payload = val_call["messages"][0]["content"]
    assert "Pricing it." not in payload and "163.44 per unit (tool result)" not in payload
    assert json.loads(payload)["question"].startswith("Price 2 lots")
    # tool results went back to the model as tool_result blocks
    second = tool_results(fake)[0]
    assert second["type"] == "tool_result" and second["tool_use_id"] == "t1"


def test_agent_withholds_on_deterministic_fail(svc):
    fake = ScriptedClaude([
        [tool("t1", "price_option", {"underlying": "NIFTY", "expiry": EXP, "strike": 27500, "option_type": "call"})],
        [text("It is worth 0.45.")],
    ], verdict="PASS")
    reply = OptionPricingAgent(svc, client=fake).ask("What is the 27500 call worth?")
    assert not reply.released and reply.status == "FAIL"
    assert reply.answer.startswith("Result withheld") and "wing vs listed quotes" in reply.answer
    assert reply.withheld_draft == "It is worth 0.45."


def test_agent_validator_can_fail_a_clean_result(svc):
    fake = ScriptedClaude([
        [tool("t1", "price_option", {"underlying": "NIFTY", "expiry": EXP, "strike": 22500, "option_type": "put"})],
        [text("Done.")],
    ], verdict="FAIL")
    reply = OptionPricingAgent(svc, client=fake).ask("Price the 22500 call")  # tool used a put: mismatch
    assert not reply.released and reply.validator_verdict == "FAIL"


def test_agent_reports_tool_errors(svc):
    fake = ScriptedClaude([
        [tool("t1", "price_option", {"underlying": "NIFTY", "expiry": "2026-12-29", "strike": 22500,
                                     "option_type": "call"})],
        [text("That expiry is not available in this data.")],
    ])
    reply = OptionPricingAgent(svc, client=fake).ask("Price the Dec 22500 call")
    assert reply.tool_calls[0]["error"] and "beyond the last fitted expiry" in reply.tool_calls[0]["error"]
    sent = tool_results(fake)[0]
    assert sent["is_error"] is True


def test_agent_requires_key(svc, monkeypatch):
    monkeypatch.setattr(Settings, "anthropic_api_key", staticmethod(lambda: ""))
    with pytest.raises(AgentUnavailable):
        OptionPricingAgent(svc)
