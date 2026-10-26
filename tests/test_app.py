"""Service, API and agent tests in replay mode (no network, no API key)."""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from optlib_app import api as api_module
from optlib_app.agent import VALIDATOR_PROMPT, AgentUnavailable, OptionPricingAgent, ungrounded_numbers, ToolRecord
from optlib_app.config import Settings
from optlib_app.llm import AnthropicLLM, LLMError, LLMReply, ToolCall
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




def no_llm(monkeypatch):
    for k in ("LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(k, "")
    monkeypatch.setattr("optlib_app.config.read_env_file", lambda path: {})


def test_api_agent_without_llm(client, monkeypatch):
    no_llm(monkeypatch)
    assert client.get("/health").json()["llm"]["configured"] is False
    r = client.post("/agent/ask", json={"question": "price the ATM call"})
    assert r.status_code == 503 and "LLM_PROVIDER" in r.json()["detail"]


# ------------------------------------------------------------------ agents (scripted, provider-neutral)

class ScriptedLLM:
    """Plays back orchestrator turns; answers validator calls with a fixed review."""
    provider, model = "scripted", "test-model"

    def __init__(self, turns, verdict="PASS"):
        self.turns = list(turns)
        self.verdict = verdict
        self.calls = []

    def chat(self, system, messages, tools=None, max_tokens=2048):
        self.calls.append({"system": system, "messages": [dict(m) for m in messages], "tools": tools})
        if system == VALIDATOR_PROMPT:
            return LLMReply(f"Inputs match the question.\nVERDICT: {self.verdict}")
        return self.turns.pop(0)


def say(text):
    return LLMReply(text)


def call(id_, name, args, text=""):
    return LLMReply(text, [ToolCall(id_, name, args)])


def price_args(strike, kind, lots=1):
    return {"underlying": "NIFTY", "expiry": EXP, "strike": strike, "option_type": kind, "lots": lots}


def test_agent_prices_and_validates(svc):
    llm = ScriptedLLM([call("t1", "price_option", price_args(22500, "call", 2), "Pricing it."),
                       say("The 22,500 call is 163.44 per unit; 2 lots of 65 are worth Rs 21,247.")])
    reply = OptionPricingAgent(svc, llm=llm).ask("Price 2 lots of the 27 Oct 22500 call")
    assert reply.released and reply.status == "WARN"  # deterministic WARNs carry through
    assert reply.validator_verdict == "PASS" and reply.deterministic_status == "WARN"
    assert reply.ungrounded_numbers == []
    assert reply.tool_calls[0]["output"]["price_per_unit"] == pytest.approx(163.44, abs=0.01)
    # the validator saw only the question and tool calls, never the orchestrator's text
    val = [c for c in llm.calls if c["system"] == VALIDATOR_PROMPT][0]
    payload = val["messages"][0]["content"]
    assert "Pricing it." not in payload and "163.44 per unit" not in payload and val["tools"] is None
    # the tool result went back as a tool message
    tool_msgs = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "t1" and not tool_msgs[0]["is_error"]


def test_agent_flags_invented_numbers(svc):
    llm = ScriptedLLM([call("t1", "price_option", price_args(22500, "call")),
                       say("It costs 163.44, so 10 lots would be about 106,237 and breakeven is 22,663.44.")])
    reply = OptionPricingAgent(svc, llm=llm).ask("Price the 22500 call")
    assert "106,237" in reply.ungrounded_numbers and "22,663.44" in reply.ungrounded_numbers
    assert "163.44" not in reply.ungrounded_numbers
    assert reply.status == "WARN" and reply.released


def test_agent_withholds_on_deterministic_fail(svc):
    llm = ScriptedLLM([call("t1", "price_option", price_args(27500, "call")), say("It is worth 0.45.")])
    reply = OptionPricingAgent(svc, llm=llm).ask("What is the 27500 call worth?")
    assert not reply.released and reply.status == "FAIL"
    assert reply.answer.startswith("Result withheld") and "wing vs listed quotes" in reply.answer
    assert reply.withheld_draft == "It is worth 0.45."


def test_agent_validator_can_fail_a_clean_result(svc):
    llm = ScriptedLLM([call("t1", "price_option", price_args(22500, "put")), say("Done.")], verdict="FAIL")
    reply = OptionPricingAgent(svc, llm=llm).ask("Price the 22500 call")  # tool used a put: mismatch
    assert not reply.released and reply.validator_verdict == "FAIL"


def test_agent_reports_tool_errors_and_bad_json(svc):
    bad = LLMReply("", [ToolCall("t2", "price_option", {}, parse_error="tool arguments are not valid JSON")])
    llm = ScriptedLLM([call("t1", "price_option", {**price_args(22500, "call"), "expiry": "2026-12-29"}), bad,
                       say("That expiry is not available in this data.")])
    reply = OptionPricingAgent(svc, llm=llm).ask("Price the Dec 22500 call")
    assert "beyond the last fitted expiry" in reply.tool_calls[0]["error"]
    assert "not valid JSON" in reply.tool_calls[1]["error"]
    tool_msgs = [m for m in llm.calls[2]["messages"] if m["role"] == "tool"]
    assert all(m["is_error"] for m in tool_msgs)


def test_agent_llm_failure_is_reported(svc):
    class Broken(ScriptedLLM):
        def chat(self, *a, **k):
            raise LLMError("groq HTTP 429: rate limit")
    reply = OptionPricingAgent(svc, llm=Broken([])).ask("Price the 22500 call")
    assert not reply.released and "429" in reply.answer


def test_agent_requires_llm(svc, monkeypatch):
    no_llm(monkeypatch)
    with pytest.raises(AgentUnavailable, match="LLM_PROVIDER"):
        OptionPricingAgent(svc)


# ------------------------------------------------------------------ Claude adapter end to end (SDK-shaped fake)

def sdk_text(t):
    return SimpleNamespace(type="text", text=t)


def sdk_tool(id_, name, inp):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


class FakeAnthropicSDK:
    def __init__(self, turns):
        self.turns, self.calls = list(turns), []
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kw):
        self.calls.append(kw)
        if kw.get("system") == VALIDATOR_PROMPT:
            return SimpleNamespace(stop_reason="end_turn", content=[sdk_text("Fine.\nVERDICT: PASS")])
        content = self.turns.pop(0)
        stop = "tool_use" if any(b.type == "tool_use" for b in content) else "end_turn"
        return SimpleNamespace(stop_reason=stop, content=content)


def test_agent_on_anthropic_adapter(svc):
    sdk = FakeAnthropicSDK([[sdk_text("Pricing."), sdk_tool("tu1", "price_option", price_args(22500, "call"))],
                            [sdk_text("The 22,500 call is 163.44 per unit.")]])
    llm = AnthropicLLM("claude-sonnet-5-5", client=sdk)
    reply = OptionPricingAgent(svc, llm=llm).ask("Price the 22500 call")
    assert reply.released and reply.provider == "anthropic"
    second = sdk.calls[1]
    assert second["messages"][1]["content"][1] == {"type": "tool_use", "id": "tu1", "name": "price_option",
                                                    "input": price_args(22500, "call")}
    result_block = second["messages"][2]["content"][0]
    assert result_block["type"] == "tool_result" and result_block["tool_use_id"] == "tu1"
    assert second["tools"][0]["name"] == "market_overview" and "input_schema" in second["tools"][0]
