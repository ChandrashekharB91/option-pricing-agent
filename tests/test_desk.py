"""Desk statistics (IV rank, open interest), news, plain-English strategies and PDF exports (replay, no network)."""
from datetime import date

import pytest
from fastapi.testclient import TestClient

from optlib_app import api as api_module
from optlib_app import intent
from optlib_app.config import Settings
from optlib_app.llm import LLMReply
from optlib_app.service import PricingService

HEADLINES = [{"title": "Nifty ends higher as banks gain; India VIX eases", "link": "https://example.com/a",
              "source": "Example News", "published": "2026-10-26T09:00:00+00:00", "relevance": 9}]
MARKET = {"spot": 22461.1, "forward": 22480.0, "atm_vol": 0.16, "as_of": date(2026, 10, 22),
          "expiries": [date(2026, 10, 27), date(2026, 11, 3), date(2026, 11, 24)]}


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    from optlib_app import insights, news

    monkeypatch.setattr(news, "headlines", lambda *a, **k: list(HEADLINES))
    monkeypatch.setattr(insights, "insights_llm", lambda get: (_ for _ in ()).throw(insights.LLMNotConfigured("off")))
    api_module._daily.clear()


@pytest.fixture(scope="module")
def svc():
    return PricingService(Settings(data_mode="replay"))


@pytest.fixture(scope="module")
def client(svc):
    api_module._service = svc
    return TestClient(api_module.app)


# ------------------------------------------------------------------ statistics

def test_iv_stats(client):
    r = client.get("/stats/iv/NIFTY").json()
    assert r["available"] and 0 <= r["rank"] <= 1 and 0 <= r["percentile"] <= 1
    assert r["low"] <= r["vix"] <= r["high"] and len(r["series"]) == len(r["dates"]) == r["days"]
    assert 0.05 < r["atm_vol"] < 0.6 and r["label"] in ("low", "middle", "high")


def test_open_interest(client):
    r = client.get("/oi/NIFTY").json()
    assert r["pcr"] > 0 and r["rows"] and r["max_pain"] in [x["strike"] for x in r["rows"]] + [r["max_pain"]]
    assert all(abs(x["strike"] - r["forward"]) <= 4 * r["expected_move"] + 1e-6 for x in r["rows"])
    assert r["expiry"] == r["expiries"][0]
    later = client.get("/oi/NIFTY", params={"expiry": r["expiries"][-1]}).json()
    assert later["expiry"] == r["expiries"][-1]


def test_news_endpoint(client):
    assert client.get("/news").json()["items"][0]["title"].startswith("Nifty")


# ------------------------------------------------------------------ plain English

@pytest.mark.parametrize("text,capital", [("I have 2 lakh", 2e5), ("capital 3L", 3e5), ("₹50,000 only", 5e4),
                                          ("use 1.5 crore", 1.5e7), ("with 75k", 75e3), ("nothing here", None)])
def test_capital(text, capital):
    assert intent.parse_capital(text) == capital


def test_levels_and_horizon():
    assert intent.parse_levels("NIFTY to 23,000 soon", 22461) == {"target": 23000}
    assert intent.parse_levels("between 22000 and 22800", 22461) == {"range": [22000, 22800]}
    assert intent.parse_levels("may fall 2% this week", 22461) == {"pct": -0.02}
    assert intent.parse_levels("I have 2 lakh", 22461) == {}  # capital is not a level
    thu = date(2026, 10, 22)  # a Thursday
    assert intent.parse_horizon("by Friday", thu)[0] == date(2026, 10, 23)
    assert intent.parse_horizon("by next Friday", thu)[0] == date(2026, 10, 30)
    assert intent.parse_horizon("in 2 weeks", thu)[0] == date(2026, 11, 5)
    assert intent.parse_horizon("by 3 Nov", thu)[0] == date(2026, 11, 3)


def test_build_request_reads_the_idea():
    b = intent.build_request("I have 2 lakh and expect NIFTY to reach 23,000 by 3 Nov, low risk", MARKET)
    r = b["request"]
    assert r["capital"] == 2e5 and r["view"] == {"source": "user", "iv_shift": 0.0, "target": 23000}
    assert r["expiry"] == "2026-11-03" and r["risk_budget"] == 0.25 and "naked" not in r["families"]
    side = intent.build_request("sideways between 22200 and 22700, 1 lakh, aggressive", MARKET)
    assert side["interpretation"]["direction"] == "neutral" and "naked" in side["request"]["families"]
    assert side["request"]["view"]["target"] == 22450
    vol = intent.build_request("volatility will rise, 5 lakh", MARKET)
    assert vol["interpretation"]["direction"] == "none" and vol["request"]["view"]["iv_shift"] == 0.02
    with pytest.raises(intent.IntentError):
        intent.build_request("hi", MARKET)


def test_model_fills_only_the_gaps():
    class Model:
        model = "fake"

        def chat(self, system, messages, tools=None, max_tokens=0):
            return LLMReply(text='{"direction": "down", "risk": "high", "objective": "none", "vol_view": "none"}')

    got = intent.classify("NIFTY looks heavy to me, 1 lakh, keep it cautious", Model())
    assert got["direction"] == "down" and got["risk"] == "low"  # the rule found "cautious"; the model gave the rest
    assert "fake" in got["read_by"]


def test_strategy_from_text_endpoint(client):
    r = client.post("/strategy/from_text", json={"text": "I have 2 lakh and expect NIFTY to reach 22,900 by "
                                                         "expiry, low risk"}).json()
    assert r["interpretation"]["capital"] == 2e5 and r["result"]["ideas"]
    assert all(i["capital_required"] <= 2e5 for i in r["result"]["ideas"])
    assert client.post("/strategy/from_text", json={"text": "........."}).status_code == 400


# ------------------------------------------------------------------ PDFs

def test_pdfs(client):
    r = client.get("/feed/pdf")
    assert r.status_code == 200 and r.content[:5] == b"%PDF-" and len(r.content) > 20000
    note = client.get("/insights/note/NIFTY").json()
    assert note["note"] is None and note["configured"] is False  # no model: no note, never an unchecked one
    idea = client.post("/strategy/from_text", json={"text": "2 lakh, NIFTY to 22,900 by expiry"}).json()
    one = idea["result"]["ideas"][0]
    p = client.post("/strategy/pdf", json={"idea": one, "context": {"text": "2 lakh", "view_source": "user"}})
    assert p.status_code == 200 and p.content[:5] == b"%PDF-"
    assert client.post("/strategy/pdf", json={"idea": {"x": 1}}).status_code == 400


# ------------------------------------------------------------------ the agent plans a brief with the new tools

def test_agent_brief_tools_and_events(svc):
    from optlib_app.agent import OptionPricingAgent, VALIDATOR_PROMPT
    from optlib_app.llm import LLMReply, ToolCall

    class Scripted:
        provider, model = "scripted", "m"

        def __init__(self):
            self.turns = [LLMReply("", [ToolCall("a", "volatility_regime", {"underlying": "NIFTY"}),
                                        ToolCall("b", "open_interest", {"underlying": "NIFTY"}),
                                        ToolCall("c", "market_news", {"limit": 3})])]

        def chat(self, system, messages, tools=None, max_tokens=0):
            if system == VALIDATOR_PROMPT:
                return LLMReply("Consistent.\nVERDICT: PASS")
            if self.turns:
                return self.turns.pop(0)
            out = [m for m in messages if m["role"] == "tool"]
            import json
            r = json.loads(out[0]["content"])
            return LLMReply(f"India VIX is {r['vix'] * 100:.2f}.")

    events = []
    reply = OptionPricingAgent(svc, llm=Scripted()).ask("Prepare the desk brief", on_event=events.append)
    names = [r["name"] for r in reply.tool_calls]
    assert names == ["volatility_regime", "open_interest", "market_news"] and all(
        r["error"] is None for r in reply.tool_calls)
    assert "series" not in reply.tool_calls[0]["output"] and "rows" not in reply.tool_calls[1]["output"]
    kinds = [e["type"] for e in events]
    assert kinds[:2] == ["thinking", "plan"] and kinds.count("tool_done") == 3 and kinds[-1] == "validated"
    assert "India VIX" in next(e["summary"] for e in events if e["type"] == "tool_done" and e["name"] == "volatility_regime")
    assert reply.status == "PASS" and not reply.ungrounded_numbers


def test_agent_stream_endpoint(client, monkeypatch):
    from optlib_app import agent as agent_mod
    from optlib_app.llm import LLMReply

    class Say:
        provider, model = "scripted", "m"

        def chat(self, *a, **k):
            return LLMReply("Nothing to compute.\nVERDICT: PASS")

    monkeypatch.setattr(agent_mod, "agent_llm", lambda role="orchestrator": Say())
    r = client.post("/agent/stream", json={"question": "hello there"})
    import json
    lines = [json.loads(x) for x in r.text.splitlines() if x]
    assert lines[0]["type"] == "thinking" and lines[-1]["type"] == "reply"
    assert lines[-1]["reply"]["answer"].startswith("Nothing")


def test_model_sees_compact_results_and_large_requests_fall_back(svc):
    from optlib_app.agent import for_llm
    from optlib_app.insights import FallbackLLM, _RESTING
    from optlib_app.llm import LLMError, LLMReply

    out = svc.price("NIFTY", svc.expiries("NIFTY")[0], 22500, "call", 1)
    v = for_llm("price_option", {"validation": out["validation"], "x": 1.234567891})
    assert set(v["validation"]) == {"status", "not_pass"} and v["x"] == 1.23457
    assert all("PASS" not in s.split(":")[0] for s in v["validation"]["not_pass"])

    class TooBig:
        model = "groq-x"

        def chat(self, *a, **k):
            raise LLMError("groq HTTP 413: Request too large for model on tokens per minute (TPM)")

    class Gem:
        model = "gemini-x"

        def chat(self, *a, **k):
            return LLMReply("ok")

    f = FallbackLLM([TooBig(), Gem()])
    assert f.chat("s", []).text == "ok" and f.model == "gemini-x"
    assert _RESTING.get("groq-x", 0) > 0  # skipped for a minute, then tried again
    _RESTING.clear()


def test_calendar_arbitrage_elsewhere_does_not_withhold_this_price():
    """Arbitrage between two later expiries is flagged (WARN) but does not withhold a price on another expiry."""
    from optlib.surface import CalendarViolation

    s = Settings()
    s.snapshot_path = FIX4 = s.snapshot_path.parent / "upstox_live_nifty_4exp_20261029.json"
    s.instruments_path = FIX4.parent / "upstox_instruments_20261029.json"
    svc4 = PricingService(s)
    surf = svc4.surface("NIFTY")
    e = [sm.slice.expiry for sm in surf.smiles]
    surf.calendar = [CalendarViolation((e[2], e[3]), 0.117, 1e-4)]
    near = svc4.price("NIFTY", e[0], 23000, "call", 1)["validation"]
    cal = next(c for c in near["checks"] if c["name"] == "calendar arbitrage")
    assert cal["status"] == "WARN" and near["status"] != "FAIL"
    far = svc4.price("NIFTY", e[3], 23000, "call", 1)["validation"]
    assert next(c for c in far["checks"] if c["name"] == "calendar arbitrage")["status"] == "FAIL"
    assert far["status"] == "FAIL"
