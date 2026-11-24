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
