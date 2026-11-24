"""AI market note: published only when grounded in the data and free of advice; news parsing and ranking."""
import pytest

from optlib_app import feed as F
from optlib_app import insights, news
from optlib_app.config import FIX, Settings
from optlib_app.llm import LLMReply
from optlib_app.service import PricingService


class ScriptedLLM:
    model, provider = "fake-model", "fake"

    def __init__(self, *texts):
        self.texts = list(texts)
        self.calls = []

    def chat(self, system, messages, tools=None, max_tokens=2048):
        self.calls.append([dict(m) for m in messages])
        return LLMReply(text=self.texts.pop(0))


@pytest.fixture(scope="module")
def feed():
    s = Settings()
    s.snapshot_path = FIX / "upstox_live_nifty_4exp_20261029.json"
    s.instruments_path = FIX / "upstox_instruments_20261029.json"
    return F.build_feed(PricingService(s))


def test_grounded_note_is_published(feed):
    f = insights.facts(feed)
    e0 = f["expiries"][0]
    text = (f"NIFTY ended at {f['spot']:,.2f}. The {e0['expiry']} expiry trades at {e0['atm_vol_pct']}% implied "
            f"volatility, pricing a move of about {e0['expected_move_points']} points.")
    note = insights.market_note(feed, [], llm=ScriptedLLM(text))
    assert note and note["text"] == text and note["model"] == "fake-model"


def test_invented_numbers_are_retried_then_dropped(feed):
    llm = ScriptedLLM("NIFTY could reach 27,431 soon.", "Volatility is 99.97% today.")
    assert insights.market_note(feed, [], llm=llm) is None
    assert len(llm.calls) == 2 and "27,431" in llm.calls[1][-1]["content"]  # the retry names the bad number


def test_advice_is_dropped(feed):
    spot = f"{insights.facts(feed)['spot']:,.2f}"
    llm = ScriptedLLM(f"NIFTY is at {spot}; traders should buy calls.", f"NIFTY is at {spot}; you must sell puts.")
    assert insights.market_note(feed, [], llm=llm) is None


def test_note_without_any_model_is_none(feed):
    assert insights.market_note(feed, [], get=lambda k, d="": d) is None


def test_news_parse_rank_and_dedupe():
    rss = """<rss><channel>
      <item><title>Nifty, Sensex fall as FIIs sell; India VIX jumps</title><link>https://a.example/1</link>
        <pubDate>Tue, 24 Nov 2026 10:00:00 +0530</pubDate></item>
      <item><title>Celebrity buys a new car</title><link>https://a.example/2</link>
        <pubDate>Tue, 24 Nov 2026 11:00:00 +0530</pubDate></item>
      <item><title>Nifty, Sensex fall as FIIs sell; India VIX jumps</title><link>https://a.example/3</link>
        <pubDate>Tue, 24 Nov 2026 09:00:00 +0530</pubDate></item>
    </channel></rss>"""
    news._cache.clear()
    out = news.headlines(5, get=lambda k, d="": "Test|https://feed.example" if k == "NEWS_FEEDS" else d,
                         fetch=lambda url: rss, max_age_hours=1e9)
    assert [o["title"] for o in out] == ["Nifty, Sensex fall as FIIs sell; India VIX jumps", "Celebrity buys a new car"]
    assert out[0]["source"] == "Test" and out[0]["relevance"] > out[1]["relevance"]


def test_gemini_falls_back_when_busy():
    from optlib_app.insights import FallbackLLM
    from optlib_app.llm import LLMError, LLMReply

    class Busy:
        model = "a"

        def chat(self, *a, **k):
            raise LLMError("gemini HTTP 503: high demand")

    class Ok:
        model = "b"

        def chat(self, *a, **k):
            return LLMReply(text="fine")

    class Bad:
        model = "c"

        def chat(self, *a, **k):
            raise LLMError("gemini HTTP 400: API key not valid")

    f = FallbackLLM([Busy(), Ok()])
    assert f.chat("s", []).text == "fine" and f.model == "b"
    import pytest
    with pytest.raises(LLMError, match="400"):  # a real error is not hidden by trying other models
        FallbackLLM([Bad(), Ok()]).chat("s", [])
