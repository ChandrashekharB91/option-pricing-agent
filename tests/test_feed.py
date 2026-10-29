"""Daily email feed: the numbers, the email, subscriptions (double opt-in, unsubscribe) and sending."""
import email
import math
import re
from pathlib import Path

import pytest
from scipy.stats import norm

from optlib_app import feed as F
from optlib_app.config import FIX, Settings
from optlib_app.feed_email import render
from optlib_app.feed_jobs import FeedJobs
from optlib_app.mailer import MailSettings, build_message
from optlib_app.service import PricingService
from optlib_app.subscribers import SubscriberStore, SubscriptionError


@pytest.fixture(scope="module")
def svc():
    s = Settings()
    s.snapshot_path = FIX / "upstox_live_nifty_4exp_20261029.json"
    s.instruments_path = FIX / "upstox_instruments_20261029.json"
    return PricingService(s)


@pytest.fixture(scope="module")
def feed(svc):
    return F.build_feed(svc)


def _settings(values: dict):
    return lambda name, default="": values.get(name, default)


def test_feed_numbers(feed):
    e0 = feed["expiries"][0]
    assert feed["date"] == "2026-10-29" and len(feed["expiries"]) == 4
    # expected move is forward x ATM vol x sqrt(T)
    assert e0["expected_move"] == pytest.approx(e0["forward"] * e0["atm_vol"] * math.sqrt(e0["days"] / 365), rel=1e-9)
    assert e0["range_lo"] < e0["forward"] < e0["range_hi"]
    # NIFTY skew: puts dearer than calls
    assert all(e["rr25"] < 0 for e in feed["expiries"])
    # probabilities fall as the level rises and sit near N(-d2)-type values around the forward
    above = [p["above"] for p in feed["probabilities"]]
    assert above == sorted(above, reverse=True) and 0.4 < above[2] < 0.6
    assert feed["open_interest"]["pcr"] > 0 and len(feed["open_interest"]["top_calls"]) == 3
    assert [r["label"] for r in feed["reference_options"]] == ["25d put", "ATM put", "ATM call", "25d call"]
    assert all(r["released"] for r in feed["reference_options"])
    assert feed["highlights"] and not any("buy" in h.lower() or "sell" in h.lower() for h in feed["highlights"])


def test_delta_strike_hits_its_delta(svc):
    sm = svc.surface("NIFTY").smiles[0]
    K = F.delta_strike(sm, 0.25, "call")
    sig, T = float(sm.fit.vol(K)), sm.time_to_expiry
    d1 = (math.log(sm.forward / K) + 0.5 * sig * sig * T) / (sig * math.sqrt(T))
    assert norm.cdf(d1) == pytest.approx(0.25, abs=1e-6)


def test_day_on_day_changes(feed, tmp_path):
    hist = tmp_path / "history.jsonl"
    prev = F.snapshot_values(feed)
    prev["date"] = "2026-10-28"
    prev["atm_vols"] = {k: v - 0.005 for k, v in prev["atm_vols"].items()}
    hist.write_text(__import__("json").dumps(prev) + "\n")
    p = F.previous(hist, feed["date"])
    ch = F.changes(feed, p)
    assert ch["against"] == "2026-10-28"
    assert ch["atm_vol"][feed["expiries"][0]["expiry"]] == pytest.approx(0.005)
    F.record(feed, hist)
    F.record(feed, hist)  # same day twice keeps one line
    assert len(hist.read_text().splitlines()) == 2


def test_email_render(feed):
    m = render(feed, unsubscribe_url="https://x.example/feed/unsubscribe?token=t", images="cid", recipient_name="Asha")
    assert "Not investment advice" in m.html and "Not investment advice".upper() in m.text.upper()
    assert "https://x.example/feed/unsubscribe?token=t" in m.html and "Hi Asha" in m.html
    assert set(m.images) == {"smile", "term"} and all(v[:4] == b"\x89PNG" for v in m.images.values())
    assert "cid:smile" in m.html and "<script" not in m.html.lower()
    assert "prefers-color-scheme: dark" in m.html
    inline = render(feed, images="data")
    assert "data:image/png;base64," in inline.html


def test_subscriptions(tmp_path):
    st = SubscriberStore(tmp_path / "s.db")
    with pytest.raises(SubscriptionError):
        st.subscribe("a@b.com", consent=False)
    with pytest.raises(SubscriptionError):
        st.subscribe("not-an-email", consent=True)
    s = st.subscribe("Asha@Example.COM", "Asha", consent=True)
    assert s.status == "pending" and s.email == "Asha@example.com"
    assert st.subscribe("asha@example.com", consent=True).email == "Asha@example.com"  # no duplicate
    assert st.counts() == {"pending": 1, "active": 0, "unsubscribed": 0}
    tok = st.get("asha@example.com").token
    assert st.confirm("wrong") is None and st.confirm(tok).status == "active"
    assert [x.email for x in st.active()] == ["Asha@example.com"]
    assert st.unsubscribe(tok).status == "unsubscribed" and not st.active()
    assert st.confirm(tok).status == "unsubscribed"  # an old confirm link cannot re-subscribe
    s2 = st.subscribe("asha@example.com", consent=True)  # signing up again needs a new confirmation
    assert s2.status == "pending" and s2.token != tok
    assert "token" not in s2.public()
    assert st.delete("asha@example.com") and st.get("asha@example.com") is None


def test_message_headers():
    ms = MailSettings(sender="feed@desk.example", sender_name="NIFTY Options Daily")
    url = "https://desk.example/feed/unsubscribe?token=" + "x" * 40
    m = build_message(ms, "a@b.com", "Subject ±", "<p>hi</p>", "hi", {"smile": b"\x89PNG..."}, unsubscribe_url=url)
    raw = email.message_from_bytes(bytes(m))
    assert raw["List-Unsubscribe"].startswith(f"<{url}>")  # plain, not encoded: Gmail's one-click needs it
    assert raw["List-Unsubscribe-Post"] == "List-Unsubscribe=One-Click"
    assert [p.get_content_type() for p in raw.walk()] == ["multipart/alternative", "text/plain", "multipart/related",
                                                          "text/html", "image/png"]


def test_end_to_end_outbox(svc, tmp_path):
    jobs = FeedJobs(svc, feed_dir=tmp_path, get=_settings({"FEED_PUBLIC_URL": "https://desk.example"}))
    assert jobs.mail.mode == "outbox"
    r = jobs.subscribe("reader@example.com", "Reader", consent=True)
    assert r["status"] == "pending"
    conf = email.message_from_bytes(next((tmp_path / "outbox").glob("*.eml")).read_bytes())
    body = next(p for p in conf.walk() if p.get_content_type() == "text/plain").get_payload(decode=True).decode()
    token = re.search(r"https://desk\.example/feed/confirm\?token=([\w-]+)", body).group(1)
    assert jobs.send()["message"] == "no active subscribers yet"  # nothing goes out before confirmation
    jobs.confirm(token)
    out = jobs.send()
    assert out["sent"] == 1 and out["failed"] == 0
    again = jobs.send()
    assert again["skipped"]  # once per day
    test = jobs.send(to=["me@example.com"])
    assert test["kind"] == "test" and test["sent"] == 1
    mails = sorted((tmp_path / "outbox").glob("*.eml"))
    daily = email.message_from_bytes(mails[-2].read_bytes())
    assert "unsubscribe?token=" + token in daily["List-Unsubscribe"]
    assert (tmp_path / "history.jsonl").exists() and jobs.status()["counts"]["active"] == 1


def test_api_feed_flow(svc, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from optlib_app import api

    jobs = FeedJobs(svc, feed_dir=tmp_path, get=_settings({}))
    api.app.dependency_overrides[api.get_service] = lambda: svc
    api.app.dependency_overrides[api.get_feed] = lambda: jobs
    try:
        c = TestClient(api.app)
        assert c.post("/feed/subscribe", json={"email": "x@y.com", "consent": False}).status_code == 400
        assert c.post("/feed/subscribe", json={"email": "x@y.com", "consent": True}).json()["status"] == "pending"
        token = jobs.store.get("x@y.com").token
        assert "confirmed" in c.get("/feed/confirm", params={"token": token}).text
        assert c.get("/feed/confirm", params={"token": "nope"}).status_code == 404
        assert c.get("/feed/subscribers").json()["counts"]["active"] == 1
        assert "NIFTY OPTIONS DAILY" in c.get("/feed/preview").text.upper()
        assert "unsubscribed" in c.post("/feed/unsubscribe", params={"token": token}).text
        page = c.get("/feed/signup")
        assert page.status_code == 200 and 'name="consent"' in page.text
        bad = c.post("/feed/signup", content="email=new%40y.com&name=N", headers={"content-type": "application/x-www-form-urlencoded"})
        assert bad.status_code == 400 and "consent" in bad.text and "new@y.com" in bad.text
        ok = c.post("/feed/signup", content="email=new%40y.com&name=N&consent=yes",
                    headers={"content-type": "application/x-www-form-urlencoded"})
        assert ok.status_code == 200 and "confirmation link" in ok.text and jobs.store.get("new@y.com").status == "pending"
        bot = c.post("/feed/signup", content="email=bot%40y.com&consent=yes&website=x",
                     headers={"content-type": "application/x-www-form-urlencoded"})
        assert bot.status_code == 200 and jobs.store.get("bot@y.com") is None
        jobs.admin_key = "k3y"
        assert c.get("/feed/subscribers").status_code == 403
        assert c.get("/feed/subscribers", headers={"X-Admin-Key": "k3y"}).status_code == 200
    finally:
        api.app.dependency_overrides.clear()
