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
    token = re.search(r"https://desk\.example/verify\?token=([\w-]+)", body).group(1)
    assert jobs.send()["message"] == "no active subscribers yet"  # nothing goes out before confirmation
    jobs.confirm(token)
    out = jobs.send()
    assert out["sent"] == 1 and out["failed"] == 0
    again = jobs.send()
    assert again["skipped"]  # once per day
    test = jobs.send(to=["me@example.com"])
    assert test["kind"] == "test" and test["sent"] == 1
    mails = [email.message_from_bytes(f.read_bytes()) for f in (tmp_path / "outbox").glob("*.eml")]
    assert len(mails) == 3  # confirmation, daily, test: none overwritten
    daily = next(m for m in mails if m["To"] == "reader@example.com" and m["Subject"].startswith("NIFTY options daily,")
                 and "[TEST]" not in m["Subject"])
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
        page = c.get("/feed/signup")  # old address forwards to the website's sign-up page
        assert page.status_code == 200 and 'name="password2"' in page.text
        bad = c.post("/feed/signup", content="email=new%40y.com&name=N", headers={"content-type": "application/x-www-form-urlencoded"})
        assert bad.status_code == 400 and "consent" in bad.text and "new@y.com" in bad.text
        ok = c.post("/feed/signup", content="email=new%40y.com&name=N&consent=yes",
                    headers={"content-type": "application/x-www-form-urlencoded"})
        assert ok.status_code == 200 and "confirmation link" in ok.text and jobs.store.get("new@y.com").status == "pending"
        # browser autofill can fill extra fields (Chrome fills "website"); a real reader must never be dropped
        auto = c.post("/feed/signup", content="email=auto%40y.com&consent=yes&website=https%3A%2F%2Fx.com",
                      headers={"content-type": "application/x-www-form-urlencoded"})
        assert auto.status_code == 200 and jobs.store.get("auto@y.com").status == "pending"
        jobs.admin_key = "k3y"
        assert c.get("/feed/subscribers").status_code == 403
        assert c.get("/feed/subscribers", headers={"X-Admin-Key": "k3y"}).status_code == 200
    finally:
        api.app.dependency_overrides.clear()


def test_website_accounts(svc, tmp_path):
    from fastapi.testclient import TestClient

    from optlib_app import api, site

    jobs = FeedJobs(svc, feed_dir=tmp_path, get=_settings({"FEED_PUBLIC_URL": "http://desk.example"}))
    api.app.dependency_overrides[api.get_service] = lambda: svc
    api.app.dependency_overrides[api.get_feed] = lambda: jobs
    form = {"content-type": "application/x-www-form-urlencoded"}
    site._failures.clear()
    try:
        c = TestClient(api.app)
        assert "Create a free account" in c.get("/").text
        assert "do not match" in c.post("/signup", content="name=A&email=a%40x.com&password=secret123&password2=other123"
                                         "&consent=yes", headers=form).text
        assert "at least 8" in c.post("/signup", content="email=a%40x.com&password=short&password2=short&consent=yes",
                                      headers=form).text
        r = c.post("/signup", content="name=Asha&email=a%40x.com&password=secret123&password2=secret123&consent=yes",
                   headers=form)
        assert r.status_code == 200 and "Hello, Asha" in r.text and "Verify your email first" in r.text
        acct = jobs.store.get("a@x.com")
        assert acct.has_account and acct.status == "pending" and "secret123" not in (acct.password_hash or "")
        assert "already exists" in c.post("/signup", content="email=a%40x.com&password=secret123&password2=secret123"
                                          "&consent=yes", headers=form).text
        # verification link from the email activates the daily email
        mail = email.message_from_bytes(sorted((tmp_path / "outbox").glob("*.eml"))[-1].read_bytes())
        body = next(p for p in mail.walk() if p.get_content_type() == "text/plain").get_payload(decode=True).decode()
        token = re.search(r"/verify\?token=([\w-]+)", body).group(1)
        assert "You receive the email" in c.get("/verify", params={"token": token}).text
        assert jobs.store.get("a@x.com").status == "active"
        # account actions need the form token
        assert "form expired" in c.post("/account", content="action=off&csrf=bad", headers=form).text
        csrf = site._csrf(c.cookies.get(site.COOKIE))
        assert "switched off" in c.post("/account", content=f"action=off&csrf={csrf}", headers=form).text
        assert jobs.store.get("a@x.com").status == "unsubscribed"
        assert "switched on" in c.post("/account", content=f"action=on&csrf={csrf}", headers=form).text
        # log out, bad password, good password, brute-force brake
        c.get("/logout")
        assert c.get("/account").url.path == "/login"
        assert "not recognised" in c.post("/login", content="email=a%40x.com&password=nope1234", headers=form).text
        assert "Hello, Asha" in c.post("/login", content="email=A%40X.com&password=secret123", headers=form).text
        c.get("/logout")
        for _ in range(5):
            c.post("/login", content="email=a%40x.com&password=wrong123", headers=form)
        assert "Too many attempts" in c.post("/login", content="email=a%40x.com&password=secret123", headers=form).text
        site._failures.clear()
        # password reset by email; unknown addresses get the same answer
        assert "If an account exists" in c.post("/forgot", content="email=nobody%40x.com", headers=form).text
        c.post("/forgot", content="email=a%40x.com", headers=form)
        reset = jobs.store.get("a@x.com").reset_token
        assert "Password saved" in c.post("/reset", content=f"token={reset}&password=newpass99&password2=newpass99",
                                          headers=form).text
        assert "Link expired" in c.post("/reset", content=f"token={reset}&password=x2345678&password2=x2345678",
                                        headers=form).text  # one use only
        assert "Today" in c.get("/account/today").text
        csrf = site._csrf(c.cookies.get(site.COOKIE))
        c.post("/account", content=f"action=delete&csrf={csrf}", headers=form)
        assert jobs.store.get("a@x.com") is None
    finally:
        api.app.dependency_overrides.clear()


def test_auto_sender_schedule(svc, tmp_path):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from optlib_app.feed_jobs import AutoSender

    jobs = FeedJobs(svc, feed_dir=tmp_path, get=_settings({}))
    off = AutoSender(jobs, get=_settings({}))
    assert not off.enabled  # replay data and no SMTP: never sends by itself
    on = AutoSender(jobs, get=_settings({"FEED_AUTO_SEND": "on", "FEED_SEND_TIME": "15:20"}))
    ist = ZoneInfo("Asia/Kolkata")
    assert not on.due(datetime(2026, 10, 30, 15, 10, tzinfo=ist))  # before the time
    assert on.due(datetime(2026, 10, 30, 15, 25, tzinfo=ist))  # Friday, in the window
    assert not on.due(datetime(2026, 10, 31, 15, 25, tzinfo=ist))  # Saturday
    assert not on.due(datetime(2026, 10, 30, 17, 5, tzinfo=ist))  # too late
