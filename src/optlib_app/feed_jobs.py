"""The daily feed end to end: sign-up with email confirmation, the daily send, previews and the send log.

Used by the API (/feed/...), the dashboard's Daily feed tab and scripts/daily_feed.py.

    FEED_PUBLIC_URL   address readers' links point to (confirm, unsubscribe). Default: PRICING_API_URL.
                      If readers cannot reach the API from their machines, links will not work for them;
                      they can still reply UNSUBSCRIBE (see README).
    FEED_ADMIN_KEY    if set, admin actions over the API (subscriber list, sending) need this key.
"""
from __future__ import annotations

import html
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from .config import ROOT, get_setting
from .feed import DISCLAIMER, build_feed, record
from .feed_email import render
from .mailer import Mailer, MailSettings, build_message
from .subscribers import CONSENT_TEXT, SubscriberStore, SubscriptionError, normalise_email

FEED_DIR = ROOT / "data" / "feed"
log = logging.getLogger("uvicorn.error")  # shows in the API console


class FeedJobs:
    def __init__(self, svc, feed_dir: Path = FEED_DIR, get=get_setting):
        self.svc = svc
        self.dir = Path(feed_dir)
        self.store = SubscriberStore(self.dir / "subscribers.db")
        self.mail = MailSettings.from_settings(get, self.dir / "outbox")
        self.public_url = (get("FEED_PUBLIC_URL", "") or get("PRICING_API_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.admin_key = get("FEED_ADMIN_KEY", "")
        self.history_path = self.dir / "history.jsonl"
        self.log_path = self.dir / "sent.jsonl"

    # links ------------------------------------------------------------------
    def link(self, action: str, token: str) -> str:
        path = {"confirm": "verify", "reset": "reset"}.get(action, f"feed/{action}")  # website pages
        return f"{self.public_url}/{path}?token={quote(token)}"

    def _simple_email(self, to: str, subject: str, heading: str, paragraphs: list[str], button: tuple[str, str],
                      unsub: str = "") -> object:
        label, url = button
        paras = "".join(f'<p style="margin:0 0 14px">{p}</p>' for p in paragraphs)
        text = "\n\n".join([heading] + [html.unescape(p) for p in paragraphs] + [f"{label}: {url}", DISCLAIMER])
        body = f"""<!doctype html><html><body style="margin:0;background:#f1f5f9;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:28px 12px">
<table role="presentation" width="560" cellpadding="0" cellspacing="0" style="max-width:560px;width:100%;background:#fff;border:1px solid #e2e8f0;border-radius:14px">
<tr><td style="background:#0b1220;padding:20px 26px;border-radius:14px 14px 0 0;color:#a5b4fc;font-size:12px;font-weight:700;letter-spacing:.14em">NIFTY OPTIONS DAILY</td></tr>
<tr><td style="padding:26px;font-size:15px;line-height:1.6;color:#0f172a"><h2 style="margin:0 0 14px;font-size:20px">{html.escape(heading)}</h2>{paras}
<p style="margin:6px 0 22px"><a href="{html.escape(url)}" style="display:inline-block;background:#4f46e5;color:#fff;text-decoration:none;padding:12px 22px;border-radius:8px;font-weight:600">{html.escape(label)}</a></p>
<p style="margin:0;font-size:12px;color:#64748b;line-height:1.5"><b>Not investment advice.</b> {html.escape(DISCLAIMER)}</p>
</td></tr></table></td></tr></table></body></html>"""
        return build_message(self.mail, to, subject, body, text, unsubscribe_url=unsub)

    def send_reset(self, email: str) -> dict:
        """Email a one-hour password-reset link. Unknown addresses get the same answer (no account probing)."""
        s = self.store.start_reset(email)
        if s:
            msg = self._simple_email(s.email, "Reset your NIFTY options daily password", "Reset your password",
                                     ["Someone asked to reset the password for this address. The link works for one "
                                      "hour. If it was not you, ignore this email: nothing changes."],
                                     ("Choose a new password", self.link("reset", s.reset_token)))
            res = Mailer(self.mail).send([msg])[0]
            log.info("feed password reset for %s: %s", s.email, res.get("file") or ("sent" if res["ok"] else res["error"]))
        where = " (outbox mode: saved in data/feed/outbox, not emailed)" if self.mail.mode == "outbox" else ""
        return {"message": f"If an account exists for that address, a reset link is on its way{where}."}

    def create_account(self, email: str, name: str, password: str, consent: bool) -> dict:
        sub = self.store.create_account(email, name, password, consent)
        log.info("feed account %s -> %s", sub.email, sub.status)
        if sub.verified:
            return {"status": "active", "email": sub.email, "message": "Account created. You are already receiving the daily email."}
        res = Mailer(self.mail).send([self._confirmation(sub)])[0]
        log.info("feed confirmation to %s: %s", sub.email, res.get("file") or ("sent" if res["ok"] else res["error"]))
        if not res["ok"]:
            return {"status": "error", "email": sub.email, "message": f"Account created, but the verification email "
                                                                      f"could not be sent: {res['error']}"}
        where = " (outbox mode: saved in data/feed/outbox, not emailed)" if self.mail.mode == "outbox" else ""
        return {"status": "pending", "email": sub.email,
                "message": f"Account created. We sent a verification link to {sub.email}{where}; the daily email "
                           "starts once you click it."}

    # sign-up ----------------------------------------------------------------
    def subscribe(self, email: str, name: str = "", consent: bool = False, source: str = "dashboard") -> dict:
        sub = self.store.subscribe(email, name, consent, source)
        log.info("feed sign-up %s -> %s (list %s)", sub.email, sub.status, self.store.path.resolve())
        if sub.status == "active":
            return {"status": "active", "email": sub.email,
                    "message": f"{sub.email} is already subscribed; nothing to do."}
        res = Mailer(self.mail).send([self._confirmation(sub)])[0]
        log.info("feed confirmation to %s: %s", sub.email, res.get("file") or ("sent" if res["ok"] else res["error"]))
        if not res["ok"]:
            return {"status": "error", "email": sub.email, "message": f"could not send the confirmation: {res['error']}"}
        where = " (outbox mode: saved in data/feed/outbox, not emailed)" if self.mail.mode == "outbox" else ""
        return {"status": "pending", "email": sub.email,
                "message": f"Almost done: we sent a confirmation link to {sub.email}{where}. "
                           "The subscription starts when it is clicked."}

    def _confirmation(self, sub):
        confirm, unsub = self.link("confirm", sub.token), self.link("unsubscribe", sub.token)
        name = f" {sub.name}" if sub.name else ""
        text = (f"Hi{name},\n\nPlease verify your email for the NIFTY options daily feed:\n{confirm}\n\n"
                f"If you did not sign up, ignore this email and nothing will be sent.\n\n{CONSENT_TEXT}\n\n{DISCLAIMER}\n")
        body = f"""<!doctype html><html><body style="margin:0;background:#f1f5f9;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:28px 12px">
<table role="presentation" width="560" cellpadding="0" cellspacing="0" style="max-width:560px;width:100%;background:#fff;border:1px solid #e2e8f0;border-radius:14px">
<tr><td style="background:#0b1220;padding:20px 26px;border-radius:14px 14px 0 0;color:#a5b4fc;font-size:12px;font-weight:700;letter-spacing:.14em">NIFTY OPTIONS DAILY</td></tr>
<tr><td style="padding:26px;font-size:15px;line-height:1.6;color:#0f172a">
<p style="margin:0 0 14px">Hi{html.escape(name)},</p>
<p style="margin:0 0 20px">Please confirm that you want the daily email with NIFTY option-market numbers: implied volatility, skew, expected moves, reference prices and open interest.</p>
<p style="margin:0 0 22px"><a href="{html.escape(confirm)}" style="display:inline-block;background:#4f46e5;color:#fff;text-decoration:none;padding:12px 22px;border-radius:8px;font-weight:600">Verify my email</a></p>
<p style="margin:0 0 14px;font-size:13px;color:#64748b">If you did not sign up, ignore this email and nothing will be sent. You can unsubscribe at any time from any feed email.</p>
<p style="margin:0;font-size:12px;color:#64748b;line-height:1.5"><b>Not investment advice.</b> {html.escape(DISCLAIMER)}</p>
</td></tr></table></td></tr></table></body></html>"""
        return build_message(self.mail, sub.email, "Verify your email for NIFTY options daily", body, text,
                             unsubscribe_url=unsub)

    def confirm(self, token: str) -> Optional[object]:
        return self.store.confirm(token)

    def unsubscribe(self, token: str) -> Optional[object]:
        return self.store.unsubscribe(token)

    # the feed ---------------------------------------------------------------
    def build(self, underlying: str = "NIFTY") -> dict:
        return build_feed(self.svc, underlying, history_path=self.history_path)

    def preview_html(self, feed: Optional[dict] = None, name: str = "") -> str:
        feed = feed or self.build()
        return render(feed, unsubscribe_url=f"{self.public_url}/feed/unsubscribe?token=preview", images="data",
                      recipient_name=name, signup_url=f"{self.public_url}/feed/signup").html

    def sent_today(self, day: str) -> bool:
        return any(r.get("date") == day and r.get("kind") == "daily" and r.get("sent", 0) > 0
                   for r in self._log())

    def _log(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        return [json.loads(x) for x in self.log_path.read_text(encoding="utf-8").splitlines() if x.strip()]

    def send(self, to: Optional[list[str]] = None, force: bool = False, feed: Optional[dict] = None) -> dict:
        """Send today's feed. ``to`` = test addresses (sent without needing a subscription); otherwise every
        active subscriber, once per day unless ``force``."""
        feed = feed or self.build()
        test = bool(to)
        if not test and self.sent_today(feed["date"]) and not force:
            return {"date": feed["date"], "kind": "daily", "sent": 0, "failed": 0, "skipped": True,
                    "message": f"the {feed['date']} feed was already sent (use force to send again)"}
        if test:
            recipients = [(normalise_email(e), "", f"{self.public_url}/feed/unsubscribe?token=test") for e in to]
        else:
            recipients = [(s.email, s.name, self.link("unsubscribe", s.token)) for s in self.store.active()]
        messages = []
        for email, name, unsub in recipients:
            m = render(feed, unsubscribe_url=unsub, images="cid", recipient_name=name,
                       signup_url=f"{self.public_url}/feed/signup")
            subject = ("[TEST] " if test else "") + m.subject
            messages.append(build_message(self.mail, email, subject, m.html, m.text, m.images, unsubscribe_url=unsub))
        results = Mailer(self.mail).send(messages) if messages else []
        record(feed, self.history_path)
        entry = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "date": feed["date"],
                 "kind": "test" if test else "daily", "mode": self.mail.mode,
                 "sent": sum(r["ok"] for r in results), "failed": sum(not r["ok"] for r in results),
                 "errors": [r for r in results if not r["ok"]][:20]}
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        msg = (f"{entry['sent']} sent, {entry['failed']} failed ({self.mail.describe()})" if recipients
               else "no active subscribers yet")
        return {**entry, "skipped": False, "message": msg,
                "files": [r["file"] for r in results if r.get("file")][:5]}

    def status(self) -> dict:
        log = self._log()
        return {"mail": self.mail.describe(), "mode": self.mail.mode, "public_url": self.public_url,
                "subscriber_db": str(self.store.path.resolve()),
                "counts": self.store.counts(), "last_send": log[-1] if log else None,
                "admin_key_required": bool(self.admin_key)}


def page(title: str, message: str, ok: bool = True) -> str:
    """Small standalone page shown after clicking a confirm or unsubscribe link."""
    col = "#15803d" if ok else "#b91c1c"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
:root{{color-scheme:light dark;--bg:#f1f5f9;--card:#fff;--ink:#0f172a;--muted:#64748b;--line:#e2e8f0}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0b1220;--card:#111a2e;--ink:#e2e8f0;--muted:#94a3b8;--line:#22304a}}}}
body{{margin:0;background:var(--bg);font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:var(--ink)}}
.c{{max-width:520px;margin:12vh auto;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:28px 26px}}
.k{{font-size:12px;letter-spacing:.14em;font-weight:700;color:#6366f1}} h1{{font-size:22px;margin:10px 0;color:{col}}}
p{{line-height:1.6;color:var(--muted)}}</style></head><body><div class="c"><div class="k">NIFTY OPTIONS DAILY</div>
<h1>{html.escape(title)}</h1><p>{html.escape(message)}</p></div></body></html>"""


__all__ = ["FeedJobs", "SubscriptionError", "page"]


def signup_page(message: str = "", ok: bool = True, email: str = "", name: str = "") -> str:
    """Public sign-up form for readers (served at /feed/signup; works without the dashboard)."""
    note = ""
    if message:
        col = "#15803d" if ok else "#b91c1c"
        note = f'<div class="msg" style="border-color:{col};color:{col}">{html.escape(message)}</div>'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Subscribe: NIFTY options daily</title><style>
:root{{color-scheme:light dark;--bg:#f1f5f9;--card:#fff;--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--soft:#f8fafc;--accent:#4f46e5}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0b1220;--card:#111a2e;--ink:#e2e8f0;--muted:#94a3b8;--line:#22304a;--soft:#0f1729;--accent:#818cf8}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:var(--ink)}}
.c{{max-width:560px;margin:7vh auto;background:var(--card);border:1px solid var(--line);border-radius:16px;overflow:hidden}}
.h{{background:linear-gradient(120deg,#0b1220,#1e1b4b);padding:22px 26px;color:#fff}}
.k{{font-size:12px;letter-spacing:.14em;font-weight:700;color:#a5b4fc}} .t{{font-size:24px;font-weight:750;margin-top:6px}}
.b{{padding:22px 26px}} p{{line-height:1.6;color:var(--muted);margin:0 0 14px;font-size:14.5px}}
label{{display:block;font-size:13px;font-weight:600;margin:14px 0 6px}}
input[type=text],input[type=email]{{width:100%;padding:11px 12px;border-radius:9px;border:1px solid var(--line);background:var(--soft);color:var(--ink);font-size:15px}}
.cb{{display:flex;gap:10px;align-items:flex-start;margin:16px 0;font-size:13.5px;line-height:1.5;color:var(--ink);font-weight:400}}
.cb input{{margin-top:3px}} button{{width:100%;padding:12px;border:0;border-radius:9px;background:var(--accent);color:#fff;font-size:15px;font-weight:650;cursor:pointer}}
.msg{{border:1px solid;border-radius:9px;padding:10px 12px;margin:0 0 14px;font-size:14px;line-height:1.5}}
.hp{{position:absolute;left:-5000px}} .d{{font-size:12px;color:var(--muted);line-height:1.55;margin-top:16px}}
ul{{margin:0 0 6px;padding-left:18px;color:var(--muted);font-size:14px;line-height:1.6}}
</style></head><body><div class="c"><div class="h"><div class="k">NIFTY OPTIONS DAILY</div>
<div class="t">Subscribe to the daily feed</div></div><div class="b">{note}
<p>One email after each trading day with the NIFTY option market in numbers:</p>
<ul><li>implied volatility by expiry and its daily change, skew and the expected move</li>
<li>market-implied odds, reference option prices with Greeks</li><li>open interest, futures basis, rates and data-quality checks</li></ul>
<form method="post" action="signup">
<label for="name">Name</label><input id="name" name="name" type="text" maxlength="80" value="{html.escape(name)}" autocomplete="name">
<label for="email">Email</label><input id="email" name="email" type="email" maxlength="254" required value="{html.escape(email)}" autocomplete="email">
<label class="cb"><input type="checkbox" name="consent" value="yes" required> {html.escape(CONSENT_TEXT)}</label>
<button type="submit">Subscribe</button></form>
<div class="d">We send a confirmation link first; nothing else arrives until you click it. Every email has a one-click
unsubscribe link. <b>Not investment advice.</b> {html.escape(DISCLAIMER)}</div></div></div></body></html>"""


class AutoSender:
    """Sends the daily feed from inside the running API, so no Task Scheduler is needed.

    FEED_AUTO_SEND   on | off | auto (default): auto = on when DATA_MODE=live and an SMTP server is configured
    FEED_SEND_TIME   HH:MM IST (default 15:20, ten minutes before the close, while quotes are live)
    Sends once per trading day (Mon-Fri), between FEED_SEND_TIME and 17:00 IST, if the API is running then.
    """

    def __init__(self, jobs: "FeedJobs", get=get_setting):
        import threading

        self.jobs = jobs
        mode = get("FEED_AUTO_SEND", "auto").lower()
        live = jobs.svc.mode == "live"
        self.enabled = mode == "on" or (mode == "auto" and live and jobs.mail.mode == "smtp")
        hh, mm = (get("FEED_SEND_TIME", "15:20") or "15:20").split(":")
        self.at = (int(hh), int(mm))
        self._done: set = set()  # days already handled by this process (sent, or nobody to send to)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="feed-auto-send", daemon=True)

    def describe(self) -> str:
        if not self.enabled:
            return "automatic daily send is off (needs DATA_MODE=live and SMTP settings, or FEED_AUTO_SEND=on)"
        return f"automatic daily send on: weekdays from {self.at[0]:02d}:{self.at[1]:02d} IST while the API runs"

    def start(self) -> None:
        log.info("feed: %s", self.describe())
        if self.enabled:
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def due(self, now) -> bool:
        start = now.replace(hour=self.at[0], minute=self.at[1], second=0, microsecond=0)
        end = now.replace(hour=17, minute=0, second=0, microsecond=0)
        day = now.date().isoformat()
        return (now.weekday() < 5 and start <= now < end and day not in self._done
                and not self.jobs.sent_today(day))

    def _run(self) -> None:
        from zoneinfo import ZoneInfo

        ist = ZoneInfo("Asia/Kolkata")
        while not self._stop.wait(30):
            now = datetime.now(ist)
            if not self.due(now):
                continue
            try:
                res = self.jobs.send()
                self._done.add(now.date().isoformat())
                log.info("feed auto-send %s: %s", now.date(), res["message"])
            except Exception as exc:  # noqa: BLE001 - keep the API up; try again in 10 minutes
                log.error("feed auto-send failed: %s", exc)
                self._stop.wait(600)
