"""Email delivery for the daily feed: SMTP when configured, otherwise an outbox folder of .eml files.

Settings (environment or .env; never commit the password):
    SMTP_HOST           e.g. smtp.gmail.com, smtp.office365.com, or your company relay. Unset = outbox mode.
    SMTP_PORT           587 (STARTTLS, default) or 465 (SSL)
    SMTP_SECURITY       starttls (default) | ssl | none
    SMTP_USER           login, usually the sending address
    SMTP_PASSWORD       for Gmail: an app password (Google account > Security > App passwords; needs 2-step)
    MAIL_FROM           sending address (default SMTP_USER)
    MAIL_FROM_NAME      display name (default "NIFTY Options Daily")
    MAIL_REPLY_TO       optional reply-to address

In outbox mode every message is written to data/feed/outbox/*.eml, which any mail program opens, so the
whole flow (sign-up, confirmation, daily send) can be tried before an SMTP account exists.
"""
from __future__ import annotations

import re
import smtplib
import ssl
import time
from datetime import datetime
from dataclasses import dataclass, field
from email import policy
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid
from pathlib import Path
from typing import Callable, Optional


# long URLs in List-Unsubscribe must stay plain text: allow lines up to the RFC 5322 limit instead of encoding
_POLICY = policy.SMTP.clone(max_line_length=998)


@dataclass
class MailSettings:
    host: str = ""
    port: int = 587
    security: str = "starttls"
    user: str = ""
    password: str = field(default="", repr=False)
    sender: str = ""
    sender_name: str = "NIFTY Options Daily"
    reply_to: str = ""
    outbox: Path = Path("data/feed/outbox")

    @classmethod
    def from_settings(cls, get: Callable[[str, str], str], outbox: Path) -> "MailSettings":
        user = get("SMTP_USER", "")
        return cls(host=get("SMTP_HOST", ""), port=int(get("SMTP_PORT", "587") or 587),
                   security=get("SMTP_SECURITY", "starttls").lower(), user=user, password=get("SMTP_PASSWORD", ""),
                   sender=get("MAIL_FROM", "") or user or "feed@localhost",
                   sender_name=get("MAIL_FROM_NAME", "NIFTY Options Daily"), reply_to=get("MAIL_REPLY_TO", ""),
                   outbox=outbox)

    @property
    def mode(self) -> str:
        return "smtp" if self.host else "outbox"

    def describe(self) -> str:
        if self.mode == "outbox":
            return f"outbox mode (no SMTP_HOST): emails are saved as .eml files in {self.outbox}"
        return f"SMTP {self.host}:{self.port} ({self.security}) as {self.sender}"


def build_message(ms: MailSettings, to: str, subject: str, html: str, text: str, images: Optional[dict] = None,
                  unsubscribe_url: str = "") -> EmailMessage:
    msg = EmailMessage(policy=_POLICY)
    msg["Subject"] = subject
    msg["From"] = formataddr((ms.sender_name, ms.sender))
    msg["To"] = to
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=ms.sender.split("@")[-1] if "@" in ms.sender else None)
    if ms.reply_to:
        msg["Reply-To"] = ms.reply_to
    if unsubscribe_url:  # one-click unsubscribe (RFC 8058), expected by Gmail and Yahoo for bulk mail
        msg["List-Unsubscribe"] = f"<{unsubscribe_url}>, <mailto:{ms.reply_to or ms.sender}?subject=unsubscribe>"
        msg["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")
    if images:
        html_part = msg.get_payload()[-1]
        for cid, png in images.items():
            html_part.add_related(png, maintype="image", subtype="png", cid=f"<{cid}>", filename=f"{cid}.png",
                                  disposition="inline")
    return msg


class Mailer:
    def __init__(self, settings: MailSettings, pause: float = 0.4):
        self.s = settings
        self.pause = pause

    def _connect(self) -> smtplib.SMTP:
        s = self.s
        ctx = ssl.create_default_context()
        if s.security == "ssl":
            conn = smtplib.SMTP_SSL(s.host, s.port, context=ctx, timeout=30)
        else:
            conn = smtplib.SMTP(s.host, s.port, timeout=30)
            if s.security == "starttls":
                conn.starttls(context=ctx)
        if s.user:
            conn.login(s.user, s.password)
        return conn

    def send(self, messages: list[EmailMessage]) -> list[dict]:
        """Send each message on one connection; returns [{to, ok, error, file?}]. Never raises for one bad address."""
        results = []
        if self.s.mode == "outbox":
            self.s.outbox.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # microseconds: two sends in one second never collide
            for i, m in enumerate(messages):
                safe = re.sub(r"[^A-Za-z0-9._-]", "_", m["To"])
                path = self.s.outbox / f"{stamp}_{i:03d}_{safe}.eml"
                path.write_bytes(bytes(m))
                results.append({"to": m["To"], "ok": True, "error": "", "file": str(path)})
            return results
        try:
            conn = self._connect()
        except (smtplib.SMTPException, OSError) as exc:
            return [{"to": m["To"], "ok": False, "error": f"connect/login failed: {exc}"} for m in messages]
        try:
            for i, m in enumerate(messages):
                try:
                    conn.send_message(m)
                    results.append({"to": m["To"], "ok": True, "error": ""})
                except smtplib.SMTPException as exc:
                    results.append({"to": m["To"], "ok": False, "error": str(exc)})
                if self.pause and i + 1 < len(messages):
                    time.sleep(self.pause)
        finally:
            try:
                conn.quit()
            except smtplib.SMTPException:
                pass
        return results
