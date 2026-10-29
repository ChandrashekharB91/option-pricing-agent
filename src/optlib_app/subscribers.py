"""Subscriber list for the daily feed: SQLite, double opt-in, one-click unsubscribe, erasure on request.

States: pending (signed up, confirmation email sent) -> active (clicked the confirmation link)
        -> unsubscribed (clicked unsubscribe; kept so the address is not mailed again by mistake).
delete() removes the row entirely (a request to erase personal data).

Each subscriber has one random token used in both the confirm and the unsubscribe link, so a link only
works for that address and needs no password.
"""
from __future__ import annotations

import re
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+'-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}$")
CONSENT_TEXT = ("I want to receive the daily options market feed by email. I understand it is market information, "
                "not investment advice, and I can unsubscribe at any time.")


class SubscriptionError(ValueError):
    pass


@dataclass
class Subscriber:
    email: str
    name: str
    status: str
    token: str
    created_at: str
    confirmed_at: Optional[str] = None
    unsubscribed_at: Optional[str] = None
    consent: str = ""
    source: str = ""

    def public(self) -> dict:
        """Everything except the token (which acts as a password for the links)."""
        return {k: v for k, v in self.__dict__.items() if k != "token"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalise_email(email: str) -> str:
    e = (email or "").strip()
    if len(e) > 254 or not EMAIL_RE.match(e):
        raise SubscriptionError("please enter a valid email address")
    local, domain = e.rsplit("@", 1)
    return f"{local}@{domain.lower()}"


class SubscriberStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS subscribers (
                email TEXT PRIMARY KEY COLLATE NOCASE, name TEXT, status TEXT NOT NULL, token TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL, confirmed_at TEXT, unsubscribed_at TEXT, consent TEXT, source TEXT)""")

    @contextmanager
    def _db(self):
        """A connection that commits on success and is always closed (Windows keeps open files locked)."""
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    @staticmethod
    def _row(r) -> Optional[Subscriber]:
        return Subscriber(**dict(r)) if r else None

    def get(self, email: str) -> Optional[Subscriber]:
        with self._db() as db:
            return self._row(db.execute("SELECT * FROM subscribers WHERE email = ?", (normalise_email(email),))
                             .fetchone())

    def by_token(self, token: str) -> Optional[Subscriber]:
        if not token:
            return None
        with self._db() as db:
            return self._row(db.execute("SELECT * FROM subscribers WHERE token = ?", (token,)).fetchone())

    def subscribe(self, email: str, name: str = "", consent: bool = False, source: str = "dashboard") -> Subscriber:
        """Create (or re-open) a pending subscription. An active subscriber stays active."""
        if not consent:
            raise SubscriptionError("please tick the consent box to subscribe")
        email = normalise_email(email)
        name = (name or "").strip()[:80]
        with self._lock, self._db() as db:
            cur = self._row(db.execute("SELECT * FROM subscribers WHERE email = ?", (email,)).fetchone())
            if cur and cur.status == "active":
                if name and name != cur.name:
                    db.execute("UPDATE subscribers SET name = ? WHERE email = ?", (name, email))
                    cur.name = name
                return cur
            token = secrets.token_urlsafe(24)
            if cur:  # pending again or re-subscribing after an unsubscribe: new token, fresh consent
                db.execute("UPDATE subscribers SET name = ?, status = 'pending', token = ?, created_at = ?, "
                           "confirmed_at = NULL, unsubscribed_at = NULL, consent = ?, source = ? WHERE email = ?",
                           (name or cur.name, token, _now(), CONSENT_TEXT, source, email))
            else:
                db.execute("INSERT INTO subscribers (email, name, status, token, created_at, consent, source) "
                           "VALUES (?, ?, 'pending', ?, ?, ?, ?)", (email, name, token, _now(), CONSENT_TEXT, source))
            return self._row(db.execute("SELECT * FROM subscribers WHERE email = ?", (email,)).fetchone())

    def confirm(self, token: str) -> Optional[Subscriber]:
        with self._lock, self._db() as db:
            s = self._row(db.execute("SELECT * FROM subscribers WHERE token = ?", (token,)).fetchone())
            if not s or s.status == "unsubscribed":
                return s
            if s.status != "active":
                db.execute("UPDATE subscribers SET status = 'active', confirmed_at = ? WHERE token = ?", (_now(), token))
            return self._row(db.execute("SELECT * FROM subscribers WHERE token = ?", (token,)).fetchone())

    def unsubscribe(self, token: str) -> Optional[Subscriber]:
        with self._lock, self._db() as db:
            s = self._row(db.execute("SELECT * FROM subscribers WHERE token = ?", (token,)).fetchone())
            if s and s.status != "unsubscribed":
                db.execute("UPDATE subscribers SET status = 'unsubscribed', unsubscribed_at = ? WHERE token = ?",
                           (_now(), token))
                s = self._row(db.execute("SELECT * FROM subscribers WHERE token = ?", (token,)).fetchone())
            return s

    def unsubscribe_email(self, email: str) -> Optional[Subscriber]:
        s = self.get(email)
        return self.unsubscribe(s.token) if s else None

    def delete(self, email: str) -> bool:
        with self._lock, self._db() as db:
            return db.execute("DELETE FROM subscribers WHERE email = ?", (normalise_email(email),)).rowcount > 0

    def all(self, status: Optional[str] = None) -> list[Subscriber]:
        q, args = "SELECT * FROM subscribers", ()
        if status:
            q, args = q + " WHERE status = ?", (status,)
        with self._db() as db:
            return [self._row(r) for r in db.execute(q + " ORDER BY created_at", args).fetchall()]

    def active(self) -> list[Subscriber]:
        return self.all("active")

    def counts(self) -> dict:
        with self._db() as db:
            rows = db.execute("SELECT status, COUNT(*) AS n FROM subscribers GROUP BY status").fetchall()
        out = {"pending": 0, "active": 0, "unsubscribed": 0}
        out.update({r["status"]: r["n"] for r in rows})
        return out
