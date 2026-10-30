"""Subscriber list for the daily feed: SQLite, double opt-in, one-click unsubscribe, erasure on request.

States: pending (signed up, confirmation email sent) -> active (clicked the confirmation link)
        -> unsubscribed (clicked unsubscribe; kept so the address is not mailed again by mistake).
delete() removes the row entirely (a request to erase personal data).

Each subscriber has one random token used in both the confirm and the unsubscribe link, so a link only
works for that address and needs no password.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+'-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}$")
CONSENT_TEXT = ("I want to receive the daily options market feed by email. I understand it is market information, "
                "not investment advice, and I can unsubscribe at any time.")


class SubscriptionError(ValueError):
    pass


MIN_PASSWORD = 8


def check_password(password: str) -> None:
    if len(password or "") < MIN_PASSWORD:
        raise SubscriptionError(f"the password needs at least {MIN_PASSWORD} characters")
    if len(password) > 256:
        raise SubscriptionError("the password is too long")


def hash_password(password: str) -> str:
    """scrypt (memory-hard, standard library) with a random salt; only the hash is stored."""
    salt = secrets.token_bytes(16)
    n, r, p = 2 ** 14, 8, 1
    h = hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, dklen=32)
    return f"scrypt${n}${r}${p}${salt.hex()}${h.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt, h = stored.split("$")
        if algo != "scrypt":
            return False
        calc = hashlib.scrypt((password or "").encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p),
                              dklen=len(bytes.fromhex(h)))
        return hmac.compare_digest(calc, bytes.fromhex(h))
    except (ValueError, TypeError):
        return False


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
    password_hash: Optional[str] = None
    reset_token: Optional[str] = None
    reset_expires: Optional[str] = None

    @property
    def has_account(self) -> bool:
        return bool(self.password_hash)

    @property
    def verified(self) -> bool:
        return bool(self.confirmed_at)

    def public(self) -> dict:
        """Everything except secrets (link token, password hash, reset token)."""
        hidden = ("token", "password_hash", "reset_token", "reset_expires")
        return {**{k: v for k, v in self.__dict__.items() if k not in hidden}, "has_account": self.has_account}


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
            cols = {r["name"] for r in db.execute("PRAGMA table_info(subscribers)")}
            for col in ("password_hash", "reset_token", "reset_expires"):  # accounts, added after the first release
                if col not in cols:
                    db.execute(f"ALTER TABLE subscribers ADD COLUMN {col} TEXT")
            db.execute("""CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, email TEXT NOT NULL,
                created_at TEXT NOT NULL, expires_at TEXT NOT NULL)""")

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
        email = normalise_email(email)
        with self._lock, self._db() as db:
            db.execute("DELETE FROM sessions WHERE email = ? COLLATE NOCASE", (email,))
            return db.execute("DELETE FROM subscribers WHERE email = ?", (email,)).rowcount > 0

    # accounts (website log-in) -------------------------------------------------
    def create_account(self, email: str, name: str, password: str, consent: bool,
                       source: str = "website") -> Subscriber:
        """Sign up with a password. The daily email starts once the address is verified by email.
        A reader who subscribed earlier without a password can set one here; an existing account cannot."""
        check_password(password)
        cur = self.get(email)
        if cur and cur.has_account:
            raise SubscriptionError("an account with this email already exists: log in, or reset the password")
        sub = cur if cur and cur.status == "active" else self.subscribe(email, name, consent, source)
        with self._lock, self._db() as db:
            db.execute("UPDATE subscribers SET password_hash = ?, name = COALESCE(NULLIF(?, ''), name) WHERE email = ?",
                       (hash_password(password), (name or "").strip()[:80], sub.email))
        return self.get(sub.email)

    def authenticate(self, email: str, password: str) -> Optional[Subscriber]:
        try:
            s = self.get(email)
        except SubscriptionError:
            return None
        if not s or not s.password_hash or not verify_password(password, s.password_hash):
            return None
        return s

    def set_password(self, email: str, password: str) -> None:
        check_password(password)
        with self._lock, self._db() as db:
            db.execute("UPDATE subscribers SET password_hash = ?, reset_token = NULL, reset_expires = NULL "
                       "WHERE email = ?", (hash_password(password), normalise_email(email)))
            db.execute("DELETE FROM sessions WHERE email = ? COLLATE NOCASE", (normalise_email(email),))

    def start_reset(self, email: str, minutes: int = 60) -> Optional[Subscriber]:
        """A one-hour password-reset token; None if the address is unknown (the caller says the same either way)."""
        try:
            s = self.get(email)
        except SubscriptionError:
            return None
        if not s:
            return None
        token = secrets.token_urlsafe(24)
        expires = (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat(timespec="seconds")
        with self._lock, self._db() as db:
            db.execute("UPDATE subscribers SET reset_token = ?, reset_expires = ? WHERE email = ?", (token, expires, s.email))
        return self.get(s.email)

    def finish_reset(self, token: str, password: str) -> Optional[Subscriber]:
        if not token:
            return None
        with self._db() as db:
            s = self._row(db.execute("SELECT * FROM subscribers WHERE reset_token = ?", (token,)).fetchone())
        if not s or not s.reset_expires or s.reset_expires < _now():
            return None
        self.set_password(s.email, password)
        if not s.confirmed_at:  # the reset link proves the reader owns the address
            self.confirm(s.token)
        return self.get(s.email)

    def set_daily(self, email: str, on: bool) -> Subscriber:
        """Account holders switch the daily email on and off; a verified address needs no new confirmation."""
        s = self.get(email)
        with self._lock, self._db() as db:
            if on and s.confirmed_at:
                db.execute("UPDATE subscribers SET status = 'active', unsubscribed_at = NULL WHERE email = ?", (s.email,))
            elif not on and s.status != "unsubscribed":
                db.execute("UPDATE subscribers SET status = 'unsubscribed', unsubscribed_at = ? WHERE email = ?",
                           (_now(), s.email))
        return self.get(s.email)

    def update_name(self, email: str, name: str) -> None:
        with self._lock, self._db() as db:
            db.execute("UPDATE subscribers SET name = ? WHERE email = ?", ((name or "").strip()[:80], normalise_email(email)))

    # sessions ---------------------------------------------------------------------
    def new_session(self, email: str, days: int = 14) -> str:
        token = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        with self._lock, self._db() as db:
            db.execute("DELETE FROM sessions WHERE expires_at < ?", (_now(),))
            db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)", (token, normalise_email(email),
                       now.isoformat(timespec="seconds"), (now + timedelta(days=days)).isoformat(timespec="seconds")))
        return token

    def session_user(self, token: str) -> Optional[Subscriber]:
        if not token:
            return None
        with self._db() as db:
            r = db.execute("SELECT email FROM sessions WHERE token = ? AND expires_at > ?", (token, _now())).fetchone()
        return self.get(r["email"]) if r else None

    def end_session(self, token: str) -> None:
        with self._lock, self._db() as db:
            db.execute("DELETE FROM sessions WHERE token = ?", (token or "",))

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
