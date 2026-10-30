"""Account endpoints for the shared dashboard (/auth/...). The dashboard calls them from its server.

Sessions are random tokens sent in the X-Session header. Passwords are stored only as scrypt hashes
(see subscribers.py). Repeated wrong passwords for one address are refused for 15 minutes.
"""
from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from .subscribers import SubscriptionError

router = APIRouter(prefix="/auth", tags=["accounts"])
_failures: dict[str, list[float]] = {}
MAX_FAILURES, WINDOW = 5, 900


def _feed(request: Request):
    from . import api

    over = request.app.dependency_overrides.get(api.get_feed)
    if over:
        return over()
    return api.get_feed(request.app.dependency_overrides.get(api.get_service, api.get_service)())


def _bad(exc: Exception, code: int = 400):
    raise HTTPException(status_code=code, detail=str(exc)[:1].upper() + str(exc)[1:])


def _public(feed, s) -> dict:
    d = s.public()
    d["verified"] = s.verified
    d["daily"] = s.status == "active"
    d["admin"] = s.email.lower() in feed.admins
    d["verification_required"] = feed.mail.mode == "smtp"  # without real email, nobody could ever verify
    return d


def _session_user(feed, token: Optional[str]):
    s = feed.store.session_user(token or "")
    if not s:
        raise HTTPException(status_code=401, detail="Please log in again.")
    return s


class SignupIn(BaseModel):
    email: str = Field(..., max_length=254)
    name: str = Field("", max_length=80)
    password: str = Field(..., max_length=256)
    accept: bool = False
    daily: bool = True


class LoginIn(BaseModel):
    email: str = Field(..., max_length=254)
    password: str = Field(..., max_length=256)


class TokenIn(BaseModel):
    token: str = Field(..., max_length=200)


class ResetIn(TokenIn):
    password: str = Field(..., max_length=256)


class EmailIn(BaseModel):
    email: str = Field(..., max_length=254)


class DailyIn(BaseModel):
    on: bool


class PasswordIn(BaseModel):
    current: str = Field(..., max_length=256)
    new: str = Field(..., max_length=256)


class NameIn(BaseModel):
    name: str = Field("", max_length=80)


@router.post("/signup")
def signup(req: SignupIn, feed=Depends(_feed)):
    if not req.accept:
        _bad(SubscriptionError("please accept the terms to create an account"))
    try:
        r = feed.create_account(req.email, req.name, req.password, consent=True, daily=req.daily)
    except SubscriptionError as exc:
        _bad(exc)
    token = feed.store.new_session(r["email"])
    return {**r, "token": token, "user": _public(feed, feed.store.get(r["email"]))}


@router.post("/login")
def login(req: LoginIn, feed=Depends(_feed)):
    key = req.email.strip().lower()
    now = time.time()
    recent = [t for t in _failures.get(key, []) if now - t < WINDOW]
    if len(recent) >= MAX_FAILURES:
        raise HTTPException(status_code=429, detail="Too many attempts. Wait 15 minutes, or reset your password.")
    s = feed.store.authenticate(req.email, req.password)
    if not s:
        _failures[key] = recent + [now]
        raise HTTPException(status_code=401, detail="Email or password not recognised.")
    _failures.pop(key, None)
    return {"token": feed.store.new_session(s.email), "user": _public(feed, s)}


@router.post("/logout")
def logout(x_session: Optional[str] = Header(default=None), feed=Depends(_feed)):
    feed.store.end_session(x_session or "")
    return {"ok": True}


@router.get("/me")
def me(x_session: Optional[str] = Header(default=None), feed=Depends(_feed)):
    return _public(feed, _session_user(feed, x_session))


@router.post("/verify")
def verify(req: TokenIn, feed=Depends(_feed)):
    s = feed.confirm(req.token)
    if not s:
        raise HTTPException(status_code=404, detail="This verification link is not valid or was replaced by a newer one.")
    out = {"user": _public(feed, s), "message": "Email verified." + (" The daily email is on." if s.status == "active" else "")}
    if s.has_account:
        out["token"] = feed.store.new_session(s.email)
    return out


@router.post("/resend")
def resend(x_session: Optional[str] = Header(default=None), feed=Depends(_feed)):
    s = _session_user(feed, x_session)
    if s.verified:
        return {"message": "Your email is already verified."}
    from .mailer import Mailer

    res = Mailer(feed.mail).send([feed._confirmation(s)])[0]
    where = " (outbox mode: saved in data/feed/outbox, not emailed)" if feed.mail.mode == "outbox" else ""
    return {"message": f"Verification link sent to {s.email}{where}." if res["ok"] else f"Could not send: {res['error']}"}


@router.post("/forgot")
def forgot(req: EmailIn, feed=Depends(_feed)):
    return feed.send_reset(req.email)


@router.post("/reset")
def reset(req: ResetIn, feed=Depends(_feed)):
    try:
        s = feed.store.finish_reset(req.token, req.password)
    except SubscriptionError as exc:
        _bad(exc)
    if not s:
        raise HTTPException(status_code=404, detail="This reset link has expired (they work for one hour). Ask for a new one.")
    return {"token": feed.store.new_session(s.email), "user": _public(feed, s), "message": "Password saved."}


@router.post("/unsubscribe")
def unsubscribe(req: TokenIn, feed=Depends(_feed)):
    s = feed.unsubscribe(req.token)
    if not s:
        raise HTTPException(status_code=404, detail="This unsubscribe link is not valid (or came from a test email).")
    return {"message": f"{s.email} will not receive any more daily emails."}


@router.post("/daily")
def daily(req: DailyIn, x_session: Optional[str] = Header(default=None), feed=Depends(_feed)):
    s = _session_user(feed, x_session)
    if req.on and not s.verified:
        raise HTTPException(status_code=400, detail="Verify your email first; we can resend the link.")
    return _public(feed, feed.store.set_daily(s.email, req.on))


@router.post("/name")
def name(req: NameIn, x_session: Optional[str] = Header(default=None), feed=Depends(_feed)):
    s = _session_user(feed, x_session)
    feed.store.update_name(s.email, req.name)
    return _public(feed, feed.store.get(s.email))


@router.post("/password")
def password(req: PasswordIn, x_session: Optional[str] = Header(default=None), feed=Depends(_feed)):
    s = _session_user(feed, x_session)
    if not feed.store.authenticate(s.email, req.current):
        raise HTTPException(status_code=400, detail="Current password not recognised; nothing changed.")
    try:
        feed.store.set_password(s.email, req.new)  # ends every session
    except SubscriptionError as exc:
        _bad(exc)
    return {"token": feed.store.new_session(s.email), "message": "Password changed."}


@router.delete("/account")
def delete(x_session: Optional[str] = Header(default=None), feed=Depends(_feed)):
    s = _session_user(feed, x_session)
    feed.store.delete(s.email)
    return {"deleted": True}
