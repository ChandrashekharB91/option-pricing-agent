"""Reader website for the daily feed, served by the API at its root address (http://127.0.0.1:8000/).

    /            landing page            /signup      create an account (name, email, password, consent)
    /verify      email verification      /login       log in            /logout
    /forgot      ask for a reset link    /reset       choose a new password
    /account     daily email on/off, name, password, delete account, today's email in the browser

Accounts live in the same SQLite file as the subscriber list. Passwords are stored only as scrypt hashes;
sessions are random tokens in an HttpOnly cookie; every form carries a token derived from the session (CSRF).
"""
from __future__ import annotations

import hashlib
import html
import time
from typing import Optional
from urllib.parse import parse_qs, quote

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .feed import DISCLAIMER
from .subscribers import CONSENT_TEXT, MIN_PASSWORD, SubscriptionError

router = APIRouter(include_in_schema=False)
COOKIE = "opa_session"
_failures: dict[str, list[float]] = {}  # email -> recent failed log-in times (simple brute-force brake)
_today: dict = {}  # cached rendering of today's email: {"at": epoch, "html": str}


def _feed(request: Request):
    """The API's FeedJobs (respects the API's dependency overrides, which tests use)."""
    from . import api

    over = request.app.dependency_overrides.get(api.get_feed)
    if over:
        return over()
    svc = request.app.dependency_overrides.get(api.get_service, api.get_service)()
    return api.get_feed(svc)


async def _form(request: Request) -> dict:
    body = (await request.body()).decode("utf-8", "replace")
    return {k: v[0] for k, v in parse_qs(body, keep_blank_values=True).items()}


def _csrf(session: str) -> str:
    return hashlib.sha256(f"csrf:{session}".encode()).hexdigest()[:32]


def _user(request: Request, feed):
    return feed.store.session_user(request.cookies.get(COOKIE, ""))


def _e(x) -> str:
    return html.escape(str(x or ""))


CSS = """
:root{--bg:#f6f7fb;--card:#fff;--soft:#f1f4fa;--ink:#0f172a;--muted:#64748b;--line:#e3e8f1;--accent:#4f46e5;--accent2:#7c3aed;
--ok:#15803d;--bad:#b91c1c;--shadow:0 1px 2px rgba(15,23,42,.04),0 8px 24px rgba(15,23,42,.06)}
:root[data-theme=dark]{--bg:#0b1220;--card:#111a2e;--soft:#0f1729;--ink:#e2e8f0;--muted:#94a3b8;--line:#22304a;--accent:#818cf8;
--accent2:#a78bfa;--ok:#4ade80;--bad:#f87171;--shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.3)}
*{box-sizing:border-box}html,body{margin:0}body{background:var(--bg);color:var(--ink);
font-family:Inter,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;line-height:1.55}
a{color:var(--accent)}.wrap{max-width:1080px;margin:0 auto;padding:0 20px}
nav{border-bottom:1px solid var(--line);background:var(--card)}nav .wrap{display:flex;align-items:center;gap:18px;height:62px}
.brand{font-weight:800;font-size:17px;color:var(--ink);text-decoration:none;letter-spacing:-.01em}.brand span{color:var(--accent)}
.sp{flex:1}nav a.l{color:var(--muted);text-decoration:none;font-weight:600;font-size:14px}nav a.l:hover{color:var(--ink)}
.btn{display:inline-block;border:0;border-radius:10px;padding:11px 18px;font-weight:650;font-size:15px;cursor:pointer;
text-decoration:none;background:var(--accent);color:#fff}.btn.ghost{background:transparent;color:var(--ink);border:1px solid var(--line)}
.btn.small{padding:7px 12px;font-size:13.5px}.btn.danger{background:var(--bad)}
.tog{background:transparent;border:1px solid var(--line);color:var(--muted);border-radius:999px;padding:6px 11px;cursor:pointer;font-size:13px}
.hero{background:linear-gradient(120deg,#0b1220 0%,#1e1b4b 70%,#312e81 100%);color:#fff;padding:72px 0 64px}
.hero .k{font-size:12.5px;letter-spacing:.16em;text-transform:uppercase;color:#a5b4fc;font-weight:700}
.hero h1{font-size:44px;line-height:1.1;margin:12px 0 14px;letter-spacing:-.02em;max-width:760px}
.hero p{font-size:18px;color:#cbd5e1;max-width:680px;margin:0 0 26px}.hero .btn.ghost{color:#fff;border-color:rgba(255,255,255,.35)}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:-34px 0 28px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px;box-shadow:var(--shadow)}
.card h3{margin:0 0 6px;font-size:17px}.card p{margin:0;color:var(--muted);font-size:14.5px}
.ic{width:38px;height:38px;border-radius:10px;display:grid;place-items:center;background:rgba(99,102,241,.12);color:var(--accent);
font-weight:800;margin-bottom:12px}
.two{display:grid;grid-template-columns:1.2fr 1fr;gap:22px;margin:10px 0 40px}
ul.chk{list-style:none;padding:0;margin:0}ul.chk li{padding:7px 0 7px 28px;position:relative;color:var(--ink);font-size:15px}
ul.chk li:before{content:"✓";position:absolute;left:0;color:var(--accent);font-weight:800}
.auth{max-width:460px;margin:48px auto}.auth h1{font-size:26px;margin:0 0 6px;letter-spacing:-.01em}.sub{color:var(--muted);margin:0 0 18px}
label{display:block;font-size:13.5px;font-weight:600;margin:14px 0 6px}
input[type=text],input[type=email],input[type=password]{width:100%;padding:11px 12px;border-radius:10px;border:1px solid var(--line);
background:var(--soft);color:var(--ink);font-size:15px}input:focus{outline:2px solid var(--accent);outline-offset:1px}
.cb{display:flex;gap:10px;align-items:flex-start;font-weight:400;font-size:13.5px;line-height:1.5;margin:16px 0}.cb input{margin-top:3px}
.full{width:100%}.msg{border:1px solid;border-radius:10px;padding:10px 13px;margin:0 0 16px;font-size:14px}
.msg.ok{color:var(--ok);border-color:var(--ok)}.msg.bad{color:var(--bad);border-color:var(--bad)}
.muted{color:var(--muted);font-size:13.5px}.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12.5px;font-weight:700;border:1px solid currentColor}
.acct{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:28px 0}.acct .card h3{margin-bottom:10px}
footer{border-top:1px solid var(--line);margin-top:30px;padding:22px 0 34px;color:var(--muted);font-size:12.5px}
iframe.mail{width:100%;height:1400px;border:1px solid var(--line);border-radius:14px;background:#fff}
@media(max-width:820px){.grid,.two,.acct{grid-template-columns:1fr}.hero h1{font-size:32px}.grid{margin-top:-24px}}
"""

THEME_JS = """<script>(function(){var k='opa-theme',t=null;try{t=localStorage.getItem(k)}catch(e){}
if(!t)t=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';document.documentElement.dataset.theme=t;
window.opaToggle=function(){var n=document.documentElement.dataset.theme==='dark'?'light':'dark';
document.documentElement.dataset.theme=n;try{localStorage.setItem(k,n)}catch(e){}};})();</script>"""


def layout(title: str, body: str, user=None, full: bool = False) -> HTMLResponse:
    links = ('<a class="l" href="/account">My account</a><a class="l" href="/logout">Log out</a>' if user else
             '<a class="l" href="/login">Log in</a><a class="btn small" href="/signup">Sign up</a>')
    main = body if full else f'<div class="wrap">{body}</div>'
    return HTMLResponse(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{_e(title)} · NIFTY Options Daily</title>
{THEME_JS}<style>{CSS}</style></head><body>
<nav><div class="wrap"><a class="brand" href="/">◆ NIFTY Options <span>Daily</span></a><div class="sp"></div>{links}
<button class="tog" onclick="opaToggle()" title="Light / dark">☀ / ☾</button></div></nav>
{main}
<footer><div class="wrap"><b>Not investment advice.</b> {_e(DISCLAIMER)}</div></footer></body></html>""")


def _msg(text: str, ok: bool = True) -> str:
    return f'<div class="msg {"ok" if ok else "bad"}">{_e(text)}</div>' if text else ""


# ------------------------------------------------------------------ pages

@router.get("/", response_class=HTMLResponse)
def home(request: Request, feed=Depends(_feed)):
    user = _user(request, feed)
    cta = ('<a class="btn" href="/account">Go to my account</a>' if user else
           '<a class="btn" href="/signup">Create a free account</a> &nbsp; <a class="btn ghost" href="/login">Log in</a>')
    body = f"""<section class="hero"><div class="wrap"><div class="k">NIFTY options · every trading day</div>
<h1>The option market in numbers, in your inbox before the close.</h1>
<p>Implied volatility, skew, expected moves, market-implied odds, reference prices with Greeks and open interest:
one clean email each trading day, built from live quotes and checked by the same validation as our pricing desk.</p>
{cta}</div></section>
<div class="wrap"><div class="grid">
<div class="card"><div class="ic">σ</div><h3>Volatility at a glance</h3><p>ATM implied vol for each expiry, its change on the day,
the term structure and the 25-delta skew, next to realised volatility and India VIX.</p></div>
<div class="card"><div class="ic">±</div><h3>Expected moves and odds</h3><p>The one-standard-deviation range the options price
to each expiry, and the market-implied chance of NIFTY settling above key levels.</p></div>
<div class="card"><div class="ic">✓</div><h3>Checked numbers</h3><p>Every price passes 13+ automated checks (parity,
arbitrage, an independent QuantLib reprice). Data-quality flags are printed, not hidden.</p></div></div>
<div class="two"><div class="card"><h3>What each email contains</h3><ul class="chk">
<li>NIFTY level, change and India VIX</li><li>Volatility by expiry: ATM vol, day change, risk reversal, butterfly, straddle</li>
<li>Smile and term-structure charts</li><li>Implied vs realised volatility</li>
<li>Market-implied probabilities at the weekly expiry</li><li>Reference option prices with delta, vega and theta</li>
<li>Open interest and put/call ratio; futures basis and MIBOR-OIS rates</li></ul></div>
<div class="card"><h3>How it works</h3><ul class="chk"><li>Create an account with your email and a password</li>
<li>Click the link we email you to verify the address</li><li>Receive the email after each trading day; read today's
issue any time in your account</li><li>Switch the email off or delete your account whenever you like</li></ul>
<p class="muted" style="margin-top:12px">Market information only: no trade recommendations.</p></div></div></div>"""
    return layout("Home", body, user, full=True)


def _signup_form(message: str = "", ok: bool = True, name: str = "", email: str = "") -> str:
    return f"""<div class="auth card"><h1>Create your account</h1><p class="sub">Free. The daily email starts once you
verify your address.</p>{_msg(message, ok)}<form method="post" action="/signup">
<label for="name">Name</label><input id="name" name="name" type="text" maxlength="80" value="{_e(name)}" autocomplete="name">
<label for="email">Email (your username)</label><input id="email" name="email" type="email" maxlength="254" required value="{_e(email)}" autocomplete="email">
<label for="pw">Password</label><input id="pw" name="password" type="password" minlength="{MIN_PASSWORD}" required autocomplete="new-password">
<label for="pw2">Repeat password</label><input id="pw2" name="password2" type="password" minlength="{MIN_PASSWORD}" required autocomplete="new-password">
<label class="cb"><input type="checkbox" name="consent" value="yes" required> {_e(CONSENT_TEXT)}</label>
<button class="btn full" type="submit">Create account</button></form>
<p class="muted" style="margin-top:14px">Already have an account? <a href="/login">Log in</a></p></div>"""


@router.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request, feed=Depends(_feed)):
    user = _user(request, feed)
    if user:
        return RedirectResponse("/account", status_code=303)
    return layout("Sign up", _signup_form())


@router.post("/signup", response_class=HTMLResponse)
async def signup(request: Request, feed=Depends(_feed)):
    f = await _form(request)
    name, email = f.get("name", ""), f.get("email", "")
    if f.get("password", "") != f.get("password2", ""):
        return layout("Sign up", _signup_form("The two passwords do not match.", False, name, email))
    try:
        r = feed.create_account(email, name, f.get("password", ""), f.get("consent") == "yes")
    except SubscriptionError as exc:
        return layout("Sign up", _signup_form(str(exc).capitalize() + ".", False, name, email))
    resp = RedirectResponse("/account?m=" + quote(r["message"]), status_code=303)
    _login(resp, feed, r["email"])
    return resp


def _login(resp, feed, email: str) -> None:
    secure = feed.public_url.startswith("https://")
    resp.set_cookie(COOKIE, feed.store.new_session(email), max_age=14 * 86400, httponly=True, samesite="lax",
                    secure=secure, path="/")


def _login_form(message: str = "", ok: bool = True, email: str = "") -> str:
    return f"""<div class="auth card"><h1>Log in</h1><p class="sub">Use the email and password you signed up with.</p>
{_msg(message, ok)}<form method="post" action="/login">
<label for="email">Email</label><input id="email" name="email" type="email" required value="{_e(email)}" autocomplete="email">
<label for="pw">Password</label><input id="pw" name="password" type="password" required autocomplete="current-password">
<div style="height:16px"></div><button class="btn full" type="submit">Log in</button></form>
<p class="muted" style="margin-top:14px"><a href="/forgot">Forgot your password?</a> · New here? <a href="/signup">Create an account</a></p></div>"""


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, m: str = "", feed=Depends(_feed)):
    if _user(request, feed):
        return RedirectResponse("/account", status_code=303)
    return layout("Log in", _login_form(m))


@router.post("/login", response_class=HTMLResponse)
async def login(request: Request, feed=Depends(_feed)):
    f = await _form(request)
    email = (f.get("email") or "").strip().lower()
    now = time.time()
    recent = [t for t in _failures.get(email, []) if now - t < 900]
    if len(recent) >= 5:
        return layout("Log in", _login_form("Too many attempts. Wait 15 minutes, or reset your password.", False, email))
    user = feed.store.authenticate(email, f.get("password", ""))
    if not user:
        _failures[email] = recent + [now]
        return layout("Log in", _login_form("Email or password not recognised. If you subscribed before accounts "
                                            "existed, use “Forgot your password?” to set one.", False, email))
    _failures.pop(email, None)
    resp = RedirectResponse("/account", status_code=303)
    _login(resp, feed, user.email)
    return resp


@router.get("/logout")
def logout(request: Request, feed=Depends(_feed)):
    feed.store.end_session(request.cookies.get(COOKIE, ""))
    resp = RedirectResponse("/login?m=" + quote("You are logged out."), status_code=303)
    resp.delete_cookie(COOKIE, path="/")
    return resp


@router.get("/verify", response_class=HTMLResponse)
def verify(request: Request, token: str = "", feed=Depends(_feed)):
    s = feed.confirm(token)
    if not s:
        return layout("Verify", '<div class="auth card"><h1>Link not recognised</h1><p class="sub">This verification link '
                                'is not valid or was replaced by a newer one. <a href="/login">Log in</a> to request a new one.</p></div>')
    msg = ("Email verified. You will receive the daily email after each trading day." if s.status == "active"
           else "Email verified. The daily email is switched off; switch it on below.")
    resp = RedirectResponse("/account?m=" + quote(msg), status_code=303)
    if s.has_account:
        _login(resp, feed, s.email)
        return resp
    return layout("Verified", f'<div class="auth card"><h1>Email verified</h1><p class="sub">{_e(msg)}</p>'
                              f'<p class="muted">Want to read issues in the browser too? <a href="/forgot">Set a password</a> '
                              f'for {_e(s.email)}.</p></div>')


@router.get("/forgot", response_class=HTMLResponse)
def forgot_page(m: str = ""):
    return layout("Reset password", f"""<div class="auth card"><h1>Reset your password</h1><p class="sub">We will email
you a link to choose a new password (it also works to set a first password).</p>{_msg(m)}<form method="post" action="/forgot">
<label for="email">Email</label><input id="email" name="email" type="email" required autocomplete="email">
<div style="height:16px"></div><button class="btn full" type="submit">Email me a reset link</button></form></div>""")


@router.post("/forgot", response_class=HTMLResponse)
async def forgot(request: Request, feed=Depends(_feed)):
    f = await _form(request)
    return RedirectResponse("/forgot?m=" + quote(feed.send_reset(f.get("email", ""))["message"]), status_code=303)


def _reset_form(token: str, message: str = "") -> str:
    return f"""<div class="auth card"><h1>Choose a new password</h1>{_msg(message, False)}<form method="post" action="/reset">
<input type="hidden" name="token" value="{_e(token)}">
<label for="pw">New password</label><input id="pw" name="password" type="password" minlength="{MIN_PASSWORD}" required autocomplete="new-password">
<label for="pw2">Repeat it</label><input id="pw2" name="password2" type="password" minlength="{MIN_PASSWORD}" required autocomplete="new-password">
<div style="height:16px"></div><button class="btn full" type="submit">Save password</button></form></div>"""


@router.get("/reset", response_class=HTMLResponse)
def reset_page(token: str = ""):
    return layout("New password", _reset_form(token))


@router.post("/reset", response_class=HTMLResponse)
async def reset(request: Request, feed=Depends(_feed)):
    f = await _form(request)
    if f.get("password", "") != f.get("password2", ""):
        return layout("New password", _reset_form(f.get("token", ""), "The two passwords do not match."))
    try:
        s = feed.store.finish_reset(f.get("token", ""), f.get("password", ""))
    except SubscriptionError as exc:
        return layout("New password", _reset_form(f.get("token", ""), str(exc).capitalize() + "."))
    if not s:
        return layout("New password", '<div class="auth card"><h1>Link expired</h1><p class="sub">Reset links work for '
                                      'one hour. <a href="/forgot">Ask for a new one</a>.</p></div>')
    resp = RedirectResponse("/account?m=" + quote("Password saved. You are logged in."), status_code=303)
    _login(resp, feed, s.email)
    return resp


@router.get("/account", response_class=HTMLResponse)
def account(request: Request, m: str = "", feed=Depends(_feed)):
    user = _user(request, feed)
    if not user:
        return RedirectResponse("/login", status_code=303)
    tok = _csrf(request.cookies.get(COOKIE, ""))
    on = user.status == "active"
    state = ('<span class="pill" style="color:var(--ok)">ON</span> You receive the email after each trading day.' if on else
             '<span class="pill" style="color:var(--muted)">OFF</span> ' + (
                 "Verify your email first: we sent you a link." if not user.verified else "You are not receiving emails."))
    toggle = "" if not user.verified else (f'<form method="post" action="/account"><input type="hidden" name="csrf" value="{tok}">'
              f'<input type="hidden" name="action" value="{"off" if on else "on"}"><button class="btn {"ghost" if on else ""}" '
              f'type="submit">{"Switch the daily email off" if on else "Switch the daily email on"}</button></form>')
    resend = ("" if user.verified else
              f'<form method="post" action="/account" style="margin-top:10px"><input type="hidden" name="csrf" value="{tok}">'
              f'<input type="hidden" name="action" value="resend"><button class="btn ghost small" type="submit">Resend the verification email</button></form>')
    body = f"""<div style="margin:30px 0 0"><h1 style="margin:0;font-size:28px;letter-spacing:-.01em">Hello{", " + _e(user.name) if user.name else ""}</h1>
<p class="sub">{_e(user.email)}</p>{_msg(m)}</div>
<div class="acct"><div class="card"><h3>Daily email</h3><p class="muted" style="margin:0 0 14px">{state}</p>{toggle}{resend}
<p style="margin:18px 0 0"><a class="btn small" href="/account/today">Read today's issue in the browser →</a></p></div>
<div class="card"><h3>Profile</h3><form method="post" action="/account"><input type="hidden" name="csrf" value="{tok}">
<input type="hidden" name="action" value="name"><label for="nm">Name</label><input id="nm" name="name" type="text" maxlength="80" value="{_e(user.name)}">
<div style="height:12px"></div><button class="btn ghost small" type="submit">Save name</button></form></div>
<div class="card"><h3>Change password</h3><form method="post" action="/account"><input type="hidden" name="csrf" value="{tok}">
<input type="hidden" name="action" value="password"><label for="cur">Current password</label><input id="cur" name="current" type="password" required autocomplete="current-password">
<label for="np">New password</label><input id="np" name="password" type="password" minlength="{MIN_PASSWORD}" required autocomplete="new-password">
<div style="height:12px"></div><button class="btn ghost small" type="submit">Change password</button></form></div>
<div class="card"><h3>Delete account</h3><p class="muted">Removes your account, your subscription and all data we hold
about you. This cannot be undone.</p><form method="post" action="/account" onsubmit="return confirm('Delete your account for good?')">
<input type="hidden" name="csrf" value="{tok}"><input type="hidden" name="action" value="delete">
<button class="btn danger small" type="submit" style="margin-top:12px">Delete my account</button></form></div></div>"""
    return layout("My account", body, user)


@router.post("/account", response_class=HTMLResponse)
async def account_action(request: Request, feed=Depends(_feed)):
    user = _user(request, feed)
    if not user:
        return RedirectResponse("/login", status_code=303)
    f = await _form(request)
    if f.get("csrf") != _csrf(request.cookies.get(COOKIE, "")):
        return RedirectResponse("/account?m=" + quote("The form expired; please try again."), status_code=303)
    action, msg = f.get("action"), ""
    if action in ("on", "off"):
        s = feed.store.set_daily(user.email, action == "on")
        msg = "Daily email switched on." if s.status == "active" else "Daily email switched off."
    elif action == "resend":
        msg = feed.subscribe(user.email, user.name, consent=True, source="website")["message"]
    elif action == "name":
        feed.store.update_name(user.email, f.get("name", ""))
        msg = "Name saved."
    elif action == "password":
        if not feed.store.authenticate(user.email, f.get("current", "")):
            msg = "Current password not recognised; nothing changed."
        else:
            try:
                feed.store.set_password(user.email, f.get("password", ""))  # also ends every session
                resp = RedirectResponse("/account?m=" + quote("Password changed."), status_code=303)
                _login(resp, feed, user.email)
                return resp
            except SubscriptionError as exc:
                msg = str(exc).capitalize() + "."
    elif action == "delete":
        feed.store.delete(user.email)
        resp = RedirectResponse("/?deleted=1", status_code=303)
        resp.delete_cookie(COOKIE, path="/")
        return resp
    return RedirectResponse("/account?m=" + quote(msg), status_code=303)


@router.get("/account/today", response_class=HTMLResponse)
def today(request: Request, feed=Depends(_feed)):
    user = _user(request, feed)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not _today or time.time() - _today["at"] > 300:  # rebuild at most every five minutes
        _today.update(at=time.time(), html=feed.preview_html())
    body = (f'<div style="margin:24px 0 14px" class="row"><h1 style="margin:0;font-size:24px">Today\'s issue</h1>'
            f'<div class="sp"></div><a class="btn ghost small" href="/account">← My account</a></div>'
            f'<iframe class="mail" title="Today\'s email" srcdoc="{_e(_today["html"])}"></iframe>')
    return layout("Today", body, user)
