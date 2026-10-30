"""Log-in for the shared dashboard: sign-up, log-in, forgot password, and the links in emails.

Email links point at the dashboard itself (FEED_LINKS=app, set by scripts/share.py):
    /?verify=<token>   /?reset=<token>   /?unsubscribe=<token>
The session token is kept in a browser cookie so a refresh does not log the person out.
"""
from __future__ import annotations

import html
import json

import streamlit as st
import streamlit.components.v1 as components

from .client import PricingAPIError

COOKIE = "opa_app_session"
TERMS = ("I understand this app shows market analytics for education and research. It is not investment advice, "
         "and the author accepts no responsibility for any loss or gain from using it.")


def _set_cookie(token: str, max_age: int = 14 * 86400) -> None:
    """Write (or with max_age 0, delete) the session cookie in the reader's browser."""
    value = json.dumps(token)
    with st.sidebar:  # not in the main area: an element appearing above the tabs would reset the open tab
        _cookie_script(value, max_age)


def _cookie_script(value: str, max_age: int) -> None:
    components.html("<script>const d = window.parent.document; const secure = window.parent.location.protocol === "
                    "'https:' ? '; Secure' : ''; d.cookie = '" + COOKIE + "=' + encodeURIComponent(" + value + ") + "
                    f"'; path=/; max-age={max_age}; SameSite=Lax' + secure;</script>", height=0)


def _start(token: str, user: dict, message: str = "") -> None:
    st.session_state.update(session=token, user=user, set_cookie=token, flash=message)


def logout(client) -> None:
    tok = st.session_state.get("session", "")
    try:
        client.auth("POST", "logout", tok)
    except PricingAPIError:
        pass
    for k in ("session", "user"):
        st.session_state.pop(k, None)
    st.session_state["clear_cookie"] = True
    st.session_state["logged_out"] = True


def _cookie_token() -> str:
    try:
        return st.context.cookies.get(COOKIE, "") or ""
    except AttributeError:
        return ""


def _links(client) -> None:
    """Handle ?verify=, ?reset=, ?unsubscribe= links from emails."""
    qp = st.query_params
    if "verify" in qp:
        try:
            r = client.auth("POST", "verify", token=qp["verify"])
            if r.get("token"):
                _start(r["token"], r["user"], r["message"])
            else:
                st.session_state["flash"] = r["message"]
        except PricingAPIError as exc:
            st.session_state["flash_error"] = exc.detail
        st.query_params.clear()
    elif "unsubscribe" in qp:
        try:
            st.session_state["flash"] = client.auth("POST", "unsubscribe", token=qp["unsubscribe"])["message"]
        except PricingAPIError as exc:
            st.session_state["flash_error"] = exc.detail
        st.query_params.clear()


def _shell(title: str, subtitle: str) -> None:
    st.markdown(f"""<div class="opa-hero" style="margin-top:8px"><div class="k">◆ Option Pricing Agent</div>
<div class="t">{html.escape(title)}</div><div class="s">{html.escape(subtitle)}</div>
<span class="opa-chip">Pricer & Greeks</span><span class="opa-chip">Vol smile</span><span class="opa-chip">Book risk & VaR</span>
<span class="opa-chip">Strategy lab</span><span class="opa-chip">Daily email</span></div>""", unsafe_allow_html=True)


def _flash(toast: bool = False) -> None:
    """Show pending messages; toast=True for the main app, where in-page alerts would shift the tabs."""
    ok, bad = st.session_state.pop("flash", None), st.session_state.pop("flash_error", None)
    if toast:
        if ok:
            st.toast(ok, icon="✅")
        if bad:
            st.toast(bad, icon="⚠️")
        return
    if ok:
        st.success(ok)
    if bad:
        st.error(bad)


def _reset_screen(client, token: str) -> None:
    _shell("Choose a new password", "The link from your email works for one hour.")
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid, st.form("reset"):
        p1 = st.text_input("New password", type="password")
        p2 = st.text_input("Repeat it", type="password")
        if st.form_submit_button("Save password", type="primary", use_container_width=True):
            if p1 != p2:
                st.error("The two passwords do not match.")
            else:
                try:
                    r = client.auth("POST", "reset", token=token, password=p1)
                    st.query_params.clear()
                    _start(r["token"], r["user"], r["message"])
                    st.rerun()
                except PricingAPIError as exc:
                    st.error(exc.detail)
    st.stop()


def _login_screen(client) -> None:
    _shell("NIFTY options, priced and checked", "Log in or create a free account to use the pricer, the smile, "
           "book risk, the strategy lab and the daily email.")
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        _flash()
        t_in, t_up, t_fp = st.tabs(["Log in", "Create account", "Forgot password"])
        with t_in, st.form("login"):
            email = st.text_input("Email", key="li_email")
            pw = st.text_input("Password", type="password", key="li_pw")
            if st.form_submit_button("Log in", type="primary", use_container_width=True):
                try:
                    r = client.auth("POST", "login", email=email, password=pw)
                    _start(r["token"], r["user"])
                    st.rerun()
                except PricingAPIError as exc:
                    st.error(exc.detail)
        with t_up, st.form("signup"):
            name = st.text_input("Name", max_chars=80)
            email = st.text_input("Email (your username)", max_chars=254)
            p1 = st.text_input("Password (at least 8 characters)", type="password")
            p2 = st.text_input("Repeat password", type="password")
            daily = st.checkbox("Also send me the daily NIFTY options email", value=True)
            accept = st.checkbox(TERMS)
            if st.form_submit_button("Create account", type="primary", use_container_width=True):
                if p1 != p2:
                    st.error("The two passwords do not match.")
                else:
                    try:
                        r = client.auth("POST", "signup", email=email, name=name, password=p1, accept=accept,
                                        daily=daily)
                        _start(r["token"], r["user"], r["message"])
                        st.rerun()
                    except PricingAPIError as exc:
                        st.error(exc.detail)
        with t_fp, st.form("forgot"):
            email = st.text_input("Email", key="fp_email")
            if st.form_submit_button("Email me a reset link", use_container_width=True):
                try:
                    st.success(client.auth("POST", "forgot", email=email)["message"])
                except PricingAPIError as exc:
                    st.error(exc.detail)
        st.caption("Market analytics only, not investment advice.")
    st.stop()


def _verify_screen(client, user: dict) -> None:
    _shell("Check your inbox", f"We sent a verification link to {user['email']}. Click it to start using the app.")
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        _flash()
        c1, c2 = st.columns(2)
        if c1.button("Resend the link", use_container_width=True):
            try:
                st.success(client.auth("POST", "resend", st.session_state["session"])["message"])
            except PricingAPIError as exc:
                st.error(exc.detail)
        if c2.button("Log out", use_container_width=True):
            logout(client)
            st.rerun()
    st.stop()


def gate(client) -> dict:
    """Return the logged-in user, or show the log-in screens and stop the page."""
    if st.session_state.pop("set_cookie", None):
        _set_cookie(st.session_state["session"])
    if st.session_state.pop("clear_cookie", None):
        _set_cookie("", max_age=0)
    _links(client)
    if "reset" in st.query_params:
        _reset_screen(client, st.query_params["reset"])
    token = st.session_state.get("session") or ("" if st.session_state.get("logged_out") else _cookie_token())
    if token:
        try:
            user = client.auth("GET", "me", token)
            st.session_state.update(session=token, user=user)
        except PricingAPIError:
            st.session_state.pop("session", None)
            user = None
        if user:
            if user.get("verification_required") and not user.get("verified"):
                _verify_screen(client, user)
            return user
    _login_screen(client)
    return {}  # not reached


def account_tab(client, user: dict) -> None:
    """My account: daily email on/off, name, password, delete."""
    tok = st.session_state["session"]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"**Signed in as** {html.escape(user['email'])}")
        on = st.toggle("Daily NIFTY options email", value=bool(user.get("daily")), disabled=not user.get("verified"),
                       help=None if user.get("verified") else "Verify your email first")
        if on != bool(user.get("daily")):
            try:
                st.session_state["user"] = client.auth("POST", "daily", tok, on=on)
                st.rerun()
            except PricingAPIError as exc:
                st.error(exc.detail)
        if not user.get("verified"):
            st.caption("Your email is not verified yet.")
            if st.button("Resend the verification link"):
                st.info(client.auth("POST", "resend", tok)["message"])
        with st.form("acct_name"):
            nm = st.text_input("Name", value=user.get("name") or "", max_chars=80)
            if st.form_submit_button("Save name"):
                st.session_state["user"] = client.auth("POST", "name", tok, name=nm)
                st.toast("Name saved.", icon="✅")
    with c2:
        with st.form("acct_pw"):
            cur = st.text_input("Current password", type="password")
            new = st.text_input("New password (at least 8 characters)", type="password")
            if st.form_submit_button("Change password"):
                try:
                    r = client.auth("POST", "password", tok, current=cur, new=new)
                    st.session_state.update(session=r["token"], set_cookie=r["token"], flash=r["message"])
                    st.rerun()
                except PricingAPIError as exc:
                    st.error(exc.detail)
        with st.expander("Delete my account"):
            st.write("Removes your account, your email subscription and all data held about you.")
            sure = st.checkbox("Yes, delete my account for good")
            if st.button("Delete account", type="primary", disabled=not sure):
                client.auth("DELETE", "account", tok)
                logout(client)
                st.rerun()
