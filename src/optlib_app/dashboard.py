"""Streamlit dashboard for the quants team. Talks to the FastAPI backend over HTTP.

Run (two terminals):
    uvicorn optlib_app.api:app
    streamlit run src/optlib_app/dashboard.py
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from optlib_app import charts, dash_auth, ui
from optlib_app.config import get_setting
from optlib_app.client import PricingAPIError, PricingClient

st.set_page_config(page_title="Option Pricing Agent", page_icon="◆", layout="wide")


def _load_hosted_secrets() -> None:
    """On Streamlit Community Cloud settings arrive as st.secrets; copy the flat ones into the environment so the
    pricing service, Upstox client, mailer and D1 store read them like a local .env."""
    import os

    try:
        for k, v in st.secrets.items():
            if isinstance(v, (str, int, float, bool)):
                os.environ.setdefault(k, str(v))
    except Exception:  # noqa: BLE001 - no secrets file locally: .env is used instead
        pass


_load_hosted_secrets()
THEME = ui.current_theme()
DARK = THEME == "dark"
ui.inject_css(THEME)

CALL, PUT, GREY = ("#818cf8", "#fb923c", "#94a3b8") if DARK else ("#4f46e5", "#ea580c", "#6b7280")
ACCENT = "#a78bfa" if DARK else "#7c3aed"
STATUS_COLOR = ui.STATUS
DEFAULT_BOOK = pd.DataFrame([
    {"expiry": "2026-10-27", "strike": 22500.0, "option_type": "call", "lots": -2.0},
    {"expiry": "2026-10-27", "strike": 22500.0, "option_type": "put", "lots": -2.0},
    {"expiry": "2026-10-27", "strike": 22000.0, "option_type": "put", "lots": 2.0},
])


@st.cache_resource
def api(url: str) -> PricingClient:
    """The pricing service. If none is running (a hosted single-process deploy such as Streamlit Community Cloud),
    start it inside this process on 127.0.0.1; EMBEDDED_API=off disables that."""
    client = PricingClient(url)
    try:
        client.health()
    except Exception:  # noqa: BLE001 - not running: start our own
        if get_setting("EMBEDDED_API", "auto").lower() != "off":
            _start_embedded_api(url, client)
    return client


def _start_embedded_api(url: str, client: PricingClient) -> None:
    import threading
    import time
    from urllib.parse import urlparse

    import uvicorn

    from optlib_app.api import app as api_app

    port = urlparse(url).port or 8000
    server = uvicorn.Server(uvicorn.Config(api_app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, name="embedded-api", daemon=True).start()
    for _ in range(120):  # up to a minute: the first build of the surface fetches live data
        try:
            client.health()
            return
        except Exception:  # noqa: BLE001
            time.sleep(0.5)


def show(fig: go.Figure, where=st, key: str | None = None) -> None:
    """Every chart goes through here so it follows the light/dark theme."""
    where.plotly_chart(charts.themed(fig, DARK), width="stretch", theme=None, key=key)


def badge(status: str) -> str:
    return f":{'green' if status == 'PASS' else 'orange' if status == 'WARN' else 'red'}[**{status}**]"


def show_error(exc: Exception):
    st.error(f"{exc.detail if isinstance(exc, PricingAPIError) else exc}")


def validation_table(report: dict):
    df = pd.DataFrame(report["checks"])[["status", "name", "detail"]]
    st.dataframe(df.style.map(lambda s: f"color: {STATUS_COLOR.get(s, 'inherit')}; font-weight: 600",
                              subset=["status"]), hide_index=True, width="stretch")


# ------------------------------------------------------------------ log-in and sidebar
client = api(PricingClient().base_url)  # fixed on the server: readers cannot point the app elsewhere
try:
    h = client.health()
except Exception as exc:  # noqa: BLE001
    st.error(f"The pricing service is not running. Start it with `python -m uvicorn optlib_app.api:app`.\n\n{exc}")
    st.stop()
USER = dash_auth.gate(client)
IS_ADMIN = bool(USER.get("admin"))
AGENT_FOR_ALL = get_setting("AGENT_FOR_ALL", "off").lower() in ("on", "true", "1", "yes")

with st.sidebar:
    st.markdown('<div class="opa-brand">◆ Option Pricing <span>Agent</span></div>'
                '<div class="opa-muted">NSE index options · pricing, risk, strategies</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(f"**{USER.get('name') or USER['email']}**" + (" · admin" if IS_ADMIN else ""))
    if st.button("Log out", width="stretch"):
        dash_auth.logout(client)
        st.rerun()
    ui.theme_switch(st)
    st.success(f"Live service · {h['data_mode']} data")
    underlyings = h["underlyings"]
    llm = h.get("llm") or {}
    if IS_ADMIN:
        st.caption(f"Agent LLM: {llm['provider']} · {llm['model']}" if llm.get("configured")
                   else "Agent LLM: not configured")
    underlying = st.selectbox("Underlying", underlyings, index=0)
    if st.button("↻ Refresh data", width="stretch", help="Fetches the latest quotes (at most once a minute)"):
        st.rerun()
    if h.get("data_mode") == "live":
        from datetime import datetime
        from zoneinfo import ZoneInfo

        _now = datetime.now(ZoneInfo("Asia/Kolkata"))
        _open = _now.weekday() < 5 and (9, 15) <= (_now.hour, _now.minute) < (15, 30)
        st.caption("Market open: quotes update every minute; press Refresh to see them." if _open else
                   "Market closed: prices stay at the last quotes until 09:15 IST on the next trading day.")
    st.divider()
    st.caption("Market analytics for education and research. Not investment advice.")

try:
    ov = client.overview(underlying)
except PricingAPIError as exc:
    st.warning(f"{underlying}: {exc.detail}")
    st.stop()

expiry_rows = ov["expiries"]
expiry_ids = [e["expiry"] for e in expiry_rows]
lot = ov["lot_size"]

as_of_txt = pd.Timestamp(ov.get("as_of") or h["as_of"]).strftime("%a %d %b %Y, %H:%M IST")
e_near = expiry_rows[0]
ui.hero(f"{underlying} options desk", f"{ov['spot']:,.2f}",
        f"Market data as of {as_of_txt} · {h['data_mode']} data · lot size {lot}",
        [f"Surface {ov['surface_status']}", f"ATM vol {e_near['atm_vol']:.2%} ({e_near['expiry']})",
         f"{len(expiry_rows)} expiries fitted", f"Curve {ov['curve']['label'].replace('MIBOR-OIS ', '')} · "
         f"{ov['curve']['age_days']} days old"])

dash_auth._flash(toast=True)  # messages from email links and account changes, as pop-ups
(tab_over, tab_market, tab_oi, tab_price, tab_book, tab_risk, tab_strat, tab_news, tab_agent, tab_feed,
 tab_acct) = st.tabs(["Overview", "Market & smile", "Open interest", "Pricer", "Book risk", "Risk (VaR, stress, P&L)",
                      "Strategy lab", "News", "Ask the agent", "Daily email", "My account"])


@st.cache_data(ttl=300, show_spinner=False)
def _news(limit: int = 20):
    return client.news(limit)


def oi_chart(o: dict, title: str) -> go.Figure:
    """Calls to the right, puts to the left of each strike (lakh units), with the forward and max pain marked."""
    rows = pd.DataFrame(o["rows"])
    fig = go.Figure()
    fig.add_trace(go.Bar(y=rows.strike, x=rows.call_oi / 1e5, orientation="h", name="call OI", marker_color=CALL,
                         hovertemplate="%{y:,.0f} call: %{x:,.1f} L<extra></extra>"))
    fig.add_trace(go.Bar(y=rows.strike, x=-rows.put_oi / 1e5, orientation="h", name="put OI", marker_color=PUT,
                         customdata=rows.put_oi / 1e5, hovertemplate="%{y:,.0f} put: %{customdata:,.1f} L<extra></extra>"))
    fig.add_hline(y=o["forward"], line=dict(color=GREY, dash="dash"), annotation_text=f"forward {o['forward']:,.0f}",
                  annotation_font_color=GREY, annotation_position="top right")
    if o.get("max_pain"):
        fig.add_hline(y=o["max_pain"], line=dict(color=ACCENT, dash="dot"),
                      annotation_text=f"max pain {o['max_pain']:,.0f}", annotation_font_color=ACCENT,
                      annotation_position="bottom left")
    fig.update_layout(title=title, barmode="relative", bargap=0.15, height=520, xaxis_title="open interest (lakh units)"
                      " · puts left, calls right", yaxis_title="strike", margin=dict(t=60, b=40),
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.08))
    fig.update_xaxes(tickvals=None, tickformat=",.0f")
    return fig


# ------------------------------------------------------------------ overview
with tab_over:
    try:
        ivs = client.iv_stats(underlying)
    except PricingAPIError as exc:
        ivs = {"available": False, "reason": exc.detail}
    try:
        oi0 = client.open_interest(underlying)
    except PricingAPIError:
        oi0 = None
    e_w = next((e for e in expiry_rows if e["days"] >= 0.5), expiry_rows[0])
    move = e_w["forward"] * e_w["atm_vol"] * (max(e_w["days"], 0.01) / 365) ** 0.5
    k = st.columns(6)
    k[0].metric(f"{underlying} spot", f"{ov['spot']:,.2f}")
    if ivs.get("available"):
        k[1].metric("India VIX", f"{ivs['vix'] * 100:.2f}",
                    f"{ivs['change_1d'] * 100:+.2f}" if ivs.get("change_1d") is not None else None,
                    delta_color="inverse")
        k[2].metric("VIX rank (1y)", f"{ivs['rank']:.0%}", ivs["label"], delta_color="off", delta_arrow="off",
                    help=f"Where today's VIX sits in its 1-year range ({ivs['low']:.2%} to {ivs['high']:.2%}); "
                         f"percentile {ivs['percentile']:.0%} of days were lower.")
    else:
        k[1].metric("India VIX", "n/a")
        k[2].metric("VIX rank (1y)", "n/a")
    k[3].metric(f"ATM IV · {e_w['expiry'][5:]}", f"{e_w['atm_vol']:.2%}", f"{e_w['days']:.1f} days",
                delta_color="off", delta_arrow="off")
    k[4].metric("Expected move", f"±{move:,.0f}", f"±{move / e_w['forward']:.2%} by expiry", delta_color="off",
                delta_arrow="off")
    if oi0:
        k[5].metric("Put/call OI", f"{oi0['pcr']:.2f}" if oi0.get("pcr") else "n/a",
                    f"max pain {oi0['max_pain']:,.0f}" if oi0.get("max_pain") else None, delta_color="off",
                    delta_arrow="off")
    else:
        k[5].metric("Put/call OI", "n/a")

    left, right = st.columns([3, 2], gap="large")
    with left:
        if ivs.get("available"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ivs["dates"], y=[v * 100 for v in ivs["series"]], name="India VIX",
                                     line=dict(color=CALL, width=2), fill="tozeroy",
                                     fillcolor="rgba(129,140,248,0.10)" if DARK else "rgba(79,70,229,0.07)"))
            fig.add_hline(y=ivs["mean"] * 100, line=dict(color=GREY, dash="dot"),
                          annotation_text=f"1-year mean {ivs['mean'] * 100:.1f}", annotation_font_color=GREY)
            fig.add_trace(go.Scatter(x=[ivs["dates"][-1]], y=[e_w["atm_vol"] * 100], mode="markers",
                                     name=f"weekly ATM IV {e_w['atm_vol']:.1%}",
                                     marker=dict(color=PUT, size=11, symbol="diamond")))
            fig.update_layout(title="India VIX, last 12 months, with today's weekly ATM implied vol", height=360,
                              yaxis_title="vol (%)", margin=dict(t=60, b=30),
                              legend=dict(orientation="h", x=1, xanchor="right", y=1.1))
            lo_y = min(min(ivs["series"]), e_w["atm_vol"]) * 100
            fig.update_yaxes(range=[lo_y * 0.85, max(max(ivs["series"]), e_w["atm_vol"]) * 100 * 1.05])
            show(fig)
        else:
            st.info(f"India VIX history unavailable: {ivs.get('reason', '')}")
        if oi0 and oi0.get("rows"):
            show(oi_chart(oi0, f"Open interest by strike · {oi0['expiry']}"), key="oi_overview")
    with right:
        note_box = st.empty()
        st.markdown("")
        st.markdown('<div class="opa-kicker">Market news</div>', unsafe_allow_html=True)
        try:
            from optlib_app.news import ago

            items = _news(20)
            if items:
                ui.news_list(items[:7], ago)
                st.caption("Top headlines from Economic Times, Mint, Business Standard and Google News, ranked for "
                           "relevance to index options. More in the **News** tab.")
            else:
                st.caption("No headlines right now.")
        except PricingAPIError as exc:
            st.caption(f"News unavailable: {exc.detail}")
        with note_box.container():
            try:
                with st.spinner("Writing today's market note…"):
                    nt = client.market_note(underlying)
            except PricingAPIError as exc:
                nt = {"note": None, "error": exc.detail}
            if nt.get("note"):
                ui.note_card(nt["note"]["text"], f"Written by AI ({nt['note'].get('model') or 'model'}) from today's "
                             "numbers; every number checked against the data. Description, not advice.")
            elif IS_ADMIN:
                st.info("AI market note: add GEMINI_API_KEY (from aistudio.google.com) to the app's secrets to turn "
                        "it on." + (f" ({nt['error']})" if nt.get("error") else ""))
        st.write("")
        if st.button("Prepare today's report (PDF)", width="stretch"):
            try:
                with st.spinner("Building the PDF…"):
                    st.session_state["daily_pdf"] = client.daily_pdf(underlying)
            except PricingAPIError as exc:
                show_error(exc)
        if st.session_state.get("daily_pdf"):
            st.download_button("⬇ Download today's report", st.session_state["daily_pdf"],
                               file_name=f"{underlying}-options-daily.pdf", mime="application/pdf", width="stretch",
                               type="primary")
    st.caption("Market analytics for education and research. Not investment advice.")

# ------------------------------------------------------------------ open interest
with tab_oi:
    o1, o2 = st.columns([1, 3])
    oi_exp = o1.selectbox("Expiry", [e for e in expiry_ids], key="oi_exp",
                          index=next((i for i, e in enumerate(expiry_rows) if e["days"] >= 0.5), 0))
    try:
        oo = client.open_interest(underlying, oi_exp)
    except PricingAPIError as exc:
        show_error(exc)
        oo = None
    if oo:
        m = st.columns(5)
        m[0].metric("Put/call ratio (OI)", f"{oo['pcr']:.2f}" if oo.get("pcr") else "n/a",
                    f"{oo['pcr'] - oo['pcr_prev']:+.2f} vs previous" if oo.get("pcr") and oo.get("pcr_prev") else None,
                    delta_color="off")
        m[1].metric("Call OI", f"{oo['call_oi'] / 1e7:,.2f} cr",
                    f"{oo['call_change'] / 1e7:+,.2f} cr" if oo.get("call_change") is not None else None,
                    delta_color="off")
        m[2].metric("Put OI", f"{oo['put_oi'] / 1e7:,.2f} cr",
                    f"{oo['put_change'] / 1e7:+,.2f} cr" if oo.get("put_change") is not None else None,
                    delta_color="off")
        m[3].metric("Max pain", f"{oo['max_pain']:,.0f}" if oo.get("max_pain") else "n/a",
                    f"{oo['max_pain'] - oo['forward']:+,.0f} vs forward" if oo.get("max_pain") else None,
                    delta_color="off", delta_arrow="off",
                    help="The settlement level at which option buyers would collect the least in total.")
        m[4].metric("1-sd move", f"±{oo['expected_move']:,.0f}", "strikes shown: ±4 sd", delta_color="off",
                    delta_arrow="off")
        a, b = st.columns([3, 2], gap="large")
        with a:
            show(oi_chart(oo, f"Open interest by strike · {oo['expiry']}"), key="oi_tab")
        with b:
            rows = pd.DataFrame(oo["rows"])
            if rows[["call_change", "put_change"]].notna().any().any():
                ch = go.Figure()
                ch.add_trace(go.Bar(x=rows.strike, y=rows.call_change.fillna(0) / 1e5, name="call OI change",
                                    marker_color=CALL))
                ch.add_trace(go.Bar(x=rows.strike, y=rows.put_change.fillna(0) / 1e5, name="put OI change",
                                    marker_color=PUT))
                ch.update_layout(title="Change in open interest on the day", height=300, barmode="group",
                                 yaxis_title="lakh units", margin=dict(t=60, b=30),
                                 legend=dict(orientation="h", x=1, xanchor="right", y=1.12))
                show(ch)
            top = pd.DataFrame({"call strike": [x["strike"] for x in oo["top_calls"]],
                                "call OI (L)": [x["oi"] / 1e5 for x in oo["top_calls"]],
                                "put strike": [x["strike"] for x in oo["top_puts"]],
                                "put OI (L)": [x["oi"] / 1e5 for x in oo["top_puts"]]})
            st.markdown("**Largest open interest**")
            st.dataframe(top.style.format({"call strike": "{:,.0f}", "put strike": "{:,.0f}", "call OI (L)": "{:,.1f}",
                                           "put OI (L)": "{:,.1f}"}), hide_index=True, width="stretch")
        st.caption("Open interest shows where positions are held, in lakh units (1 L = 100,000). It is not a level the "
                   "index must respect. Change is against the previous session's close.")

# ------------------------------------------------------------------ news
with tab_news:
    st.markdown("#### Market news")
    st.caption("Headlines from the publishers' public feeds, ranked for relevance to index options (NIFTY, Sensex, "
               "RBI, FII flows, volatility, F&O) and recency. Links open the publisher's site. Refreshed every 10 "
               "minutes.")
    try:
        from optlib_app.news import ago

        items = _news(30)
    except PricingAPIError as exc:
        items = []
        show_error(exc)
    if items:
        q = st.text_input("Filter", placeholder="e.g. RBI, Bank Nifty, crude", key="news_q")
        shown = [n for n in items if not q or q.lower() in n["title"].lower()]
        c1, c2 = st.columns(2, gap="large")
        with c1:
            ui.news_list(shown[::2], ago)
        with c2:
            ui.news_list(shown[1::2], ago)
        if not shown:
            st.caption("No headline matches the filter.")
    else:
        st.info("No headlines available right now.")

# ------------------------------------------------------------------ market
with tab_market:
    c = st.columns(5)
    c[0].metric("Spot", f"{ov['spot']:,.2f}")
    c[1].metric("Lot size", lot)
    c[2].metric("Discount curve", ov["curve"]["label"].replace("MIBOR-OIS ", ""),
                f"{ov['curve']['age_days']} days old", delta_color="off", delta_arrow="off")
    c[3].metric("Expiries fitted", len(expiry_rows))
    c[4].metric("Surface status", ov["surface_status"],
                f"{sum(e['svi_status'] == 'PASS' for e in expiry_rows)} of {len(expiry_rows)} smiles PASS",
                delta_color="off", delta_arrow="off")

    table = pd.DataFrame([{
        "expiry": e["expiry"], "days": e["days"], "forward": e["forward"], "rate": e["rate"], "ATM vol": e["atm_vol"],
        "fwd vs future": (f"{e['forward_check']['gap']:+.3%} {e['forward_check']['verdict']}"
                          if e["forward_check"] else "no future"),
        "QA": e["qa_status"], "SVI RMSE (vol pts)": e["svi_rmse_vol_pts"], "SVI ATM gap": e["svi_atm_gap_vol_pts"],
        "SVI": e["svi_status"]} for e in expiry_rows])
    st.dataframe(table.style.format({"days": "{:.2f}", "forward": "{:,.2f}", "rate": "{:.3%}", "ATM vol": "{:.2%}",
                                     "SVI RMSE (vol pts)": "{:.3f}", "SVI ATM gap": "{:+.3f}"})
                 .map(lambda s: f"color: {STATUS_COLOR.get(s, 'inherit')}; font-weight: 600", subset=["QA", "SVI"]),
                 hide_index=True, width="stretch")

    c1, c2 = st.columns([1, 2])
    exp = c1.selectbox("Expiry", expiry_ids, key="smile_exp")
    width = c2.slider("Strikes shown (± points from forward)", 500, 5000, 1800, 100)
    sm = client.smile(exp, underlying)
    q = pd.DataFrame(sm["quotes"])
    q = q[(q.strike - sm["forward"]).abs() <= width]
    curve = pd.DataFrame(sm["svi_curve"])
    curve = curve[(curve.strike - sm["forward"]).abs() <= width]
    band = "rgba(129,140,248,0.22)" if DARK else "rgba(79,70,229,0.15)"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q.strike, y=q.iv_ask * 100, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=q.strike, y=q.iv_bid * 100, fill="tonexty", line=dict(width=0),
                             fillcolor=band, name="bid-ask"))
    fig.add_trace(go.Scatter(x=q.strike, y=q.iv_mid * 100, mode="markers", marker=dict(color=CALL, size=5),
                             name="market mid", customdata=q[["type", "bid", "ask"]],
                             hovertemplate="%{x:,.0f} %{customdata[0]}<br>vol %{y:.2f}%<br>"
                                           "bid %{customdata[1]:.2f} / ask %{customdata[2]:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=curve.strike, y=curve.vol * 100, line=dict(color=ACCENT, width=2.5), name="SVI fit"))
    fig.add_vline(x=sm["forward"], line=dict(color=GREY, dash="dash"),
                  annotation_text=f"forward {sm['forward']:,.0f}", annotation_font_color=GREY)
    lo_k, hi_k = sm["fit_range"]
    fig.add_vrect(x0=lo_k, x1=hi_k, fillcolor=ACCENT, opacity=0.06, line_width=0,
                  annotation_text="SVI fit range (±3 expected moves)", annotation_position="bottom left",
                  annotation_font_color=GREY)
    fig.update_layout(title=f"{underlying} smile · {exp} · ATM {sm['atm_vol']:.2%}", xaxis_title="strike",
                      yaxis_title="implied vol (%)", height=460, margin=dict(t=70, b=40),
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.08))
    show(fig)
    with st.expander("Chain QA"):
        st.code(sm["qa_summary"], language=None)
        if sm["dropped"]:
            st.dataframe(pd.DataFrame(sm["dropped"]), hide_index=True)

# ------------------------------------------------------------------ pricer
with tab_price:
    with st.form("price_form"):
        c = st.columns(4)
        p_exp = c[0].selectbox("Expiry", expiry_ids)
        atm = round(expiry_rows[expiry_ids.index(p_exp)]["forward"] / 50) * 50
        p_strike = c[1].number_input("Strike", value=float(atm), step=50.0)
        p_type = c[2].radio("Type", ["call", "put"], horizontal=True)
        p_lots = c[3].number_input("Lots (negative = short)", value=1.0, step=1.0)
        go_price = st.form_submit_button("Price and validate", type="primary")
    if go_price:
        try:
            out = client.price(p_exp, p_strike, p_type, p_lots, underlying)
            r, v = out["result"], out["validation"]
            if not v["released"]:
                st.error("Result withheld: a validation check failed. See the report below.")
            c = st.columns(5)
            c[0].metric("Price per unit", f"{r['price']:,.2f}")
            c[1].metric("Position value", f"₹{r['value']:,.0f}", f"{p_lots:g} lots × {lot}", delta_color="off", delta_arrow="off")
            c[2].metric("Implied vol", f"{r['vol']:.2%}", r["vol_source"], delta_color="off", delta_arrow="off")
            c[3].metric("Forward", f"{r['forward']:,.2f}", f"rate {r['rate']:.3%}", delta_color="off", delta_arrow="off")
            c[4].metric("Validation", v["status"], "released" if v["released"] else "withheld", delta_color="off", delta_arrow="off")
            g = pd.DataFrame({"per unit": r["greeks"], "position": r["position_greeks"]})
            g.index.name = "Greek"
            left, right = st.columns([1, 2])
            left.dataframe(g.style.format(lambda x: f"{x:,.2f}" if abs(x) >= 1 else f"{x:.6f}"),
                           width="stretch")
            left.caption("vega per vol point · theta per calendar day · rho per bp · position = per unit × quantity")
            with right:
                validation_table(v)
        except PricingAPIError as exc:
            show_error(exc)

# ------------------------------------------------------------------ book
with tab_book:
    st.caption("Edit the book (lots: positive long, negative short), then run. Default: short 22,500 straddle "
               "with a 22,000 put hedge.")
    book = st.data_editor(DEFAULT_BOOK.assign(expiry=expiry_ids[0]), num_rows="dynamic", width="stretch",
                          column_config={"option_type": st.column_config.SelectboxColumn(options=["call", "put"]),
                                         "expiry": st.column_config.SelectboxColumn(options=expiry_ids)})
    if st.button("Run book risk", type="primary"):
        positions = [dict(underlying=underlying, **row) for row in book.dropna().to_dict("records")]
        try:
            b = client.book(positions)
            t = b["totals"]
            c = st.columns(6)
            c[0].metric("Value", f"₹{t['value']:,.0f}")
            c[1].metric("Delta (units)", f"{t['delta']:,.1f}")
            c[2].metric("Gamma", f"{t['gamma']:,.4f}")
            c[3].metric("Vega / vol pt", f"₹{t['vega']:,.0f}")
            c[4].metric("Theta / day", f"₹{t['theta']:,.0f}")
            c[5].metric("Validation", b["status"])
            st.dataframe(pd.DataFrame(b["positions"]).style.format(
                {"lots": "{:g}", "quantity": "{:,.0f}", "price": "{:,.2f}", "vol": "{:.2%}", "value": "{:,.0f}",
                 "delta": "{:,.2f}", "gamma": "{:.4f}",
                 "vega": "{:,.0f}", "theta": "{:,.0f}", "rho": "{:,.2f}", "strike": "{:,.0f}"}),
                hide_index=True, width="stretch")
            sc = client.scenarios(positions)
            z = pd.DataFrame(sc["pnl"], index=[f"{v * 100:+.0f} vol" for v in sc["vol_shocks"]],
                             columns=[f"{s * 100:+.0f}%" for s in sc["spot_shocks"]])
            heat = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=z.index, colorscale="RdYlGn", zmid=0,
                                        text=[[f"{v / 1000:,.1f}k" for v in row] for row in z.values],
                                        texttemplate="%{text}", hovertemplate="spot %{x}, %{y}: ₹%{z:,.0f}<extra></extra>"))
            heat.update_layout(title="P&L by spot move and vol shift (₹, full revaluation, sticky strike)",
                               xaxis_title="spot move", yaxis_title="vol shift (points)", height=420,
                               margin=dict(t=50, b=40))
            heat.update_xaxes(type="category")
            heat.update_yaxes(type="category")
            show(heat)
        except PricingAPIError as exc:
            show_error(exc)

# ------------------------------------------------------------------ risk (phase 2)
with tab_risk:
    st.caption("Uses the book from the **Book risk** tab. VaR and ES by historical simulation with full revaluation; "
               "stress tests replay real market days and hypothetical shocks; P&L explain uses NSE end-of-day prices.")
    positions = [dict(underlying=underlying, **row) for row in book.dropna().to_dict("records")]
    c = st.columns([1, 1, 2])
    window = c[0].select_slider("VaR history (days)", options=[250, 500, 750, 1000, 1500], value=500)
    run = c[1].button("Run risk", type="primary")
    if run:
        st.session_state["risk"] = None
        try:
            st.session_state["risk"] = client.risk(positions, window)
        except PricingAPIError as exc:
            show_error(exc)
    r = st.session_state.get("risk")
    if r:
        v, bt = r["var"], r["backtest"]
        m = st.columns(6)
        m[0].metric("1-day VaR 99%", f"₹{v['var']:,.0f}")
        m[1].metric("ES 97.5%", f"₹{v['es']:,.0f}")
        m[2].metric("Worst day in window", f"₹{v['worst']['pnl']:,.0f}", v["worst"]["date"], delta_color="off", delta_arrow="off")
        m[3].metric("Worst stress", f"₹{r['stress'][0]['pnl']:,.0f}", r["stress"][0]["scenario"][:28],
                    delta_color="off", delta_arrow="off")
        m[4].metric("Backtest exceptions", f"{bt.get('exceptions', '-')} / {bt.get('days', '-')}",
                    f"traffic light {bt.get('traffic_light', 'n/a')}", delta_color="off", delta_arrow="off")
        m[5].metric("Risk status", r["status"])
        st.caption(f"History through {r['history_last_date']} (no look-ahead into the pricing day) · "
                   f"window {v['first_date']} to {v['last_date']}")

        sub = st.tabs(["Limits & Greeks", "VaR & ES", "Backtest", "Stress tests", "P&L explain", "Report"])
        with sub[0]:
            show(charts.limits_bar(r["limits"]))
            st.caption("Limits are placeholders in config/risk_limits.json; the risk team sets them.")
            st.dataframe(pd.DataFrame(r["by_expiry"]).style.format(
                {"value": "{:,.0f}", "delta": "{:,.1f}", "gamma": "{:.4f}", "vega": "{:,.0f}", "theta": "{:,.0f}"}),
                hide_index=True, width="stretch")
        with sub[1]:
            left, right = st.columns([3, 2])
            show(charts.var_histogram(v), left)
            show(charts.contributions_bar(v), right)
            right.caption("Average P&L of each position on the worst 2.5% of days; red adds to the loss, "
                          "green offsets it. The bars add up to ES.")
            st.caption(v["method"])
        with sub[2]:
            if "exceptions" in bt:
                show(charts.backtest_chart(bt))
                st.markdown(f"Kupiec (frequency) p = **{bt['kupiec_p']:.2f}** · Christoffersen (clustering) "
                            f"p = **{bt['christoffersen_p']:.2f}** · values below 0.05 reject the model. "
                            f"_{bt['note']}._")
            else:
                st.info(bt.get("error", "backtest unavailable"))
        with sub[3]:
            show(charts.stress_bar(r["stress"]))
            st.dataframe(pd.DataFrame([{k: x[k] for k in ("scenario", "kind", "index_move", "vol_mult", "pnl")}
                                       for x in r["stress"]]).style.format(
                {"index_move": "{:+.2%}", "vol_mult": "x{:.2f}", "pnl": "{:,.0f}"}), hide_index=True,
                width="stretch")
        with sub[4]:
            days = client.eod_days()
            if days:
                cc = st.columns(3)
                d0 = cc[0].selectbox("From close of", days, index=0)
                d1 = cc[1].selectbox("To close of", days, index=len(days) - 1)
            else:
                cc = st.columns(3)
                d0 = cc[0].date_input("From close of").isoformat()
                d1 = cc[1].date_input("To close of").isoformat()
            if cc[2].button("Explain P&L"):
                try:
                    e = client.pnl_explain(positions, d0, d1)
                    show(charts.pnl_waterfall(e))
                    st.dataframe(pd.DataFrame(e["legs"])[["label", "close0", "close1", "vol0", "vol1", "actual", "delta",
                                                          "gamma", "vega", "theta", "cross", "unexplained"]].style.format(
                        {"close0": "{:,.2f}", "close1": "{:,.2f}", "vol0": "{:.2%}", "vol1": "{:.2%}",
                         **{k: "{:,.0f}" for k in ("actual", "delta", "gamma", "vega", "theta", "cross",
                                                   "unexplained")}}), hide_index=True, width="stretch")
                    for n in e["notes"]:
                        st.caption(n)
                except PricingAPIError as exc:
                    show_error(exc)
        with sub[5]:
            st.write("A self-contained HTML report of everything above (charts load Plotly from its CDN).")
            days = client.eod_days()
            if st.button("Build report"):
                try:
                    html_doc = client.risk_report(positions, days[0] if len(days) > 1 else None,
                                                  days[-1] if len(days) > 1 else None)
                    st.download_button("Download risk report (HTML)", html_doc, file_name="risk_report.html",
                                       mime="text/html")
                except PricingAPIError as exc:
                    show_error(exc)
    else:
        st.info("Press **Run risk** to compute VaR, ES, the backtest, stress tests and limits for the book.")


# ------------------------------------------------------------------ strategies (phase 3)
VIEW_LABELS = {"blend": "Blend: my view + model", "user": "My view", "model": "Model (realised vol)",
               "market": "Market-implied"}


def max_profit_text(i: dict, rupee: str = "") -> str:
    """Unlimited, or the cap; when the cap needs NIFTY near zero, show the profit within a 3-sd move with *."""
    if i["max_profit"] is None:
        return "unlimited"
    p3 = i.get("max_profit_3sd")
    if p3 is not None and i["max_profit"] > 1.5 * p3 + 100:
        return f"{rupee}{p3:,.0f}*"
    return f"{rupee}{i['max_profit']:,.0f}"


def idea_rows(ideas: list, source: str, with_reasons: bool = False) -> pd.DataFrame:
    out = []
    for n, i in enumerate(ideas, 1):
        v = i["views"][source]
        row = {"#": n, "strategy": i["strategy"]["label"],
               "why this strategy": (i.get("rationale") or {}).get("short", ""),
               "lots": i["lots"], "capital needed": i["capital_required"],
               "max loss": "open-ended" if i["unlimited_loss"] else f"{i['max_loss']:,.0f}",
               "max profit": max_profit_text(i), "expected P&L": v["expected"], "chance of profit": v["prob_profit"],
               "ES 95%": v["es95"], "breakevens": ", ".join(f"{b:,.0f}" for b in i["breakevens"]),
               "expected, market view": i["views"]["market"]["expected"]}
        if with_reasons:
            short = [r.replace("chance of profit ", "chance ").replace(" is below ", " < ")
                     .replace("expected P&L Rs ", "expected ₹").replace(" is not above zero", " ≤ 0")
                     for r in i.get("reasons", [])]
            row = {"#": n, "strategy": row["strategy"], "why not recommended": "; ".join(short),
                   **{k: v2 for k, v2 in row.items() if k not in ("#", "strategy")}}
        out.append(row)
    return pd.DataFrame(out)


IDEA_FORMAT = {"capital needed": "{:,.0f}", "expected P&L": "{:,.0f}", "chance of profit": "{:.0%}",
               "ES 95%": "{:,.0f}", "expected, market view": "{:,.0f}"}
IDEA_COLUMNS = {"why this strategy": st.column_config.TextColumn(width="large"),
                "why not recommended": st.column_config.TextColumn(width="medium")}


def show_strategy(p: dict, source: str, view_summary: dict | None):
    v = p["views"][source]
    c = st.columns(6)
    c[0].metric("Capital needed", f"₹{p['capital_required']:,.0f}")
    c[1].metric("Max loss", "open-ended" if p["unlimited_loss"] else f"₹{p['max_loss']:,.0f}",
                help=None if not p["unlimited_loss"] else f"₹{p['stress_loss']:,.0f} in a 3-sd move by expiry")
    c[2].metric("Max profit", max_profit_text(p, "₹"),
                help=None if p["max_profit"] is None else f"₹{p['max_profit']:,.0f} if NIFTY went to zero; "
                f"₹{p['max_profit_3sd']:,.0f} within a 3-sd move by expiry")
    c[3].metric("Expected P&L", f"₹{v['expected']:,.0f}", help=f"under the {source} view, after all charges")
    c[4].metric("Chance of profit", f"{v['prob_profit']:.0%}")
    c[5].metric("ES 95%", f"₹{v['es95']:,.0f}", help="average loss in the worst 5% of outcomes")
    show(charts.payoff_chart(p, {**view_summary, "name": source} if view_summary else None))
    left, right = st.columns([3, 2])
    legs = pd.DataFrame([{"leg": l["label"], "bid": l["bid"], "ask": l["ask"], "fill price": l["price"],
                          "vol": l["vol"]} for l in p["legs"]])
    left.dataframe(legs.style.format({"bid": "{:,.2f}", "ask": "{:,.2f}", "fill price": "{:,.2f}",
                                      "vol": lambda x: "" if x is None or x != x else f"{x:.2%}"}),
                   hide_index=True, width="stretch")
    m = p["margin"]
    costs = pd.DataFrame([
        {"item": "Premium paid (+) / received (-)", "Rs": p["premium"]},
        {"item": "of which bid-ask cost vs mid", "Rs": p["premium"] - p["fair_premium"]},
        {"item": "Charges (brokerage, STT, fees, GST)", "Rs": p["entry_charges"]},
        {"item": "Margin estimate (SPAN-style + exposure)", "Rs": m["total"]},
        {"item": "Capital needed", "Rs": p["capital_required"]}])
    right.dataframe(costs.style.format({"Rs": "{:,.0f}"}), hide_index=True, width="stretch")
    g = p["greeks"]
    st.caption(f"Position Greeks: delta {g['delta']:,.1f} units · gamma {g['gamma']:,.3f} · "
               f"vega ₹{g['vega']:,.0f} per vol pt · theta ₹{g['theta']:,.0f} per day. "
               f"Market view: expected ₹{p['views']['market']['expected']:,.0f}, model view: "
               f"₹{p['views']['model']['expected']:,.0f}. {m['note']}.")


with tab_strat:
    try:
        meta = client.strategy_meta(underlying)
    except PricingAPIError as exc:
        show_error(exc)
        st.stop()
    with st.container(border=True):
        st.markdown('<div class="opa-kicker">Describe your idea</div>', unsafe_allow_html=True)
        st.caption("Write it in plain English: capital, where you think the index goes (a level, a % move or a "
                   "range), by when, and how much risk you accept. The sentence sets the parameters, and everything "
                   "assumed is listed with the result. Or skip this and set the parameters below.")
        examples = ["I have 2 lakh and expect NIFTY to reach 24,500 by next Friday, low risk",
                    "NIFTY will stay between 23,800 and 24,400 till expiry, capital 3L, high probability",
                    "Bearish: NIFTY may fall 2% this week, ₹50,000, no naked selling",
                    "Expect a big move either way after the RBI policy, 1 lakh"]
        ex = st.selectbox("Examples", ["(write your own)"] + examples, key="intent_ex")
        text = st.text_area("Your idea", value="" if ex.startswith("(") else ex, height=90, key="intent_text",
                            placeholder=examples[0])
        if st.button("Find strategies for this idea", type="primary", key="intent_go"):
            try:
                with st.spinner("Reading the idea and pricing every strategy…"):
                    st.session_state["intent"] = client.strategy_from_text(text, underlying)
                st.session_state.pop("intent_pdf", None)
            except PricingAPIError as exc:
                show_error(exc)
        res = st.session_state.get("intent")
        if res:
            it, rr = res["interpretation"], res["result"]
            st.markdown("##### How the idea was read")
            c = st.columns(6)
            c[0].metric("Capital", f"₹{it['capital']:,.0f}")
            c[1].metric("Direction", {"up": "Up", "down": "Down", "neutral": "Sideways", "volatile": "Big move",
                                      "none": "No view"}.get(it["direction"], it["direction"]))
            c[2].metric("Target" if not it.get("range") else "Range",
                        (f"{it['range'][0]:,.0f}–{it['range'][1]:,.0f}" if it.get("range")
                         else f"{it['target']:,.0f}" if it.get("target") else "market view"))
            c[3].metric("Horizon", pd.Timestamp(it["horizon"]).strftime("%d %b"), f"expiry {it['expiry'][5:]}",
                        delta_color="off", delta_arrow="off")
            c[4].metric("Risk", it["risk"].capitalize(), f"loss ≤ {rr['risk_budget']:.0%}",
                        delta_color="off", delta_arrow="off")
            c[5].metric("Rank by", {"risk_adjusted": "Reward/risk", "probability": "Probability",
                                    "expected_return": "Return"}[it["objective"]])
            if res["assumptions"]:
                st.markdown("\n".join(f"- {a}" for a in res["assumptions"]))
            st.caption(f"Read by: {it['read_by']}. Numbers come only from your sentence or the market data.")
            src_i = rr["view"]["source"]
            if rr["ideas"]:
                st.markdown("##### Ideas")
                st.dataframe(idea_rows(rr["ideas"], src_i).style.format(IDEA_FORMAT), hide_index=True,
                             width="stretch", column_config=IDEA_COLUMNS)
                pick_i = st.selectbox("Inspect", range(len(rr["ideas"])), key="intent_pick",
                                      format_func=lambda k: rr["ideas"][k]["strategy"]["label"])
                idea_i = rr["ideas"][pick_i]
                st.markdown(f"**{idea_i['strategy']['label']}**. {idea_i['why']}")
                show_strategy(idea_i, src_i, rr["views"][src_i])
                pdf_ctx = {"text": res["text"], "assumptions": res["assumptions"], "spot": it["spot"],
                           "underlying": underlying, "view_source": src_i}
                if st.button("Prepare PDF of this strategy", key="intent_pdf_btn"):
                    st.session_state["intent_pdf"] = client.strategy_pdf(idea_i, pdf_ctx)
                if st.session_state.get("intent_pdf"):
                    st.download_button("⬇ Download strategy PDF", st.session_state["intent_pdf"],
                                       file_name="strategy.pdf", mime="application/pdf", key="intent_pdf_dl")
            else:
                st.warning("No strategy fits this idea within the capital and risk limit. Try more capital, a "
                           "higher risk level or a later date.")
            st.caption(rr["disclaimer"])

    st.markdown('<div class="opa-kicker">Or set the parameters</div>', unsafe_allow_html=True)
    exp_row = expiry_rows[0]
    st.caption("Enter the capital and your view; the engine prices every strategy at the bid/ask you would pay, "
               "adds all charges and an estimated margin, sizes it to your capital, and projects the P&L.")
    a1, a2, a3, a4, a5 = st.columns(5)
    capital = a1.number_input("Capital (₹)", min_value=1000.0, value=20000.0, step=5000.0, format="%.0f",
                              help="Any amount; strategies are sized to fit it")
    s_exp = a2.selectbox("Expiry", meta["expiries"], key="strat_exp")
    erow = next(e for e in expiry_rows if e["expiry"] == s_exp)
    objective = a3.selectbox("Rank by", list(meta["objectives"]), format_func=lambda k: meta["objectives"][k])
    budget_pct = a4.slider("Max loss, % of capital", 10, 100, 50, 5)
    min_pop_pct = a5.slider("Min chance of profit", 0, 95, 50, 5, format="%d%%",
                            help="Recommend only ideas with at least this chance of profit AND expected P&L above "
                                 "zero under the chosen view")
    b1, b2, b3, b4, b5 = st.columns(5)
    source = b1.selectbox("Market view", list(VIEW_LABELS), format_func=VIEW_LABELS.get)
    target = b2.number_input("My target level", value=float(round(erow["forward"] / 50) * 50), step=50.0,
                             format="%.0f", help="Index level you expect at the horizon (my view)")
    as_of = pd.Timestamp(ov.get("as_of") or h["as_of"]).date()
    horizon = b3.date_input("Horizon", value=pd.Timestamp(s_exp).date(), min_value=as_of,
                            max_value=pd.Timestamp(s_exp).date())
    my_vol = b4.number_input("My vol (%)", value=round(erow["atm_vol"] * 100, 1), step=0.5, format="%.1f")
    iv_shift = b5.number_input("IV change by horizon (vol pts)", value=0.0, step=1.0, format="%.1f")
    families = st.multiselect("Strategy families", list(meta["families"]), default=list(meta["families"]),
                              format_func=meta["families"].get)
    view = {"source": source, "target": target, "horizon": str(horizon), "vol": my_vol / 100, "iv_shift": iv_shift / 100}

    rec_tab, build_tab, hedge_tab = st.tabs(["Recommender", "Strategy builder", "Hedge finder"])
    with rec_tab:
        if st.button("Find strategies", type="primary"):
            try:
                st.session_state["rec"] = client.recommend(capital, view, s_exp, families, budget_pct / 100, objective,
                                                           min_prob_profit=min_pop_pct / 100)
            except PricingAPIError as exc:
                show_error(exc)
        rec = st.session_state.get("rec")
        if rec:
            src = rec["view"]["source"]
            vs = rec["views"][src]
            m = st.columns(5)
            m[0].metric("Forward", f"{rec['forward']:,.0f}")
            m[1].metric("ATM implied vol", f"{rec['atm_vol']:.1%}")
            m[2].metric("Model realised vol", "n/a" if rec["realised_vol"] is None else f"{rec['realised_vol']:.1%}")
            m[3].metric("View: 90% range", f"{vs['p5'] / 1000:.1f}k-{vs['p95'] / 1000:.1f}k",
                        help=f"{VIEW_LABELS[src]}: {vs['p5']:,.0f} to {vs['p95']:,.0f}; mean {vs['mean']:,.0f}")
            m[4].metric("Recommended", f"{rec['recommended_count']} of {rec['feasible']}",
                        help=f"{rec['feasible']} of {rec['considered']} strategies fit the capital")
            st.caption(rec["rule"])
            if rec["note"]:
                st.warning(rec["note"])
            choices = [("Recommended", i) for i in rec["ideas"]] + [("Not recommended", i) for i in rec["near_misses"]]
            if rec["ideas"]:
                st.markdown("#### Recommended")
                st.dataframe(idea_rows(rec["ideas"], src).style.format(IDEA_FORMAT), hide_index=True,
                             width="stretch", column_config=IDEA_COLUMNS)
            if rec["near_misses"]:
                with st.expander(f"Not recommended: the {len(rec['near_misses'])} closest, and why",
                                 expanded=not rec["ideas"]):
                    st.dataframe(idea_rows(rec["near_misses"], src, with_reasons=True).style.format(IDEA_FORMAT),
                                 hide_index=True, width="stretch", column_config=IDEA_COLUMNS)
            if any(max_profit_text(i).endswith("*") for _, i in choices):
                st.caption("\\* profit within a 3-sd move by expiry; the larger figure needs NIFTY to fall to zero.")
            if choices:
                pick = st.selectbox("Inspect", range(len(choices)),
                                    format_func=lambda k: f"{choices[k][0]}: {choices[k][1]['strategy']['label']}")
                tag, idea = choices[pick]
                r = idea.get("rationale") or {}
                st.markdown(f"**{idea['strategy']['label']}** ({idea['strategy']['family_label']}, {tag.lower()}). "
                            f"{idea['why']}")
                st.markdown(f"**Why this strategy.** {r.get('basis', '')} {r.get('fit', '')}")
                if idea.get("reasons"):
                    st.error("Not recommended: " + "; ".join(idea["reasons"]) + ".")
                show_strategy(idea, src, vs)
                if st.button("Prepare PDF of this strategy", key="rec_pdf_btn"):
                    st.session_state["rec_pdf"] = client.strategy_pdf(idea, {"underlying": underlying,
                                                                            "spot": ov["spot"], "view_source": src})
                if st.session_state.get("rec_pdf"):
                    st.download_button("⬇ Download strategy PDF", st.session_state["rec_pdf"], file_name="strategy.pdf",
                                       mime="application/pdf", key="rec_pdf_dl")
            if rec["blocked"]:
                with st.expander("Strategies that do not fit this capital"):
                    st.dataframe(pd.DataFrame(rec["blocked"]).rename(columns={"min_capital": "cheapest needs (₹)"})
                                 .style.format({"cheapest needs (₹)": "{:,.0f}"}), hide_index=True,
                                 width="stretch")
        else:
            st.info("Set the capital and view above, then press **Find strategies**.")

    with build_tab:
        st.caption("Build any combination and see its P&L before trading it. Sides: buy or sell; ratio = lots of "
                   "that leg per unit.")
        k0 = float(round(erow["forward"] / 100) * 100)
        legs_df = st.data_editor(pd.DataFrame([
            {"instrument": "call", "side": "buy", "strike": k0, "ratio": 1},
            {"instrument": "call", "side": "sell", "strike": k0 + 200, "ratio": 1}]), num_rows="dynamic",
            width="stretch", key="legs", column_config={
                "instrument": st.column_config.SelectboxColumn(options=["call", "put", "future"]),
                "side": st.column_config.SelectboxColumn(options=["buy", "sell"])})
        lots_b = st.number_input("Lots", min_value=1, max_value=50, value=1)
        if st.button("Project P&L", type="primary"):
            legs = []
            for r in legs_df.to_dict("records"):
                if not r.get("instrument"):
                    continue
                leg = {"instrument": r["instrument"], "side": r["side"], "expiry": s_exp, "ratio": int(r["ratio"] or 1)}
                if r["instrument"] != "future":
                    leg["strike"] = float(r["strike"])
                legs.append(leg)
            try:
                st.session_state["proj"] = client.project_strategy(legs, int(lots_b), view)
            except PricingAPIError as exc:
                show_error(exc)
        proj = st.session_state.get("proj")
        if proj:
            st.markdown(proj["why"])
            st.markdown(f"**Why this strategy.** {proj['rationale']['basis']} {proj['rationale']['fit']}")
            show_strategy(proj, view["source"], proj["views_summary"][view["source"]])
            show(charts.pnl_heatmap(proj["grid"]))

    with hedge_tab:
        st.caption("Uses the book from the **Book risk** tab. Searches futures, single calls or puts, call + put "
                   "pairs (both wings, unequal lots allowed) and spreads, 1 to 3 lots each, within your cash budget. "
                   "Each hedge is judged on the loss if the index rises and if it falls (3-sd moves to expiry).")
        h1, h2 = st.columns(2)
        hb = h1.number_input("Hedge budget (₹, premium + extra margin)", min_value=1000.0, value=50000.0,
                             step=5000.0, format="%.0f")
        rank_by = h2.selectbox("Choose hedges by", list(meta["hedge_ranks"]), format_func=meta["hedge_ranks"].get)
        if st.button("Find hedges", type="primary"):
            positions = [dict(underlying=underlying, **row) for row in book.dropna().to_dict("records")]
            try:
                st.session_state["hedge"] = client.hedge(positions, hb, view, rank_by, top=5)
            except PricingAPIError as exc:
                show_error(exc)
        hg = st.session_state.get("hedge")
        if hg:
            bf = hg["before"]
            st.markdown(f"**Book today:** loss ₹{bf['loss_if_up']:,.0f} if the index rises 3 sd, "
                        f"₹{bf['loss_if_down']:,.0f} if it falls 3 sd; ES95 ₹{bf['es95']:,.0f} ({hg['view']} view); "
                        f"delta {bf['greeks']['delta']:,.1f}, vega ₹{bf['greeks']['vega']:,.0f} per vol pt.")
            if hg["hedges"]:
                st.dataframe(pd.DataFrame([{
                    "#": n, "hedge": x["label"], "cash needed": x["cash_needed"], "loss if up": x["loss_if_up"],
                    "loss if down": x["loss_if_down"], "ES95 after": x["es95"], "delta after": x["greeks"]["delta"],
                    "vega after": x["greeks"]["vega"], "expected P&L after": x["expected"]}
                    for n, x in enumerate(hg["hedges"], 1)]).style.format(
                    {c: "{:,.0f}" for c in ("cash needed", "loss if up", "loss if down", "ES95 after", "vega after",
                                            "expected P&L after")} | {"delta after": "{:,.1f}"}),
                    hide_index=True, width="stretch")
                pick = st.selectbox("Inspect hedge", range(len(hg["hedges"])),
                                    format_func=lambda i: f"{i + 1}. {hg['hedges'][i]['label']}")
                x = hg["hedges"][pick]
                left, right = st.columns([3, 2])
                show(charts.hedge_payoff(hg, pick), left)
                right.markdown("**Trades to place**")
                right.dataframe(pd.DataFrame(x["legs"]).rename(columns={"price": "fill price"}).style.format(
                    {"fill price": "{:,.2f}", "bid": "{:,.2f}", "ask": "{:,.2f}"}), hide_index=True,
                    width="stretch")
                mtxt = (f"margin released ₹{-x['extra_margin']:,.0f}" if x["extra_margin"] < 0
                        else f"extra margin ₹{x['extra_margin']:,.0f}")
                right.caption(f"Premium ₹{x['premium']:,.0f} · {mtxt} · charges ₹{x['charges']:,.0f}. "
                              f"{hg['note']}")
                show(charts.hedge_bar(hg))
            else:
                st.warning("No hedge fits this budget.")

    st.divider()
    st.warning(f"**Disclaimer.** {meta['disclaimer']} Margins are estimates; confirm with the broker before trading.")

# ------------------------------------------------------------------ agent
with tab_agent:
    if not (IS_ADMIN or AGENT_FOR_ALL):  # the agent runs on the owner's LLM key
        st.info("The AI agent is available to administrators for now.")
    else:
        st.caption("Ask in plain language, e.g. *Price 2 lots of the 22,500 call and explain the Greeks*. "
                   "Every number comes from the pricing tools; an independent validator reviews each answer.")
        if not (h.get("llm") or {}).get("configured"):
            st.info("The agent needs an LLM. In the repo's `.env` set `LLM_PROVIDER` (e.g. groq, gemini, openrouter, "
                    "ollama, anthropic), `LLM_API_KEY` and `LLM_MODEL`, restart the API, then check it with "
                    "`python scripts/check_llm.py`.\n\n" + (h.get("llm") or {}).get("message", ""))
        st.session_state.setdefault("chat", [])
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        question = st.chat_input("Ask about a price, Greeks, the smile or the book")
        if question:
            st.session_state.chat.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Pricing and validating..."):
                    try:
                        history = st.session_state.chat[:-1]
                        reply = client.ask(question, history)
                        st.markdown(reply["answer"])
                        st.markdown(f"Status {badge(reply['status'])} · validator: {reply['validator_verdict']} · "
                                    f"{reply.get('provider', '')} {reply.get('model', '')}")
                        if reply.get("ungrounded_numbers"):
                            st.warning("Numbers not found in any tool result (treat with care): "
                                       + ", ".join(reply["ungrounded_numbers"]))
                        with st.expander("Independent validator review"):
                            st.write(reply["validator_review"])
                        with st.expander(f"Tool calls ({len(reply['tool_calls'])})"):
                            st.json(reply["tool_calls"], expanded=False)
                        st.session_state.chat.append({"role": "assistant", "content": reply["answer"]})
                    except PricingAPIError as exc:
                        show_error(exc)

# ------------------------------------------------------------------ daily feed (email)
with tab_feed:
    left, right = st.columns([2, 3], gap="large")
    with left:
        ui.card("NIFTY options daily, by email",
                "One email after each trading day with the numbers on this dashboard: ATM implied vol and its daily "
                "change, skew (25-delta risk reversal and butterfly), the expected move to each expiry, "
                "market-implied odds, reference prices with Greeks, open interest, futures basis and rates, and the "
                "data-quality checks behind them. Market information only, no trade ideas.")
        st.write("")
        on_now = bool(USER.get("daily"))
        state = ("✅ **Subscribed.** The email goes to **{}** after each trading day (about 15:20 IST)."
                 if on_now else "**Not subscribed.** Switch it on below to get the email at **{}**.")
        st.markdown(state.format(USER["email"]))
        on = st.toggle("Send me the daily email", value=on_now, disabled=not USER.get("verified"),
                       help=None if USER.get("verified") else "Verify your email first (see My account)")
        if on != on_now:
            try:
                st.session_state["user"] = client.auth("POST", "daily", st.session_state["session"], on=on)
                st.rerun()
            except PricingAPIError as exc:
                show_error(exc)
        if not USER.get("verified"):
            st.caption("Verify your email address first: use **Resend the verification link** in *My account*.")
        st.caption("Every email has a one-click unsubscribe link.")
        with st.form("subscribe_other", clear_on_submit=True):
            st.markdown("**Send it to another address too**")
            o_email = st.text_input("Email", max_chars=254, placeholder="e.g. your work email",
                                    label_visibility="collapsed")
            o_ok = st.checkbox("The owner of this address agrees to receive the daily email (it is market "
                               "information, not investment advice).")
            if st.form_submit_button("Subscribe this address", width="stretch"):
                try:
                    r = client.feed_subscribe(o_email, "", o_ok)
                    (st.success if r["status"] in ("pending", "active") else st.error)(
                        r["message"] + ("" if r["status"] != "pending" else " The owner of that inbox confirms it."))
                except PricingAPIError as exc:
                    show_error(exc)

    with right:
        st.markdown("**Preview of today's email**")
        pv1, pv2 = st.columns([3, 1])
        p_name = pv1.text_input("Greeting name for the preview", value="", placeholder="optional",
                                label_visibility="collapsed")
        if pv2.button("Build preview", width="stretch") or "feed_preview" in st.session_state:
            try:
                if "feed_preview" not in st.session_state or st.session_state.get("feed_preview_name") != p_name:
                    with st.spinner("Building today's feed..."):
                        st.session_state["feed_preview"] = client.feed_preview(p_name)
                        st.session_state["feed_preview_name"] = p_name
                ui.embed(st.session_state["feed_preview"], height=1100)
                st.download_button("Download preview (HTML)", st.session_state["feed_preview"],
                                   file_name="nifty_options_daily_preview.html", mime="text/html")
            except PricingAPIError as exc:
                show_error(exc)
        else:
            st.info("Press **Build preview** to render today's email with live numbers.")

    if IS_ADMIN:
        with st.expander("Admin: subscribers and sending"):
            key = st.text_input("Admin key", type="password", help="Needed only if FEED_ADMIN_KEY is set in .env")
            try:
                fs = client.feed_status(key)
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Active", fs["counts"]["active"])
                a2.metric("Awaiting confirmation", fs["counts"]["pending"])
                a3.metric("Unsubscribed", fs["counts"]["unsubscribed"])
                ls = fs.get("last_send") or {}
                a4.metric("Last send", ls.get("date", "never"),
                          f"{ls.get('sent', 0)} sent, {ls.get('failed', 0)} failed" if ls else None, delta_color="off", delta_arrow="off")
                st.caption(f"Mail: {fs['mail']} · links point to {fs['public_url']}")
                subs = client.feed_subscribers(key)["subscribers"]
                if subs:
                    st.dataframe(pd.DataFrame(subs)[["email", "name", "status", "created_at", "confirmed_at",
                                                     "unsubscribed_at", "source"]], hide_index=True,
                                 width="stretch")
                t1, t2 = st.columns([3, 1])
                test_to = t1.text_input("Send a test copy to", placeholder="you@company.com")
                if t2.button("Send test", width="stretch") and test_to:
                    st.info(client.feed_send([test_to], admin_key=key)["message"])
                b1, b2 = st.columns([3, 1])
                sure = b1.checkbox(f"Send today's feed to all {fs['counts']['active']} active subscribers now")
                if b2.button("Send to all", type="primary", disabled=not sure, width="stretch"):
                    st.info(client.feed_send(admin_key=key)["message"])
                d1, d2 = st.columns([3, 1])
                del_email = d1.text_input("Erase a subscriber (removes all their data)", placeholder="email")
                if d2.button("Erase", width="stretch") and del_email:
                    st.info("Erased." if client.feed_delete(del_email, key)["deleted"] else "No such subscriber.")
            except PricingAPIError as exc:
                show_error(exc)

    st.warning("**Not investment advice.** The feed is market information and analytics for education and "
               "research. It is not a recommendation to buy, sell or hold any security or derivative, and the "
               "author accepts no responsibility for any loss or gain from using it.")

# ------------------------------------------------------------------ my account
with tab_acct:
    dash_auth.account_tab(client, USER)
