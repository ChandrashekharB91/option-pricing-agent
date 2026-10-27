"""Streamlit dashboard for the quants team. Talks to the FastAPI backend over HTTP.

Run (two terminals):
    uvicorn optlib_app.api:app
    streamlit run src/optlib_app/dashboard.py
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from optlib_app import charts
from optlib_app.client import PricingAPIError, PricingClient

CALL, PUT, ACCENT, GREY = "#2563eb", "#ea580c", "#7c3aed", "#6b7280"
STATUS_COLOR = {"PASS": "#16a34a", "WARN": "#d97706", "FAIL": "#dc2626", "SKIP": "#6b7280"}
DEFAULT_BOOK = pd.DataFrame([
    {"expiry": "2026-10-27", "strike": 22500.0, "option_type": "call", "lots": -2.0},
    {"expiry": "2026-10-27", "strike": 22500.0, "option_type": "put", "lots": -2.0},
    {"expiry": "2026-10-27", "strike": 22000.0, "option_type": "put", "lots": 2.0},
])

st.set_page_config(page_title="Option Pricing Agent", layout="wide")


@st.cache_resource
def api(url: str) -> PricingClient:
    return PricingClient(url)


def badge(status: str) -> str:
    return f":{'green' if status == 'PASS' else 'orange' if status == 'WARN' else 'red'}[**{status}**]"


def show_error(exc: Exception):
    st.error(f"{exc.detail if isinstance(exc, PricingAPIError) else exc}")


def validation_table(report: dict):
    df = pd.DataFrame(report["checks"])[["status", "name", "detail"]]
    st.dataframe(df.style.map(lambda s: f"color: {STATUS_COLOR.get(s, 'inherit')}; font-weight: 600",
                              subset=["status"]), hide_index=True, use_container_width=True)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Option Pricing Agent")
    url = st.text_input("API URL", value=PricingClient().base_url)
    client = api(url)
    try:
        h = client.health()
        st.success(f"API connected · {h['data_mode']} data")
        st.caption(f"Market data as of {h['as_of'][:16].replace('T', ' ')} IST")
        underlyings = h["underlyings"]
        llm = h.get("llm") or {}
        if llm.get("configured"):
            st.caption(f"Agent LLM: {llm['provider']} · {llm['model']}")
        else:
            st.caption("Agent LLM: not configured")
    except Exception as exc:  # noqa: BLE001
        st.error(f"API not reachable at {url}. Start it with `uvicorn optlib_app.api:app`.\n\n{exc}")
        st.stop()
    underlying = st.selectbox("Underlying", underlyings, index=0)

try:
    ov = client.overview(underlying)
except PricingAPIError as exc:
    st.warning(f"{underlying}: {exc.detail}")
    st.stop()

expiry_rows = ov["expiries"]
expiry_ids = [e["expiry"] for e in expiry_rows]
lot = ov["lot_size"]

tab_market, tab_price, tab_book, tab_risk, tab_agent = st.tabs(
    ["Market & smile", "Pricer", "Book risk", "Risk (VaR, stress, P&L)", "Ask the agent"])

# ------------------------------------------------------------------ market
with tab_market:
    c = st.columns(5)
    c[0].metric("Spot", f"{ov['spot']:,.2f}")
    c[1].metric("Lot size", lot)
    c[2].metric("Discount curve", ov["curve"]["label"].replace("MIBOR-OIS ", ""),
                f"{ov['curve']['age_days']} days old", delta_color="off")
    c[3].metric("Expiries fitted", len(expiry_rows))
    c[4].markdown(f"Surface status<br>{badge(ov['surface_status'])}", unsafe_allow_html=True)

    table = pd.DataFrame([{
        "expiry": e["expiry"], "days": e["days"], "forward": e["forward"], "rate": e["rate"], "ATM vol": e["atm_vol"],
        "fwd vs future": (f"{e['forward_check']['gap']:+.3%} {e['forward_check']['verdict']}"
                          if e["forward_check"] else "no future"),
        "QA": e["qa_status"], "SVI RMSE (vol pts)": e["svi_rmse_vol_pts"], "SVI ATM gap": e["svi_atm_gap_vol_pts"],
        "SVI": e["svi_status"]} for e in expiry_rows])
    st.dataframe(table.style.format({"days": "{:.2f}", "forward": "{:,.2f}", "rate": "{:.3%}", "ATM vol": "{:.2%}",
                                     "SVI RMSE (vol pts)": "{:.3f}", "SVI ATM gap": "{:+.3f}"}),
                 hide_index=True, use_container_width=True)

    exp = st.selectbox("Expiry", expiry_ids, key="smile_exp")
    sm = client.smile(exp, underlying)
    q = pd.DataFrame(sm["quotes"])
    width = st.slider("Strikes shown (± points from forward)", 500, 5000, 1800, 100)
    q = q[(q.strike - sm["forward"]).abs() <= width]
    curve = pd.DataFrame(sm["svi_curve"])
    curve = curve[(curve.strike - sm["forward"]).abs() <= width]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q.strike, y=q.iv_ask * 100, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=q.strike, y=q.iv_bid * 100, fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(37,99,235,0.18)", name="bid-ask"))
    fig.add_trace(go.Scatter(x=q.strike, y=q.iv_mid * 100, mode="markers", marker=dict(color=CALL, size=5),
                             name="market mid", customdata=q[["type", "bid", "ask"]],
                             hovertemplate="%{x:,.0f} %{customdata[0]}<br>vol %{y:.2f}%<br>"
                                           "bid %{customdata[1]:.2f} / ask %{customdata[2]:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=curve.strike, y=curve.vol * 100, line=dict(color=ACCENT, width=2), name="SVI fit"))
    fig.add_vline(x=sm["forward"], line=dict(color=GREY, dash="dash"),
                  annotation_text=f"forward {sm['forward']:,.0f}")
    fig.update_layout(title=f"{underlying} smile · {exp} · ATM {sm['atm_vol']:.2%}", xaxis_title="strike",
                      yaxis_title="implied vol (%)", height=440, margin=dict(t=50, b=40), legend=dict(x=0.01, y=0.99))
    st.plotly_chart(fig, use_container_width=True)
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
            c[1].metric("Position value", f"₹{r['value']:,.0f}", f"{p_lots:g} lots × {lot}", delta_color="off")
            c[2].metric("Implied vol", f"{r['vol']:.2%}", r["vol_source"], delta_color="off")
            c[3].metric("Forward", f"{r['forward']:,.2f}", f"rate {r['rate']:.3%}", delta_color="off")
            c[4].markdown(f"Validation<br>{badge(v['status'])}", unsafe_allow_html=True)
            g = pd.DataFrame({"per unit": r["greeks"], "position": r["position_greeks"]})
            g.index.name = "Greek"
            left, right = st.columns([1, 2])
            left.dataframe(g.style.format(lambda x: f"{x:,.2f}" if abs(x) >= 1 else f"{x:.6f}"),
                           use_container_width=True)
            left.caption("vega per vol point · theta per calendar day · rho per bp · position = per unit × quantity")
            with right:
                validation_table(v)
        except PricingAPIError as exc:
            show_error(exc)

# ------------------------------------------------------------------ book
with tab_book:
    st.caption("Edit the book (lots: positive long, negative short), then run. Default: short 22,500 straddle "
               "with a 22,000 put hedge.")
    book = st.data_editor(DEFAULT_BOOK.assign(expiry=expiry_ids[0]), num_rows="dynamic", use_container_width=True,
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
            c[5].markdown(f"Validation<br>{badge(b['status'])}", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(b["positions"]).style.format(
                {"lots": "{:g}", "quantity": "{:,.0f}", "price": "{:,.2f}", "vol": "{:.2%}", "value": "{:,.0f}",
                 "delta": "{:,.2f}", "gamma": "{:.4f}",
                 "vega": "{:,.0f}", "theta": "{:,.0f}", "rho": "{:,.2f}", "strike": "{:,.0f}"}),
                hide_index=True, use_container_width=True)
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
            st.plotly_chart(heat, use_container_width=True)
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
        m[2].metric("Worst day in window", f"₹{v['worst']['pnl']:,.0f}", v["worst"]["date"], delta_color="off")
        m[3].metric("Worst stress", f"₹{r['stress'][0]['pnl']:,.0f}", r["stress"][0]["scenario"][:28],
                    delta_color="off")
        m[4].metric("Backtest exceptions", f"{bt.get('exceptions', '-')} / {bt.get('days', '-')}",
                    f"traffic light {bt.get('traffic_light', 'n/a')}", delta_color="off")
        m[5].markdown(f"Risk status<br>{badge(r['status'])}", unsafe_allow_html=True)
        st.caption(f"History through {r['history_last_date']} (no look-ahead into the pricing day) · "
                   f"window {v['first_date']} to {v['last_date']}")

        sub = st.tabs(["Limits & Greeks", "VaR & ES", "Backtest", "Stress tests", "P&L explain", "Report"])
        with sub[0]:
            st.plotly_chart(charts.limits_bar(r["limits"]), use_container_width=True)
            st.caption("Limits are placeholders in config/risk_limits.json; the risk team sets them.")
            st.dataframe(pd.DataFrame(r["by_expiry"]).style.format(
                {"value": "{:,.0f}", "delta": "{:,.1f}", "gamma": "{:.4f}", "vega": "{:,.0f}", "theta": "{:,.0f}"}),
                hide_index=True, use_container_width=True)
        with sub[1]:
            left, right = st.columns([3, 2])
            left.plotly_chart(charts.var_histogram(v), use_container_width=True)
            right.plotly_chart(charts.contributions_bar(v), use_container_width=True)
            right.caption("Average P&L of each position on the worst 2.5% of days; red adds to the loss, "
                          "green offsets it. The bars add up to ES.")
            st.caption(v["method"])
        with sub[2]:
            if "exceptions" in bt:
                st.plotly_chart(charts.backtest_chart(bt), use_container_width=True)
                st.markdown(f"Kupiec (frequency) p = **{bt['kupiec_p']:.2f}** · Christoffersen (clustering) "
                            f"p = **{bt['christoffersen_p']:.2f}** · values below 0.05 reject the model. "
                            f"_{bt['note']}._")
            else:
                st.info(bt.get("error", "backtest unavailable"))
        with sub[3]:
            st.plotly_chart(charts.stress_bar(r["stress"]), use_container_width=True)
            st.dataframe(pd.DataFrame([{k: x[k] for k in ("scenario", "kind", "index_move", "vol_mult", "pnl")}
                                       for x in r["stress"]]).style.format(
                {"index_move": "{:+.2%}", "vol_mult": "x{:.2f}", "pnl": "{:,.0f}"}), hide_index=True,
                use_container_width=True)
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
                    st.plotly_chart(charts.pnl_waterfall(e), use_container_width=True)
                    st.dataframe(pd.DataFrame(e["legs"])[["label", "close0", "close1", "vol0", "vol1", "actual", "delta",
                                                          "gamma", "vega", "theta", "cross", "unexplained"]].style.format(
                        {"close0": "{:,.2f}", "close1": "{:,.2f}", "vol0": "{:.2%}", "vol1": "{:.2%}",
                         **{k: "{:,.0f}" for k in ("actual", "delta", "gamma", "vega", "theta", "cross",
                                                   "unexplained")}}), hide_index=True, use_container_width=True)
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

# ------------------------------------------------------------------ agent
with tab_agent:
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
