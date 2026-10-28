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

tab_market, tab_price, tab_book, tab_risk, tab_strat, tab_agent = st.tabs(
    ["Market & smile", "Pricer", "Book risk", "Risk (VaR, stress, P&L)", "Strategy lab", "Ask the agent"])

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
    st.plotly_chart(charts.payoff_chart(p, {**view_summary, "name": source} if view_summary else None),
                    use_container_width=True)
    left, right = st.columns([3, 2])
    legs = pd.DataFrame([{"leg": l["label"], "bid": l["bid"], "ask": l["ask"], "fill price": l["price"],
                          "vol": l["vol"]} for l in p["legs"]])
    left.dataframe(legs.style.format({"bid": "{:,.2f}", "ask": "{:,.2f}", "fill price": "{:,.2f}",
                                      "vol": lambda x: "" if x is None or x != x else f"{x:.2%}"}),
                   hide_index=True, use_container_width=True)
    m = p["margin"]
    costs = pd.DataFrame([
        {"item": "Premium paid (+) / received (-)", "Rs": p["premium"]},
        {"item": "of which bid-ask cost vs mid", "Rs": p["premium"] - p["fair_premium"]},
        {"item": "Charges (brokerage, STT, fees, GST)", "Rs": p["entry_charges"]},
        {"item": "Margin estimate (SPAN-style + exposure)", "Rs": m["total"]},
        {"item": "Capital needed", "Rs": p["capital_required"]}])
    right.dataframe(costs.style.format({"Rs": "{:,.0f}"}), hide_index=True, use_container_width=True)
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
                             use_container_width=True, column_config=IDEA_COLUMNS)
            if rec["near_misses"]:
                with st.expander(f"Not recommended: the {len(rec['near_misses'])} closest, and why",
                                 expanded=not rec["ideas"]):
                    st.dataframe(idea_rows(rec["near_misses"], src, with_reasons=True).style.format(IDEA_FORMAT),
                                 hide_index=True, use_container_width=True, column_config=IDEA_COLUMNS)
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
            if rec["blocked"]:
                with st.expander("Strategies that do not fit this capital"):
                    st.dataframe(pd.DataFrame(rec["blocked"]).rename(columns={"min_capital": "cheapest needs (₹)"})
                                 .style.format({"cheapest needs (₹)": "{:,.0f}"}), hide_index=True,
                                 use_container_width=True)
        else:
            st.info("Set the capital and view above, then press **Find strategies**.")

    with build_tab:
        st.caption("Build any combination and see its P&L before trading it. Sides: buy or sell; ratio = lots of "
                   "that leg per unit.")
        k0 = float(round(erow["forward"] / 100) * 100)
        legs_df = st.data_editor(pd.DataFrame([
            {"instrument": "call", "side": "buy", "strike": k0, "ratio": 1},
            {"instrument": "call", "side": "sell", "strike": k0 + 200, "ratio": 1}]), num_rows="dynamic",
            use_container_width=True, key="legs", column_config={
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
            st.plotly_chart(charts.pnl_heatmap(proj["grid"]), use_container_width=True)

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
                    hide_index=True, use_container_width=True)
                pick = st.selectbox("Inspect hedge", range(len(hg["hedges"])),
                                    format_func=lambda i: f"{i + 1}. {hg['hedges'][i]['label']}")
                x = hg["hedges"][pick]
                left, right = st.columns([3, 2])
                left.plotly_chart(charts.hedge_payoff(hg, pick), use_container_width=True)
                right.markdown("**Trades to place**")
                right.dataframe(pd.DataFrame(x["legs"]).rename(columns={"price": "fill price"}).style.format(
                    {"fill price": "{:,.2f}", "bid": "{:,.2f}", "ask": "{:,.2f}"}), hide_index=True,
                    use_container_width=True)
                mtxt = (f"margin released ₹{-x['extra_margin']:,.0f}" if x["extra_margin"] < 0
                        else f"extra margin ₹{x['extra_margin']:,.0f}")
                right.caption(f"Premium ₹{x['premium']:,.0f} · {mtxt} · charges ₹{x['charges']:,.0f}. "
                              f"{hg['note']}")
                st.plotly_chart(charts.hedge_bar(hg), use_container_width=True)
            else:
                st.warning("No hedge fits this budget.")

    st.divider()
    st.warning(f"**Disclaimer.** {meta['disclaimer']} Margins are estimates; confirm with the broker before trading.")

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
