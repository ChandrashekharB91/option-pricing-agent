"""Plotly charts shared by the dashboard and the HTML risk report (same picture in both places)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

LOSS, GAIN, ACCENT, GREY, AMBER = "#dc2626", "#16a34a", "#7c3aed", "#6b7280", "#d97706"
LAYOUT = dict(height=380, margin=dict(t=50, b=40, l=60, r=20), template="plotly_white")


def _rs(x: float) -> str:
    a = abs(x)
    if a >= 1e7:
        s = f"{a / 1e7:.2f} cr"
    elif a >= 1e5:
        s = f"{a / 1e5:.2f} L"
    else:
        s = f"{a:,.0f}"
    return ("-" if x < 0 else "") + "Rs " + s


def var_histogram(var: dict) -> go.Figure:
    pnl = np.asarray(var["pnl"])
    fig = go.Figure(go.Histogram(x=pnl, nbinsx=60, marker_color=GREY, opacity=0.75, name="scenario P&L",
                                 hovertemplate="P&L %{x:,.0f}<br>%{y} days<extra></extra>"))
    # ES lies further into the loss tail than VaR: label ES on its left, VaR on its right, so they never overlap
    for x, label, color, pos in ((-var["var"], f"VaR {var['var_level']:.0%}: {_rs(var['var'])}", LOSS, "top right"),
                                 (-var["es"], f"ES {var['es_level']:.1%}: {_rs(var['es'])}", AMBER, "top left")):
        fig.add_vline(x=x, line=dict(color=color, width=2, dash="dash"), annotation_text=label,
                      annotation_position=pos, annotation_font_color=color)
    fig.update_layout(title=f"1-day P&L over {var['window']} historical days ({var['first_date']} to {var['last_date']})",
                      xaxis_title="P&L (Rs)", yaxis_title="days", showlegend=False, **LAYOUT)
    return fig


def contributions_bar(var: dict) -> go.Figure:
    c = sorted(var["es_contributions"], key=lambda x: x["contribution"])
    fig = go.Figure(go.Bar(x=[x["contribution"] for x in c], y=[x["label"] for x in c], orientation="h",
                           marker_color=[LOSS if x["contribution"] > 0 else GAIN for x in c],
                           hovertemplate="%{y}: %{x:,.0f}<extra></extra>"))
    fig.update_layout(title=f"Contribution to ES ({_rs(var['es'])})",
                      xaxis_title="Rs", **{**LAYOUT, "height": max(220, 60 + 40 * len(c))})
    return fig


def backtest_chart(bt: dict) -> go.Figure:
    pnl, var = np.asarray(bt["pnl"]), np.asarray(bt["var"])
    hits = pnl < -var
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bt["dates"], y=pnl, marker_color=np.where(hits, LOSS, "#cbd5e1"), name="hypothetical P&L",
                         hovertemplate="%{x}: %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=bt["dates"], y=-var, line=dict(color=ACCENT, width=1.5), name="-VaR 99%"))
    fig.update_layout(title=f"Backtest: {bt['exceptions']} exceptions in {bt['days']} days "
                            f"(expected {bt['expected']:.1f}) · traffic light {bt['traffic_light'].upper()}",
                      yaxis_title="Rs", legend=dict(orientation="h", y=-0.15), **LAYOUT)
    return fig


def stress_bar(stress: list) -> go.Figure:
    rows = sorted(stress, key=lambda r: r["pnl"], reverse=True)
    labels = [f"{r['scenario']} [{r['index_move']:+.1%}, vol x{r['vol_mult']:.2f}]" for r in rows]
    fig = go.Figure(go.Bar(x=[r["pnl"] for r in rows], y=labels, orientation="h",
                           marker_color=[LOSS if r["pnl"] < 0 else GAIN for r in rows],
                           marker_pattern_shape=["" if r["kind"] == "historical" else "/" for r in rows],
                           hovertemplate="%{y}<br>P&L %{x:,.0f}<extra></extra>"))
    fig.update_layout(title="Stress tests (solid: real historical days replayed; hatched: hypothetical)",
                      xaxis_title="P&L (Rs)", **{**LAYOUT, "height": 90 + 30 * len(rows), "margin": dict(t=50, b=40, l=380, r=20)})
    return fig


def pnl_waterfall(e: dict) -> go.Figure:
    t = e["totals"]
    steps = [("Delta", t["delta"]), ("Gamma", t["gamma"]), ("Vega", t["vega"]), ("Theta", t["theta"]),
             ("Cross (vanna, volga)", t["cross"]), ("Rate", t["rate"]), ("Unexplained", t["unexplained"])]
    fig = go.Figure(go.Waterfall(
        x=[s for s, _ in steps] + ["Actual P&L"], y=[v for _, v in steps] + [t["actual"]],
        measure=["relative"] * len(steps) + ["total"], text=[_rs(v) for _, v in steps] + [_rs(t["actual"])],
        textposition="outside", increasing=dict(marker_color=GAIN), decreasing=dict(marker_color=LOSS),
        totals=dict(marker_color=ACCENT), connector=dict(line=dict(color=GREY, width=1))))
    fig.update_layout(title=f"P&L explain {e['day0']} to {e['day1']}: unexplained {e['unexplained_share']:.1%} "
                            f"of explained ({e['status']})", yaxis_title="Rs", showlegend=False, **LAYOUT)
    return fig


def limits_bar(limits: list) -> go.Figure:
    colors = {"PASS": GAIN, "WARN": AMBER, "FAIL": LOSS}
    fig = go.Figure(go.Bar(x=[min(l["utilisation"], 1.5) * 100 for l in limits], y=[l["limit"] for l in limits],
                           orientation="h", marker_color=[colors[l["status"]] for l in limits],
                           text=[f"{l['utilisation']:.0%}" for l in limits], textposition="outside",
                           customdata=[[l["value"], l["max"], l["unit"]] for l in limits],
                           hovertemplate="%{y}: %{customdata[0]:,.2f} of %{customdata[1]:,.2f} %{customdata[2]}<extra></extra>"))
    fig.add_vline(x=100, line=dict(color=LOSS, dash="dash"))
    fig.add_vline(x=80, line=dict(color=AMBER, dash="dot"))
    fig.update_layout(title="Limit utilisation (amber from 80%, red above 100%)", xaxis_title="% of limit",
                      xaxis_range=[0, 160], **{**LAYOUT, "height": 80 + 45 * len(limits)})
    return fig


# ------------------------------------------------------------------ strategies (phase 3)

def payoff_chart(p: dict, view: dict | None = None, horizon_label: str = "at horizon") -> go.Figure:
    """P&L of a strategy across index levels: at expiry, at the view's horizon and right now."""
    c = p["curves"]
    F = c["F"]
    fig = go.Figure()
    if view:  # 90% range of the index under the chosen view
        fig.add_vrect(x0=view["p5"], x1=view["p95"], fillcolor=ACCENT, opacity=0.07, line_width=0,
                      annotation_text=f"90% range, {view['name']} view", annotation_position="top left",
                      annotation_font_color=GREY)
    fig.add_hline(y=0, line=dict(color=GREY, width=1))
    fig.add_trace(go.Scatter(x=F, y=c["now"], name="now", line=dict(color=GREY, width=1, dash="dot"),
                             hovertemplate="index %{x:,.0f}: Rs %{y:,.0f}<extra>now</extra>"))
    if c["horizon"] != c["expiry"]:
        fig.add_trace(go.Scatter(x=F, y=c["horizon"], name=horizon_label, line=dict(color=AMBER, width=2),
                                 hovertemplate="index %{x:,.0f}: Rs %{y:,.0f}<extra>horizon</extra>"))
    fig.add_trace(go.Scatter(x=F, y=c["expiry"], name="at expiry", line=dict(color=ACCENT, width=3),
                             hovertemplate="index %{x:,.0f}: Rs %{y:,.0f}<extra>expiry</extra>"))
    for b in p["breakevens"]:
        fig.add_vline(x=b, line=dict(color=GREY, width=1, dash="dash"),
                      annotation_text=f"breakeven {b:,.0f}", annotation_position="bottom right",
                      annotation_font_color=GREY)
    fig.update_layout(title=f"{p['strategy']['label']} x {p['lots']} lot(s): P&L after all charges",
                      xaxis_title="NIFTY forward", yaxis_title="P&L (Rs)", legend=dict(orientation="h", y=-0.2),
                      **{**LAYOUT, "height": 420})
    return fig


def pnl_heatmap(grid: dict) -> go.Figure:
    z = np.asarray(grid["pnl"]).T  # rows: index level, columns: time
    fig = go.Figure(go.Heatmap(z=z, x=grid["times"], y=[f"{f:,.0f}" for f in grid["F"]], colorscale="RdYlGn",
                               zmid=0, hovertemplate="%{x}, index %{y}: Rs %{z:,.0f}<extra></extra>",
                               colorbar=dict(title="Rs")))
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    fig.update_layout(title="P&L by index level and date (sticky-strike vols, after charges)",
                      xaxis_title="date (IST)", yaxis_title="NIFTY forward", **{**LAYOUT, "height": 480})
    return fig


def hedge_bar(h: dict) -> go.Figure:
    """Loss if the index rises and if it falls (3-sd moves to expiry), before and after each hedge."""
    rows = [("Book today", h["before"])] + [(x["label"], x) for x in h["hedges"]]
    names = [r[0] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=[r[1]["loss_if_up"] for r in rows], orientation="h", name="if the index rises",
                         marker_color=LOSS, text=[_rs(r[1]["loss_if_up"]) for r in rows], textposition="outside"))
    fig.add_trace(go.Bar(y=names, x=[r[1]["loss_if_down"] for r in rows], orientation="h", name="if the index falls",
                         marker_color=AMBER, text=[_rs(r[1]["loss_if_down"]) for r in rows], textposition="outside"))
    top = max(max(r[1]["loss_if_up"], r[1]["loss_if_down"]) for r in rows)
    fig.update_layout(title="Worst loss by expiry for a 3-sd move, before and after each hedge", barmode="group",
                      xaxis_title="loss (Rs)", xaxis_range=[0, 1.3 * top], yaxis=dict(autorange="reversed"),
                      legend=dict(orientation="h", y=-0.2),
                      **{**LAYOUT, "height": 160 + 70 * len(rows), "margin": dict(t=50, b=60, l=420, r=60)})
    return fig


def hedge_payoff(h: dict, i: int) -> go.Figure:
    """Book P&L at expiry across index levels: today's book vs the book with hedge i."""
    F, x = h["curves"]["F"], h["hedges"][i]
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=GREY, width=1))
    fig.add_trace(go.Scatter(x=F, y=h["curves"]["before"], name="book today", line=dict(color=LOSS, width=2, dash="dot"),
                             hovertemplate="index %{x:,.0f}: Rs %{y:,.0f}<extra>before</extra>"))
    fig.add_trace(go.Scatter(x=F, y=x["curve_after"], name="with the hedge", line=dict(color=GAIN, width=3),
                             hovertemplate="index %{x:,.0f}: Rs %{y:,.0f}<extra>after</extra>"))
    fig.update_layout(title=f"Book P&L at expiry: {x['label']}", xaxis_title="NIFTY at expiry",
                      yaxis_title="P&L from today (Rs)", legend=dict(orientation="h", y=-0.2), **{**LAYOUT, "height": 420})
    return fig
