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
