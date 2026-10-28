"""Self-contained HTML report for the strategy backtest (Plotly from CDN)."""
from __future__ import annotations

import html

import plotly.graph_objects as go

from optlib.recommender import DISCLAIMER

from .charts import ACCENT, AMBER, GAIN, GREY, LAYOUT, LOSS

VIEW_NAMES = {"model": "Model view (no drift, EWMA realised vol)",
              "market": "Market-implied view (option-implied probabilities)"}


def reliability_chart(cal: list[dict], view: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GREY, dash="dash", width=1),
                             name="perfect calibration", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[c["predicted"] for c in cal], y=[c["realised"] for c in cal], mode="markers+lines",
        marker=dict(size=[max(6, min(28, 4 + c["n"] ** 0.5 / 3)) for c in cal], color=ACCENT),
        line=dict(color=ACCENT, width=1.5), name="strategies",
        error_y=dict(type="data", array=[1.96 * c["stderr"] for c in cal], color=ACCENT, thickness=1),
        customdata=[[c["bucket"], c["n"]] for c in cal],
        hovertemplate="bucket %{customdata[0]}: predicted %{x:.0%}, won %{y:.0%} (n=%{customdata[1]:,})<extra></extra>"))
    fig.update_layout(title=f"Chance of profit: predicted vs realised, {view} view (marker size = number of ideas)",
                      xaxis=dict(title="predicted chance of profit", tickformat=".0%", range=[0, 1]),
                      yaxis=dict(title="share that actually made money", tickformat=".0%", range=[0, 1]),
                      legend=dict(orientation="h", y=-0.2), **{**LAYOUT, "height": 460})
    return fig


def strategy_chart(rows: list[dict], view: str) -> go.Figure:
    rows = sorted(rows, key=lambda r: r["realised_pnl_pts"])
    names = [r["strategy"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=[r["predicted_pnl_pts"] for r in rows], orientation="h", name="predicted",
                         marker_color=GREY))
    fig.add_trace(go.Bar(y=names, x=[r["realised_pnl_pts"] for r in rows], orientation="h", name="realised",
                         marker_color=[GAIN if r["realised_pnl_pts"] > 0 else LOSS for r in rows]))
    fig.update_layout(title=f"Average P&L per trade, index points per unit, {view} view (after charges)",
                      barmode="group", xaxis_title="points per unit (x lot size = rupees per lot)",
                      legend=dict(orientation="h", y=-0.12),
                      **{**LAYOUT, "height": 160 + 26 * len(rows), "margin": dict(t=50, b=60, l=170, r=30)})
    return fig


def equity_chart(eq: list[dict], view: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=[e["day"] for e in eq], y=[e["cum"] for e in eq], mode="lines",
                               line=dict(color=ACCENT, width=2), customdata=[[e["label"], e["pnl"]] for e in eq],
                               hovertemplate="%{x}: %{customdata[0]} %{customdata[1]:,.0f}; total Rs %{y:,.0f}"
                                             "<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=GREY, width=1))
    fig.update_layout(title=f"Following the top recommended idea each day, 1 lot, {view} view (cumulative Rs)",
                      xaxis_title="entry day", yaxis_title="cumulative P&L (Rs)", **{**LAYOUT, "height": 380})
    return fig


def _table(rows: list[dict], cols: list[tuple[str, str, str]]) -> str:
    head = "".join(f"<th>{html.escape(c[1])}</th>" for c in cols)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f"<td>{format(r.get(c[0]), c[2]) if r.get(c[0]) is not None else ''}</td>"
                                 for c in cols) + "</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _fig(fig: go.Figure, first: list) -> str:
    out = fig.to_html(full_html=False, include_plotlyjs="cdn" if not first else False)
    first.append(1)
    return out


def render(s: dict) -> str:
    first: list = []
    dc = s["data_checks"]
    parts = [f"""<!doctype html><html><head><meta charset="utf-8"><title>Strategy backtest</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#111;background:#fff}}
table{{border-collapse:collapse;width:100%;margin:8px 0 20px;font-size:13px}}th,td{{border-bottom:1px solid #e5e7eb;padding:5px 8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}th{{background:#f9fafb}}h2{{margin-top:36px}}.note{{color:#4b5563;font-size:13px}}
.disc{{background:#fffbeb;border:1px solid #fde68a;padding:10px 14px;border-radius:8px;font-size:13px}}</style></head><body>
<h1>Strategy engine backtest: NIFTY weekly options</h1>
<p>{s['period'][0]} to {s['period'][1]}: {s['days_used']} of {s['days']} trading days usable,
{s['ideas']:,} strategy ideas, each settled at NIFTY's actual close on expiry day after all charges.</p>
<h2>Data checks</h2>
<table><tbody>
<tr><td>Forward vs index close (average, largest)</td><td>{dc['basis_mean']:+.2%}, {dc['basis_abs_max']:.2%}</td></tr>
<tr><td>ATM implied vol, average (India VIX average)</td><td>{dc['atm_vol_mean']:.1%} ({dc['vix_mean']:.1%})</td></tr>
<tr><td>Correlation of ATM implied vol with India VIX</td><td>{dc['atm_vs_vix_corr']:.2f}</td></tr>
<tr><td>Model realised vol, average</td><td>{dc['realised_vol_mean']:.1%}</td></tr>
<tr><td>Actual move to expiry / implied move (std dev; 1.0 = implied vol was right)</td><td>{dc['move_over_implied_sd']:.2f}</td></tr>
<tr><td>Moves beyond 2 implied standard deviations (normal: 4.6%)</td><td>{dc['share_moves_beyond_2sd']:.1%}</td></tr>
</tbody></table>"""]
    for v, d in s["views"].items():
        r = d["rule"]
        parts.append(f"<h2>{html.escape(VIEW_NAMES.get(v, v))}</h2>")
        parts.append(f"<p>Brier score {d['brier']:.3f} (lower is better; 0.25 = coin flip).</p>")
        parts.append(_fig(reliability_chart(d["calibration"], v), first))
        parts.append(_table(d["calibration"], [("bucket", "predicted bucket", ""), ("n", "ideas", ",d"),
                                               ("predicted", "average predicted", ".1%"),
                                               ("realised", "share that won", ".1%")]))
        parts.append(_fig(strategy_chart(d["by_strategy"], v), first))
        parts.append(_table(d["by_strategy"], [
            ("strategy", "strategy", ""), ("family", "family", ""), ("n", "trades", ",d"),
            ("predicted_pop", "predicted chance", ".0%"), ("realised_win_rate", "won", ".0%"),
            ("predicted_pnl_pts", "predicted pts", "+.1f"), ("realised_pnl_pts", "realised pts", "+.1f"),
            ("realised_t", "t-stat", "+.1f"), ("worst_pts", "worst pts", ",.0f")]))
        if r.get("ideas"):
            t = r["top"]
            parts.append(f"""<h3>Recommendation rule (expected P&L &gt; 0 and chance of profit &ge;
{s['assumptions']['min_prob_profit']:.0%})</h3>
<table><tbody>
<tr><td>Ideas that passed</td><td>{r['ideas']:,} on {r['days_with_idea']} of {r['days']} days</td></tr>
<tr><td>Chance of profit: predicted / realised</td><td>{r['predicted_pop']:.0%} / {r['realised_win_rate']:.0%}</td></tr>
<tr><td>Average P&L per trade, points per unit: predicted / realised</td><td>{r['predicted_pnl_pts']:+.1f} / {r['realised_pnl_pts']:+.1f}</td></tr>
<tr><td>Top idea each day: trades, win rate</td><td>{t['n']}, {t['win_rate']:.0%}</td></tr>
<tr><td>Top idea each day: total, worst trade, max drawdown (Rs, 1 lot)</td><td>{t['total_rs']:,.0f}, {t['worst_rs']:,.0f}, {t['max_drawdown_rs']:,.0f}</td></tr>
<tr><td>Strategies chosen</td><td>{html.escape(', '.join(f'{k} {n}' for k, n in list(t['by_strategy'].items())[:6]))}</td></tr>
</tbody></table>""")
            if t["equity"]:
                parts.append(_fig(equity_chart(t["equity"], v), first))
        else:
            parts.append("<p>No idea passed the recommendation rule under this view.</p>")
    a = s["assumptions"]
    parts.append(f"""<h2>Method and assumptions</h2><p class="note">Each trading day at the close, the nearest weekly
NIFTY expiry ({a['expiry_window_days'][0]:g} to {a['expiry_window_days'][1]:g} days away) is rebuilt from NSE's bhavcopy: parity forward,
implied vols and the SVI smile. Entry is at the close plus or minus {a['half_spread_rel']:.2%} (at least one tick); strikes
with no volume that day are not tradeable. Charges use today's schedule for every day ({a['charges']}); flat
{a['rate']:.1%} rate. Each strategy is settled at the index close on expiry day. Ideas are per strategy unit (1 lot);
capital sizing is not applied. The model view uses only closes before the entry day.</p>
<p class="disc"><b>Disclaimer.</b> {html.escape(DISCLAIMER)} Past results do not guarantee future results.</p>
</body></html>""")
    return "\n".join(parts)
