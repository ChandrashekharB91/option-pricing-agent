"""Daily risk report: one self-contained HTML file (charts load Plotly from its CDN)."""
from __future__ import annotations

import html
from datetime import datetime
from typing import Optional

from . import charts

STATUS_COLOR = {"PASS": "#16a34a", "WARN": "#d97706", "FAIL": "#dc2626", "green": "#16a34a", "amber": "#d97706",
                "red": "#dc2626"}


def _badge(s: str) -> str:
    return f'<span class="badge" style="background:{STATUS_COLOR.get(s, "#6b7280")}">{html.escape(s.upper())}</span>'


def _table(rows: list[dict], cols: list[tuple[str, str, str]]) -> str:
    head = "".join(f"<th>{html.escape(t)}</th>" for _, t, _ in cols)
    body = []
    for r in rows:
        cells = []
        for key, _, fmt in cols:
            v = r.get(key)
            txt = _badge(v) if fmt == "status" else (fmt.format(v) if isinstance(v, (int, float)) and fmt else str(v))
            cells.append(f"<td>{txt if fmt == 'status' else html.escape(txt)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def summary_text(risk: dict, pnl: Optional[dict]) -> str:
    v, bt, t = risk["var"], risk["backtest"], risk["book"]["totals"]
    worst = risk["stress"][0]
    breaches = [l["limit"] for l in risk["limits"] if l["status"] == "FAIL"]
    near = [l["limit"] for l in risk["limits"] if l["status"] == "WARN"]
    parts = [
        f"The book is worth {charts._rs(t['value'])} with net delta {t['delta']:,.1f} index units, vega "
        f"{charts._rs(t['vega'])} per vol point and theta {charts._rs(t['theta'])} per day.",
        f"One-day 99% VaR is {charts._rs(v['var'])} and 97.5% Expected Shortfall {charts._rs(v['es'])}, from "
        f"{v['window']} days of history; the worst day in that window ({v['worst']['date']}) would lose "
        f"{charts._rs(-v['worst']['pnl'])}.",
        f"The worst stress test is \"{worst['scenario']}\" at {charts._rs(worst['pnl'])}.",
    ]
    if "exceptions" in bt:
        parts.append(f"The VaR backtest shows {bt['exceptions']} exceptions in {bt['days']} days (expected "
                     f"{bt['expected']:.1f}), traffic light {bt['traffic_light']}.")
    parts.append(f"Limits breached: {', '.join(breaches)}." if breaches else
                 (f"Limits near their maximum: {', '.join(near)}." if near else "All limits are within 80% of maximum."))
    if pnl and "totals" in pnl:
        p = pnl["totals"]
        parts.append(f"From {pnl['day0']} to {pnl['day1']} the book made {charts._rs(p['actual'])}: delta "
                     f"{charts._rs(p['delta'])}, gamma {charts._rs(p['gamma'])}, vega {charts._rs(p['vega'])}, theta "
                     f"{charts._rs(p['theta'])}, unexplained {pnl['unexplained_share']:.1%} ({pnl['status']}).")
    return " ".join(parts)


def render_risk_report(risk: dict, pnl: Optional[dict] = None, commentary: str = "") -> str:
    figs = [charts.limits_bar(risk["limits"]), charts.var_histogram(risk["var"]),
            charts.contributions_bar(risk["var"])]
    if "exceptions" in risk["backtest"]:
        figs.append(charts.backtest_chart(risk["backtest"]))
    figs.append(charts.stress_bar(risk["stress"]))
    if pnl and "totals" in pnl:
        figs.append(charts.pnl_waterfall(pnl))
    plots = [f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False) for i, f in enumerate(figs)]

    positions = _table(risk["book"]["positions"], [
        ("underlying", "Index", ""), ("expiry", "Expiry", ""), ("strike", "Strike", "{:,.0f}"),
        ("option_type", "Type", ""), ("lots", "Lots", "{:g}"), ("price", "Price", "{:,.2f}"), ("vol", "Vol", "{:.2%}"),
        ("value", "Value Rs", "{:,.0f}"), ("delta", "Delta", "{:,.1f}"), ("gamma", "Gamma", "{:.4f}"),
        ("vega", "Vega Rs/pt", "{:,.0f}"), ("theta", "Theta Rs/day", "{:,.0f}"), ("validation", "Validation", "status")])
    by_exp = _table(risk["by_expiry"], [("expiry", "Expiry", ""), ("value", "Value Rs", "{:,.0f}"),
                                        ("delta", "Delta", "{:,.1f}"), ("gamma", "Gamma", "{:.4f}"),
                                        ("vega", "Vega Rs/pt", "{:,.0f}"), ("theta", "Theta Rs/day", "{:,.0f}")])
    bt = risk["backtest"]
    bt_html = ("<p>Backtest unavailable: " + html.escape(bt.get("error", "")) + "</p>") if "exceptions" not in bt else (
        f"<p>Kupiec (exception frequency) p = {bt['kupiec_p']:.2f}; Christoffersen (clustering) p = "
        f"{bt['christoffersen_p']:.2f}. p-values below 0.05 reject the VaR model. {html.escape(bt['note'])}.</p>")
    explain_html = ""
    if pnl and "error" in pnl:
        explain_html = f"<h2>P&amp;L explain</h2><p>{html.escape(pnl['error'])}</p>"
    elif pnl:
        explain_html = "<h2>P&amp;L explain</h2>" + plots.pop() + _table(pnl["legs"], [
            ("label", "Position", ""), ("actual", "Actual", "{:,.0f}"), ("delta", "Delta", "{:,.0f}"),
            ("gamma", "Gamma", "{:,.0f}"), ("vega", "Vega", "{:,.0f}"), ("theta", "Theta", "{:,.0f}"),
            ("cross", "Cross", "{:,.0f}"), ("unexplained", "Unexplained", "{:,.0f}")]) + \
            "".join(f"<p class='note'>{html.escape(n)}</p>" for n in pnl.get("notes", []))
    stress_plot = plots.pop()
    bt_plot = plots.pop() if "exceptions" in bt else ""
    limits_plot, hist_plot, contrib_plot = plots[0], plots[1], plots[2]
    comm = f"<h2>Commentary</h2><p>{html.escape(commentary)}</p>" if commentary else ""

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Daily Risk Report</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#111827}}
h1{{margin-bottom:4px}} h2{{margin-top:32px;border-bottom:1px solid #e5e7eb;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}} th,td{{padding:6px 8px;border-bottom:1px solid #e5e7eb;text-align:right}}
th:first-child,td:first-child{{text-align:left}} th{{background:#f9fafb}}
.badge{{color:#fff;border-radius:4px;padding:2px 6px;font-size:12px;font-weight:600}}
.meta{{color:#6b7280}} .note{{color:#6b7280;font-size:12px}} .summary{{background:#f9fafb;padding:12px 16px;border-radius:8px;line-height:1.5}}
</style></head><body>
<h1>Daily Risk Report {_badge(risk['status'])}</h1>
<p class="meta">Market data as of {html.escape(str(risk['as_of'])[:16].replace('T', ' '))} IST ({html.escape(risk['data_mode'])} data) ·
history to {html.escape(str(risk['history_last_date']))} · generated {datetime.now():%d %b %Y %H:%M}</p>
<div class="summary">{html.escape(summary_text(risk, pnl))}</div>
{comm}
<h2>Limits</h2>{limits_plot}
<h2>Positions</h2>{positions}<h3>By expiry</h3>{by_exp}
<h2>Value at Risk and Expected Shortfall</h2>{hist_plot}{contrib_plot}
<h2>VaR backtest</h2>{bt_plot}{bt_html}
<h2>Stress tests</h2>{stress_plot}
{explain_html}
<h2>Method</h2><p class="note">{html.escape(risk['var']['method'])}. VaR 1-day horizon includes time decay;
stress tests are instantaneous. India VIX is computed from NIFTY options and is a proxy for other indices.
Limits are placeholders set in config/risk_limits.json. All prices from the in-house library, validated per position.</p>
</body></html>"""
