"""PDF exports: the daily market report (same content as the email) and a one-strategy report.

Built with reportlab and matplotlib, both in the app's requirements; every page carries the disclaimer.
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

from optlib.recommender import DISCLAIMER

INK, MUTED, ACCENT, LINE, SOFT = "#0f172a", "#64748b", "#4f46e5", "#e2e8f0", "#f8fafc"


def _rl():
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)
    return locals()


def _styles(r):
    base = r["getSampleStyleSheet"]()
    PS = r["ParagraphStyle"]
    c = r["colors"]
    return {
        "title": PS("t", parent=base["Title"], fontName="Helvetica-Bold", fontSize=18, leading=22, alignment=0,
                    textColor=c.HexColor(INK), spaceAfter=2),
        "sub": PS("s", parent=base["Normal"], fontSize=9, leading=12, textColor=c.HexColor(MUTED)),
        "h": PS("h", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14,
                textColor=c.HexColor(ACCENT), spaceBefore=10, spaceAfter=4),
        "p": PS("p", parent=base["Normal"], fontSize=9.5, leading=13.5, textColor=c.HexColor(INK)),
        "note": PS("n", parent=base["Normal"], fontSize=7.8, leading=10.5, textColor=c.HexColor(MUTED)),
        "cell": PS("c", parent=base["Normal"], fontSize=8.3, leading=10.5, textColor=c.HexColor(INK)),
        "kpi": PS("k", parent=base["Normal"], fontSize=8.3, leading=15, textColor=c.HexColor(INK)),
        "disc": PS("d", parent=base["Normal"], fontSize=8, leading=11, textColor=c.HexColor("#7c2d12")),
    }


def _table(r, head, rows, widths=None, align_right_from=1):
    c = r["colors"]
    t = r["Table"]([head] + rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.2), ("FONT", (0, 1), (-1, -1), "Helvetica", 8.3),
        ("TEXTCOLOR", (0, 0), (-1, 0), c.HexColor(MUTED)), ("TEXTCOLOR", (0, 1), (-1, -1), c.HexColor(INK)),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, c.HexColor("#cbd5e1")),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, c.HexColor(LINE)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if align_right_from is not None:
        style.append(("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"))
    for i in range(1, len(rows) + 1):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), c.HexColor(SOFT)))
    t.setStyle(r["TableStyle"](style))
    return t


def _kpis(r, st, items):
    """A row of KPI tiles: [(label, value, sub), ...]."""
    c = r["colors"]
    cells = [[r["Paragraph"](f'<font size="7.5" color="{MUTED}">{l.upper()}</font><br/>'
                             f'<font size="13" color="{INK}"><b>{v}</b></font><br/>'
                             f'<font size="7.5" color="{MUTED}">{s}</font>', st["kpi"]) for l, v, s in items]]
    w = 180 * r["mm"] / len(items)
    t = r["Table"](cells, colWidths=[w] * len(items), hAlign="LEFT")
    t.setStyle(r["TableStyle"]([("BOX", (0, 0), (-1, -1), 0.5, c.HexColor(LINE)),
                                ("INNERGRID", (0, 0), (-1, -1), 0.5, c.HexColor(LINE)),
                                ("BACKGROUND", (0, 0), (-1, -1), c.HexColor(SOFT)),
                                ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                                ("LEFTPADDING", (0, 0), (-1, -1), 7)]))
    return t


def _img(r, png: bytes, width_mm: float = 180):
    from reportlab.lib.utils import ImageReader

    ir = ImageReader(io.BytesIO(png))
    w, h = ir.getSize()
    width = width_mm * r["mm"]
    return r["Image"](io.BytesIO(png), width=width, height=width * h / w)


def _disclaimer(r, st):
    c = r["colors"]
    t = r["Table"]([[r["Paragraph"](f"<b>Not investment advice.</b> {DISCLAIMER}", st["disc"])]],
                   colWidths=[180 * r["mm"]], hAlign="LEFT")
    t.setStyle(r["TableStyle"]([("BACKGROUND", (0, 0), (-1, -1), c.HexColor("#fff7ed")),
                                ("BOX", (0, 0), (-1, -1), 0.5, c.HexColor("#fed7aa")),
                                ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def _doc(title: str):
    r = _rl()
    buf = io.BytesIO()

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(r["colors"].HexColor(MUTED))
        canvas.drawString(15 * r["mm"], 9 * r["mm"],
                          f"{title} · model output for analysis, not investment advice · page {doc.page}")
        canvas.restoreState()

    doc = r["SimpleDocTemplate"](buf, pagesize=r["A4"], leftMargin=15 * r["mm"], rightMargin=15 * r["mm"],
                                 topMargin=14 * r["mm"], bottomMargin=16 * r["mm"], title=title,
                                 author="NIFTY Options Daily")
    return r, buf, doc, footer


def _d(iso: str, fmt: str = "%d %b") -> str:
    return datetime.fromisoformat(iso[:10]).strftime(fmt)


# ------------------------------------------------------------------ daily report

def daily_report(feed: dict) -> bytes:
    from .feed_email import charts

    r, buf, doc, footer = _doc(f"{feed['underlying']} options daily {feed['date']}")
    st = _styles(r)
    P, S = r["Paragraph"], r["Spacer"]
    e0 = feed["expiries"][0]
    as_of = datetime.fromisoformat(feed["as_of"])
    src = "Upstox live quotes" if feed["data_mode"] == "live" else "recorded Upstox snapshot (replay)"
    story = [P(f"{feed['underlying']} options daily", st["title"]),
             P(f"{as_of:%A, %d %B %Y} · snapshot {as_of:%H:%M} IST · {src} · surface {feed['surface_status']}",
               st["sub"]), S(1, 8)]
    chg = (f"{feed['change']:+,.2f} ({feed['change_pct']:+.2%})" if feed.get("change") is not None else "")
    ivr = feed.get("iv_rank") or {}
    oi = feed.get("open_interest") or {}
    story.append(_kpis(r, st, [
        (feed["underlying"], f"{feed['spot']:,.2f}", chg),
        ("India VIX", f"{feed['hist_vix'] * 100:.2f}" if feed.get("hist_vix") else "n/a",
         f"1y rank {ivr['rank']:.0%}" if ivr.get("available") else ""),
        (f"ATM vol {_d(e0['expiry'])}", f"{e0['atm_vol']:.2%}", f"{e0['days']:.1f} days"),
        ("Expected move", f"±{e0['expected_move']:,.0f}", f"{e0['range_lo']:,.0f}–{e0['range_hi']:,.0f}"),
        ("Put/call OI", f"{oi['pcr']:.2f}" if oi.get("pcr") else "n/a",
         f"max pain {oi['max_pain']:,.0f}" if oi.get("max_pain") else ""),
    ]))
    note = feed.get("ai_note")
    if note:
        story += [P("AI market note", st["h"]), P(note["text"], st["p"]),
                  P(f"Written by AI ({note.get('model') or 'language model'}) from the numbers in this report; every "
                    f"number was checked against the data. Description, not advice.", st["note"])]
    story += [P("Today in brief", st["h"])] + [P(f"• {h}", st["p"]) for h in feed["highlights"]]

    ch = feed["changes"]
    rows = [[_d(e["expiry"]), f"{e['days']:.1f}", f"{e['forward']:,.0f}", f"{e['atm_vol']:.2%}",
             (f"{ch['atm_vol'][e['expiry']] * 100:+.2f}" if e["expiry"] in ch.get("atm_vol", {}) else "–"),
             f"{e['rr25'] * 100:+.2f}", f"{e['bf25'] * 100:+.2f}", f"±{e['expected_move']:,.0f}",
             f"{e['straddle']:,.1f}"] for e in feed["expiries"]]
    story += [P("Volatility by expiry", st["h"]),
              _table(r, ["Expiry", "Days", "Forward", "ATM vol", "Day chg", "25d RR", "25d BF", "1-sd move",
                         "Straddle"], rows),
              P("Day change, RR and BF in vol points. 1-sd move = forward × ATM vol × sqrt(time).", st["note"])]
    imgs = charts(feed)
    if imgs.get("term"):
        story += [S(1, 4), _img(r, imgs["term"], 140)]
    if imgs.get("smile"):
        story += [r["KeepTogether"]([P("The smile", st["h"]), _img(r, imgs["smile"], 150),
                                     P("Lines: fitted SVI smile within ±3 expected moves; dots: market mid implied "
                                       "vols.", st["note"])])]

    rv = [["India VIX (30-day implied)", f"{feed['hist_vix']:.2%}" if feed.get("hist_vix") else "n/a"],
          [f"ATM implied vol, {_d(e0['expiry'])}", f"{e0['atm_vol']:.2%}"],
          ["Realised vol, 20 days", f"{feed['hist_rv20']:.2%}" if feed.get("hist_rv20") else "n/a"],
          ["Realised vol, 60 days", f"{feed['hist_rv60']:.2%}" if feed.get("hist_rv60") else "n/a"],
          ["Implied minus 20-day realised", f"{feed['vrp'] * 100:+.2f} vol pts" if feed.get("vrp") is not None else "n/a"]]
    if ivr.get("available"):
        rv += [["India VIX 1-year rank / percentile", f"{ivr['rank']:.0%} / {ivr['percentile']:.0%} ({ivr['label']})"],
               ["India VIX 1-year range", f"{ivr['low']:.2%} to {ivr['high']:.2%}"]]
    story += [r["KeepTogether"]([P("Implied vs realised", st["h"]), _table(r, ["Measure", "Value"], rv)])]

    prob = [[f"{p['level']:,.0f}", f"{p['sds']:+.0f} sd" if p["sds"] else "forward", f"{p['above']:.0%}",
             f"{p['below']:.0%}"] for p in feed["probabilities"]]
    story += [r["KeepTogether"]([P(f"Market-implied odds at the {_d(e0['expiry'])} expiry", st["h"]),
                                 _table(r, ["Level", "Distance", "Settles above", "Settles below"], prob),
                                 P("Risk-neutral probabilities from option prices: what the market prices, not a "
                                   "forecast.", st["note"])])]
    ref = [[x["label"], f"{x['strike']:,.0f}", f"{x['price']:,.2f}", f"{x['vol']:.2%}", f"{x['delta']:+.3f}",
            f"{x['vega']:.2f}", f"{x['theta']:.2f}", f"{x['lot_value']:,.0f}", x["status"]]
           for x in feed["reference_options"]]
    story += [r["KeepTogether"]([P(f"Reference prices · {_d(e0['expiry'])} expiry", st["h"]),
                                 _table(r, ["Option", "Strike", "Price", "Vol", "Delta", "Vega", "Theta", "Rs/lot",
                                            "Check"], ref),
                                 P(f"Model prices on mid-market vols; Greeks per unit. Lot size {feed['lot_size']}.",
                                   st["note"])])]
    if oi:
        top = [[f"{c['strike']:,.0f}", f"{c['oi'] / 1e5:,.1f} L", f"{p['strike']:,.0f}", f"{p['oi'] / 1e5:,.1f} L"]
               for c, p in zip(oi["top_calls"][:3], oi["top_puts"][:3])]
        story += [r["KeepTogether"]([P(f"Open interest · {_d(e0['expiry'])} expiry", st["h"]),
                                     _table(r, ["Call strike", "Call OI", "Put strike", "Put OI"], top),
                                     P(f"Put/call ratio {oi['pcr']:.2f}" if oi.get("pcr") else "", st["note"])])]
    news = feed.get("news") or []
    if news:
        story += [P("Market news", st["h"])]
        for n in news[:6]:
            story.append(P(f'<link href="{n["link"]}" color="{INK}"><b>{_esc(n["title"])}</b></link> '
                           f'<font color="{MUTED}" size="7.5">· {_esc(n["source"])}</font>', st["p"]))
        story.append(P("Headlines and links from the publishers' public feeds.", st["note"]))
    story += [S(1, 10), _disclaimer(r, st)]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ------------------------------------------------------------------ strategy report

def _payoff_png(p: dict) -> Optional[bytes]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker  # noqa: F401
    except ImportError:
        return None
    cv = p.get("curves") or {}
    if not cv.get("F"):
        return None
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    F = cv["F"]
    ax.axhline(0, color="#94a3b8", lw=0.8)
    if cv.get("expiry"):
        ax.plot(F, cv["expiry"], color="#4f46e5", lw=2, label="at expiry")
    if cv.get("horizon") and cv.get("horizon") != cv.get("expiry"):
        ax.plot(F, cv["horizon"], color="#0e7490", lw=1.4, ls="--", label="at the horizon")
    if cv.get("now"):
        ax.plot(F, cv["now"], color="#b45309", lw=1.2, ls=":", label="today")
    for b in p.get("breakevens") or []:
        ax.axvline(b, color="#94a3b8", lw=0.7, ls="--")
    ax.set_xlabel("index level")
    ax.set_ylabel("P&L (Rs)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e5:,.1f} L" if abs(v) >= 1e5
                                                                   else f"{v:,.0f}"))
    ax.grid(axis="y", color="#eef2f7", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _rs(x) -> str:
    return "unlimited" if x is None else f"Rs {x:,.0f}"


def strategy_report(p: dict, context: Optional[dict] = None) -> bytes:
    """One idea from the recommender or the custom builder. ``context``: market and request info for the header."""
    context = context or {}
    label = p["strategy"]["label"]
    r, buf, doc, footer = _doc(label)
    st = _styles(r)
    P, S = r["Paragraph"], r["Spacer"]
    story = [P(label, st["title"]),
             P(f"{p['strategy'].get('family_label', '')} · {p['lots']} lot(s) of {p['lot_size']} · "
               f"prepared {datetime.now():%d %b %Y %H:%M}"
               + (f" · {context['underlying']} {context['spot']:,.2f}" if context.get("spot") else ""), st["sub"]),
             S(1, 8)]
    if context.get("text"):
        story += [P("Your idea", st["h"]), P(_esc(context["text"]), st["p"])]
    if context.get("assumptions"):
        story += [P("How it was read", st["h"])] + [P(f"• {_esc(a)}", st["p"]) for a in context["assumptions"]]
    view = (p.get("views") or {})
    vkey = context.get("view_source") or next(iter(view), None)
    vv = view.get(vkey, {}) if vkey else {}
    story.append(S(1, 6))
    story.append(_kpis(r, st, [
        ("Capital needed", _rs(p.get("capital_required")), "premium + margin + costs"),
        ("Max loss", _rs(p.get("max_loss")), "at expiry"),
        ("Max profit", _rs(p.get("max_profit")), "at expiry"),
        ("Chance of profit", f"{vv['prob_profit']:.0%}" if vv.get("prob_profit") is not None else "n/a",
         f"{vkey} view" if vkey else ""),
        ("Expected P&L", f"Rs {vv['expected']:,.0f}" if vv.get("expected") is not None else "n/a",
         f"{vkey} view" if vkey else ""),
    ]))
    legs = [[l.get("label", ""), l["instrument"], l["side"], f"{l['strike']:,.0f}" if l.get("strike") else "–",
             _d(l["expiry"]) if l.get("expiry") else "", f"{l['price']:,.2f}" if l.get("price") is not None else "",
             f"{l['vol']:.2%}" if l.get("vol") else ""] for l in p.get("legs", [])]
    story += [P("Legs", st["h"]), _table(r, ["Leg", "Type", "Side", "Strike", "Expiry", "Price", "Vol"], legs, None, 3)]
    png = _payoff_png(p)
    if png:
        story += [P("Payoff", st["h"]), _img(r, png, 150),
                  P("Breakevens: " + (", ".join(f"{b:,.0f}" for b in p.get("breakevens") or []) or "none"),
                    st["note"])]
    g = p.get("greeks") or {}
    if g:
        story += [P("Position Greeks", st["h"]),
                  _table(r, ["Delta", "Gamma", "Vega", "Theta"],
                         [[f"{g.get('delta', 0):,.2f}", f"{g.get('gamma', 0):,.4f}", f"{g.get('vega', 0):,.0f}",
                           f"{g.get('theta', 0):,.0f}"]], None, 0)]
    if view:
        vrows = [[k, f"Rs {v['expected']:,.0f}", f"{v['prob_profit']:.0%}", f"Rs {v['es95']:,.0f}",
                  f"Rs {v['p5']:,.0f}", f"Rs {v['p95']:,.0f}"] for k, v in view.items()]
        story += [P("Outcomes under each view", st["h"]),
                  _table(r, ["View", "Expected P&L", "P(profit)", "ES 95%", "5th pct", "95th pct"], vrows),
                  P("market: risk-neutral density from the smile; model: historical vol; user: your target. "
                    "ES 95%: average loss in the worst 5% of outcomes.", st["note"])]
    why = p.get("rationale") or p.get("why")
    if why:
        story += [P("Why it was shortlisted", st["h"])]
        for w in (why if isinstance(why, list) else [why]):
            story.append(P(f"• {_esc(str(w))}", st["p"]))
    costs = [["Premium", f"Rs {p.get('premium', 0):,.0f}"], ["Slippage vs fair", f"Rs {p.get('slippage', 0):,.0f}"],
             ["Entry charges", f"Rs {p.get('entry_charges', 0):,.0f}"],
             ["Margin", f"Rs {(p.get('margin') or {}).get('total', 0):,.0f}"],
             ["Stress loss", _rs(p.get("stress_loss"))]]
    story += [r["KeepTogether"]([P("Costs and stress", st["h"]), _table(r, ["Item", "Amount"], costs)]),
              S(1, 10), _disclaimer(r, st)]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
