"""Render the daily feed as an email: HTML (table layout, inline styles, light and dark), plain text and charts.

    msg = render(feed, unsubscribe_url="https://.../feed/unsubscribe?token=...", images="cid")
    msg.subject, msg.html, msg.text, msg.images   # images: {cid: png bytes}; images="data" inlines them

Email clients ignore scripts and most CSS, so the layout is nested tables with inline styles. A
prefers-color-scheme block re-colours the template in clients that honour it (Apple Mail, Outlook apps);
the rest show the light version. Charts are PNGs drawn with matplotlib.
"""
from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass, field
from datetime import date, datetime

from .feed import DISCLAIMER

# palette (light); the <style> block below swaps these in dark mode
INK, MUTED, LINE, BG, CARD, SOFT = "#0f172a", "#64748b", "#e2e8f0", "#f1f5f9", "#ffffff", "#f8fafc"
ACCENT, UP, DOWN, WARN = "#4f46e5", "#15803d", "#b91c1c", "#b45309"
HEAD = "#0b1220"
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"


@dataclass
class EmailMessageParts:
    subject: str
    html: str
    text: str
    images: dict = field(default_factory=dict)  # cid -> png bytes


# ------------------------------------------------------------------ formatting helpers

def _e(x) -> str:
    return html.escape(str(x))


def _d(iso: str, fmt: str = "%d %b") -> str:
    return date.fromisoformat(iso[:10]).strftime(fmt)


def _chg(v, fmt: str, scale: float = 1.0, colour: bool = False, suffix: str = "") -> str:
    if v is None:
        return f'<span style="color:{MUTED}" class="muted">n/a</span>'
    x = v * scale
    arrow = "&#9650;" if x > 0 else "&#9660;" if x < 0 else "&#9679;"
    col = (UP if x > 0 else DOWN if x < 0 else MUTED) if colour else MUTED
    cls = ("up" if x > 0 else "down" if x < 0 else "muted") if colour else "muted"
    return f'<span style="color:{col};white-space:nowrap" class="{cls}">{arrow} {format(x, fmt)}{suffix}</span>'


def _pill(status: str) -> str:
    col = {"PASS": UP, "WARN": WARN, "FAIL": DOWN}.get(status, MUTED)
    return (f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid {col};'
            f'color:{col};font-size:11px;font-weight:700;letter-spacing:.04em">{_e(status)}</span>')


def _section(title: str, body: str, note: str = "") -> str:
    n = (f'<p style="margin:8px 0 0;font-size:12px;line-height:1.5;color:{MUTED}" class="muted">{note}</p>'
         if note else "")
    return f"""
<tr><td style="padding:22px 28px 0" class="pad">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
    <tr><td style="border-left:3px solid {ACCENT};padding:0 0 0 10px;font-size:12px;font-weight:700;
        letter-spacing:.08em;text-transform:uppercase;color:{ACCENT}" class="accent">{_e(title)}</td></tr>
  </table>
  <div style="height:10px;line-height:10px">&nbsp;</div>
  {body}{n}
</td></tr>"""


def _table(head: list[str], rows: list[list[str]], align: list[str] | None = None, hide: tuple = ()) -> str:
    """hide: column indices dropped on narrow (phone) screens."""
    align = align or (["left"] + ["right"] * (len(head) - 1))
    hm = [" hm" if i in hide else "" for i in range(len(head))]
    th = "".join(f'<th align="{a}" style="padding:7px 8px;font-size:11px;font-weight:600;color:{MUTED};'
                 f'border-bottom:1px solid {LINE};text-transform:uppercase;letter-spacing:.03em" class="muted line t{x}">'
                 f'{h}</th>' for h, a, x in zip(head, align, hm))
    body = ""
    for i, r in enumerate(rows):
        bg = SOFT if i % 2 else CARD
        cls = "zebra" if i % 2 else "card"
        body += f'<tr style="background:{bg}" class="{cls}">' + "".join(
            f'<td align="{a}" style="padding:7px 8px;font-size:13px;color:{INK};border-bottom:1px solid {LINE};'
            f'font-variant-numeric:tabular-nums" class="ink line t{x}">{c}</td>' for c, a, x in zip(r, align, hm)) + "</tr>"
    return (f'<table role="presentation" width="100%" cellspacing="0" cellpadding="0" '
            f'style="border-collapse:collapse;border:1px solid {LINE};border-radius:8px" class="line">'
            f"<tr>{th}</tr>{body}</table>")


def _tile(label: str, value: str, sub: str) -> str:
    return (f'<td width="50%" valign="top" style="padding:6px">'
            f'<table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:{SOFT};'
            f'border:1px solid {LINE};border-radius:10px" class="tile line"><tr><td style="padding:14px 16px">'
            f'<div style="font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:{MUTED}"'
            f' class="muted">{label}</div>'
            f'<div style="font-size:24px;font-weight:700;color:{INK};margin-top:4px;font-variant-numeric:tabular-nums"'
            f' class="ink">{value}</div>'
            f'<div style="font-size:12px;color:{MUTED};margin-top:2px" class="muted">{sub}</div>'
            f"</td></tr></table></td>")


# ------------------------------------------------------------------ charts

def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


def charts(feed: dict) -> dict:
    """{name: png}: the smile of each expiry, and the ATM vol term structure with the 25-delta skew."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # charts are optional; the tables carry every number
        return {}
    out = {}
    shades = ["#4f46e5", "#7c83f0", "#a5a9f5", "#c7c9f9"]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#475569",
                         "xtick.color": "#475569", "ytick.color": "#475569"})
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    for i, s in enumerate(feed["smiles"]):
        col = shades[min(i, len(shades) - 1)]
        k = [x / s["forward"] * 100 - 100 for x in s["strikes"]]
        ax.plot(k, [v * 100 for v in s["vols"]], color=col, lw=2.2 if i == 0 else 1.4,
                label=f"{_d(s['expiry'])} ({s['days']:.0f}d)")
        km = [x / s["forward"] * 100 - 100 for x in s["mkt_strikes"]]
        ax.scatter(km, [v * 100 for v in s["mkt_vols"]], s=6, color=col, alpha=0.55, linewidths=0)
    ax.axvline(0, color="#94a3b8", lw=0.8, ls="--")
    ax.set_xlabel("strike vs forward (%)")
    ax.set_ylabel("implied vol (%)")
    ax.grid(axis="y", color="#eef2f7", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    out["smile"] = _png(fig)
    plt.close(fig)

    ex = feed["expiries"]
    fig, ax = plt.subplots(figsize=(6.2, 2.3))
    days = [e["days"] for e in ex]
    ax.plot(days, [e["atm_vol"] * 100 for e in ex], color="#4f46e5", marker="o", lw=2, label="ATM")
    ax.plot(days, [e["put25_vol"] * 100 for e in ex], color="#b45309", marker="o", lw=1.2, ls="--",
            label="25-delta put")
    ax.plot(days, [e["call25_vol"] * 100 for e in ex], color="#0e7490", marker="o", lw=1.2, ls="--",
            label="25-delta call")
    for e in ex:
        ax.annotate(f"{e['atm_vol'] * 100:.1f}", (e["days"], e["atm_vol"] * 100), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8, color="#4f46e5")
    ax.set_xlabel("days to expiry")
    ax.set_ylabel("implied vol (%)")
    ax.grid(axis="y", color="#eef2f7", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    out["term"] = _png(fig)
    plt.close(fig)
    return out


def _img(name: str, images: dict, mode: str, alt: str) -> str:
    if name not in images:
        return ""
    src = f"cid:{name}" if mode == "cid" else "data:image/png;base64," + base64.b64encode(images[name]).decode()
    return (f'<img src="{src}" width="584" alt="{_e(alt)}" style="display:block;width:100%;max-width:584px;'
            f'height:auto;border:1px solid {LINE};border-radius:8px;background:#ffffff" class="line">')


# ------------------------------------------------------------------ the email

def subject(feed: dict) -> str:
    e0 = feed["expiries"][0]
    chg = f" {feed['change_pct']:+.2%}" if feed.get("change_pct") is not None else ""
    return (f"{feed['underlying']} options daily, {_d(feed['date'], '%d %b %Y')}: {feed['spot']:,.0f}{chg} · "
            f"weekly IV {e0['atm_vol']:.1%} · move ±{e0['expected_move']:,.0f}")


def render(feed: dict, unsubscribe_url: str = "", images: str = "cid", recipient_name: str = "",
           manage_url: str = "") -> EmailMessageParts:
    imgs = charts(feed)
    e0 = feed["expiries"][0]
    ch = feed["changes"]
    as_of = datetime.fromisoformat(feed["as_of"])
    src = "Upstox live quotes" if feed["data_mode"] == "live" else "recorded Upstox snapshot (replay)"
    dv0 = ch["atm_vol"].get(e0["expiry"])

    tiles = (
        "<tr>"
        + _tile(f"{_e(feed['underlying'])}", f"{feed['spot']:,.2f}",
                (_chg(feed["change"], "+,.2f", colour=True) + f" &nbsp;{feed['change_pct']:+.2%} vs {_d(feed['hist_prev_date'])}"
                 if feed.get("change") is not None else "previous close n/a"))
        + _tile("India VIX", f"{feed['hist_vix'] * 100:.2f}" if feed.get("hist_vix") else "n/a",
                (f"close of {_d(feed['hist_vix_date'])} &nbsp;"
                 + _chg((feed["hist_vix"] - feed["hist_vix_prev"]) if feed.get("hist_vix_prev") else None, "+.2f", 100))
                if feed.get("hist_vix") else "")
        + "</tr><tr>"
        + _tile(f"ATM vol · {_d(e0['expiry'])}", f"{e0['atm_vol']:.2%}",
                (_chg(dv0, "+.2f", 100, suffix=" vol pts") + " on the day") if dv0 is not None
                else f"{e0['days']:.1f} days to expiry")
        + _tile("Expected move by expiry", f"±{e0['expected_move']:,.0f}",
                f"±{e0['expected_move_pct']:.2%} · {e0['range_lo']:,.0f} to {e0['range_hi']:,.0f}")
        + "</tr>")
    kpis = f'<tr><td style="padding:16px 22px 0" class="pad"><table role="presentation" width="100%" cellspacing="0" cellpadding="0">{tiles}</table></td></tr>'

    bullets = "".join(f'<li style="margin:0 0 8px;font-size:14px;line-height:1.55;color:{INK}" class="ink">{_e(b)}</li>'
                      for b in feed["highlights"])
    sec_high = _section("Today in brief", f'<ul style="margin:0;padding:0 0 0 18px">{bullets}</ul>')

    vol_rows = [[f"<b>{_d(e['expiry'])}</b>", f"{e['days']:.1f}", f"{e['forward']:,.0f}", f"{e['atm_vol']:.2%}",
                 _chg(ch["atm_vol"].get(e["expiry"]), "+.2f", 100), f"{e['rr25'] * 100:+.2f}",
                 f"{e['bf25'] * 100:+.2f}", f"±{e['expected_move']:,.0f}", f"{e['straddle']:,.1f}"]
                for e in feed["expiries"]]
    sec_vol = _section(
        "Volatility by expiry",
        _table(["Expiry", "Days", "Fwd", "ATM vol", "Day chg", "25d RR", "25d BF", "1-sd move", "Straddle"],
               vol_rows, hide=(1, 6, 8))
        + '<div style="height:12px;line-height:12px">&nbsp;</div>' + _img("term", imgs, images, "ATM vol term structure"),
        "Vols in %; day change, RR and BF in vol points. 25d RR: 25-delta call vol minus 25-delta put vol (negative = puts "
        "dearer). 25d BF: average of the two minus ATM vol. 1-sd move: forward × ATM vol × √time. ATM straddle: "
        "call plus put at the strike nearest the forward, in index points." +
        (f" Day changes against {_d(ch['against'])}." if ch.get("against") else ""))

    sec_smile = _section("The smile", _img("smile", imgs, images, "Implied vol smile by expiry"),
                         "Lines: fitted SVI smile within ±3 expected moves; dots: market mid implied vols.")

    rv = [["India VIX (30-day implied, NSE)", f"{feed['hist_vix']:.2%}" if feed.get("hist_vix") else "n/a"],
          [f"ATM implied vol, {_d(e0['expiry'])} expiry", f"{e0['atm_vol']:.2%}"],
          ["Realised vol, last 20 trading days", f"{feed['hist_rv20']:.2%}" if feed.get("hist_rv20") else "n/a"],
          ["Realised vol, last 60 trading days", f"{feed['hist_rv60']:.2%}" if feed.get("hist_rv60") else "n/a"],
          ["Realised vol, EWMA (λ = 0.94)", f"{feed['hist_ewma']:.2%}" if feed.get("hist_ewma") else "n/a"],
          ["Implied minus 20-day realised", f"{feed['vrp'] * 100:+.2f} vol pts" if feed.get("vrp") is not None else "n/a"]]
    sec_rv = _section("Implied vs realised", _table(["Measure", "Value"], rv),
                      f"Realised vols are annualised close-to-close, through the {_d(feed['hist_prev_date'])} close."
                      if feed.get("hist_prev_date") else "")

    prob_rows = [[f"{p['level']:,.0f}", f"{p['sds']:+.0f} sd" if p["sds"] else "at the forward",
                  f"{p['above']:.0%}", f"{p['below']:.0%}"] for p in feed["probabilities"]]
    sec_prob = _section(
        f"Market-implied odds at the {_d(e0['expiry'])} expiry",
        _table([f"{_e(feed['underlying'])} level", "Distance", "Settles above", "Settles below"], prob_rows),
        "Risk-neutral probabilities read from option prices (the slope of call prices across strikes). They "
        "describe what the market is pricing, not a forecast.")

    ref_rows = [[f"<b>{_e(r['label'])}</b>", f"{r['strike']:,.0f}", f"{r['price']:,.2f}", f"{r['vol']:.2%}",
                 f"{r['delta']:+.3f}", f"{r['vega']:.2f}", f"{r['theta']:.2f}",
                 f"{r['lot_value']:,.0f}", _pill(r["status"])] for r in feed["reference_options"]]
    sec_ref = _section(
        f"Reference prices · {_d(e0['expiry'])} expiry",
        _table(["Option", "Strike", "Price", "Vol", "Delta", "Vega", "Theta", "₹/lot", "Check"], ref_rows, hide=(5, 7)),
        f"Our model prices on mid-market vols, with Greeks per unit (vega per vol point, theta per calendar day). "
        f"Lot size {feed['lot_size']}. Check: result of the 13+ automated validation checks on each price.")

    oi = feed.get("open_interest")
    sec_oi = ""
    if oi:
        top = []
        for i in range(3):
            c = oi["top_calls"][i] if i < len(oi["top_calls"]) else None
            p = oi["top_puts"][i] if i < len(oi["top_puts"]) else None
            top.append([f"{c['strike']:,.0f}" if c else "", f"{c['oi'] / 1e5:,.1f} L" if c else "",
                        _chg(c["change"] / 1e5 if c and c["change"] is not None else None, "+,.1f", suffix=" L") if c else "",
                        f"{p['strike']:,.0f}" if p else "", f"{p['oi'] / 1e5:,.1f} L" if p else "",
                        _chg(p["change"] / 1e5 if p and p["change"] is not None else None, "+,.1f", suffix=" L") if p else ""])
        summary = _table(["", "Calls", "Puts", "Put/call"], [
            ["Open interest (units)", f"{oi['call_oi'] / 1e7:,.2f} cr", f"{oi['put_oi'] / 1e7:,.2f} cr",
             f"{oi['pcr']:.2f}" if oi["pcr"] else "n/a"],
            ["Change on the day", _chg(oi["call_oi_change"] / 1e7 if oi["call_oi_change"] is not None else None, "+,.2f", suffix=" cr"),
             _chg(oi["put_oi_change"] / 1e7 if oi["put_oi_change"] is not None else None, "+,.2f", suffix=" cr"),
             f"{oi['pcr_prev']:.2f} prev" if oi.get("pcr_prev") else ""]])
        sec_oi = _section(
            f"Open interest · {_d(e0['expiry'])} expiry",
            summary + '<div style="height:10px;line-height:10px">&nbsp;</div>'
            + _table(["Call strike", "OI", "Change", "Put strike", "OI", "Change"], top,
                     ["left", "right", "right", "left", "right", "right"]),
            "Largest open interest by strike, in lakh units. Open interest shows where positions are held; it is "
            "not a level the index must respect.")

    fut_rows = [[_d(f["expiry"], "%d %b %Y"), f"{f['price']:,.2f}", f"{f['basis']:+,.2f}",
                 f"{f['carry']:.2%}" if f.get("carry") is not None else "n/a"] for f in feed["futures"]]
    rate_rows = [[r["tenor"], f"{r['zero']:.2%}"] for r in feed["rates"]]
    sec_fut = _section(
        "Futures and rates",
        (_table(["Future", "Price", "Basis to spot", "Implied carry p.a."], fut_rows) if fut_rows else "")
        + '<div style="height:10px;line-height:10px">&nbsp;</div>'
        + _table(["MIBOR-OIS tenor", "Zero rate (cont.)"], rate_rows),
        f"Discount curve {_e(feed['curve_label'])} ({feed['curve_age_days']} days old); option forwards come from "
        f"put-call parity, not from the futures.")

    k = feed["checks"]
    health_rows = [[f"<b>{_d(e['expiry'])}</b>", str(e["quotes_kept"]), _pill(e["qa_status"]), f"{e['svi_rmse']:.2f}",
                    _pill(e["svi_status"])] for e in feed["expiries"]]
    sec_health = _section(
        "Data quality",
        _table(["Expiry", "Quotes used", "Chain QA", "Fit error (vol pts)", "Smile fit"], health_rows),
        f"Surface {feed['surface_status']}; calendar arbitrage {'none' if not feed['calendar_violations'] else feed['calendar_violations']}; "
        f"{k['references_released']} of {k['references']} reference prices passed validation. WARN means a check "
        f"found something worth knowing (for example, quotes with no trades today); FAIL would withhold the number.")

    glossary = _section("How to read this", (
        f'<p style="margin:0;font-size:12.5px;line-height:1.6;color:{MUTED}" class="muted">'
        "<b>Implied vol</b>: the volatility that makes the model price equal the market price. "
        "<b>ATM</b>: at the money, the strike nearest the forward. <b>Forward</b>: the index level implied by "
        "call and put prices for that expiry. <b>Delta</b>: price change per 1-point index move; <b>gamma</b>: "
        "change in delta; <b>vega</b>: price change per 1 vol point; <b>theta</b>: price change per day. "
        "<b>Realised vol</b>: how much the index actually moved, annualised.</p>"))

    hello = f"Hi {_e(recipient_name)}, here" if recipient_name else "Here"
    unsub = (f'<a href="{_e(unsubscribe_url)}" style="color:{MUTED};text-decoration:underline" class="muted">'
             f"Unsubscribe</a>" if unsubscribe_url else "Reply with UNSUBSCRIBE to stop these emails")
    footer = f"""
<tr><td style="padding:26px 28px 6px" class="pad">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:#fff7ed;border:1px solid #fed7aa;border-radius:10px" class="disc">
  <tr><td style="padding:14px 16px;font-size:12px;line-height:1.6;color:#7c2d12" class="disctext">
  <b>Not investment advice.</b> {_e(DISCLAIMER)}</td></tr></table>
</td></tr>
<tr><td style="padding:14px 28px 28px;font-size:11.5px;line-height:1.6;color:{MUTED}" class="pad muted">
  Data: {_e(src)} at {as_of.strftime('%H:%M')} IST; NSE India VIX and index closes; FBIL MIBOR-OIS. Prices are
  model values on mid-market vols, not quotes you can trade on. You receive this because you subscribed to the
  {_e(feed['underlying'])} options daily feed. {unsub}.
</td></tr>"""

    style = f"""
<style>
  :root {{ color-scheme: light dark; supported-color-schemes: light dark; }}
  @media (prefers-color-scheme: dark) {{
    .bg {{ background:#0b1220 !important; }}
    .card, .card td {{ background:#111a2e !important; }}
    .zebra, .zebra td {{ background:#0f1729 !important; }}
    .tile {{ background:#0f1729 !important; }}
    .ink {{ color:#e2e8f0 !important; }}
    .muted {{ color:#94a3b8 !important; }}
    .line {{ border-color:#22304a !important; }}
    .accent {{ color:#a5b4fc !important; border-color:#818cf8 !important; }}
    .up {{ color:#4ade80 !important; }} .down {{ color:#f87171 !important; }}
    .disc {{ background:#2a1a0e !important; border-color:#7c2d12 !important; }}
    .disctext {{ color:#fed7aa !important; }}
  }}
  @media only screen and (max-width:620px) {{ .pad {{ padding-left:12px !important; padding-right:12px !important; }}
    .t {{ font-size:10.5px !important; padding:5px 3px !important; letter-spacing:0 !important; }}
    .hm {{ display:none !important; }} .outer {{ padding:0 !important; }} }}
</style>"""

    preheader = _e(feed["highlights"][0] if feed["highlights"] else subject(feed))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark"><meta name="supported-color-schemes" content="light dark">
<title>{_e(subject(feed))}</title>{style}</head>
<body style="margin:0;padding:0;background:{BG};font-family:{FONT}" class="bg">
<div style="display:none;max-height:0;overflow:hidden;opacity:0">{preheader}</div>
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:{BG}" class="bg"><tr><td align="center" style="padding:24px 10px" class="outer">
<table role="presentation" width="640" cellspacing="0" cellpadding="0" style="width:100%;max-width:640px;background:{CARD};border-radius:14px;overflow:hidden;border:1px solid {LINE}" class="card line">
<tr><td style="background:{HEAD};padding:24px 28px" class="pad">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0"><tr>
    <td style="font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#a5b4fc;font-weight:700">{_e(feed['underlying'])} options daily</td>
    <td align="right" style="font-size:12px;color:#94a3b8">{_pill(feed['surface_status'])}</td></tr>
  <tr><td colspan="2" style="padding-top:8px;font-size:24px;font-weight:700;color:#ffffff">{as_of.strftime('%A, %d %B %Y')}</td></tr>
  <tr><td colspan="2" style="padding-top:4px;font-size:13px;color:#cbd5e1">Snapshot {as_of.strftime('%H:%M')} IST · {_e(src)}</td></tr></table>
</td></tr>
<tr><td style="padding:18px 28px 0;font-size:14px;line-height:1.55;color:{INK}" class="pad ink">{hello} is today's summary of the {_e(feed['underlying'])} options market: volatility, skew, expected moves, reference prices and open interest. Information only, not advice.</td></tr>
{kpis}{sec_high}{sec_vol}{sec_smile}{sec_rv}{sec_prob}{sec_ref}{sec_oi}{sec_fut}{sec_health}{glossary}{footer}
</table></td></tr></table></body></html>"""

    return EmailMessageParts(subject=subject(feed), html=page, text=render_text(feed, unsubscribe_url), images=imgs)


def render_text(feed: dict, unsubscribe_url: str = "") -> str:
    e0 = feed["expiries"][0]
    lines = [f"{feed['underlying']} OPTIONS DAILY - {_d(feed['date'], '%d %b %Y')} (snapshot {feed['as_of'][11:16]} IST)",
             "Information only, not investment advice.", ""]
    lines += [f"- {h}" for h in feed["highlights"]] + ["", "VOLATILITY BY EXPIRY"]
    lines.append(f"{'Expiry':<8}{'Days':>6}{'Forward':>10}{'ATM vol':>9}{'25d RR':>8}{'1-sd move':>11}")
    for e in feed["expiries"]:
        lines.append(f"{_d(e['expiry']):<8}{e['days']:>6.1f}{e['forward']:>10,.0f}{e['atm_vol']:>9.2%}"
                     f"{e['rr25'] * 100:>+8.2f}{'±' + format(e['expected_move'], ',.0f'):>11}")
    lines += ["", f"MARKET-IMPLIED ODDS AT THE {_d(e0['expiry']).upper()} EXPIRY (risk-neutral, not a forecast)"]
    lines += [f"  above {p['level']:,.0f}: {p['above']:.0%}" for p in feed["probabilities"]]
    lines += ["", "REFERENCE PRICES (per unit)"]
    lines += [f"  {r['label']:<14}{r['strike']:>8,.0f}  {r['price']:>8.2f}  vol {r['vol']:.2%}  delta {r['delta']:+.3f}  "
              f"[{r['status']}]" for r in feed["reference_options"]]
    lines += ["", "NOT INVESTMENT ADVICE. " + DISCLAIMER, ""]
    lines.append(f"Unsubscribe: {unsubscribe_url}" if unsubscribe_url else "Reply UNSUBSCRIBE to stop these emails.")
    return "\n".join(lines)
