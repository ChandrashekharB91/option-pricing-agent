"""Daily market feed: the day's option-market numbers, assembled for the email digest.

Everything here is descriptive analytics from the same PricingService the dashboard uses: index level,
implied and realised volatility, skew, expected moves, market-implied probabilities, reference option prices
with Greeks, open interest and data-quality checks. It contains no trade ideas: the strategy recommender stays
on the internal dashboard.

    feed = build_feed(PricingService())          # dict of values
    record(feed, path)                           # keep today's values for tomorrow's day-on-day changes
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import norm

from optlib.black import black76_price
from optlib.pricing import price_option
from optlib.recommender import ewma_vol
from optlib.validator import validate

DISCLAIMER = ("This email is market information and analytics for education and research only. It is not "
              "investment advice, a research report or a recommendation to buy, sell or hold any security or "
              "derivative, and it does not consider anyone's objectives or circumstances. Figures come from "
              "third-party market data and models that can be wrong, delayed or incomplete. Options carry a high "
              "risk of loss. Consult a SEBI-registered adviser before acting. The author and the firm accept no "
              "responsibility for any loss or gain from using it.")

PROB_SDS = (-2.0, -1.0, 0.0, 1.0, 2.0)  # probability levels, in expected moves from the forward
RATE_TENORS = ("O/N", "1M", "3M", "6M", "1Y")


def _round_to(x: float, step: float = 50.0) -> float:
    return float(round(x / step) * step)


def delta_strike(smile, delta: float, option_type: str) -> float:
    """Strike whose Black-76 forward delta equals ``delta`` (calls 0..1, puts -1..0), on the smile's own vols."""
    F, T = smile.forward, smile.time_to_expiry
    target = delta if option_type == "call" else 1.0 + delta  # N(d1) for the put
    z = norm.ppf(target)
    sig = smile.slice.atm_vol()
    K = F
    for _ in range(50):
        K_new = F * math.exp(-z * sig * math.sqrt(T) + 0.5 * sig * sig * T)
        sig = float(smile.fit.vol(K_new))
        if abs(K_new - K) < 1e-6:
            break
        K = K_new
    return float(K_new)


def prob_above(smile, K: float, h: float = 10.0) -> float:
    """Market-implied (risk-neutral) probability that the index settles above K: minus the strike derivative of
    the undiscounted call price on the fitted smile (central difference)."""
    F, T = smile.forward, smile.time_to_expiry

    def c(k):
        return float(black76_price(F, k, T, 0.0, float(smile.fit.vol(k)), "call"))

    return float(min(1.0, max(0.0, -(c(K + h) - c(K - h)) / (2 * h))))


def _skew(smile) -> dict:
    kc, kp = delta_strike(smile, 0.25, "call"), delta_strike(smile, -0.25, "put")
    vc, vp, va = float(smile.fit.vol(kc)), float(smile.fit.vol(kp)), float(smile.fit.vol(smile.forward))
    return {"call25_strike": kc, "put25_strike": kp, "call25_vol": vc, "put25_vol": vp,
            "rr25": vc - vp, "bf25": 0.5 * (vc + vp) - va}


def _realised(closes: np.ndarray, days: int) -> Optional[float]:
    c = closes[np.isfinite(closes)]
    if c.size < days + 1:
        return None
    r = np.diff(np.log(c[-(days + 1):]))
    return float(np.std(r, ddof=1) * math.sqrt(252))


def _history_block(svc, as_of: date) -> dict:
    out = {"prev_close": None, "prev_date": None, "vix": None, "vix_prev": None, "vix_date": None,
           "rv20": None, "rv60": None, "ewma": None}
    try:
        h = svc.history()
    except Exception:  # noqa: BLE001 - history is optional for the feed
        return out
    dates = [d if isinstance(d, date) else date.fromisoformat(str(d)) for d in h.dates]
    idx = [i for i, d in enumerate(dates) if d < as_of]
    if not idx:
        return out
    last = idx[-1]
    nifty = np.asarray(h.series("NIFTY"), float)[: last + 1]
    out.update(prev_close=float(nifty[-1]), prev_date=dates[last].isoformat(),
               rv20=_realised(nifty, 20), rv60=_realised(nifty, 60), ewma=ewma_vol(nifty))
    try:
        vix = np.asarray(h.series("INDIAVIX"), float)[: last + 1]
        ok = np.where(np.isfinite(vix))[0]
        if ok.size:
            out.update(vix=float(vix[ok[-1]]) / 100, vix_date=dates[ok[-1]].isoformat(),
                       vix_prev=float(vix[ok[-2]]) / 100 if ok.size > 1 else None)
    except (KeyError, ValueError):
        pass
    return out


def _open_interest(chain) -> dict:
    calls = [q for q in chain.quotes if q.option_type == "call"]
    puts = [q for q in chain.quotes if q.option_type == "put"]
    c_oi, p_oi = sum(q.oi for q in calls), sum(q.oi for q in puts)
    c_prev, p_prev = sum(q.prev_oi for q in calls), sum(q.prev_oi for q in puts)

    def top(qs):
        return [{"strike": q.strike, "oi": q.oi, "change": q.oi - q.prev_oi if q.prev_oi else None}
                for q in sorted(qs, key=lambda q: q.oi, reverse=True)[:3] if q.oi > 0]

    return {"call_oi": c_oi, "put_oi": p_oi, "pcr": p_oi / c_oi if c_oi else None,
            "pcr_prev": (p_prev / c_prev) if c_prev and p_prev else None,
            "call_oi_change": c_oi - c_prev if c_prev else None, "put_oi_change": p_oi - p_prev if p_prev else None,
            "top_calls": top(calls), "top_puts": top(puts)}


def _reference_options(svc, surf, smile, lot: int) -> list[dict]:
    exp = date.fromisoformat(smile.slice.expiry)
    atm = _round_to(smile.forward)
    sk = _skew(smile)
    legs = [("25d put", _round_to(sk["put25_strike"]), "put"), ("ATM put", atm, "put"),
            ("ATM call", atm, "call"), ("25d call", _round_to(sk["call25_strike"]), "call")]
    rows = []
    for label, K, typ in legs:
        try:
            res = price_option(surf, K, exp, typ, quantity=lot)
            rep = validate(res, surf)
        except ValueError:
            continue
        g = res.greeks
        rows.append({"label": label, "strike": K, "type": typ, "price": res.price, "vol": res.vol,
                     "vol_source": res.vol_source, "delta": g.get("delta"), "gamma": g.get("gamma"),
                     "vega": g.get("vega"), "theta": g.get("theta"), "lot_value": res.price * lot,
                     "status": rep.status, "released": rep.released})
    return rows


def build_feed(svc, underlying: str = "NIFTY", max_expiries: int = 4, history_path: Optional[Path] = None) -> dict:
    surf = svc.surface(underlying, max_expiries=max_expiries)
    as_of = surf.as_of if isinstance(surf.as_of, datetime) else datetime.fromisoformat(str(surf.as_of))
    lot = svc.lot_size(underlying)
    hist = _history_block(svc, as_of.date())
    smiles = surf.smiles
    near = smiles[0]

    expiries = []
    for s in smiles:
        T = s.time_to_expiry
        atm_vol = s.slice.atm_vol()
        sd = s.forward * atm_vol * math.sqrt(T)
        sk = _skew(s)
        atm_k = _round_to(s.forward)
        straddle = (float(black76_price(s.forward, atm_k, T, s.slice.rate, float(s.vol(atm_k)[0]), "call"))
                    + float(black76_price(s.forward, atm_k, T, s.slice.rate, float(s.vol(atm_k)[0]), "put")))
        expiries.append({
            "expiry": s.slice.expiry, "days": T * 365, "forward": s.forward, "basis": s.forward - surf.spot,
            "atm_vol": atm_vol, "expected_move": sd, "expected_move_pct": sd / s.forward,
            "range_lo": s.forward - sd, "range_hi": s.forward + sd, "atm_strike": atm_k, "straddle": straddle,
            **sk, "svi_status": s.fit.status, "svi_rmse": s.fit.rmse * 100, "qa_status": s.slice.qa.status,
            "quotes_kept": int(len(s.slice.strikes)),
        })

    e0 = expiries[0]
    probs = []
    for n in PROB_SDS:
        K = _round_to(near.forward + n * e0["expected_move"])
        probs.append({"level": K, "sds": n, "above": prob_above(near, K), "below": 1 - prob_above(near, K)})

    try:
        oi = _open_interest(svc.source().option_chain(underlying, date.fromisoformat(near.slice.expiry)))
    except Exception:  # noqa: BLE001 - OI is a nice-to-have
        oi = None
    try:
        futures = [{"expiry": f.expiry.isoformat(), "price": f.mid if math.isfinite(f.mid) else f.last,
                    "basis": (f.mid if math.isfinite(f.mid) else f.last) - surf.spot, "oi": f.oi}
                   for f in svc.source().futures(underlying)]
        for f in futures:
            days = (date.fromisoformat(f["expiry"]) - as_of.date()).days
            f["days"] = days
            f["carry"] = (math.log(f["price"] / surf.spot) * 365 / days) if days > 0 and f["price"] > 0 else None
    except Exception:  # noqa: BLE001
        futures = []

    c = surf.curve
    rates = []
    for tenor, (d, _df) in c.pillars.items():
        if tenor not in RATE_TENORS:
            continue
        rates.append({"tenor": tenor, "maturity": d.isoformat() if hasattr(d, "isoformat") else str(d),
                      "zero": c.zero_rate(c.time(d))})

    refs = _reference_options(svc, surf, near, lot)
    feed = {
        "underlying": surf.underlying, "as_of": as_of.isoformat(), "date": as_of.date().isoformat(),
        "data_mode": svc.mode, "lot_size": lot, "spot": surf.spot, **{f"hist_{k}": v for k, v in hist.items()},
        "change": surf.spot - hist["prev_close"] if hist["prev_close"] else None,
        "change_pct": surf.spot / hist["prev_close"] - 1 if hist["prev_close"] else None,
        "surface_status": surf.status, "calendar_violations": len(surf.calendar),
        "curve_label": c.label, "curve_age_days": c.age_days(),
        "expiries": expiries, "probabilities": probs, "open_interest": oi, "futures": futures, "rates": rates,
        "reference_options": refs, "lot_note": f"Lot size {lot}; Greeks per unit (vega per vol point, theta per day).",
        "smiles": [_smile_points(s) for s in smiles],
    }
    feed["vrp"] = (e0["atm_vol"] - hist["rv20"]) if hist["rv20"] else None
    feed["previous"] = previous(history_path, feed["date"]) if history_path else None
    feed["changes"] = changes(feed, feed["previous"])
    feed["highlights"] = highlights(feed)
    feed["checks"] = {
        "expiries_pass": sum(e["svi_status"] == "PASS" for e in expiries), "expiries": len(expiries),
        "qa_pass": sum(e["qa_status"] == "PASS" for e in expiries),
        "references_released": sum(r["released"] for r in refs), "references": len(refs),
    }
    return feed


def _smile_points(s) -> dict:
    """SVI vols within the fit range plus the market mids there, for the smile chart."""
    lo, hi = s.fit.fit_range
    lo, hi = max(lo, -0.15), min(hi, 0.15)
    k = np.linspace(lo, hi, 60)
    K = s.forward * np.exp(k)
    q = s.slice
    km = q.log_moneyness
    m = (km >= lo) & (km <= hi)
    return {"expiry": q.expiry, "days": s.time_to_expiry * 365, "forward": s.forward,
            "strikes": K.tolist(), "vols": [float(v) for v in s.fit.vol(K)],
            "mkt_strikes": q.strikes[m].tolist(), "mkt_vols": q.iv_mid[m].tolist()}


# ------------------------------------------------------------------ day-on-day memory

def snapshot_values(feed: dict) -> dict:
    return {"date": feed["date"], "as_of": feed["as_of"], "spot": feed["spot"], "vix": feed["hist_vix"],
            "atm_vols": {e["expiry"]: e["atm_vol"] for e in feed["expiries"]},
            "rr25": {e["expiry"]: e["rr25"] for e in feed["expiries"]},
            "pcr": (feed["open_interest"] or {}).get("pcr")}


def record(feed: dict, path: Path) -> None:
    """Keep one line per date (the latest run of the day wins)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in _read(path) if r.get("date") != feed["date"]]
    rows.append(snapshot_values(feed))
    rows.sort(key=lambda r: r["date"])
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _read(path: Optional[Path]) -> list[dict]:
    if not path or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def previous(path: Optional[Path], day: str) -> Optional[dict]:
    rows = [r for r in _read(path) if r["date"] < day]
    return rows[-1] if rows else None


def changes(feed: dict, prev: Optional[dict]) -> dict:
    """Day-on-day changes against the last recorded feed (same expiry for vols and skew)."""
    out = {"against": prev["date"] if prev else None, "atm_vol": {}, "rr25": {}, "pcr": None}
    if not prev:
        return out
    for e in feed["expiries"]:
        if e["expiry"] in prev.get("atm_vols", {}):
            out["atm_vol"][e["expiry"]] = e["atm_vol"] - prev["atm_vols"][e["expiry"]]
        if e["expiry"] in prev.get("rr25", {}):
            out["rr25"][e["expiry"]] = e["rr25"] - prev["rr25"][e["expiry"]]
    pcr = (feed["open_interest"] or {}).get("pcr")
    if pcr is not None and prev.get("pcr") is not None:
        out["pcr"] = pcr - prev["pcr"]
    return out


def highlights(feed: dict) -> list[str]:
    """Plain factual sentences for the top of the email. Descriptive only: no direction, no advice."""
    out = []
    e0 = feed["expiries"][0]
    if feed["change"] is not None:
        out.append(f"{feed['underlying']} at {feed['spot']:,.2f}, {feed['change']:+,.2f} points "
                   f"({feed['change_pct']:+.2%}) from the {_d(feed['hist_prev_date'])} close of "
                   f"{feed['hist_prev_close']:,.2f}.")
    dv = feed["changes"]["atm_vol"].get(e0["expiry"])
    move = f", {dv * 100:+.2f} vol pts on the day" if dv is not None else ""
    out.append(f"{_d(e0['expiry'])} expiry: ATM implied vol {e0['atm_vol']:.2%}{move}; the options price a move of "
               f"about ±{e0['expected_move']:,.0f} points (±{e0['expected_move_pct']:.2%}) by expiry, "
               f"a one-standard-deviation range of {e0['range_lo']:,.0f} to {e0['range_hi']:,.0f}.")
    if feed["vrp"] is not None:
        rel = "above" if feed["vrp"] >= 0 else "below"
        out.append(f"Weekly implied vol is {abs(feed['vrp']) * 100:.1f} vol pts {rel} 20-day realised vol "
                   f"({feed['hist_rv20']:.1%}).")
    out.append(f"Skew: 25-delta puts trade {abs(e0['rr25']) * 100:.2f} vol pts "
               f"{'above' if e0['rr25'] < 0 else 'below'} 25-delta calls on the {_d(e0['expiry'])} expiry.")
    oi = feed["open_interest"]
    if oi and oi["pcr"] is not None:
        prev = f" ({oi['pcr_prev']:.2f} at the previous close)" if oi.get("pcr_prev") else ""
        calls = ", ".join(f"{t['strike']:,.0f}" for t in oi["top_calls"][:2])
        puts = ", ".join(f"{t['strike']:,.0f}" for t in oi["top_puts"][:2])
        out.append(f"Put/call open-interest ratio {oi['pcr']:.2f}{prev}; largest call OI at {calls}, "
                   f"largest put OI at {puts}.")
    if len(feed["expiries"]) > 1:
        last = feed["expiries"][-1]
        shape = "upward" if last["atm_vol"] > e0["atm_vol"] else "downward"
        out.append(f"Term structure slopes {shape}: {e0['atm_vol']:.1%} for {e0['days']:.0f} days to "
                   f"{last['atm_vol']:.1%} for {last['days']:.0f} days.")
    return out


def _d(iso: str) -> str:
    return date.fromisoformat(iso).strftime("%d %b")
