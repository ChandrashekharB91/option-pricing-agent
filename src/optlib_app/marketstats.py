"""Desk statistics: implied-volatility rank and the open-interest profile of an expiry.

IV rank uses India VIX (NSE's 30-day implied volatility of NIFTY) over the last year, the standard proxy when a
long history of our own fitted ATM vols is not available:
    rank       = (today - 1y low) / (1y high - 1y low)        0% = cheapest of the year, 100% = dearest
    percentile = share of the last year's closes below today

Open interest per strike from the option chain: put/call ratio, the strikes with the largest call and put OI,
OI change on the day, and max pain (the settlement level at which option buyers as a whole would receive the
least, a common positioning reference, not a forecast).
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np

YEAR = 252


def iv_rank(history, as_of: date, atm_vol: Optional[float] = None) -> dict:
    dates = [d if isinstance(d, date) else date.fromisoformat(str(d)) for d in history.dates]
    vix = np.asarray(history.series("INDIAVIX"), float)
    keep = [i for i, d in enumerate(dates) if d <= as_of and np.isfinite(vix[i])]
    if len(keep) < 20:
        return {"available": False, "reason": "not enough India VIX history"}
    idx = keep[-YEAR:]
    s = vix[idx] / 100.0
    cur = float(s[-1])
    lo, hi = float(s.min()), float(s.max())
    out = {
        "available": True, "as_of": dates[idx[-1]].isoformat(), "vix": cur, "low": lo, "high": hi,
        "mean": float(s.mean()), "rank": (cur - lo) / (hi - lo) if hi > lo else 0.5,
        "percentile": float((s[:-1] < cur).mean()) if len(s) > 1 else 0.5,
        "days": len(s), "dates": [dates[i].isoformat() for i in idx], "series": [float(v) for v in s],
        "change_1d": float(s[-1] - s[-2]) if len(s) > 1 else None,
    }
    if atm_vol is not None:
        out["atm_vol"] = float(atm_vol)
        out["atm_vs_vix"] = float(atm_vol) - cur
        out["atm_rank"] = min(1.0, max(0.0, (float(atm_vol) - lo) / (hi - lo))) if hi > lo else None
    out["label"] = ("high" if out["rank"] >= 0.7 else "low" if out["rank"] <= 0.3 else "middle")
    return out


def max_pain(strikes: list[float], call_oi: dict, put_oi: dict) -> Optional[float]:
    """Settlement strike that minimises the total intrinsic value paid to option holders."""
    if not strikes:
        return None
    ks = np.array(sorted(strikes))
    c = np.array([call_oi.get(k, 0) for k in ks], float)
    p = np.array([put_oi.get(k, 0) for k in ks], float)
    pain = [float((c * np.maximum(s - ks, 0)).sum() + (p * np.maximum(ks - s, 0)).sum()) for s in ks]
    return float(ks[int(np.argmin(pain))])


def oi_profile(chain, forward: float, width_sd: Optional[float] = None, sd: Optional[float] = None) -> dict:
    """OI by strike for one expiry. ``width_sd`` and ``sd`` limit the strikes returned for charts (all strikes are
    used for the totals and max pain)."""
    calls = {q.strike: q for q in chain.quotes if q.option_type == "call"}
    puts = {q.strike: q for q in chain.quotes if q.option_type == "put"}
    strikes = sorted(set(calls) | set(puts))
    c_oi = {k: float(calls[k].oi) for k in calls}
    p_oi = {k: float(puts[k].oi) for k in puts}
    c_prev = {k: float(calls[k].prev_oi) for k in calls}
    p_prev = {k: float(puts[k].prev_oi) for k in puts}
    tc, tp = sum(c_oi.values()), sum(p_oi.values())
    pc, pp = sum(c_prev.values()), sum(p_prev.values())
    rows = []
    for k in strikes:
        if width_sd and sd and abs(k - forward) > width_sd * sd:
            continue
        rows.append({"strike": k, "call_oi": c_oi.get(k, 0.0), "put_oi": p_oi.get(k, 0.0),
                     "call_change": (c_oi.get(k, 0.0) - c_prev.get(k, 0.0)) if c_prev.get(k) else None,
                     "put_change": (p_oi.get(k, 0.0) - p_prev.get(k, 0.0)) if p_prev.get(k) else None})

    def top(d):
        return [{"strike": k, "oi": v} for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:3] if v > 0]

    return {
        "expiry": str(chain.expiry), "forward": forward, "call_oi": tc, "put_oi": tp,
        "pcr": tp / tc if tc else None, "pcr_prev": pp / pc if pc and pp else None,
        "call_change": tc - pc if pc else None, "put_change": tp - pp if pp else None,
        "max_pain": max_pain(strikes, c_oi, p_oi), "top_calls": top(c_oi), "top_puts": top(p_oi), "rows": rows,
    }
