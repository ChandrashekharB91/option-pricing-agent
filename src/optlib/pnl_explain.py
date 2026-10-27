"""Day-over-day P&L explain from NSE end-of-day prices.

For each position: actual P&L = (close on day 1 - close on day 0) x quantity. It is split with day-0
Black-76 Greeks into
    delta  = delta x dF                 gamma = 1/2 gamma x dF^2
    vega   = vega x d(vol)              theta = theta x dt   (calendar time passing)
    cross  = vanna x dF x d(vol) + 1/2 volga x d(vol)^2
    rate   = rho x d(r)
    unexplained = actual - all of the above
where F is the parity-implied forward of the option's expiry on each day and vol the implied vol from that
day's close. The unexplained share (of the gross explained P&L) is checked against the spec: warn above
5%, fail above 15%.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

import numpy as np

from .black import black76_greeks
from .curves import DiscountCurve
from .data.bhavcopy import BhavRow
from .data.models import IST, NSE_CLOSE, year_fraction
from .iv import implied_vol_black76

WARN_SHARE, FAIL_SHARE = 0.05, 0.15
COMPONENTS = ("delta", "gamma", "vega", "theta", "cross", "rate")


def _close_time(day: date) -> datetime:
    return datetime.combine(day, NSE_CLOSE, tzinfo=IST)


def eod_forward(rows: list[BhavRow], symbol: str, expiry: date, df: float, n: int = 7) -> float:
    """Median parity forward from the n strikes nearest the index close with both call and put traded."""
    opts = [r for r in rows if r.symbol == symbol and r.kind == "option" and r.expiry == expiry]
    if not opts:
        raise ValueError(f"no {symbol} options for {expiry} in the end-of-day file")
    spot = opts[0].underlying
    by_k: dict[float, dict] = {}
    for r in opts:
        by_k.setdefault(r.strike, {})[r.option_type] = r
    pairs = [(k, d["call"], d["put"]) for k, d in by_k.items()
             if "call" in d and "put" in d and d["call"].volume > 0 and d["put"].volume > 0
             and d["call"].close > 0 and d["put"].close > 0]
    if len(pairs) < 3:
        raise ValueError(f"too few traded call/put pairs for {symbol} {expiry}")
    near = sorted(pairs, key=lambda p: abs(p[0] - spot))[:n]
    return statistics.median(k + (c.close - p.close) / df for k, c, p in near)


@dataclass
class LegExplain:
    label: str
    quantity: float
    close0: float
    close1: float
    forward0: float
    forward1: float
    vol0: float
    vol1: float
    traded1: bool
    actual: float
    delta: float
    gamma: float
    vega: float
    theta: float
    cross: float
    rate: float
    unexplained: float


@dataclass
class PnLExplain:
    day0: str
    day1: str
    legs: list[LegExplain]
    totals: dict
    unexplained_share: float
    status: str
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def explain(positions: list[dict], rows0: list[BhavRow], rows1: list[BhavRow], curve0: DiscountCurve,
            curve1: DiscountCurve) -> PnLExplain:
    """positions: [{"underlying", "expiry", "strike", "option_type", "lots"}]; lot size from the day-0 file."""
    if not rows0 or not rows1:
        raise ValueError("end-of-day data missing for one of the days")
    d0, d1 = rows0[0].trade_date, rows1[0].trade_date
    if d1 <= d0:
        raise ValueError("day 1 must be after day 0")
    t0, t1 = _close_time(d0), _close_time(d1)
    idx0 = {(r.symbol, r.expiry, r.strike, r.option_type): r for r in rows0 if r.kind == "option"}
    idx1 = {(r.symbol, r.expiry, r.strike, r.option_type): r for r in rows1 if r.kind == "option"}
    fwd_cache: dict = {}
    legs, notes = [], []
    for p in positions:
        sym = str(p.get("underlying", "NIFTY")).upper()
        exp = p["expiry"] if isinstance(p["expiry"], date) else date.fromisoformat(str(p["expiry"])[:10])
        key = (sym, exp, float(p["strike"]), str(p["option_type"]).lower())
        a, b = idx0.get(key), idx1.get(key)
        if a is None or b is None:
            raise ValueError(f"{sym} {exp} {p['strike']} {p['option_type']} not in both end-of-day files")
        if exp <= d1:
            raise ValueError(f"{sym} {exp} expires on or before {d1}")
        qty = float(p["lots"]) * a.lot_size
        T0, T1 = year_fraction(t0, exp), year_fraction(t1, exp)
        r0 = curve0.rate_for_option(t0, _close_time(exp))
        r1 = curve1.rate_for_option(t1, _close_time(exp))
        for (day, rows, T, r) in ((d0, rows0, T0, r0), (d1, rows1, T1, r1)):
            if (sym, exp, day) not in fwd_cache:
                fwd_cache[(sym, exp, day)] = eod_forward(rows, sym, exp, math.exp(-r * T))
        F0, F1 = fwd_cache[(sym, exp, d0)], fwd_cache[(sym, exp, d1)]
        v0 = float(implied_vol_black76(a.close, F0, key[2], T0, r0, key[3]))
        v1 = float(implied_vol_black76(b.close, F1, key[2], T1, r1, key[3]))
        if not (math.isfinite(v0) and math.isfinite(v1)):
            raise ValueError(f"no implied vol for {key[2]:,.0f} {key[3]} (close below intrinsic?)")
        g = black76_greeks(F0, key[2], T0, r0, v0, key[3])
        dF, dv, dt, dr = F1 - F0, v1 - v0, T0 - T1, r1 - r0
        comp = {"delta": g.delta * dF, "gamma": 0.5 * g.gamma * dF ** 2, "vega": g.vega * dv,
                "theta": g.theta * dt, "cross": g.vanna * dF * dv + 0.5 * g.volga * dv ** 2, "rate": g.rho * dr}
        comp = {k: float(v) * qty for k, v in comp.items()}
        actual = (b.close - a.close) * qty
        if b.volume == 0:
            notes.append(f"{key[2]:,.0f} {key[3]} did not trade on {d1}: its close may be stale")
        legs.append(LegExplain(label=f"{key[2]:,.0f} {key[3]} {exp}", quantity=qty, close0=a.close, close1=b.close,
                               forward0=F0, forward1=F1, vol0=v0, vol1=v1, traded1=b.volume > 0, actual=actual,
                               **comp, unexplained=actual - sum(comp.values())))
    totals = {k: sum(getattr(l, k) for l in legs) for k in ("actual", *COMPONENTS, "unexplained")}
    gross = sum(abs(totals[k]) for k in COMPONENTS)
    share = abs(totals["unexplained"]) / gross if gross > 0 else 0.0
    status = "FAIL" if share > FAIL_SHARE else ("WARN" if share > WARN_SHARE else "PASS")
    notes.insert(0, "Large one-day moves leave more unexplained: a Taylor expansion with day-0 Greeks "
                    "cannot fully capture them; full revaluation is in the actual P&L.")
    return PnLExplain(day0=d0.isoformat(), day1=d1.isoformat(), legs=legs, totals=totals,
                      unexplained_share=share, status=status, notes=notes)
