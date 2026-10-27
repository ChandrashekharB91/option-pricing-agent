"""P&L projector: what a strategy makes or loses across index levels, dates and views of the market.

Inputs are a strategy, a ``MarketState`` (forward, rate, sticky-strike smile, live bid/ask, lot size)
and one or more ``Distribution``s of the forward at the horizon. Entry is at executable prices (buy at
the ask, sell at the bid) plus all charges; exit before expiry is at model value less half the entry
spread and exit charges; at expiry options settle at intrinsic value with STT on exercised longs.

Views (``Distribution``):
  market  risk-neutral density read off the SVI smile (Breeden-Litzenberger); under it every strategy
          is worth about zero before costs, so it is the honest baseline
  model   lognormal centred on today's forward with an EWMA forecast of realised volatility
  user    lognormal whose mean is the user's target level, at the user's volatility
  blend   half user, half model
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Callable, Optional

import numpy as np
from scipy.stats import norm

from .black import black76_greeks, black76_price
from .charges import Charges
from .data.models import IST, nse_expiry_instant
from .margin import MarginParams, estimate_margin
from .strategies import Strategy, StrategyLeg

_YEAR = 365.0 * 86400


class NotQuoted(ValueError):
    """A leg has no live two-sided quote, so the strategy cannot be entered at a known price."""


@dataclass
class MarketState:
    underlying: str
    as_of: datetime
    expiry: date
    T: float  # years from as_of to the 15:30 IST expiry
    forward: float
    rate: float
    spot: float
    lot_size: int
    atm_vol: float
    vol_at: Callable[[float], float]  # sticky-strike implied vol at expiry
    svi_w: Callable[[np.ndarray], np.ndarray]  # SVI total variance at T, by log-moneyness
    quotes: dict  # (strike, "call"|"put") -> (bid, ask)
    future: Optional[tuple] = None  # (bid, ask) of the future for this expiry
    realised_vol: Optional[float] = None  # EWMA forecast from index history
    history_note: str = ""

    @property
    def is_expiry_day(self) -> bool:
        return self.as_of.date() == self.expiry

    @property
    def sd(self) -> float:
        """Expected move to expiry in index points."""
        return self.forward * self.atm_vol * math.sqrt(self.T)

    def years_to(self, horizon: date) -> float:
        end = nse_expiry_instant(horizon) if horizon == self.expiry else datetime.combine(horizon, time(15, 30), IST)
        return min(max((end - self.as_of).total_seconds() / _YEAR, 0.0), self.T)

    def value(self, leg: StrategyLeg, F, t: float, dvol: float = 0.0):
        """Per-unit value of one long leg at forward F with t years left and the smile shifted by dvol."""
        F = np.asarray(F, float)
        if leg.instrument == "future":
            return F
        K = leg.strike
        if t <= 1e-10:
            return np.maximum(F - K, 0.0) if leg.instrument == "call" else np.maximum(K - F, 0.0)
        vol = max(self.vol_at(K) + dvol, 1e-4)
        return black76_price(F, K, t, self.rate, vol, leg.instrument)

    def entry(self, leg: StrategyLeg) -> dict:
        """Executable price (buy at ask, sell at bid), mid and spread for one leg."""
        if leg.instrument == "future":
            if not self.future or any(math.isnan(x) for x in self.future):
                raise NotQuoted(f"no two-sided quote for the {self.expiry} future")
            bid, ask = self.future
        else:
            q = self.quotes.get((leg.strike, leg.instrument))
            if q is None or any(math.isnan(x) for x in q) or q[1] < q[0] or q[0] <= 0:
                raise NotQuoted(f"no two-sided quote for {leg.strike:,.0f} {leg.instrument}")
            bid, ask = q
        return {"price": ask if leg.side > 0 else bid, "mid": 0.5 * (bid + ask), "spread": ask - bid,
                "bid": bid, "ask": ask}


# ------------------------------------------------------------------ distributions

@dataclass
class Distribution:
    name: str
    F: np.ndarray  # forward at the horizon
    p: np.ndarray  # probabilities, sum to 1
    detail: str = ""

    def mean(self) -> float:
        return float(np.sum(self.p * self.F))

    def quantile(self, q: float) -> float:
        c = np.cumsum(self.p)
        return float(np.interp(q, c, self.F))

    def to_dict(self) -> dict:
        return {"name": self.name, "mean": self.mean(), "p5": self.quantile(0.05), "p50": self.quantile(0.5),
                "p95": self.quantile(0.95), "detail": self.detail}


def horizon_grid(state: MarketState, t: float, vols=(), n: int = 1601) -> np.ndarray:
    s = max([state.atm_vol, *[v for v in vols if v]]) * math.sqrt(max(t, 1 / 365))
    half = max(7.0 * s, 0.03)
    return state.forward * np.exp(np.linspace(-half, half, n))


def _normalise(p) -> np.ndarray:
    p = np.clip(np.nan_to_num(np.asarray(p, float)), 0.0, None)
    tot = p.sum()
    if tot <= 0:
        raise ValueError("empty distribution")
    return p / tot


def lognormal(name: str, F: np.ndarray, mean: float, vol: float, t: float, detail: str = "") -> Distribution:
    t = max(t, 1 / 365 / 24)
    s = vol * math.sqrt(t)
    mu = math.log(mean) - 0.5 * s * s
    x = np.log(F)
    dens = norm.pdf(x, mu, s) * np.gradient(x)
    return Distribution(name, F, _normalise(dens), detail)


def market_implied(state: MarketState, F: np.ndarray, t: float) -> Distribution:
    """Risk-neutral density of the forward at the horizon from the SVI smile (Breeden-Litzenberger:
    density = d2C/dK2), with total variance scaled by t / T. SVI gives a smooth, arbitrage-free shape;
    ``project`` re-centres each leg on its market mid so this view reprices the quotes exactly."""
    t = max(t, 1 / 365 / 24)
    k = np.log(F / state.forward)
    w = np.maximum(np.asarray(state.svi_w(k), float) * t / state.T, 1e-12)
    sig = np.sqrt(w)
    d1 = -k / sig + 0.5 * sig
    C = state.forward * norm.cdf(d1) - F * norm.cdf(d1 - sig)
    dens = np.gradient(np.gradient(C, F), F)
    return Distribution("market", F, _normalise(dens * np.gradient(F)),
                        "risk-neutral density from the option smile, re-centred on market prices")


def build_views(state: MarketState, horizon_t: float, target: Optional[float] = None,
                user_vol: Optional[float] = None) -> dict[str, Distribution]:
    model_vol = state.realised_vol or state.atm_vol
    uvol = user_vol or state.atm_vol
    F = horizon_grid(state, horizon_t, (model_vol, uvol))
    views = {"market": market_implied(state, F, horizon_t),
             "model": lognormal("model", F, state.forward, model_vol, horizon_t,
                                f"no drift, EWMA realised vol {model_vol:.1%} ({state.history_note})")}
    tgt = target or state.forward
    views["user"] = lognormal("user", F, tgt, uvol, horizon_t, f"target {tgt:,.0f}, vol {uvol:.1%}")
    views["blend"] = Distribution("blend", F, _normalise(0.5 * views["user"].p + 0.5 * views["model"].p),
                                  "half your view, half the model view")
    return views


# ------------------------------------------------------------------ projection

def _pnl_curve(strategy: Strategy, state: MarketState, entries, F, t_rem: float, dvol: float, units: float,
               charges: Charges):
    """Rupee P&L at forward(s) F with t_rem years left, after entry prices and exit/settlement costs
    (entry charges are subtracted by the caller)."""
    pnl = np.zeros_like(np.asarray(F, float))
    exit_cost = np.zeros_like(pnl)
    for leg, en in zip(strategy.legs, entries):
        q = leg.ratio * units
        v = state.value(leg, F, t_rem, dvol)
        pnl = pnl + leg.side * (v - en["price"]) * q
        if t_rem > 1e-10:  # trade out: charges on the closing order and half the spread
            exit_cost = exit_cost + charges.order(leg.instrument, -leg.side, v, q)["total"] + 0.5 * en["spread"] * q
        elif leg.instrument != "future" and leg.side > 0:  # long option settles: STT on exercise
            exit_cost = exit_cost + charges.stt_option_exercise_intrinsic * v * q
    return pnl - exit_cost


def _tail(pnl, p, level=0.95) -> float:
    """Expected shortfall (as a positive loss) of a discrete P&L distribution."""
    order = np.argsort(pnl)
    x, w = pnl[order], p[order]
    c = np.cumsum(w)
    a = 1.0 - level
    take = np.clip(a - (c - w), 0.0, w)
    return float(-np.sum(take * x) / a)


def greeks(strategy: Strategy, state: MarketState, units: float) -> dict:
    out = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    for leg in strategy.legs:
        q = leg.signed * units
        if leg.instrument == "future":
            out["delta"] += q
            continue
        g = black76_greeks(state.forward, leg.strike, state.T, state.rate, state.vol_at(leg.strike),
                           leg.instrument).reporting_units()
        for k in out:
            out[k] += float(g[k]) * q
    return out


def project(strategy: Strategy, state: MarketState, views: dict[str, Distribution], horizon_t: float,
            charges: Charges, margin: MarginParams, lots: int = 1, iv_shift: float = 0.0,
            curves: bool = True) -> dict:
    """Full projection of ``lots`` units of a strategy. Rupee amounts throughout."""
    entries = [state.entry(l) for l in strategy.legs]
    units = state.lot_size * lots
    t_rem = max(state.T - horizon_t, 0.0)
    entry_charges = sum(charges.order(l.instrument, l.side, e["price"], l.ratio * units)["total"]
                        for l, e in zip(strategy.legs, entries))
    premium = sum(l.side * e["price"] * l.ratio * units for l, e in zip(strategy.legs, entries)
                  if l.instrument != "future")  # + paid, - received
    fair = sum(l.side * e["mid"] * l.ratio * units for l, e in zip(strategy.legs, entries)
               if l.instrument != "future")
    slippage = sum(0.5 * e["spread"] * l.ratio * units for l, e in zip(strategy.legs, entries))
    m = estimate_margin(strategy.legs, state.lot_size, state, margin, lots)
    capital = max(premium, 0.0) + m["total"] + entry_charges

    stats = {}
    for name, d in views.items():
        pnl = _pnl_curve(strategy, state, entries, d.F, t_rem, iv_shift, units, charges) - entry_charges
        if name == "market":
            # pricing identity: under the market's own measure each leg is worth its mid today, so remove
            # the gap between the SVI density's value of each leg and its market mid (SVI fitting error)
            pnl = pnl + sum(l.side * l.ratio * units * (e["mid"] - float(np.sum(d.p * state.value(l, d.F, t_rem, 0.0))))
                            for l, e in zip(strategy.legs, entries))
        stats[name] = {"expected": float(np.sum(d.p * pnl)), "prob_profit": float(np.sum(d.p[pnl > 0])),
                       "es95": _tail(pnl, d.p, 0.95), "p5": float(_quantile(pnl, d.p, 0.05)),
                       "p95": float(_quantile(pnl, d.p, 0.95))}

    # expiry payoff on a wide grid: max profit / loss, breakevens, open-ended risk
    wide = state.forward * np.exp(np.linspace(-0.6, 0.6, 2401))
    at_exp = _pnl_curve(strategy, state, entries, wide, 0.0, 0.0, units, charges) - entry_charges
    near = (wide > state.forward * 0.8) & (wide < state.forward * 1.2)
    slope_up = (at_exp[-1] - at_exp[-50]) / (wide[-1] - wide[-50])
    slope_dn = (at_exp[49] - at_exp[0]) / (wide[49] - wide[0])
    # open-ended if the payoff keeps moving by at least 5% of a unit per index point at the edges
    # (smaller slopes come from STT on exercise, not from the structure)
    tol = 0.05 * units
    # upside is truly unlimited; downside is capped at an index of zero but still open-ended in practice
    unlimited_loss = slope_up < -tol or slope_dn > tol
    unlimited_profit = slope_up > tol
    at_zero = float((_pnl_curve(strategy, state, entries, np.array([1e-6]), 0.0, 0.0, units, charges)
                     - entry_charges)[0])
    xs, ys = wide[near], at_exp[near]
    cross = np.flatnonzero(np.diff(np.sign(ys)) != 0)
    breakevens = [float(xs[i] - ys[i] * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])) for i in cross
                  if ys[i + 1] != ys[i]]
    out = {
        "strategy": strategy.to_dict(), "lots": lots, "units": units, "lot_size": state.lot_size,
        "legs": [{**l.to_dict(), "label": l.label, **{k: e[k] for k in ("price", "mid", "bid", "ask", "spread")},
                  "vol": None if l.instrument == "future" else state.vol_at(l.strike)}
                 for l, e in zip(strategy.legs, entries)],
        "premium": premium, "fair_premium": fair, "slippage": slippage, "entry_charges": entry_charges,
        "margin": m, "capital_required": capital,
        "max_profit": None if unlimited_profit else float(max(np.max(at_exp), at_zero)),
        "max_loss": None if unlimited_loss else float(-min(np.min(at_exp), at_zero)),
        "loss_if_index_to_zero": float(-min(at_zero, 0.0)),
        "loss_at_20pct": float(-min(at_exp[near].min(), 0.0)),
        "stress_loss": _stress_loss(strategy, state, entries, units, charges, entry_charges),
        "unlimited_loss": bool(unlimited_loss), "unlimited_profit": bool(unlimited_profit),
        "breakevens": breakevens, "views": stats, "greeks": greeks(strategy, state, units),
        "horizon_years": horizon_t, "iv_shift": iv_shift,
    }
    if curves:
        sd_rel = state.atm_vol * math.sqrt(state.T)
        g = state.forward * np.exp(np.linspace(-4 * sd_rel, 4 * sd_rel, 161))
        out["curves"] = {
            "F": g.tolist(),
            "expiry": (_pnl_curve(strategy, state, entries, g, 0.0, 0.0, units, charges) - entry_charges).tolist(),
            "horizon": (_pnl_curve(strategy, state, entries, g, t_rem, iv_shift, units, charges)
                        - entry_charges).tolist(),
            "now": (_pnl_curve(strategy, state, entries, g, state.T, 0.0, units, charges) - entry_charges).tolist(),
        }
    return out


def _stress_loss(strategy, state, entries, units, charges, entry_charges) -> float:
    """Worst loss at expiry for index moves up to 3 expected moves either way (a positive number)."""
    sd_rel = state.atm_vol * math.sqrt(state.T)
    F = state.forward * np.exp(np.linspace(-3 * sd_rel, 3 * sd_rel, 121))
    pnl = _pnl_curve(strategy, state, entries, F, 0.0, 0.0, units, charges) - entry_charges
    return float(-min(pnl.min(), 0.0))


def _quantile(x, p, q) -> float:
    o = np.argsort(x)
    return float(np.interp(q, np.cumsum(p[o]), x[o]))


def pnl_grid(strategy: Strategy, state: MarketState, lots: int, charges: Charges, n_spot: int = 41,
             iv_shift: float = 0.0) -> dict:
    """P&L heatmap: index level (rows) x date (columns): now, then each weekday close up to expiry."""
    entries = [state.entry(l) for l in strategy.legs]
    units = state.lot_size * lots
    entry_charges = sum(charges.order(l.instrument, l.side, e["price"], l.ratio * units)["total"]
                        for l, e in zip(strategy.legs, entries))
    sd_rel = state.atm_vol * math.sqrt(state.T)
    F = state.forward * np.exp(np.linspace(-3 * sd_rel, 3 * sd_rel, n_spot))
    stamps = [state.as_of]
    for n in range((state.expiry - state.as_of.date()).days + 1):
        d = date.fromordinal(state.as_of.date().toordinal() + n)
        close = datetime.combine(d, time(15, 30), IST)
        if d.weekday() < 5 and close > state.as_of:
            stamps.append(close)
    z = []
    for ts in stamps:
        t_rem = max(state.T - (ts - state.as_of).total_seconds() / _YEAR, 0.0)
        z.append((_pnl_curve(strategy, state, entries, F, t_rem, iv_shift, units, charges) - entry_charges).tolist())
    labels = ["now"] + [ts.strftime("%a %d %b close") for ts in stamps[1:]]
    return {"F": F.tolist(), "times": labels, "pnl": z}
