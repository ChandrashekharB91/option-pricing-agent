"""Strategy recommender and hedge finder (Phase 3). Internal analysis tool, not investment advice.

``recommend`` takes the capital the user enters and a market view, generates every strategy in the
library at listed strikes, prices each at executable prices with charges and an estimated margin,
sizes it to the capital and a risk budget, projects its P&L under the chosen view, and ranks.
``find_hedges`` searches futures and option trades that cut the expected shortfall of an existing book.
Every result carries ``DISCLAIMER``.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .charges import Charges
from .margin import MarginParams, estimate_margin
from .projector import (MarketState, NotQuoted, _pnl_curve, _tail, build_views, greeks, pnl_grid, project)
from .strategies import FAMILIES, Strategy, StrategyLeg, StrikeGrid, generate_candidates

DISCLAIMER = ("This is not financial advice. The recommendations are model output for internal analysis only. "
              "The author is not responsible for any losses or gains from decisions based on them.")
VIEW_SOURCES = ("user", "market", "model", "blend")
OBJECTIVES = {
    "risk_adjusted": "expected profit per rupee at risk",
    "expected_return": "expected profit per rupee of capital used",
    "probability": "probability of profit",
}
MAX_LOTS = 50


@dataclass
class View:
    source: str = "blend"  # user | market | model | blend
    target: Optional[float] = None  # index level expected at the horizon (user view)
    horizon: Optional[date] = None  # default: expiry
    vol: Optional[float] = None  # user's volatility for the move (annual, decimal); default ATM implied
    iv_shift: float = 0.0  # change in implied vol by the horizon (decimal, 0.02 = +2 vol points)

    def __post_init__(self):
        if self.source not in VIEW_SOURCES:
            raise ValueError(f"view source must be one of {VIEW_SOURCES}")


# ------------------------------------------------------------------ market state

def ewma_vol(closes, lam: float = 0.94, days: int = 250) -> float:
    c = np.asarray(closes, float)
    c = c[np.isfinite(c)]
    r = np.diff(np.log(c))[-days:]
    var = r[0] ** 2
    for x in r[1:]:
        var = lam * var + (1 - lam) * x * x
    return float(math.sqrt(var * 252))


def market_state(surface, smile, chain, futures, lot_size: int, history=None) -> MarketState:
    """Assemble what the projector needs from a fitted smile, the raw chain and the futures quotes."""
    quotes = {(q.strike, q.option_type): (q.bid, q.ask) for q in chain.quotes if q.two_sided and q.bid > 0}
    fut = next((f for f in futures if f.expiry == smile.expiry), None)
    F = smile.forward
    rv, note = None, ""
    if history is not None:
        h = history.before(surface.as_of.date())
        try:
            rv = ewma_vol(h.series(surface.underlying))
            note = f"index closes to {h.dates[-1]}"
        except (KeyError, IndexError, ValueError):
            rv = None
    return MarketState(
        underlying=surface.underlying, as_of=surface.as_of, expiry=smile.expiry, T=smile.time_to_expiry,
        forward=F, rate=smile.slice.rate, spot=surface.spot, lot_size=lot_size, atm_vol=smile.vol(F)[0],
        vol_at=lambda K, sm=smile: float(sm.vol(K)[0]), svi_w=smile.fit.params.total_variance, quotes=quotes,
        future=(fut.bid, fut.ask) if fut else None, realised_vol=rv, history_note=note)


# ------------------------------------------------------------------ recommend

def _risk(one: dict) -> float:
    """Rupees at risk per position: the maximum loss, or for open-ended strategies the 3-sd stress loss."""
    return one["stress_loss"] if one["unlimited_loss"] else (one["max_loss"] or 0.0)


def _size(one: dict, capital: float, risk_budget: float, margin: MarginParams) -> tuple[int, str]:
    need = one["capital_required"] * (1 + margin.capital_buffer)
    if need <= 0:
        return 0, "no capital requirement computed"
    by_cap = math.floor(capital / need)
    risk = _risk(one)
    by_risk = math.floor(risk_budget * capital / risk) if risk > 0 else MAX_LOTS
    lots = int(min(by_cap, by_risk, MAX_LOTS))
    if lots >= 1:
        return lots, ""
    if by_cap < 1:
        return 0, f"needs about Rs {need:,.0f} per lot (premium, margin, charges and a {margin.capital_buffer:.0%} buffer)"
    return 0, f"one lot risks Rs {risk:,.0f}, above the risk budget of Rs {risk_budget * capital:,.0f}"


def _score(objective: str, st: dict, risk: float, capital_used: float) -> float:
    if objective == "probability":
        return st["prob_profit"] + 1e-9 * st["expected"]
    if objective == "expected_return":
        return st["expected"] / max(capital_used, 1.0)
    return st["expected"] / max(risk, 1.0)


def recommend(state: MarketState, capital: float, view: View, charges: Charges, margin: MarginParams,
              families: Optional[Iterable[str]] = None, risk_budget: float = 0.5, objective: str = "risk_adjusted",
              top: int = 8, per_name: int = 2) -> dict:
    if capital <= 0:
        raise ValueError("capital must be positive")
    if not 0 < risk_budget <= 1:
        raise ValueError("risk budget is a share of capital between 0 and 1")
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {list(OBJECTIVES)}")
    fams = list(families or FAMILIES)
    t_h = state.years_to(view.horizon or state.expiry)
    views = build_views(state, t_h, view.target, view.vol)
    calls = sorted(k for (k, t) in state.quotes if t == "call")
    puts = sorted(k for (k, t) in state.quotes if t == "put")
    grid = StrikeGrid(state.forward, state.sd, calls, puts, state.expiry)
    candidates = generate_candidates(grid, fams)

    ranked, blocked, skipped = [], {}, 0
    for s in candidates:
        try:
            one = project(s, state, views, t_h, charges, margin, lots=1, iv_shift=view.iv_shift, curves=False)
        except NotQuoted:
            skipped += 1
            continue
        lots, why = _size(one, capital, risk_budget, margin)
        if lots < 1:
            b = blocked.setdefault(s.family, {"family": FAMILIES[s.family], "count": 0, "min_capital": math.inf,
                                              "example": ""})
            b["count"] += 1
            need = one["capital_required"] * (1 + margin.capital_buffer)
            if need < b["min_capital"]:
                b["min_capital"], b["example"] = need, f"{s.label}: {why}"
            continue
        st = one["views"][view.source]
        risk = _risk(one) * lots
        used = one["capital_required"] * lots
        scaled = {k: v * lots if k != "prob_profit" else v for k, v in st.items()}
        ranked.append((_score(objective, scaled, risk, used), s, lots, scaled))

    ranked.sort(key=lambda x: x[0], reverse=True)
    if any(st["expected"] > 0 for _, _, _, st in ranked):  # show losing ideas only when nothing wins
        ranked = [r for r in ranked if r[3]["expected"] > 0]
    ideas, per = [], {}
    for score, s, lots, st in ranked:
        if per.get(s.name, 0) >= per_name:
            continue
        per[s.name] = per.get(s.name, 0) + 1
        full = project(s, state, views, t_h, charges, margin, lots=lots, iv_shift=view.iv_shift)
        full["score"] = score
        full["why"] = _explain(full, view.source)
        ideas.append(full)
        if len(ideas) >= top:
            break
    positive = [i for i in ideas if i["views"][view.source]["expected"] > 0]
    return {
        "as_of": state.as_of.isoformat(), "underlying": state.underlying, "expiry": state.expiry.isoformat(),
        "forward": state.forward, "spot": state.spot, "atm_vol": state.atm_vol, "realised_vol": state.realised_vol,
        "lot_size": state.lot_size, "capital": capital, "risk_budget": risk_budget, "objective": objective,
        "objective_label": OBJECTIVES[objective], "view": {**asdict(view), "horizon": (view.horizon or state.expiry).isoformat()},
        "views": {k: v.to_dict() for k, v in views.items()}, "families": fams,
        "considered": len(candidates), "not_quoted": skipped, "feasible": len(ranked),
        "blocked": [{**b, "min_capital": None if math.isinf(b["min_capital"]) else b["min_capital"]}
                    for b in blocked.values()],
        "ideas": ideas, "any_positive": bool(positive),
        "note": ("" if positive else
                 "No idea has a positive expected profit under this view after costs. Expected profit comes only "
                 "from a view that differs from the market's; "
                 + ("under the market-implied view every strategy loses its costs on average."
                    if view.source == "market" else
                    f"here the view's mean ({views[view.source].mean():,.0f}) is close to the forward "
                    f"({state.forward:,.0f}), so costs dominate.")),
        "disclaimer": DISCLAIMER,
    }


def _explain(p: dict, source: str) -> str:
    st = p["views"][source]
    loss = (f"open-ended loss (Rs {p['stress_loss']:,.0f} in a 3-sd move)" if p["unlimited_loss"]
            else f"maximum loss Rs {p['max_loss']:,.0f}")
    be = ", ".join(f"{b:,.0f}" for b in p["breakevens"]) or "none in range"
    return (f"{p['lots']} lot(s); expected P&L Rs {st['expected']:,.0f}, {st['prob_profit']:.0%} chance of profit, "
            f"{loss}; breakeven {be}; capital used Rs {p['capital_required']:,.0f}.")


def project_custom(state: MarketState, legs: list[dict], lots: int, view: View, charges: Charges,
                   margin: MarginParams, name: str = "Custom strategy") -> dict:
    s = Strategy.custom([StrategyLeg.from_dict(d) for d in legs], name)
    if s.legs[0].expiry != state.expiry:
        raise ValueError(f"legs must expire on {state.expiry}")
    t_h = state.years_to(view.horizon or state.expiry)
    views = build_views(state, t_h, view.target, view.vol)
    out = project(s, state, views, t_h, charges, margin, lots=lots, iv_shift=view.iv_shift)
    out["grid"] = pnl_grid(s, state, lots, charges, iv_shift=view.iv_shift)
    out["views_summary"] = {k: v.to_dict() for k, v in views.items()}
    out["why"] = _explain(out, view.source)
    out["disclaimer"] = DISCLAIMER
    return out


# ------------------------------------------------------------------ hedges

def _book_pnl(legs, state, F, t_rem, units_per_lot, dvol=0.0):
    """Rupee P&L of an existing book from today's mid to the horizon (no entry cost)."""
    pnl = np.zeros_like(F)
    for l in legs:
        mid = state.entry(l)["mid"]
        pnl = pnl + l.signed * (state.value(l, F, t_rem, dvol) - mid) * units_per_lot
    return pnl


def hedge_candidates(state: MarketState) -> list[Strategy]:
    calls = sorted(k for (k, t) in state.quotes if t == "call")
    puts = sorted(k for (k, t) in state.quotes if t == "put")
    g = StrikeGrid(state.forward, state.sd, calls, puts, state.expiry)
    e, out = state.expiry, []

    def lots(n):
        return f"{n} lot{'s' if n > 1 else ''}"

    for n in (1, 2, 3):
        out += [Strategy(f"Buy {lots(n)} of the future", "futures", "hedge", (StrategyLeg("future", 1, e, None, n),)),
                Strategy(f"Sell {lots(n)} of the future", "futures", "hedge", (StrategyLeg("future", -1, e, None, n),))]
        for z in (0.0, 0.5, 1.0):
            kp, kc = g.at(-z, "put"), g.at(z, "call")
            if kp:
                out.append(Strategy(f"Buy {lots(n)} of the {kp:,.0f} put", "buy", "hedge",
                                    (StrategyLeg("put", 1, e, kp, n),)))
            if kc:
                out.append(Strategy(f"Buy {lots(n)} of the {kc:,.0f} call", "buy", "hedge",
                                    (StrategyLeg("call", 1, e, kc, n),)))
        for z, w in ((0.0, 1.0), (0.5, 1.0)):
            p1, p2, c1, c2 = g.at(-z, "put"), g.at(-z - w, "put"), g.at(z, "call"), g.at(z + w, "call")
            if p1 and p2 and p1 > p2:
                out.append(Strategy(f"Buy {lots(n)} of the {p1:,.0f}/{p2:,.0f} put spread", "debit", "hedge",
                                    (StrategyLeg("put", 1, e, p1, n), StrategyLeg("put", -1, e, p2, n))))
            if c1 and c2 and c2 > c1:
                out.append(Strategy(f"Buy {lots(n)} of the {c1:,.0f}/{c2:,.0f} call spread", "debit", "hedge",
                                    (StrategyLeg("call", 1, e, c1, n), StrategyLeg("call", -1, e, c2, n))))
    return out


def find_hedges(state: MarketState, positions: list[dict], view: View, charges: Charges, margin: MarginParams,
                budget: float, top: int = 3) -> dict:
    """Search single hedge trades (futures, options, spreads; 1 to 3 lots) that most reduce the book's ES95
    at the horizon under the chosen view, within a cash budget for premium plus extra margin."""
    book = []
    for p in positions:
        exp = p["expiry"] if isinstance(p["expiry"], date) else date.fromisoformat(str(p["expiry"])[:10])
        if exp != state.expiry:
            raise ValueError(f"the hedge finder handles one expiry at a time ({state.expiry}); got {exp}")
        lots = float(p["lots"])
        if lots == 0 or lots != int(lots):
            raise ValueError("positions must be whole, non-zero lots")
        book.append(StrategyLeg(p["option_type"], 1 if lots > 0 else -1, exp, float(p["strike"]), abs(int(lots))))
    if not book:
        raise ValueError("the book is empty")
    t_h = state.years_to(view.horizon or state.expiry)
    t_rem = max(state.T - t_h, 0.0)
    d = build_views(state, t_h, view.target, view.vol)[view.source]
    units = state.lot_size
    base = _book_pnl(book, state, d.F, t_rem, units, view.iv_shift)
    base_margin = estimate_margin(book, units, state, margin)["total"]
    before = {"es95": _tail(base, d.p), "expected": float(np.sum(d.p * base)),
              "stress_loss": _book_stress(book, state, units), "greeks": _legs_greeks(book, state, units),
              "margin": base_margin}
    results = []
    for h in hedge_candidates(state):
        try:
            entries = [state.entry(l) for l in h.legs]
        except NotQuoted:
            continue
        ch = sum(charges.order(l.instrument, l.side, e["price"], l.ratio * units)["total"]
                 for l, e in zip(h.legs, entries))
        hp = _pnl_curve(h, state, entries, d.F, t_rem, view.iv_shift, units, charges) - ch
        premium = sum(l.side * e["price"] * l.ratio * units for l, e in zip(h.legs, entries) if l.instrument != "future")
        extra_margin = estimate_margin(book + list(h.legs), units, state, margin)["total"] - base_margin
        cash = max(premium, 0.0) + max(extra_margin, 0.0) + ch
        if cash > budget:
            continue
        tot = base + hp
        results.append({"hedge": h.to_dict(), "label": h.name, "legs": [l.label for l in h.legs],
            "cash_needed": cash, "premium": premium, "extra_margin": extra_margin, "charges": ch,
            "es95": _tail(tot, d.p), "expected": float(np.sum(d.p * tot)),
            "stress_loss": _book_stress(book + list(h.legs), state, units),
            "greeks": _legs_greeks(book + list(h.legs), state, units)})
    results.sort(key=lambda r: (r["es95"], r["cash_needed"]))
    return {"as_of": state.as_of.isoformat(), "expiry": state.expiry.isoformat(), "view": view.source,
            "budget": budget, "before": before, "hedges": results[:top], "searched": len(results),
            "disclaimer": DISCLAIMER}


def _legs_greeks(legs, state, units) -> dict:
    return greeks(Strategy("book", "custom", "hedge", tuple(legs)), state, units)


def _book_stress(legs, state, units) -> float:
    sd = state.atm_vol * math.sqrt(state.T)
    F = state.forward * np.exp(np.linspace(-3 * sd, 3 * sd, 121))
    pnl = _book_pnl(legs, state, F, 0.0, units)
    return float(-min(pnl.min(), 0.0))


# ------------------------------------------------------------------ audit log

def log_event(path: Path, kind: str, request: dict, result: dict) -> None:
    """Append one line per recommendation or hedge search, so every proposal can be reviewed later."""
    keep = {k: result.get(k) for k in ("as_of", "expiry", "forward", "capital", "objective", "feasible")}
    keep["ideas"] = [{"label": i["strategy"]["label"], "lots": i["lots"], "score": i.get("score")}
                     for i in result.get("ideas", [])]
    keep["hedges"] = [h["label"] for h in result.get("hedges", [])]
    rec = {"logged_at": datetime.now().isoformat(timespec="seconds"), "kind": kind, "request": request, "result": keep}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
