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


BASIS = {  # what each strategy is for: market view, volatility stance, risk shape
    "Long call": "Bullish. Pays a premium for upside; the loss is capped at that premium; gains if implied vol rises.",
    "Long put": "Bearish, or insurance against a fall. Loss capped at the premium paid; gains if implied vol rises.",
    "Long straddle": "Expects a big move either way (e.g. before an event). Loss capped at the premium; the move "
                     "must beat both breakevens.",
    "Long strangle": "Expects a big move either way, cheaper than a straddle but with wider breakevens.",
    "Bull call spread": "Moderately bullish. Cheaper than a call; profit capped at the upper strike; little "
                        "exposure to vol.",
    "Bear put spread": "Moderately bearish. Cheaper than a put; profit capped at the lower strike; little "
                       "exposure to vol.",
    "Long call butterfly": "Expects NIFTY to settle near the middle strike. Small cost, capped loss, peak profit "
                           "at the middle.",
    "Bull put spread": "Mildly bullish or not bearish. Collects premium; loss capped by the lower put; earns time "
                       "decay.",
    "Bear call spread": "Mildly bearish or not bullish. Collects premium; loss capped by the upper call; earns time "
                        "decay.",
    "Iron condor": "Range-bound. Collects premium on both sides; loss capped by the wings; earns time decay and "
                   "gains if vol falls.",
    "Iron butterfly": "Expects NIFTY to pin near the centre strike. More premium than a condor, narrower profit "
                      "zone; capped loss.",
    "Short straddle": "Expects very little movement. Collects the most premium; losses are open-ended in a big "
                      "move either way.",
    "Short strangle": "Expects a range. Collects premium with wider breakevens than a straddle; open-ended losses "
                      "beyond them.",
    "Short put": "Bullish to neutral; paid to take on downside risk. Large loss if NIFTY falls sharply.",
    "Short call": "Bearish to neutral; paid to take on upside risk. Open-ended loss if NIFTY rallies.",
    "Long future": "Bullish, one-for-one with NIFTY; gains and losses are open-ended.",
    "Short future": "Bearish, one-for-one with NIFTY; gains and losses are open-ended.",
    "Protective put": "Bullish with insurance: a long future whose downside is capped by a bought put.",
    "Protective call": "Bearish with insurance: a short future whose upside loss is capped by a bought call.",
    "Covered call": "Neutral to mildly bullish: a long future plus a sold call; earns premium but gives up upside "
                    "above the strike.",
    "Collar": "Bullish with limits: a long future between a bought put and a sold call; the call helps pay for "
              "the put.",
}


STANCE = {  # one-phrase version of BASIS for tables
    "Long call": "Bullish", "Long put": "Bearish", "Long straddle": "Big move either way",
    "Long strangle": "Big move either way", "Bull call spread": "Moderately bullish",
    "Bear put spread": "Moderately bearish", "Long call butterfly": "Settles near the middle strike",
    "Bull put spread": "Mildly bullish, collects premium", "Bear call spread": "Mildly bearish, collects premium",
    "Iron condor": "Range-bound, capped loss", "Iron butterfly": "Pins near the centre, capped loss",
    "Short straddle": "Very little movement, open-ended loss", "Short strangle": "Range-bound, open-ended loss",
    "Short put": "Bullish to neutral, collects premium", "Short call": "Bearish to neutral, collects premium",
    "Long future": "Bullish, one-for-one", "Short future": "Bearish, one-for-one",
    "Protective put": "Bullish with downside insurance", "Protective call": "Bearish with upside insurance",
    "Covered call": "Neutral to mildly bullish, earns premium", "Collar": "Bullish within a band",
}


def rationale(p: dict, state: MarketState, dist, source: str) -> dict:
    """Why this strategy: its basis (what it is for) and how it fits the chosen view and today's vols."""
    name = p["strategy"]["name"]
    basis = BASIS.get(name, "Custom combination of the legs shown.")
    F = state.forward
    mean = dist.mean()
    move = mean / F - 1
    be = sorted(p["breakevens"])
    view_txt = f"the {source} view expects {mean:,.0f} ({move:+.1%})"
    view = p["strategy"].get("view", "")
    if not be:
        fit = f"No breakeven within 20% of the forward; {view_txt}."
    elif len(be) == 1:
        side = "above" if view in ("bullish",) or (view not in ("bearish",) and p["greeks"]["delta"] > 0) else "below"
        fit = f"Needs NIFTY {side} {be[0]:,.0f} ({be[0] / F - 1:+.1%}) at the horizon; {view_txt}."
    else:
        lo, hi = be[0], be[-1]
        if name in ("Long straddle", "Long strangle") or (name not in BASIS and p["greeks"]["gamma"] > 0):
            fit = (f"Needs NIFTY outside {lo:,.0f} to {hi:,.0f} ({lo / F - 1:+.1%} / {hi / F - 1:+.1%}); "
                   f"{view_txt}.")
        else:
            fit = (f"Profits if NIFTY stays between {lo:,.0f} and {hi:,.0f} ({lo / F - 1:+.1%} / {hi / F - 1:+.1%}); "
                   f"{view_txt}.")
    vega = p["greeks"]["vega"] / max(p.get("lots", 1), 1)  # Rs per vol point per lot
    if state.realised_vol and abs(vega) < 150:
        vol = " Little net exposure to volatility."
    elif state.realised_vol:
        rich = state.atm_vol > state.realised_vol
        stance = "sells" if vega < 0 else "buys"
        helps = (vega < 0) == rich
        vol = (f" It {stance} volatility: implied {state.atm_vol:.1%} vs model {state.realised_vol:.1%}, which "
               f"{'favours' if helps else 'works against'} it.")
    else:
        vol = ""
    stance = STANCE.get(name, "Custom combination")
    if not be:
        need = "no breakeven nearby"
    elif len(be) == 1:
        need = f"{'above' if 'above' in fit[:30] else 'below'} {be[0]:,.0f} ({be[0] / F - 1:+.1%})"
    else:
        need = (f"{'outside' if 'outside' in fit[:30] else 'between'} {be[0]:,.0f}-{be[-1]:,.0f}")
    short = f"{stance}; needs NIFTY {need}; view {mean:,.0f} ({move:+.1%})"
    return {"basis": basis, "fit": fit + vol, "text": f"{basis} {fit}{vol}", "short": short}


def recommend(state: MarketState, capital: float, view: View, charges: Charges, margin: MarginParams,
              families: Optional[Iterable[str]] = None, risk_budget: float = 0.5, objective: str = "risk_adjusted",
              top: int = 8, per_name: int = 2, min_prob_profit: float = 0.5, near_misses: int = 5) -> dict:
    """Rank strategies for the capital and view. An idea is *recommended* only if its expected P&L is above zero
    and its chance of profit is at least ``min_prob_profit`` under the chosen view (and it fits the capital and
    risk budget). The best ideas that fail are returned separately as near misses, with the reasons."""
    if capital <= 0:
        raise ValueError("capital must be positive")
    if not 0 < risk_budget <= 1:
        raise ValueError("risk budget is a share of capital between 0 and 1")
    if not 0 <= min_prob_profit < 1:
        raise ValueError("minimum chance of profit is a share between 0 and 1")
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
        quick = scaled["expected"] <= 0 or scaled["prob_profit"] < min_prob_profit  # 1-lot screen; re-checked at size
        ranked.append((_score(objective, scaled, risk, used), s, lots, scaled, quick))

    ranked.sort(key=lambda x: x[0], reverse=True)
    dist = views[view.source]

    def check(st: dict) -> list[str]:
        reasons = []
        if st["expected"] <= 0:
            reasons.append(f"expected P&L Rs {st['expected']:,.0f} is not above zero")
        if st["prob_profit"] < min_prob_profit:
            pp = st["prob_profit"]
            shown = f"{pp:.0%}" if round(pp * 100) < round(min_prob_profit * 100) else f"{pp:.1%}"
            reasons.append(f"chance of profit {shown} is below {min_prob_profit:.0%}")
        return reasons

    # the rule is applied to the full-size projection, so the reasons match the numbers shown
    ideas, misses, per_ok, per_miss = [], [], {}, {}
    for score, s, lots, _st, quick in ranked:
        if len(ideas) >= top and len(misses) >= near_misses:
            break
        bucket, per, limit = (ideas, per_ok, top) if not quick else (misses, per_miss, near_misses)
        if len(bucket) >= limit or per.get(s.name, 0) >= per_name:
            continue
        full = project(s, state, views, t_h, charges, margin, lots=lots, iv_shift=view.iv_shift)
        reasons = check(full["views"][view.source])
        bucket, per, limit = (ideas, per_ok, top) if not reasons else (misses, per_miss, near_misses)
        if len(bucket) >= limit or per.get(s.name, 0) >= per_name:
            continue
        per[s.name] = per.get(s.name, 0) + 1
        full.update(score=score, recommended=not reasons, reasons=reasons, why=_explain(full, view.source),
                    rationale=rationale(full, state, dist, view.source))
        bucket.append(full)
    if ideas:
        note = ""
    else:
        note = (f"No recommendation for this capital and view: no strategy that fits has both a positive expected "
                f"P&L and at least a {min_prob_profit:.0%} chance of profit. ")
        if view.source == "market":
            note += "Under the market-implied view every strategy loses its costs on average."
        elif abs(dist.mean() / state.forward - 1) < 0.002:
            note += (f"The view's mean ({dist.mean():,.0f}) is close to the forward ({state.forward:,.0f}), so costs "
                     "dominate; set a target that differs from the forward.")
        else:
            note += ("Strategies with a high chance of profit (spreads, hedged selling) need more capital for "
                     "margin; increase the capital, lower the threshold, or change the view.")
    return {
        "as_of": state.as_of.isoformat(), "underlying": state.underlying, "expiry": state.expiry.isoformat(),
        "forward": state.forward, "spot": state.spot, "atm_vol": state.atm_vol, "realised_vol": state.realised_vol,
        "lot_size": state.lot_size, "capital": capital, "risk_budget": risk_budget, "objective": objective,
        "objective_label": OBJECTIVES[objective], "min_prob_profit": min_prob_profit,
        "view": {**asdict(view), "horizon": (view.horizon or state.expiry).isoformat()},
        "views": {k: v.to_dict() for k, v in views.items()}, "families": fams,
        "considered": len(candidates), "not_quoted": skipped, "feasible": len(ranked),
        "recommended_count": sum(1 for r in ranked if not r[4]),
        "blocked": [{**b, "min_capital": None if math.isinf(b["min_capital"]) else b["min_capital"]}
                    for b in blocked.values()],
        "ideas": ideas, "near_misses": misses, "any_positive": bool(ideas), "note": note,
        "rule": (f"Recommended only if expected P&L > 0 and chance of profit >= {min_prob_profit:.0%} under the "
                 f"{view.source} view, within the capital and a maximum loss of {risk_budget:.0%} of it."),
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
    out["rationale"] = rationale(out, state, views[view.source], view.source)
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
        for zc in (0.5, 1.0):  # both wings at once: n lots of a call plus m lots of a put
            for zp in (0.5, 1.0):
                kc, kp = g.at(zc, "call"), g.at(-zp, "put")
                if not (kc and kp):
                    continue
                for m in (1, 2, 3):
                    out.append(Strategy(f"Buy {lots(n)} of the {kc:,.0f} call + {lots(m)} of the {kp:,.0f} put",
                                        "buy", "hedge", (StrategyLeg("call", 1, e, kc, n), StrategyLeg("put", 1, e, kp, m))))
        for z, w in ((0.0, 1.0), (0.5, 1.0)):
            p1, p2, c1, c2 = g.at(-z, "put"), g.at(-z - w, "put"), g.at(z, "call"), g.at(z + w, "call")
            if p1 and p2 and p1 > p2:
                out.append(Strategy(f"Buy {lots(n)} of the {p1:,.0f}/{p2:,.0f} put spread", "debit", "hedge",
                                    (StrategyLeg("put", 1, e, p1, n), StrategyLeg("put", -1, e, p2, n))))
            if c1 and c2 and c2 > c1:
                out.append(Strategy(f"Buy {lots(n)} of the {c1:,.0f}/{c2:,.0f} call spread", "debit", "hedge",
                                    (StrategyLeg("call", 1, e, c1, n), StrategyLeg("call", -1, e, c2, n))))
    return out


HEDGE_RANKS = {
    "worst_case": "smallest loss in a 3-sd move either way (protects both directions)",
    "es95": "smallest expected shortfall (95%) under the chosen view",
    "per_rupee": "most worst-case loss removed per rupee spent",
}


def _two_sided(F, pnl, F0) -> tuple[float, float]:
    up, dn = pnl[F >= F0], pnl[F <= F0]
    return float(-min(up.min(), 0.0)), float(-min(dn.min(), 0.0))


def find_hedges(state: MarketState, positions: list[dict], view: View, charges: Charges, margin: MarginParams,
                budget: float, top: int = 3, rank_by: str = "worst_case") -> dict:
    """Search hedge trades for a book: futures, single calls or puts, call + put pairs (unequal lots allowed)
    and spreads, 1 to 3 lots each, within a cash budget for premium plus extra margin.

    Each hedge is judged on the book's P&L at expiry for index moves up to 3 expected moves either way
    (loss if the index rises, loss if it falls) and on ES95 at the horizon under the chosen view."""
    if rank_by not in HEDGE_RANKS:
        raise ValueError(f"rank_by must be one of {list(HEDGE_RANKS)}")
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
    sd = state.atm_vol * math.sqrt(state.T)
    Fx = state.forward * np.exp(np.linspace(-3 * sd, 3 * sd, 121))  # expiry stress grid
    base = _book_pnl(book, state, d.F, t_rem, units, view.iv_shift)
    base_x = _book_pnl(book, state, Fx, 0.0, units)
    base_margin = estimate_margin(book, units, state, margin)["total"]
    up0, dn0 = _two_sided(Fx, base_x, state.forward)
    before = {"es95": _tail(base, d.p), "expected": float(np.sum(d.p * base)), "loss_if_up": up0,
              "loss_if_down": dn0, "worst_case": max(up0, dn0), "greeks": _legs_greeks(book, state, units),
              "margin": base_margin}
    results = []
    for h in hedge_candidates(state):
        try:
            entries = [state.entry(l) for l in h.legs]
        except NotQuoted:
            continue
        ch = sum(charges.order(l.instrument, l.side, e["price"], l.ratio * units)["total"]
                 for l, e in zip(h.legs, entries))
        premium = sum(l.side * e["price"] * l.ratio * units for l, e in zip(h.legs, entries) if l.instrument != "future")
        extra_margin = estimate_margin(book + list(h.legs), units, state, margin)["total"] - base_margin
        cash = max(premium, 0.0) + max(extra_margin, 0.0) + ch
        if cash > budget:
            continue
        tot = base + _pnl_curve(h, state, entries, d.F, t_rem, view.iv_shift, units, charges) - ch
        tot_x = base_x + _pnl_curve(h, state, entries, Fx, 0.0, 0.0, units, charges) - ch
        up, dn = _two_sided(Fx, tot_x, state.forward)
        results.append({
            "hedge": h.to_dict(), "label": h.name,
            "legs": [{"leg": l.label, "price": e["price"], "bid": e["bid"], "ask": e["ask"]}
                     for l, e in zip(h.legs, entries)],
            "cash_needed": cash, "premium": premium, "extra_margin": extra_margin, "charges": ch,
            "es95": _tail(tot, d.p), "expected": float(np.sum(d.p * tot)), "loss_if_up": up, "loss_if_down": dn,
            "worst_case": max(up, dn), "greeks": _legs_greeks(book + list(h.legs), state, units),
            "curve_after": tot_x.tolist()})
    key = {"worst_case": lambda r: (r["worst_case"], r["es95"], r["cash_needed"]),
           "es95": lambda r: (r["es95"], r["worst_case"], r["cash_needed"]),
           "per_rupee": lambda r: (-(before["worst_case"] - r["worst_case"]) / max(r["cash_needed"], 1.0),
                                   r["worst_case"])}[rank_by]
    results.sort(key=key)
    best = results[:top]
    for r in results[top:]:
        r.pop("curve_after", None)
    return {"as_of": state.as_of.isoformat(), "expiry": state.expiry.isoformat(), "view": view.source,
            "budget": budget, "rank_by": rank_by, "rank_label": HEDGE_RANKS[rank_by], "before": before,
            "hedges": best, "searched": len(results), "curves": {"F": Fx.tolist(), "before": base_x.tolist()},
            "note": "Book P&L is measured from today's mid prices; hedges are entered at the bid/ask with all charges.",
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
