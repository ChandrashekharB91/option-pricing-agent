"""Backtest and calibration of the strategy engine on NSE end-of-day history.

Question answered: when the engine says a strategy has a 60% chance of profit and an expected P&L of
Rs X, does that happen? For every trading day in the sample:

1. Rebuild that day's market at the close from the F&O bhavcopy: the nearest weekly NIFTY expiry, the
   parity forward, implied vols and the SVI smile, with a bid/ask spread assumed around each close
   (closing prices only; no live quotes exist in the archive).
2. Run every strategy in the library through the projector exactly as the recommender does, under the
   model and the market-implied views, and record the predictions (chance of profit, expected P&L, ES95).
3. Settle each strategy at NIFTY's actual close on expiry day (NSE settles index options on the index
   close), after entry at the assumed bid/ask, all charges and STT on exercise.

Outputs: one row per (day, strategy) with predictions and outcome; calibration by probability bucket;
predicted vs realised P&L by strategy; and how ideas that passed the recommendation rule performed.

Assumptions (stated in the report): flat 6.5% rate (immaterial for weekly options); the current charges
schedule for every day; spread = max(1 tick, 0.25% of price) each side of the close; strikes with no
volume that day are not tradeable. No look-ahead: the model view's realised vol uses closes before the day.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from .charges import Charges
from .curves import flat_curve
from .data.bhavcopy import BhavRow
from .data.models import IST, NSE_CLOSE, OptionChain, OptionQuote
from .margin import MarginParams
from .projector import NotQuoted, _pnl_curve, build_views, project
from .recommender import _risk, market_state
from .strategies import FAMILIES, StrikeGrid, generate_candidates
from .surface import build_surface

TICK = 0.05


@dataclass
class BacktestConfig:
    symbol: str = "NIFTY"
    rate: float = 0.065
    half_spread_rel: float = 0.0025  # each side of the close, as a share of price
    min_days: float = 0.9  # skip expiries closer than this (expiry-day trading is a different game)
    max_days: float = 8.0  # nearest weekly only
    min_prob_profit: float = 0.5
    views: tuple = ("model", "market")
    families: Optional[tuple] = None


class _DaySource:
    """Minimal market-data source over one day's end-of-day rows (what build_surface needs)."""

    def __init__(self, rows: list[BhavRow], spot: float, as_of: datetime, cfg: BacktestConfig):
        self.rows, self.spot, self.as_of, self.cfg = rows, spot, as_of, cfg

    def expiries(self, underlying):
        return sorted({r.expiry for r in self.rows if r.kind == "option"})

    def futures(self, underlying):
        from .data.models import FutureQuote

        out = []
        for r in self.rows:
            if r.kind == "future" and r.close > 0:
                h = max(TICK, self.cfg.half_spread_rel * 0.1 * r.close)
                out.append(FutureQuote(f"{r.symbol}{r.expiry}", r.symbol, r.expiry, r.close, r.close - h, r.close + h,
                                       r.oi, int(r.volume), self.as_of))
        return out

    def option_chain(self, underlying, expiry):
        quotes = []
        for r in self.rows:
            if r.kind != "option" or r.expiry != expiry or r.close <= 0:
                continue
            if r.volume <= 0:  # no trade today: the close is a stale reference price, not tradeable
                bid = ask = math.nan
            else:
                h = max(TICK, self.cfg.half_spread_rel * r.close)
                bid, ask = max(r.close - h, TICK), r.close + h
            quotes.append(OptionQuote(f"{r.symbol}{r.expiry}{r.strike}{r.option_type}", expiry, r.strike,
                                      r.option_type, bid=bid, ask=ask, volume=int(r.volume), oi=r.oi, last=r.close))
        return OptionChain(underlying, underlying, expiry, self.spot, self.as_of, "bhavcopy", tuple(quotes))


def day_state(rows: list[BhavRow], day: date, spot: float, history, cfg: BacktestConfig):
    """MarketState for the nearest eligible expiry at the close of ``day``, or None if not usable."""
    as_of = datetime.combine(day, NSE_CLOSE, tzinfo=IST)
    src = _DaySource([r for r in rows if r.symbol == cfg.symbol], spot, as_of, cfg)
    exps = [e for e in src.expiries(cfg.symbol) if cfg.min_days <= (e - day).days + 0.0 <= cfg.max_days]
    if not exps:
        return None
    e = exps[0]
    curve = flat_curve(day, cfg.rate)
    surf = build_surface(src, curve, cfg.symbol, expiries=[e])
    sm = surf.smiles[0]
    lot = next((r.lot_size for r in rows if r.kind == "option" and r.expiry == e and r.lot_size), 0)
    chain = src.option_chain(cfg.symbol, e)
    st = market_state(surf, sm, chain, src.futures(cfg.symbol), lot, history)
    # the model view must not see the day's own close: market_state uses history.before(day) already
    return st


def run_day(rows: list[BhavRow], day: date, history, settle: Callable[[date], Optional[float]],
            cfg: BacktestConfig, charges: Charges, margin: MarginParams) -> list[dict]:
    spot = _close(history, day, cfg.symbol)
    if spot is None:
        return []
    try:
        st = day_state(rows, day, spot, history, cfg)
    except (ValueError, ZeroDivisionError, IndexError, KeyError):
        return []
    if st is None or st.lot_size <= 0:
        return []
    S_T = settle(st.expiry)
    if S_T is None:
        return []
    views = build_views(st, st.T)
    views = {k: v for k, v in views.items() if k in cfg.views}
    calls = sorted(k for (k, t) in st.quotes if t == "call")
    puts = sorted(k for (k, t) in st.quotes if t == "put")
    grid = StrikeGrid(st.forward, st.sd, calls, puts, st.expiry)
    out = []
    for s in generate_candidates(grid, cfg.families or FAMILIES):
        try:
            p = project(s, st, views, st.T, charges, margin, lots=1, curves=False)
        except NotQuoted:
            continue
        entries = [st.entry(l) for l in s.legs]
        realised = float((_pnl_curve(s, st, entries, np.array([S_T]), 0.0, 0.0, st.lot_size, charges)
                          - p["entry_charges"])[0])
        row = {"day": day.isoformat(), "expiry": st.expiry.isoformat(), "days": round(st.T * 365, 2),
               "forward": st.forward, "spot": spot, "settle": S_T, "move": S_T / st.forward - 1,
               "atm_vol": st.atm_vol, "realised_vol": st.realised_vol, "lot_size": st.lot_size,
               "strategy": s.name, "label": s.label, "family": s.family, "capital": p["capital_required"],
               "max_loss": p["max_loss"], "risk": _risk(p), "unlimited_loss": p["unlimited_loss"],
               "breakevens": p["breakevens"], "realised_pnl": realised, "won": realised > 0}
        for v, d in p["views"].items():
            row[f"{v}_pop"] = d["prob_profit"]
            row[f"{v}_expected"] = d["expected"]
            row[f"{v}_es95"] = d["es95"]
            row[f"{v}_recommended"] = d["expected"] > 0 and d["prob_profit"] >= cfg.min_prob_profit
        out.append(row)
    return out


def _close(history, day: date, symbol: str) -> Optional[float]:
    try:
        i = history.dates.index(day)
    except ValueError:
        return None
    v = float(history.series(symbol)[i])
    return v if math.isfinite(v) else None


def settle_from_history(history, symbol: str = "NIFTY") -> Callable[[date], Optional[float]]:
    return lambda d: _close(history, d, symbol)


# ------------------------------------------------------------------ analysis

def calibration(rows: list[dict], view: str, bins: Iterable[float] = np.linspace(0, 1, 11)) -> list[dict]:
    """Predicted chance of profit vs the share that actually won, by probability bucket."""
    edges = list(bins)
    p = np.array([r[f"{view}_pop"] for r in rows])
    w = np.array([r["won"] for r in rows], float)
    out = []
    for lo, hi in zip(edges, edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < edges[-1] else (p <= hi))
        if m.sum() == 0:
            continue
        n = int(m.sum())
        rate = float(w[m].mean())
        out.append({"bucket": f"{lo:.0%}-{hi:.0%}", "n": n, "predicted": float(p[m].mean()), "realised": rate,
                    "stderr": math.sqrt(max(rate * (1 - rate), 1e-9) / n)})
    return out


def brier(rows: list[dict], view: str) -> float:
    p = np.array([r[f"{view}_pop"] for r in rows])
    w = np.array([r["won"] for r in rows], float)
    return float(np.mean((p - w) ** 2))


def by_strategy(rows: list[dict], view: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["strategy"], []).append(r)
    out = []
    for name, rs in groups.items():
        real = np.array([r["realised_pnl"] / r["lot_size"] for r in rs])  # index points per unit
        pred = np.array([r[f"{view}_expected"] / r["lot_size"] for r in rs])
        pop = np.array([r[f"{view}_pop"] for r in rs])
        out.append({"strategy": name, "family": FAMILIES.get(rs[0]["family"], rs[0]["family"]), "n": len(rs),
                    "predicted_pop": float(pop.mean()), "realised_win_rate": float(np.mean([r["won"] for r in rs])),
                    "predicted_pnl_pts": float(pred.mean()), "realised_pnl_pts": float(real.mean()),
                    "realised_t": float(real.mean() / (real.std(ddof=1) / math.sqrt(len(real)))) if len(real) > 2
                    and real.std(ddof=1) > 0 else 0.0,
                    "worst_pts": float(real.min()), "best_pts": float(real.max())})
    return sorted(out, key=lambda x: x["realised_pnl_pts"], reverse=True)


def rule_performance(rows: list[dict], view: str) -> dict:
    """How ideas that passed the recommendation rule did, and the best-ranked idea per day."""
    rec = [r for r in rows if r[f"{view}_recommended"]]
    days = sorted({r["day"] for r in rows})
    if not rec:
        return {"ideas": 0, "days": len(days)}
    pts = np.array([r["realised_pnl"] / r["lot_size"] for r in rec])
    pred = np.array([r[f"{view}_expected"] / r["lot_size"] for r in rec])
    top = []  # one idea per day: highest expected profit per rupee at risk among those that pass
    for d in days:
        cand = [r for r in rec if r["day"] == d]
        if cand:
            top.append(max(cand, key=lambda r: r[f"{view}_expected"] / max(r["risk"], 1.0)))
    tp = np.array([r["realised_pnl"] / r["lot_size"] for r in top])
    eq = np.cumsum([r["realised_pnl"] for r in top])
    peak = np.maximum.accumulate(eq) if len(eq) else eq
    return {
        "ideas": len(rec), "days": len(days), "days_with_idea": len(top),
        "predicted_pop": float(np.mean([r[f"{view}_pop"] for r in rec])),
        "realised_win_rate": float(np.mean([r["won"] for r in rec])),
        "predicted_pnl_pts": float(pred.mean()), "realised_pnl_pts": float(pts.mean()),
        "top": {"n": len(top), "win_rate": float(np.mean([r["won"] for r in top])) if top else 0.0,
                "mean_pts": float(tp.mean()) if len(tp) else 0.0, "total_rs": float(eq[-1]) if len(eq) else 0.0,
                "worst_rs": float(min(r["realised_pnl"] for r in top)) if top else 0.0,
                "max_drawdown_rs": float((peak - eq).max()) if len(eq) else 0.0,
                "by_strategy": _count([r["strategy"] for r in top]),
                "equity": [{"day": r["day"], "label": r["label"], "pnl": r["realised_pnl"], "cum": float(c)}
                           for r, c in zip(top, eq)]},
    }


def _count(xs) -> dict:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def data_checks(rows: list[dict], history) -> dict:
    """Sanity of the rebuilt markets: forward vs index close, ATM implied vol vs India VIX."""
    per_day = {}
    for r in rows:
        per_day.setdefault(r["day"], r)
    days = sorted(per_day)
    basis = np.array([per_day[d]["forward"] / per_day[d]["spot"] - 1 for d in days])
    atm = np.array([per_day[d]["atm_vol"] for d in days])
    vix = []
    for d in days:
        try:
            i = history.dates.index(date.fromisoformat(d))
            vix.append(float(history.series("INDIAVIX")[i]) / 100)
        except (ValueError, KeyError):
            vix.append(math.nan)
    vix = np.array(vix)
    ok = np.isfinite(vix)
    rv = np.array([per_day[d]["realised_vol"] or math.nan for d in days])
    moves = np.array([per_day[d]["move"] for d in days])
    T = np.array([per_day[d]["days"] for d in days]) / 365
    z = moves / (atm * np.sqrt(T))
    return {"days": len(days), "basis_mean": float(np.mean(basis)), "basis_abs_max": float(np.max(np.abs(basis))),
            "atm_vol_mean": float(np.mean(atm)), "vix_mean": float(np.nanmean(vix)),
            "atm_vs_vix_corr": float(np.corrcoef(atm[ok], vix[ok])[0, 1]) if ok.sum() > 2 else None,
            "realised_vol_mean": float(np.nanmean(rv)),
            "move_over_implied_sd": float(np.std(z)),
            "share_moves_beyond_2sd": float(np.mean(np.abs(z) > 2))}
