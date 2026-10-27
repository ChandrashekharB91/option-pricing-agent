"""Book risk: historical-simulation VaR and Expected Shortfall, backtesting, stress tests, limits.

Scenario model (one day of history = one scenario):
  - index move: the day's log return, applied to each option's forward;
  - volatility move: the day's relative change in India VIX, applied to each option's implied vol
    (sticky strike, so the smile keeps its shape). India VIX is computed from NIFTY options; for the other
    indices it is a proxy;
  - time: VaR uses a 1-day horizon (theta included); stress tests are instantaneous.
Every option is fully revalued with Black-76 in every scenario; no Greek approximations.

VaR/ES:   99% 1-day VaR = minus the 1st percentile of scenario P&L; 97.5% ES = minus the mean of the
          worst 2.5% (Basel FRTB convention). ES contributions per position add up exactly to ES.
Backtest: today's book is held fixed and revalued on each of the last 250 days (hypothetical P&L), against
          the VaR estimated from the preceding window. Kupiec (frequency) and Christoffersen (independence)
          tests, and the Basel traffic light (green 0-4, amber 5-9, red 10+ exceptions in 250 days at 99%).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import chi2

from .black import black76_price
from .data.history import History

VOL_SERIES = "INDIAVIX"
DAY = 1.0 / 365.0

NAMED_EVENTS = {  # moves verified against the stored NIFTY / India VIX history
    date(2015, 8, 24): "China devaluation sell-off",
    date(2019, 9, 20): "Corporate tax cut rally",
    date(2020, 3, 12): "COVID sell-off (WHO pandemic declaration)",
    date(2020, 3, 23): "COVID crash (largest one-day fall)",
    date(2020, 4, 7): "COVID rebound (largest one-day rise)",
    date(2021, 2, 1): "Union Budget 2021 rally",
    date(2022, 2, 24): "Russia-Ukraine invasion",
    date(2024, 6, 3): "Exit-poll rally 2024",
    date(2024, 6, 4): "General election results 2024",
}

HYPOTHETICAL = [  # (label, index move, vol multiplier): assumptions, not history
    ("Gap down 10%, vol x2", -0.10, 2.00),
    ("Gap down 5%, vol +50%", -0.05, 1.50),
    ("Gap down 3%, vol +25%", -0.03, 1.25),
    ("Vol spike +50%, index flat", 0.00, 1.50),
    ("Vol crush -30%, index flat", 0.00, 0.70),
    ("Gap up 3%, vol -10%", 0.03, 0.90),
    ("Gap up 5%, vol -20%", 0.05, 0.80),
]


# ------------------------------------------------------------------ positions and revaluation

@dataclass(frozen=True)
class Leg:
    label: str
    underlying: str
    expiry: str
    strike: float
    option_type: str
    quantity: float  # signed units
    forward: float
    rate: float
    time_to_expiry: float
    vol: float
    price: float  # per unit, today

    @classmethod
    def from_result(cls, r: dict) -> "Leg":
        """From a PricingResult.to_dict()."""
        lots = r["quantity"]
        return cls(label=f"{r['underlying']} {r['strike']:,.0f} {r['option_type']} {r['expiry']}",
                   underlying=r["underlying"], expiry=r["expiry"],
                   strike=r["strike"], option_type=r["option_type"], quantity=lots, forward=r["forward"],
                   rate=r["rate"], time_to_expiry=r["time_to_expiry"], vol=r["vol"], price=r["price"])


def revalue(legs: list[Leg], log_ret, vol_mult, dt: float = 0.0) -> np.ndarray:
    """P&L matrix (scenarios x legs) in Rs from full Black-76 revaluation.

    ``log_ret`` is one array applied to every leg, or a dict {underlying: array} on common dates."""
    vol_mult = np.atleast_1d(np.asarray(vol_mult, float))[:, None]
    if isinstance(log_ret, dict):
        moves = np.column_stack([np.asarray(log_ret[l.underlying], float) for l in legs])
    else:
        moves = np.atleast_1d(np.asarray(log_ret, float))[:, None]
    F = np.array([l.forward for l in legs])[None, :] * np.exp(moves)
    K = np.array([l.strike for l in legs])[None, :]
    T = np.maximum(np.array([l.time_to_expiry for l in legs]) - dt, 0.0)[None, :]
    r = np.array([l.rate for l in legs])[None, :]
    vol = np.array([l.vol for l in legs])[None, :] * vol_mult
    types = np.array([l.option_type for l in legs])[None, :]
    F, K, T, r, vol, types = np.broadcast_arrays(F, K, T, r, vol, types)
    new = black76_price(F, K, T, r, vol, types)
    base = np.array([l.price for l in legs])[None, :]
    qty = np.array([l.quantity for l in legs])[None, :]
    return (new - base) * qty


# ------------------------------------------------------------------ scenarios

@dataclass
class ScenarioSet:
    dates: list[date]
    log_ret: dict  # underlying -> log returns on these dates
    vol_mult: np.ndarray

    def total_move(self, underlying: str) -> np.ndarray:
        return self.log_ret[underlying]


def historical_scenarios(history: History, underlyings, window: Optional[int] = 500) -> ScenarioSet:
    """One scenario per trading day on which every underlying and India VIX have data: index log returns
    and the India VIX ratio (most recent ``window`` days)."""
    names = [underlyings] if isinstance(underlyings, str) else list(dict.fromkeys(underlyings))
    vix = history.series(VOL_SERIES)
    ok = np.isfinite(vix[1:]) & np.isfinite(vix[:-1])
    for n in names:
        px = history.series(n)
        ok &= np.isfinite(px[1:]) & np.isfinite(px[:-1])
    idx = np.flatnonzero(ok) + 1
    if window:
        idx = idx[-window:]
    if idx.size < 50:
        raise ValueError(f"only {idx.size} usable days of {', '.join(names)} / India VIX history")
    rets = {n: np.log(history.series(n)[idx] / history.series(n)[idx - 1]) for n in names}
    return ScenarioSet([history.dates[i] for i in idx], rets, vix[idx] / vix[idx - 1])


# ------------------------------------------------------------------ VaR / ES

@dataclass
class VarResult:
    var: float
    es: float
    var_level: float
    es_level: float
    window: int
    horizon_days: int
    first_date: str
    last_date: str
    pnl: list  # scenario P&L of the book, oldest first
    dates: list
    worst: dict
    es_contributions: list  # [{"label", "contribution"}], sums to es
    method: str = ("historical simulation, full Black-76 revaluation; index log return and relative India VIX "
                   "change per day; sticky strike")

    def to_dict(self) -> dict:
        return asdict(self)


def var_es(legs: list[Leg], scen: ScenarioSet, var_level: float = 0.99, es_level: float = 0.975,
           horizon_days: int = 1) -> VarResult:
    P = revalue(legs, scen.log_ret, scen.vol_mult, dt=horizon_days * DAY)
    total = P.sum(axis=1)
    var = -float(np.quantile(total, 1 - var_level))
    cut = np.quantile(total, 1 - es_level)
    tail = total <= cut
    es = -float(total[tail].mean())
    contrib = -P[tail].mean(axis=0)
    i = int(np.argmin(total))
    return VarResult(var=var, es=es, var_level=var_level, es_level=es_level, window=len(total),
                     horizon_days=horizon_days, first_date=scen.dates[0].isoformat(),
                     last_date=scen.dates[-1].isoformat(), pnl=total.tolist(),
                     dates=[d.isoformat() for d in scen.dates],
                     worst={"date": scen.dates[i].isoformat(), "pnl": float(total[i]),
                            "index_moves": {n: float(math.expm1(v[i])) for n, v in scen.log_ret.items()},
                            "vol_mult": float(scen.vol_mult[i])},
                     es_contributions=[{"label": l.label, "contribution": float(c)} for l, c in zip(legs, contrib)])


# ------------------------------------------------------------------ backtest

def _xlogy(x, y):
    return 0.0 if x == 0 else x * math.log(y)


def kupiec(exceptions: int, n: int, p: float) -> tuple[float, float]:
    """Proportion-of-failures likelihood ratio and p-value (chi-square, 1 dof)."""
    x = exceptions
    phat = x / n
    lr = -2 * (_xlogy(n - x, 1 - p) + _xlogy(x, p) - _xlogy(n - x, 1 - phat) - _xlogy(x, phat))
    return lr, float(chi2.sf(lr, 1))


def christoffersen(hits: np.ndarray) -> tuple[float, float]:
    """Independence of exceptions (clustering): likelihood ratio and p-value (chi-square, 1 dof)."""
    h = np.asarray(hits, int)
    a, b = h[:-1], h[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    pi0 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi1 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    l1 = _xlogy(n00, 1 - pi0) + _xlogy(n01, pi0) + _xlogy(n10, 1 - pi1) + _xlogy(n11, pi1)
    lr = max(-2 * (l0 - l1), 0.0)
    return lr, float(chi2.sf(lr, 1))


def traffic_light(exceptions: int, n: int = 250) -> str:
    """Basel zones for 99% VaR over 250 days (scaled for other lengths)."""
    x = exceptions * 250 / n
    return "green" if x <= 4 else ("amber" if x <= 9 else "red")


@dataclass
class BacktestResult:
    exceptions: int
    days: int
    expected: float
    var_level: float
    kupiec_lr: float
    kupiec_p: float
    christoffersen_lr: float
    christoffersen_p: float
    traffic_light: str
    dates: list
    pnl: list
    var: list
    note: str = ("hypothetical P&L: today's book held fixed and revalued on each past day's market move, "
                 "against VaR from the preceding window")

    def to_dict(self) -> dict:
        return asdict(self)


def backtest(legs: list[Leg], history: History, window: int = 500, test_days: int = 250,
             var_level: float = 0.99, horizon_days: int = 1) -> BacktestResult:
    scen = historical_scenarios(history, [l.underlying for l in legs], window + test_days)
    total = revalue(legs, scen.log_ret, scen.vol_mult, dt=horizon_days * DAY).sum(axis=1)
    n_win = len(total) - test_days
    if n_win < 100:
        raise ValueError("not enough history for the backtest window")
    var = np.array([-np.quantile(total[i:i + n_win], 1 - var_level) for i in range(test_days)])
    realised = total[n_win:]
    hits = realised < -var
    x = int(hits.sum())
    k_lr, k_p = kupiec(x, test_days, 1 - var_level)
    c_lr, c_p = christoffersen(hits)
    return BacktestResult(exceptions=x, days=test_days, expected=test_days * (1 - var_level), var_level=var_level,
                          kupiec_lr=k_lr, kupiec_p=k_p, christoffersen_lr=c_lr, christoffersen_p=c_p,
                          traffic_light=traffic_light(x, test_days),
                          dates=[d.isoformat() for d in scen.dates[n_win:]], pnl=realised.tolist(), var=var.tolist())


# ------------------------------------------------------------------ stress tests

def stress_tests(legs: list[Leg], history: Optional[History] = None) -> list[dict]:
    """Named historical days replayed on today's book (each index with its own move that day), plus
    hypothetical shocks applied to every index. Instantaneous."""
    names = list(dict.fromkeys(l.underlying for l in legs))
    rows = []
    if history is not None:
        try:
            scen = historical_scenarios(history, names, window=None)
        except ValueError:
            scen = None
        lookup = {d: i for i, d in enumerate(scen.dates)} if scen else {}
        for d, label in NAMED_EVENTS.items():
            i = lookup.get(d)
            if i is not None:
                rows.append({"scenario": f"{label} ({d:%d %b %Y})", "kind": "historical",
                             "index_move": float(math.expm1(scen.log_ret[names[0]][i])),
                             "vol_mult": float(scen.vol_mult[i]),
                             "log_ret": {n: float(scen.log_ret[n][i]) for n in names}})
    for label, move, mult in HYPOTHETICAL:
        rows.append({"scenario": label, "kind": "hypothetical", "index_move": move, "vol_mult": mult,
                     "log_ret": {n: math.log1p(move) for n in names}})
    P = revalue(legs, {n: [r["log_ret"][n] for r in rows] for n in names}, [r["vol_mult"] for r in rows])
    for r, p in zip(rows, P):
        r["pnl"] = float(p.sum())
        r["by_leg"] = [{"label": l.label, "pnl": float(x)} for l, x in zip(legs, p)]
        r["index_moves"] = {n: float(math.expm1(v)) for n, v in r.pop("log_ret").items()}
    return sorted(rows, key=lambda r: r["pnl"])


# ------------------------------------------------------------------ limits

@dataclass
class Limits:
    """Placeholders: the risk team sets these (repo file config/risk_limits.json overrides them)."""
    delta_units: float = 500.0  # |net delta| in index units (vs forward)
    gamma: float = 1.0  # |net gamma| per index point
    vega_rs: float = 50_000.0  # |net vega| in Rs per vol point
    theta_rs: float = 50_000.0  # |net theta| in Rs per day
    var99_rs: float = 500_000.0  # 1-day 99% VaR in Rs
    stress_rs: float = 1_500_000.0  # worst stress-test loss in Rs
    warn_at: float = 0.80  # utilisation that turns a limit amber

    @classmethod
    def load(cls, path: Optional[Path]) -> "Limits":
        if path and Path(path).is_file():
            return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        return cls()


def check_limits(totals: dict, var: Optional[float], worst_stress: Optional[float], limits: Limits) -> list[dict]:
    items = [("Net delta", abs(totals["delta"]), limits.delta_units, "index units"),
             ("Net gamma", abs(totals["gamma"]), limits.gamma, "per point"),
             ("Net vega", abs(totals["vega"]), limits.vega_rs, "Rs per vol pt"),
             ("Net theta", abs(totals["theta"]), limits.theta_rs, "Rs per day")]
    if var is not None:
        items.append(("1-day 99% VaR", var, limits.var99_rs, "Rs"))
    if worst_stress is not None:
        items.append(("Worst stress loss", max(-worst_stress, 0.0), limits.stress_rs, "Rs"))
    out = []
    for name, value, limit, unit in items:
        u = value / limit if limit else math.inf
        out.append({"limit": name, "value": float(value), "max": float(limit), "unit": unit, "utilisation": float(u),
                    "status": "FAIL" if u > 1 else ("WARN" if u >= limits.warn_at else "PASS")})
    return out


def greeks_by_expiry(positions: list[dict]) -> list[dict]:
    """Sum position Greeks and value per expiry (positions as returned by the book service)."""
    out: dict[str, dict] = {}
    for p in positions:
        e = out.setdefault(p["expiry"], {"expiry": p["expiry"], "value": 0.0, "delta": 0.0, "gamma": 0.0,
                                         "vega": 0.0, "theta": 0.0})
        for k in ("value", "delta", "gamma", "vega", "theta"):
            e[k] += p[k]
    return [out[k] for k in sorted(out)]
