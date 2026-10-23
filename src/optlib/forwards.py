"""Forward estimation from option prices (put-call parity) and checks against futures.

For European options, C - P = DF x (F - K) at every strike, so each strike with live
call and put quotes gives one forward estimate F_k = K + (C - P) / DF. Near-the-money
strikes are tightest; the estimate is a spread-weighted median with outliers removed.
This captures dividends, borrow and futures basis together, which spot x carry misses.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import numpy as np

from .data.models import OptionChain

FORWARD_WARN = 0.0005  # spec: parity forward vs future, warn above 0.05%
FORWARD_FAIL = 0.0020  # fail above 0.20%


@dataclass(frozen=True)
class ForwardEstimate:
    forward: float
    discount_factor: float
    strikes: tuple[float, ...]
    per_strike: tuple[float, ...]
    dispersion: float  # max - min of per-strike forwards used (index points)
    spot: float
    time_to_expiry: float

    @property
    def n_strikes(self) -> int:
        return len(self.strikes)

    @property
    def implied_carry(self) -> float:
        """Annualised ln(F / spot) / T: everything spot x carry would have to explain."""
        return math.log(self.forward / self.spot) / self.time_to_expiry if self.time_to_expiry > 0 else math.nan


@dataclass(frozen=True)
class ForwardCheck:
    forward: float
    future: float
    gap: float  # (forward - future) / future
    verdict: str  # PASS | WARN | FAIL


def _weighted_median(values, weights) -> float:
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cdf = np.cumsum(w) / np.sum(w)
    return float(v[np.searchsorted(cdf, 0.5)])


def parity_forward(chain: OptionChain, discount_factor: float, *, max_strikes: int = 11,
                   outlier_points: float = 5.0) -> ForwardEstimate:
    """Spread-weighted median of per-strike parity forwards, nearest the money first.

    Two passes: centre on spot, then re-centre on the first estimate. Strikes whose
    forward is more than ``max(outlier_points, 4 x MAD)`` from the median are dropped.
    """
    pairs = []
    for k in chain.strikes:
        c, p = chain.quote(k, "call"), chain.quote(k, "put")
        if c and p and c.two_sided and p.two_sided:
            pairs.append((k, k + (c.mid - p.mid) / discount_factor, c.spread + p.spread))
    if len(pairs) < 3:
        raise ValueError(f"need at least 3 strikes with live calls and puts, found {len(pairs)}")

    centre = chain.spot
    for _ in range(2):
        near = sorted(pairs, key=lambda x: abs(x[0] - centre))[:max_strikes]
        est = _weighted_median([f for _, f, _ in near], [1.0 / max(s, 1e-6) for _, _, s in near])
        centre = est

    mad = statistics.median(abs(f - est) for _, f, _ in near)
    keep = [x for x in near if abs(x[1] - est) <= max(outlier_points, 4.0 * mad)]
    est = _weighted_median([f for _, f, _ in keep], [1.0 / max(s, 1e-6) for _, _, s in keep])
    keep.sort()
    fwds = [f for _, f, _ in keep]
    return ForwardEstimate(forward=est, discount_factor=discount_factor, strikes=tuple(k for k, _, _ in keep),
                           per_strike=tuple(fwds), dispersion=max(fwds) - min(fwds), spot=chain.spot,
                           time_to_expiry=chain.time_to_expiry())


def parity_regression(chain: OptionChain, strikes) -> tuple[float, float]:
    """Least-squares fit of C - P = DF x F - DF x K across strikes.

    Returns (forward, discount_factor) implied by the options alone. Useful as a check on
    the curve for longer expiries; poorly determined for expiries of a few days.
    """
    ks, y = [], []
    for k in strikes:
        c, p = chain.quote(k, "call"), chain.quote(k, "put")
        if c and p and c.two_sided and p.two_sided:
            ks.append(k)
            y.append(c.mid - p.mid)
    if len(ks) < 3:
        raise ValueError("need at least 3 strikes")
    slope, intercept = np.polyfit(np.asarray(ks), np.asarray(y), 1)
    df = -slope
    return float(intercept / df), float(df)


def check_against_future(forward: float, future_price: float, *, warn: float = FORWARD_WARN,
                         fail: float = FORWARD_FAIL) -> ForwardCheck:
    gap = (forward - future_price) / future_price
    verdict = "PASS" if abs(gap) <= warn else ("WARN" if abs(gap) <= fail else "FAIL")
    return ForwardCheck(forward=forward, future=future_price, gap=gap, verdict=verdict)
