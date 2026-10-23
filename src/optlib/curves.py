"""INR discount curve bootstrapped from FBIL MIBOR-OIS rates.

Conventions (FBIL / CCIL MIBOR-OIS):
- Floating leg: FBIL overnight MIBOR, Actual/365.
- Tenors up to 1 year: single (bullet) exchange at maturity; fixed leg R x tau, tau = days/365.
- Tenors above 1 year: semi-annual fixed payments, Actual/365.
- Start date: next business day after the curve date (T+1); the stub from the curve date to
  the start is discounted at overnight MIBOR.
- Business days are weekdays only; Indian market holidays are not modelled (effect on a
  discount factor is a fraction of a basis point).

Interpolation: log-linear in discount factors (piecewise-constant forward rates), flat
forward beyond the last pillar. Times are Actual/365 from the curve's reference date.
"""
from __future__ import annotations

import bisect
import calendar
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Mapping, Optional

from scipy.optimize import brentq

from .data.models import IST

_SECONDS_PER_YEAR = 365.0 * 24 * 3600


# ------------------------------------------------------------------ dates

def next_business_day(d: date) -> date:
    d = d + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def following(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def tenor_months(tenor: str) -> int:
    """'1M' -> 1, '9M' -> 9, '1Y' -> 12, '5Y' -> 60."""
    t = tenor.strip().upper()
    n, unit = int(t[:-1]), t[-1]
    if unit == "M":
        return n
    if unit == "Y":
        return 12 * n
    raise ValueError(f"unsupported tenor {tenor!r}")


def _yf(start: date, end: date) -> float:
    return (end - start).days / 365.0


# ------------------------------------------------------------------ inputs

@dataclass(frozen=True)
class OISQuotes:
    """One day's MIBOR-OIS fixings, rates in percent as FBIL publishes them."""

    fixing_date: date
    overnight: float  # FBIL overnight MIBOR, %
    tenors: Mapping[str, float]  # e.g. {"1M": 5.70, ..., "5Y": 6.77}, %
    source: str = "FBIL"

    def __post_init__(self):
        if not self.tenors:
            raise ValueError("no OIS tenors given")
        rates = [self.overnight, *self.tenors.values()]
        for t in self.tenors:
            tenor_months(t)
        if not all(-5.0 < r < 50.0 for r in rates):
            raise ValueError(f"rates out of range: {rates} (expected percent, e.g. 5.70)")
        if max(abs(r) for r in rates) < 0.5:
            raise ValueError(f"rates {rates} look like decimals; pass percent, e.g. 5.70 not 0.057")


# ------------------------------------------------------------------ curve

@dataclass
class DiscountCurve:
    reference_date: date
    times: list[float]  # pillar times (years), strictly increasing, first is 0
    dfs: list[float]  # discount factors at pillars, first is 1
    source_date: Optional[date] = None  # date of the market data used
    label: str = ""
    pillars: dict = field(default_factory=dict)  # tenor -> (date, df) for audit

    def __post_init__(self):
        if self.times[0] != 0.0 or self.dfs[0] != 1.0:
            raise ValueError("curve must start at t=0 with df=1")
        if any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("pillar times must increase")
        self._log = [math.log(x) for x in self.dfs]

    # core ----------------------------------------------------------------
    def df(self, t: float) -> float:
        """Discount factor from the reference date to t years (Act/365)."""
        if t <= 0:
            return 1.0
        times, logs = self.times, self._log
        if t >= times[-1]:
            # flat forward from the last segment
            slope = (logs[-1] - logs[-2]) / (times[-1] - times[-2])
            return math.exp(logs[-1] + slope * (t - times[-1]))
        i = bisect.bisect_right(times, t) - 1
        w = (t - times[i]) / (times[i + 1] - times[i])
        return math.exp(logs[i] + w * (logs[i + 1] - logs[i]))

    def zero_rate(self, t: float) -> float:
        """Continuously compounded zero rate to t years."""
        if t <= 0:
            t = 1e-6
        return -math.log(self.df(t)) / t

    def forward_rate(self, t1: float, t2: float) -> float:
        return math.log(self.df(t1) / self.df(t2)) / (t2 - t1)

    # datetimes -------------------------------------------------------------
    def time(self, when: datetime | date) -> float:
        ref = datetime.combine(self.reference_date, datetime.min.time(), tzinfo=IST)
        if isinstance(when, datetime):
            if when.tzinfo is None:
                raise ValueError("datetime must be timezone-aware")
            return (when - ref).total_seconds() / _SECONDS_PER_YEAR
        return _yf(self.reference_date, when)

    def discount(self, start: datetime | date, end: datetime | date) -> float:
        """Discount factor from ``start`` to ``end``."""
        return self.df(self.time(end)) / self.df(self.time(start))

    def rate_for_option(self, start: datetime, end: datetime) -> float:
        """Continuous rate r with exp(-r T) = discount(start, end), T = Act/365 seconds-based.
        This is the ``r`` to pass to Black-76 for an option alive from start to end."""
        T = (end - start).total_seconds() / _SECONDS_PER_YEAR
        if T <= 0:
            return self.zero_rate(1e-6)
        return -math.log(self.discount(start, end)) / T

    def age_days(self) -> Optional[int]:
        return None if self.source_date is None else (self.reference_date - self.source_date).days


def flat_curve(reference_date: date, rate: float, label: str = "flat") -> DiscountCurve:
    """Continuously compounded flat curve (for tests and what-ifs)."""
    return DiscountCurve(reference_date, [0.0, 50.0], [1.0, math.exp(-rate * 50.0)], label=label)


def bootstrap_mibor_ois(quotes: OISQuotes, reference_date: Optional[date] = None) -> DiscountCurve:
    """Bootstrap a discount curve from MIBOR-OIS quotes.

    ``reference_date`` defaults to the fixing date. Passing a later date builds the curve
    from today with the latest available rates held constant (used when the published
    rates lag the pricing date); the curve records ``source_date`` and ``age_days``.
    """
    ref = reference_date or quotes.fixing_date
    start = next_business_day(ref)
    on = quotes.overnight / 100.0
    df_start = 1.0 / (1.0 + on * _yf(ref, start))

    times = [0.0, _yf(ref, start)]
    dfs = [1.0, df_start]
    pillars = {"O/N": (start, df_start)}

    def curve_now(extra_t=None, extra_df=None):
        ts, ds = list(times), list(dfs)
        if extra_t is not None:
            ts.append(extra_t)
            ds.append(extra_df)
        return DiscountCurve(ref, ts, ds)

    for tenor, rate_pct in sorted(quotes.tenors.items(), key=lambda kv: tenor_months(kv[0])):
        n = tenor_months(tenor)
        R = rate_pct / 100.0
        maturity = following(add_months(start, n))
        t_mat = _yf(ref, maturity)
        if t_mat <= times[-1]:
            continue
        if n <= 12:
            df_mat = df_start / (1.0 + R * _yf(start, maturity))
        else:
            pay_dates = [following(add_months(start, 6 * i)) for i in range(1, n // 6 + 1)]

            def par_error(df_guess: float) -> float:
                c = curve_now(t_mat, df_guess)
                fixed, prev = 0.0, start
                for d in pay_dates:
                    fixed += R * _yf(prev, d) * c.df(_yf(ref, d))
                    prev = d
                floating = df_start - df_guess  # PV of compounded overnight from start to maturity
                return fixed - floating

            df_mat = brentq(par_error, 1e-6, df_start, xtol=1e-15)
        times.append(t_mat)
        dfs.append(df_mat)
        pillars[tenor] = (maturity, df_mat)

    return DiscountCurve(ref, times, dfs, source_date=quotes.fixing_date,
                         label=f"MIBOR-OIS {quotes.fixing_date.isoformat()}", pillars=pillars)
