"""Volatility smile per expiry and the surface across expiries.

Pricing rule for one expiry (``ExpirySmile.vol``):
  - a listed strike that passed QA is priced at its own market mid vol (the market is the answer);
  - an unlisted strike inside the quoted range uses SVI corrected by the SVI-to-market gap of its
    neighbours, so the smile passes through every market point;
  - beyond the quoted range, the SVI wings (smooth, butterfly-checked, Lee-bounded).
The gap between SVI and the market at listed strikes is reported as model risk. Raw SVI cannot
match very short-dated NIFTY smiles to within their 0.01-0.03 vol-point bid-ask bands (on the
23 Oct 2026 four-day slice it is 0.2 vol points low at the money), which is why listed strikes use
the market directly.

Across expiries (``VolSurface.vol``): total variance is interpolated linearly in time at fixed
log-moneyness against each expiry's own forward (the forward itself is interpolated in log terms
from spot). Before the first expiry, vol is held at the first expiry's level; after the last,
at the last expiry's level. Calendar arbitrage (total variance falling with maturity) is checked.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np

from .curves import DiscountCurve
from .data.models import nse_expiry_instant, year_fraction
from .slices import VolSlice, build_vol_slice
from .svi import SVIFit, fit_slice

_CAL_GRID = np.linspace(-0.3, 0.3, 121)
CAL_TOL = 1e-6  # total variance tolerance for calendar checks


@dataclass
class ExpirySmile:
    slice: VolSlice
    fit: SVIFit

    @property
    def expiry(self) -> date:
        return date.fromisoformat(self.slice.expiry)

    @property
    def time_to_expiry(self) -> float:
        return self.slice.time_to_expiry

    @property
    def forward(self) -> float:
        return self.slice.forward.forward

    def market_vol(self, strike: float) -> Optional[float]:
        hit = np.flatnonzero(np.isclose(self.slice.strikes, strike, rtol=0, atol=1e-6))
        if hit.size and np.isfinite(self.slice.iv_mid[hit[0]]):
            return float(self.slice.iv_mid[hit[0]])
        return None

    def vol(self, strike: float) -> tuple[float, str]:
        """(implied vol, source).

        'market'        listed strike that passed QA: its own mid vol
        'svi_adjusted'  between listed strikes: SVI plus the SVI-to-market gap interpolated
                        linearly from the two neighbouring listed strikes (continuous with the market)
        'svi'           outside the quoted range: SVI wings
        """
        mv = self.market_vol(strike)
        if mv is not None:
            return mv, "market"
        svi = float(self.fit.vol(strike))
        K = self.slice.strikes
        if K.size and K[0] < strike < K[-1]:
            gap = float(np.interp(strike, K, self.slice.iv_mid - np.asarray(self.fit.vol(K))))
            return svi + gap, "svi_adjusted"
        return svi, "svi"

    def adjusted_convexity(self, step: float = 5.0) -> tuple[int, float]:
        """Butterfly check of the market-consistent layer on a fine strike grid inside the quoted range.
        Returns (number of negative butterflies, worst one in index points). Small values reflect quote
        noise; pure SVI (used across expiries and for exotics) is checked separately via g(k)."""
        from .black import black76_price

        K = self.slice.strikes
        grid = np.arange(K[0], K[-1] + 1e-9, step)
        vols = np.array([self.vol(k)[0] for k in grid])
        c = black76_price(self.forward, grid, self.time_to_expiry, self.slice.rate, vols, "call")
        fly = np.diff(c, 2)
        neg = fly[fly < -1e-6]
        return int(neg.size), float(-neg.min()) if neg.size else 0.0

    def gaps(self) -> np.ndarray:
        """SVI minus market vol at each listed (QA-kept) strike."""
        return np.asarray(self.fit.vol(self.slice.strikes)) - self.slice.iv_mid

    def summary(self) -> str:
        g = self.gaps()
        atm = self.fit.atm_gap(self.slice.atm_vol())
        return (f"{self.fit.summary()}\n  SVI vs market: ATM {atm * 100:+.3f} vol pts, largest gap "
                f"{np.max(np.abs(g)) * 100:.3f} at {self.slice.strikes[int(np.argmax(np.abs(g)))]:,.0f}")


@dataclass(frozen=True)
class CalendarViolation:
    expiries: tuple[str, str]
    k: float
    size: float  # total variance shortfall of the later expiry


@dataclass
class VolSurface:
    underlying: str
    as_of: datetime
    spot: float
    smiles: list[ExpirySmile]
    curve: DiscountCurve
    calendar: list[CalendarViolation] = field(default_factory=list)

    def __post_init__(self):
        self.smiles.sort(key=lambda s: s.time_to_expiry)
        self.calendar = self.calendar_check()

    # time and forward ---------------------------------------------------------
    def time_to(self, expiry: date | datetime | float) -> float:
        if isinstance(expiry, (int, float)):
            return float(expiry)
        if isinstance(expiry, datetime):
            return max((expiry - self.as_of).total_seconds(), 0.0) / (365 * 86400)
        return year_fraction(self.as_of, expiry)

    def forward(self, T: float) -> float:
        ts = [0.0] + [s.time_to_expiry for s in self.smiles]
        logf = [math.log(self.spot)] + [math.log(s.forward) for s in self.smiles]
        if T <= ts[-1]:
            return math.exp(float(np.interp(T, ts, logf)))
        slope = (logf[-1] - logf[-2]) / (ts[-1] - ts[-2])  # extend the last carry
        return math.exp(logf[-1] + slope * (T - ts[-1]))

    def rate(self, T: float) -> float:
        end = datetime.fromtimestamp(self.as_of.timestamp() + T * 365 * 86400, tz=self.as_of.tzinfo)
        return self.curve.rate_for_option(self.as_of, end)

    # vols -------------------------------------------------------------------
    def total_variance(self, k, T: float):
        sm = self.smiles
        k = np.asarray(k, float)
        if T <= sm[0].time_to_expiry:
            return sm[0].fit.params.total_variance(k) * T / sm[0].time_to_expiry
        if T >= sm[-1].time_to_expiry:
            return sm[-1].fit.params.total_variance(k) * T / sm[-1].time_to_expiry
        i = max(j for j, s in enumerate(sm) if s.time_to_expiry <= T)
        a, b = sm[i], sm[i + 1]
        x = (T - a.time_to_expiry) / (b.time_to_expiry - a.time_to_expiry)
        return (1 - x) * a.fit.params.total_variance(k) + x * b.fit.params.total_variance(k)

    def vol(self, strike: float, expiry: date | datetime | float) -> tuple[float, str]:
        """(implied vol, source) for any strike and expiry. Source: 'market', 'svi' or 'interpolated'."""
        T = self.time_to(expiry)
        for s in self.smiles:
            if abs(s.time_to_expiry - T) < 1e-9:
                return s.vol(strike)
        k = math.log(strike / self.forward(T))
        w = float(self.total_variance(k, T))
        return math.sqrt(max(w, 0.0) / T), "interpolated"

    # checks -----------------------------------------------------------------
    def calendar_check(self) -> list[CalendarViolation]:
        out = []
        for a, b in zip(self.smiles, self.smiles[1:]):
            wa = a.fit.params.total_variance(_CAL_GRID)
            wb = b.fit.params.total_variance(_CAL_GRID)
            short = wa - wb
            i = int(np.argmax(short))
            if short[i] > CAL_TOL:
                out.append(CalendarViolation((a.slice.expiry, b.slice.expiry), float(_CAL_GRID[i]), float(short[i])))
        return out

    @property
    def status(self) -> str:
        states = [s.fit.status for s in self.smiles] + [s.slice.qa.status for s in self.smiles]
        if self.calendar or "FAIL" in states:
            return "FAIL"
        return "WARN" if "WARN" in states else "PASS"

    def summary(self) -> str:
        lines = [f"Surface {self.underlying} at {self.as_of:%d %b %Y %H:%M}: {len(self.smiles)} expiries -> {self.status}"]
        for s in self.smiles:
            lines.append(f"  {s.slice.expiry}  T {s.time_to_expiry * 365:6.2f}d  F {s.forward:,.2f}  "
                         f"ATM {s.slice.atm_vol():.2%}  SVI RMSE {s.fit.rmse * 100:.3f} pts  QA {s.slice.qa.status}  "
                         f"fit {s.fit.status}")
        lines += [f"  calendar arbitrage between {v.expiries[0]} and {v.expiries[1]} at k={v.k:+.3f} "
                  f"(variance short by {v.size:.2e})" for v in self.calendar]
        return "\n".join(lines)


def build_surface(source, curve: DiscountCurve, underlying: str, expiries: Optional[list] = None,
                  max_expiries: int = 4, qa_config=None) -> VolSurface:
    """Fetch chains, build slices and SVI fits, and assemble the surface."""
    expiries = expiries or source.expiries(underlying)[:max_expiries]
    futures = {f.expiry: f for f in source.futures(underlying)}
    smiles, spot, as_of = [], None, None
    for e in expiries:
        chain = source.option_chain(underlying, e)
        fut = futures.get(chain.expiry)
        fut_px = None if fut is None else (fut.mid if not math.isnan(fut.mid) else fut.last)
        s = build_vol_slice(chain, curve, future_price=fut_px, config=qa_config)
        smiles.append(ExpirySmile(s, fit_slice(s)))
        spot, as_of = chain.spot, as_of or chain.as_of
    return VolSurface(underlying, as_of, spot, smiles, curve)
