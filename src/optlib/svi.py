"""SVI smile fit for one expiry, with butterfly-arbitrage checks.

Raw SVI (Gatheral) gives total implied variance w = sigma_iv^2 T as a function of
log-moneyness k = ln(K / F):

    w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + s^2))

Fitting uses the quasi-explicit method (Zeliade, 2009): for fixed (m, s), w is linear in
three transformed parameters, so the inner problem is a bounded linear least-squares solve and
only (m, s) are searched numerically. Bounds enforce w >= 0 and Roger Lee's wing limit
b (1 + |rho|) <= 2 (total-variance slope at most 2), which rules out arbitrage in the far wings.

Residuals are measured in implied-vol terms and weighted by the slice weights (inverse bid-ask
width, halved for no-volume quotes). Butterfly arbitrage is checked with Gatheral's density
condition g(k) >= 0; a penalty in the outer search pushes the fit away from violations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import lsq_linear, minimize

RMSE_WARN, RMSE_FAIL = 0.005, 0.015  # spec: fit RMSE 0.5 / 1.5 vol points
NEAR_TOL = 0.0025  # a fitted vol counts as "near" the market if within bid-ask or 0.25 vol pt of it
NEAR_WARN, NEAR_FAIL = 0.90, 0.75  # share of quotes near the market: warn below 90%, fail below 75%
G_TOL = -1e-6  # numerical tolerance on Gatheral's g
_K_GRID = np.linspace(-1.5, 1.5, 601)  # where arbitrage conditions are checked


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    s: float

    def total_variance(self, k):
        k = np.asarray(k, float)
        x = k - self.m
        return self.a + self.b * (self.rho * x + np.sqrt(x * x + self.s * self.s))

    def dw(self, k):
        x = np.asarray(k, float) - self.m
        return self.b * (self.rho + x / np.sqrt(x * x + self.s * self.s))

    def d2w(self, k):
        x = np.asarray(k, float) - self.m
        return self.b * self.s * self.s / (x * x + self.s * self.s) ** 1.5

    def g(self, k):
        """Gatheral's butterfly condition: the implied risk-neutral density is >= 0 iff g >= 0."""
        k = np.asarray(k, float)
        w, w1, w2 = self.total_variance(k), self.dw(k), self.d2w(k)
        return (1 - k * w1 / (2 * w)) ** 2 - w1 * w1 / 4 * (1 / w + 0.25) + w2 / 2

    def min_variance(self) -> float:
        return self.a + self.b * self.s * math.sqrt(max(1 - self.rho * self.rho, 0.0))

    def as_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "rho": self.rho, "m": self.m, "s": self.s}


@dataclass
class SVIFit:
    params: SVIParams
    time_to_expiry: float
    forward: float
    k: np.ndarray  # log-moneyness of the quotes fitted
    iv_market: np.ndarray
    iv_bid: np.ndarray
    iv_ask: np.ndarray
    weights: np.ndarray
    expiry: str = ""
    warnings: list = field(default_factory=list)

    # model -----------------------------------------------------------------
    def vol_k(self, k):
        w = self.params.total_variance(k)
        return np.sqrt(np.maximum(w, 0.0) / self.time_to_expiry)

    def vol(self, strike):
        """Fitted implied vol at any strike (listed or not)."""
        return self.vol_k(np.log(np.asarray(strike, float) / self.forward))

    @property
    def iv_model(self) -> np.ndarray:
        return self.vol_k(self.k)

    # diagnostics -------------------------------------------------------------
    @property
    def errors(self) -> np.ndarray:
        return self.iv_model - self.iv_market

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.errors ** 2)))

    @property
    def weighted_rmse(self) -> float:
        w = self.weights / self.weights.sum()
        return float(np.sqrt(np.sum(w * self.errors ** 2)))

    @property
    def max_abs_error(self) -> float:
        return float(np.max(np.abs(self.errors)))

    def share_inside(self, tolerance: float = 0.0) -> float:
        """Share of quotes whose fitted vol lies within [bid vol - tol, ask vol + tol].
        Quotes with no bid vol (bid below intrinsic) count from 0."""
        bid = np.where(np.isfinite(self.iv_bid), self.iv_bid, 0.0)
        ask = np.where(np.isfinite(self.iv_ask), self.iv_ask, np.inf)
        m = self.iv_model
        return float(np.mean((m >= bid - tolerance) & (m <= ask + tolerance)))

    def min_g(self) -> float:
        return float(np.min(self.params.g(_K_GRID)))

    @property
    def butterfly_free(self) -> bool:
        return self.min_g() >= G_TOL

    def atm_gap(self, market_atm_vol: float) -> float:
        """Fitted minus market at-the-money vol."""
        return float(self.vol_k(0.0)) - market_atm_vol

    @property
    def status(self) -> str:
        near = self.share_inside(NEAR_TOL)
        if self.rmse > RMSE_FAIL or near < NEAR_FAIL or not self.butterfly_free:
            return "FAIL"
        return "WARN" if self.rmse > RMSE_WARN or near < NEAR_WARN or self.warnings else "PASS"

    def summary(self) -> str:
        p = self.params
        return (f"SVI {self.expiry}: RMSE {self.rmse * 100:.3f} vol pts (weighted {self.weighted_rmse * 100:.3f}), "
                f"max error {self.max_abs_error * 100:.3f}, inside bid-ask {self.share_inside():.0%} "
                f"(within 0.25 pt {self.share_inside(0.0025):.0%}), min g {self.min_g():.4f} -> {self.status}\n"
                f"  a={p.a:.6f} b={p.b:.5f} rho={p.rho:.4f} m={p.m:.5f} s={p.s:.5f}  "
                f"ATM vol {float(self.vol_k(0.0)):.2%}")


# ------------------------------------------------------------------ fitting

def _inner(m: float, s: float, k, w_mkt, wt, w_max):
    """Best (a, u, v) for fixed (m, s): w = a + u (z + y)/2 + v (z - y)/2, y = (k-m)/s, z = sqrt(y^2+1).
    u = b s (1 + rho), v = b s (1 - rho); bounds 0 <= u, v <= 2 s (Lee), a <= max w.
    a may be negative (needed for steep short-dated smiles); non-negative total variance,
    a + b s sqrt(1 - rho^2) >= 0, is enforced by the outer penalty."""
    y = (k - m) / s
    z = np.sqrt(y * y + 1.0)
    A = np.column_stack([np.ones_like(y), 0.5 * (z + y), 0.5 * (z - y)]) * wt[:, None]
    res = lsq_linear(A, w_mkt * wt, bounds=([-4.0 * s, 0.0, 0.0], [w_max, 2 * s, 2 * s]), method="bvls")
    a, u, v = res.x
    c, d = 0.5 * (u + v), 0.5 * (u - v)
    b = c / s
    rho = d / c if c > 0 else 0.0
    return SVIParams(a, b, float(np.clip(rho, -0.999999, 0.999999)), m, s), res.cost


def fit_svi(k, iv, T: float, *, weights=None, iv_bid=None, iv_ask=None, forward: float = 1.0,
            expiry: str = "", arbitrage_penalty: float = 1e3) -> SVIFit:
    """Fit raw SVI to one expiry's implied vols at log-moneyness k."""
    k = np.asarray(k, float)
    iv = np.asarray(iv, float)
    ok = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    if ok.sum() < 5:
        raise ValueError(f"SVI needs at least 5 quotes, got {ok.sum()}")
    weights = np.ones_like(iv) if weights is None else np.asarray(weights, float)
    k, iv, weights = k[ok], iv[ok], weights[ok]
    iv_bid = np.full_like(iv, np.nan) if iv_bid is None else np.asarray(iv_bid, float)[ok]
    iv_ask = np.full_like(iv, np.nan) if iv_ask is None else np.asarray(iv_ask, float)[ok]

    w_mkt = iv * iv * T
    # residual in vol units: d(vol) = d(w) / (2 vol T)
    wt = np.sqrt(weights / weights.sum()) / (2.0 * iv * T)
    w_max = float(w_mkt.max())
    span = float(k.max() - k.min())

    def objective(x):
        m, log_s = x
        s = math.exp(log_s)
        p, cost = _inner(m, s, k, w_mkt, wt, w_max)
        floor = p.min_variance()
        if floor <= 0:  # negative variance somewhere: infeasible, push away proportionally
            return cost + 1e6 * (1e-12 - floor) + 1.0
        g_min = float(np.min(p.g(_K_GRID)))
        return cost + arbitrage_penalty * min(g_min, 0.0) ** 2

    best = None
    for m0 in np.linspace(k.min() + 0.25 * span, k.max() - 0.25 * span, 5):
        for s0 in (0.05 * span, 0.2 * span, 0.6 * span):
            r = minimize(objective, [m0, math.log(max(s0, 1e-4))], method="Nelder-Mead",
                         options={"xatol": 1e-7, "fatol": 1e-14, "maxiter": 2000})
            if best is None or r.fun < best.fun:
                best = r
    m, log_s = best.x
    params, _ = _inner(m, math.exp(log_s), k, w_mkt, wt, w_max)
    fit = SVIFit(params, T, forward, k, iv, iv_bid, iv_ask, weights, expiry=expiry)
    if params.min_variance() <= 0:
        fit.warnings.append("fit reaches zero variance: check the slice")
    for name, at_bound in (("left", params.b * (1 - params.rho) >= 2 - 1e-9),
                           ("right", params.b * (1 + params.rho) >= 2 - 1e-9)):
        if at_bound:
            fit.warnings.append(f"{name} wing at Roger Lee's slope limit")
    return fit


def fit_slice(vol_slice) -> SVIFit:
    """Fit SVI to a ``VolSlice`` from ``optlib.slices.build_vol_slice``."""
    s = vol_slice
    return fit_svi(s.log_moneyness, s.iv_mid, s.time_to_expiry, weights=s.weights, iv_bid=s.iv_bid,
                   iv_ask=s.iv_ask, forward=s.forward.forward, expiry=s.expiry)
