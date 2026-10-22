"""Implied volatility for European options (Black-76 and BSM).

Method: convert every quote to its out-of-the-money equivalent via put-call parity
(better conditioned), then a safeguarded Newton iteration on the undiscounted Black
price with a shrinking bisection bracket. The Manaster-Koehler starting point
sqrt(2|ln(F/K)|/T) keeps Newton monotone; bisection takes over whenever a Newton step
leaves the bracket, so every quote inside the no-arbitrage bounds converges.

Status codes (``return_status=True``):
    0 converged, 1 below intrinsic, 2 above the no-arbitrage maximum,
    3 did not converge, 4 expired or invalid inputs.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .black import _is_call, _pdf

OK, BELOW_INTRINSIC, ABOVE_MAX, NO_CONVERGENCE, INVALID = 0, 1, 2, 3, 4

_VOL_MIN, _VOL_MAX = 1e-8, 20.0


def _black_undiscounted(F, K, T, sigma, w):
    vol_t = sigma * np.sqrt(T)
    d1 = np.log(F / K) / vol_t + 0.5 * vol_t
    d2 = d1 - vol_t
    price = w * (F * ndtr(w * d1) - K * ndtr(w * d2))
    vega = F * _pdf(d1) * np.sqrt(T)
    return price, vega


def implied_vol_black76(price, F, K, T, r, option_type="call", *, tol=1e-12, max_iter=100,
                        return_status=False):
    """Implied Black-76 volatility of a European option on forward/future F."""
    price, F, K, T, r, is_call = np.broadcast_arrays(
        np.asarray(price, float), np.asarray(F, float), np.asarray(K, float),
        np.asarray(T, float), np.asarray(r, float), _is_call(option_type),
    )
    shape = price.shape
    price, F, K, T, r, is_call = (a.ravel() for a in (price, F, K, T, r, is_call))
    n = price.size

    vol = np.full(n, np.nan)
    status = np.full(n, INVALID, dtype=int)

    valid = (T > 0) & (F > 0) & (K > 0) & np.isfinite(price)
    df = np.exp(-r * T)
    undisc = np.where(valid, price / np.where(valid, df, 1.0), np.nan)

    # Switch to the OTM option via parity: c - p = F - K (undiscounted)
    w_given = np.where(is_call, 1.0, -1.0)
    w_otm = np.where(K >= F, 1.0, -1.0)
    p_otm = np.where(w_given == w_otm, undisc, undisc - w_given * (F - K))
    upper = np.where(w_otm > 0, F, K)

    abs_tol = tol * np.maximum(np.abs(undisc), 1e-300)
    below = valid & (p_otm <= abs_tol)
    above = valid & (p_otm >= upper)
    status[below] = BELOW_INTRINSIC
    status[above] = ABOVE_MAX
    todo = valid & ~below & ~above

    if np.any(todo):
        idx = np.flatnonzero(todo)
        f_, k_, t_, w_, p_ = F[idx], K[idx], T[idx], w_otm[idx], p_otm[idx]
        lo = np.full(idx.size, _VOL_MIN)
        hi = np.full(idx.size, _VOL_MAX)
        x = np.abs(np.log(f_ / k_))
        guess = np.maximum(np.sqrt(2.0 * x / t_), np.sqrt(2.0 * np.pi / t_) * p_ / f_)
        sig = np.clip(guess, 1e-4, 5.0)
        done = np.zeros(idx.size, dtype=bool)
        target_tol = tol * p_

        for _ in range(max_iter):
            model, vega = _black_undiscounted(f_, k_, t_, sig, w_)
            diff = model - p_
            done |= np.abs(diff) <= target_tol
            if np.all(done):
                break
            hi = np.where(~done & (diff > 0), sig, hi)
            lo = np.where(~done & (diff < 0), sig, lo)
            with np.errstate(divide="ignore", invalid="ignore"):
                newton = sig - diff / vega
            ok_step = np.isfinite(newton) & (newton > lo) & (newton < hi)
            nxt = np.where(ok_step, newton, 0.5 * (lo + hi))
            done |= (hi - lo) <= 1e-15 * np.maximum(hi, 1.0)
            sig = np.where(done, sig, nxt)

        model, _ = _black_undiscounted(f_, k_, t_, sig, w_)
        converged = np.abs(model - p_) <= np.maximum(target_tol, 1e-14 * f_)
        vol[idx] = np.where(converged, sig, np.nan)
        status[idx] = np.where(converged, OK, NO_CONVERGENCE)

    vol = vol.reshape(shape)
    status = status.reshape(shape)
    if vol.ndim == 0:
        vol, status = vol.item(), int(status)
    return (vol, status) if return_status else vol


def implied_vol_bsm(price, S, K, T, r, q, option_type="call", **kwargs):
    """Implied BSM volatility on spot S with continuous dividend yield q."""
    S, T, r, q = (np.asarray(a, float) for a in (S, T, r, q))
    F = S * np.exp((r - q) * T)
    return implied_vol_black76(price, F, K, T, r, option_type, **kwargs)
