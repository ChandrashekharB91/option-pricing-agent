"""Closed-form European option pricing: generalised Black-Scholes (cost of carry b).

One core covers both phase-1 models:
    BSM on spot      b = r - q
    Black-76 on F    b = 0, underlying = F

All Greeks are returned in raw units (per 1.00 change in the input, per year for
time). Use ``Greeks.reporting_units()`` for desk units (vol point, calendar day, bp).
Time sensitivities (theta, charm) are d/dt with calendar time passing, i.e. -d/dT.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
from scipy.special import ndtr

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _is_call(option_type) -> np.ndarray:
    """Accept 'call'/'put', 'C'/'P', booleans, or arrays of these."""
    arr = np.asarray(option_type)
    if arr.dtype == bool:
        return arr
    lowered = np.char.lower(arr.astype(str))
    calls = np.isin(lowered, ["call", "c"])
    puts = np.isin(lowered, ["put", "p"])
    if not np.all(calls | puts):
        raise ValueError(f"option_type must be call/put, got {option_type!r}")
    return calls


def _validate(S, K, T, sigma) -> None:
    if np.any(S <= 0) or np.any(K <= 0):
        raise ValueError("underlying and strike must be positive")
    if np.any(T < 0) or np.any(sigma < 0):
        raise ValueError("time to expiry and volatility must be non-negative")


@dataclass(frozen=True)
class Greeks:
    """Price and Greeks in raw units. ``phi`` is None for Black-76 (no dividend yield)."""

    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray
    rho: np.ndarray
    vanna: np.ndarray
    volga: np.ndarray
    charm: np.ndarray
    speed: np.ndarray
    zomma: np.ndarray
    phi: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}

    def reporting_units(self) -> dict:
        """Desk units: vega/vanna per vol point, volga per vol point^2, theta/charm per
        calendar day (365), rho/phi per basis point."""
        out = self.to_dict()
        out["vega"] = self.vega / 100.0
        out["vanna"] = self.vanna / 100.0
        out["volga"] = self.volga / 1.0e4
        out["zomma"] = self.zomma / 100.0
        out["theta"] = self.theta / 365.0
        out["charm"] = self.charm / 365.0
        out["rho"] = self.rho / 1.0e4
        if self.phi is not None:
            out["phi"] = self.phi / 1.0e4
        return out


def _generalised(S, K, T, sigma, r, b, is_call):
    """Price and Greeks of the generalised BS model (Haug). Returns dict of arrays."""
    S, K, T, sigma, r, b, is_call = np.broadcast_arrays(
        np.asarray(S, float), np.asarray(K, float), np.asarray(T, float),
        np.asarray(sigma, float), np.asarray(r, float), np.asarray(b, float), is_call,
    )
    _validate(S, K, T, sigma)

    live = (T > 0) & (sigma > 0)
    Ts = np.where(live, T, 1.0)
    vs = np.where(live, sigma, 1.0)

    sqrtT = np.sqrt(Ts)
    vol_t = vs * sqrtT
    carry_df = np.exp((b - r) * T)  # e^{(b-r)T}
    df = np.exp(-r * T)
    w = np.where(is_call, 1.0, -1.0)

    d1 = (np.log(S / K) + (b + 0.5 * vs * vs) * Ts) / vol_t
    d2 = d1 - vol_t
    nd1 = _pdf(d1)
    Nwd1 = ndtr(w * d1)
    Nwd2 = ndtr(w * d2)

    price = w * (S * carry_df * Nwd1 - K * df * Nwd2)
    delta = w * carry_df * Nwd1
    gamma = carry_df * nd1 / (S * vol_t)
    vega = S * carry_df * nd1 * sqrtT
    theta = (-S * carry_df * nd1 * vs / (2.0 * sqrtT)
             - w * (b - r) * S * carry_df * Nwd1
             - w * r * K * df * Nwd2)
    vanna = -carry_df * nd1 * d2 / vs
    volga = vega * d1 * d2 / vs
    charm = -carry_df * (nd1 * (b / vol_t - d2 / (2.0 * Ts)) + w * (b - r) * Nwd1)
    speed = -gamma / S * (1.0 + d1 / vol_t)
    zomma = gamma * (d1 * d2 - 1.0) / vs
    carry_sens = w * Ts * S * carry_df * Nwd1  # dV/db
    rho_spot = w * K * Ts * df * Nwd2  # BSM dV/dr with b = r - q

    # Expired or zero-vol options: discounted forward intrinsic, step delta, other Greeks 0.
    fwd = S * np.exp(b * T)
    itm = w * (fwd - K) > 0
    dead_price = df * np.maximum(w * (fwd - K), 0.0)
    dead_delta = np.where(itm, w * carry_df, 0.0)
    zero = np.zeros_like(price)

    def pick(alive_val, dead_val=zero):
        return np.where(live, alive_val, dead_val)

    return {
        "price": pick(price, dead_price),
        "delta": pick(delta, dead_delta),
        "gamma": pick(gamma),
        "vega": pick(vega),
        "theta": pick(theta),
        "vanna": pick(vanna),
        "volga": pick(volga),
        "charm": pick(charm),
        "speed": pick(speed),
        "zomma": pick(zomma),
        "carry_sens": pick(carry_sens, np.where(itm, w * T * S * carry_df, 0.0)),
        "rho_spot": pick(rho_spot, np.where(itm, w * K * T * df, 0.0)),
        "T": T,
    }


def _squeeze(x):
    x = np.asarray(x)
    return x.item() if x.ndim == 0 else x


# ---------------------------------------------------------------- BSM (spot)

def bsm_price(S, K, T, r, q, sigma, option_type="call"):
    """Black-Scholes-Merton price on spot with continuous dividend yield q."""
    res = _generalised(S, K, T, sigma, r, np.asarray(r, float) - np.asarray(q, float), _is_call(option_type))
    return _squeeze(res["price"])


def bsm_greeks(S, K, T, r, q, sigma, option_type="call") -> Greeks:
    """BSM price and Greeks w.r.t. spot. rho = dV/dr (q fixed), phi = dV/dq."""
    res = _generalised(S, K, T, sigma, r, np.asarray(r, float) - np.asarray(q, float), _is_call(option_type))
    rho = res["rho_spot"]
    return Greeks(
        price=_squeeze(res["price"]), delta=_squeeze(res["delta"]), gamma=_squeeze(res["gamma"]),
        vega=_squeeze(res["vega"]), theta=_squeeze(res["theta"]), rho=_squeeze(rho),
        vanna=_squeeze(res["vanna"]), volga=_squeeze(res["volga"]), charm=_squeeze(res["charm"]),
        speed=_squeeze(res["speed"]), zomma=_squeeze(res["zomma"]),
        phi=_squeeze(-res["carry_sens"]),
    )


# ---------------------------------------------------------------- Black-76 (forward / futures)

def black76_price(F, K, T, r, sigma, option_type="call"):
    """Black-76 price of a European option on a forward or future F, discounted at r."""
    res = _generalised(F, K, T, sigma, r, 0.0, _is_call(option_type))
    return _squeeze(res["price"])


def black76_greeks(F, K, T, r, sigma, option_type="call") -> Greeks:
    """Black-76 price and Greeks w.r.t. the forward F. rho = dV/dr with F fixed = -T V."""
    res = _generalised(F, K, T, sigma, r, 0.0, _is_call(option_type))
    rho = -res["T"] * res["price"]
    return Greeks(
        price=_squeeze(res["price"]), delta=_squeeze(res["delta"]), gamma=_squeeze(res["gamma"]),
        vega=_squeeze(res["vega"]), theta=_squeeze(res["theta"]), rho=_squeeze(rho),
        vanna=_squeeze(res["vanna"]), volga=_squeeze(res["volga"]), charm=_squeeze(res["charm"]),
        speed=_squeeze(res["speed"]), zomma=_squeeze(res["zomma"]), phi=None,
    )
