"""One expiry, end to end: chain + curve -> rate, forward, QA, implied vols.

``build_vol_slice`` is the function the agent's tools call; its output is the input to
the SVI fit (bid/mid/ask vols, log-moneyness, total variance, fit weights).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .black import black76_greeks
from .curves import DiscountCurve
from .data.models import OptionChain, nse_expiry_instant
from .forwards import ForwardCheck, ForwardEstimate, check_against_future, parity_forward
from .iv import implied_vol_black76
from .qa import QAConfig, QAReport, review_chain


@dataclass
class VolSlice:
    underlying: str
    expiry: str
    as_of: str
    time_to_expiry: float
    rate: float
    discount_factor: float
    forward: ForwardEstimate
    forward_check: Optional[ForwardCheck]
    qa: QAReport
    strikes: np.ndarray
    option_types: np.ndarray
    bid: np.ndarray
    mid: np.ndarray
    ask: np.ndarray
    iv_bid: np.ndarray
    iv_mid: np.ndarray
    iv_ask: np.ndarray
    vega: np.ndarray  # Black-76 vega at iv_mid, per 1.00 vol
    no_volume: np.ndarray  # flag: no trades today

    @property
    def log_moneyness(self) -> np.ndarray:
        return np.log(self.strikes / self.forward.forward)

    @property
    def total_variance(self) -> np.ndarray:
        return self.iv_mid ** 2 * self.time_to_expiry

    @property
    def weights(self) -> np.ndarray:
        """Fit weights: inverse bid-ask width in vol, halved for quotes with no volume today."""
        width = self.iv_ask - self.iv_bid
        finite = np.isfinite(width)
        fallback = width[finite].max() if finite.any() else 1.0  # bid below intrinsic: treat as widest
        w = 1.0 / np.maximum(np.where(finite, width, fallback), 1e-4)
        return np.where(self.no_volume, 0.5 * w, w)

    def atm_vol(self) -> float:
        return float(np.interp(0.0, self.log_moneyness, self.iv_mid))

    def to_records(self) -> list[dict]:
        k = self.log_moneyness
        return [{"strike": float(self.strikes[i]), "type": str(self.option_types[i]), "log_moneyness": float(k[i]),
                 "bid": float(self.bid[i]), "mid": float(self.mid[i]), "ask": float(self.ask[i]),
                 "iv_bid": float(self.iv_bid[i]), "iv_mid": float(self.iv_mid[i]), "iv_ask": float(self.iv_ask[i]),
                 "vega": float(self.vega[i]), "no_volume": bool(self.no_volume[i])}
                for i in range(len(self.strikes))]


def build_vol_slice(chain: OptionChain, curve: DiscountCurve, *, future_price: Optional[float] = None,
                    config: Optional[QAConfig] = None) -> VolSlice:
    T = chain.time_to_expiry()
    if T <= 0:
        raise ValueError(f"{chain.underlying} {chain.expiry} has expired")
    expiry_dt = nse_expiry_instant(chain.expiry)
    r = curve.rate_for_option(chain.as_of, expiry_dt)
    df = math.exp(-r * T)

    fwd = parity_forward(chain, df)
    check = check_against_future(fwd.forward, future_price) if future_price else None
    report = review_chain(chain, fwd.forward, df, config, curve_age_days=curve.age_days())
    if check and check.verdict != "PASS":
        report.warnings.append(f"{'FAIL' if check.verdict == 'FAIL' else 'WARN'} parity forward vs future "
                               f"gap {check.gap:+.3%}")

    kept = report.kept
    K = np.array([q.strike for q in kept], float)
    types = np.array([q.option_type for q in kept])
    bid = np.array([q.bid for q in kept], float)
    ask = np.array([q.ask for q in kept], float)
    mid = 0.5 * (bid + ask)
    F = fwd.forward
    iv_mid = implied_vol_black76(mid, F, K, T, r, types) if len(K) else np.array([])
    iv_bid = implied_vol_black76(bid, F, K, T, r, types) if len(K) else np.array([])
    iv_ask = implied_vol_black76(ask, F, K, T, r, types) if len(K) else np.array([])
    vega = black76_greeks(F, K, T, r, np.nan_to_num(iv_mid, nan=0.2), types).vega if len(K) else np.array([])
    no_vol = np.array(["no_volume_today" in d.flags for d in report.decisions if d.keep], bool)

    return VolSlice(underlying=chain.underlying, expiry=chain.expiry.isoformat(), as_of=chain.as_of.isoformat(),
                    time_to_expiry=T, rate=r, discount_factor=df, forward=fwd, forward_check=check, qa=report,
                    strikes=K, option_types=types, bid=bid, mid=mid, ask=ask, iv_bid=np.atleast_1d(iv_bid),
                    iv_mid=np.atleast_1d(iv_mid), iv_ask=np.atleast_1d(iv_ask), vega=np.atleast_1d(vega),
                    no_volume=no_vol)
