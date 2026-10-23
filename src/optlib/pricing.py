"""Price one European index option off the surface: the function behind the agent's price_option tool.

Every input that produced the number is carried in the result (forward, rate, vol and where the
vol came from, curve date, snapshot time), so the validator can re-check it and any run can be
reproduced.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date

from .black import black76_greeks
from .data.models import nse_expiry_instant
from .surface import VolSurface


@dataclass
class PricingResult:
    underlying: str
    strike: float
    expiry: str
    option_type: str
    quantity: float  # number of units (lots x lot size)
    as_of: str
    spot: float
    forward: float
    rate: float
    time_to_expiry: float
    vol: float
    vol_source: str  # market | svi_adjusted | svi | interpolated
    price: float  # per unit, index points
    greeks: dict  # per unit, desk units (vega per vol pt, theta per day, rho per bp)
    curve_label: str = ""
    curve_age_days: int | None = None
    inputs: dict = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.price * self.quantity

    def position_greeks(self) -> dict:
        return {k: v * self.quantity for k, v in self.greeks.items()}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["value"] = self.value
        d["position_greeks"] = self.position_greeks()
        return d


def price_option(surface: VolSurface, strike: float, expiry: date, option_type: str,
                 quantity: float = 1.0, allow_extrapolation: bool = False) -> PricingResult:
    """Price one option off the surface. Expiries after the last fitted expiry are refused unless
    ``allow_extrapolation``: holding a short-dated smile flat out to a later expiry is not a price."""
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    T = surface.time_to(nse_expiry_instant(expiry))
    if T <= 0:
        raise ValueError(f"{expiry} has expired")
    last = surface.smiles[-1]
    if T > last.time_to_expiry + 1e-9 and not allow_extrapolation:
        raise ValueError(f"{expiry} is beyond the last fitted expiry ({last.slice.expiry}); "
                         "build the surface with more expiries")
    smile = next((s for s in surface.smiles if s.expiry == expiry), None)
    if smile is not None:
        F, r = smile.forward, smile.slice.rate
        T = smile.time_to_expiry
    else:
        F, r = surface.forward(T), surface.rate(T)
    vol, source = surface.vol(strike, expiry if smile is not None else T)
    g = black76_greeks(F, strike, T, r, vol, option_type)
    return PricingResult(
        underlying=surface.underlying, strike=float(strike), expiry=expiry.isoformat(), option_type=option_type,
        quantity=float(quantity), as_of=surface.as_of.isoformat(), spot=surface.spot, forward=F, rate=r,
        time_to_expiry=T, vol=vol, vol_source=source, price=float(g.price),
        greeks={k: float(v) for k, v in g.reporting_units().items() if k != "price"},
        curve_label=surface.curve.label, curve_age_days=surface.curve.age_days(),
        inputs={"discount_factor": math.exp(-r * T), "listed_expiry": smile is not None},
    )
