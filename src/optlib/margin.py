"""SPAN-style margin estimate for a strategy, used only to decide what fits the capital entered.

Not the exchange's SPAN. It follows the same idea: revalue the whole position under a grid of
price moves (up to the price scan range) and implied-vol shifts (the vol scan range), plus two extreme
moves counted at 35%, and take the worst loss. Short options and futures then add exposure margin
(and an extra amount on expiry day, per SEBI's October 2024 measures). Long options need only their
premium, which the projector counts separately. Parameters are in ``config/margin.json``; the real
numbers come from NSE's daily SPAN risk parameter files and a broker's margin calculator.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MarginParams:
    price_scan_range: float = 0.056
    vol_scan_range: float = 0.04
    extreme_move_multiple: float = 2.0
    extreme_move_cover: float = 0.35
    exposure_margin: float = 0.02
    expiry_day_extra_elm: float = 0.02
    capital_buffer: float = 0.10

    @classmethod
    def load(cls, path: Path) -> "MarginParams":
        raw = json.loads(Path(path).read_text())
        names = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in raw.items() if k in names})


def estimate_margin(legs, quantity_per_lot: float, state, params: MarginParams, lots: float = 1.0) -> dict:
    """Margin in rupees for ``lots`` units of a strategy. ``state`` is a projector.MarketState."""
    shorts = [l for l in legs if l.side < 0 and l.instrument != "future"]
    futs = [l for l in legs if l.instrument == "future"]
    if not shorts and not futs:
        return {"span": 0.0, "exposure": 0.0, "total": 0.0, "note": "long options only: premium paid upfront"}
    psr, vsr = params.price_scan_range, params.vol_scan_range
    moves = np.concatenate([np.linspace(-psr, psr, 7), [-params.extreme_move_multiple * psr,
                                                       params.extreme_move_multiple * psr]])
    cover = np.concatenate([np.ones(7), [params.extreme_move_cover] * 2])
    F0 = state.forward
    now = sum(l.signed * state.value(l, F0, state.T, 0.0) for l in legs)
    worst = 0.0
    for dv in (-vsr, vsr):
        F = F0 * (1.0 + moves)
        v = sum(l.signed * state.value(l, F, state.T, dv) for l in legs)
        loss = -(v - now) * cover
        worst = max(worst, float(np.max(loss)))
    qty = quantity_per_lot * lots
    span = worst * qty
    elm = params.exposure_margin + (params.expiry_day_extra_elm if state.is_expiry_day else 0.0)
    exposure = sum(l.ratio * F0 * qty * (elm if l.instrument != "future" else params.exposure_margin)
                   for l in shorts + futs)
    return {"span": span, "exposure": exposure, "total": span + exposure,
            "note": "SPAN-style estimate; confirm with the broker's margin calculator"}
