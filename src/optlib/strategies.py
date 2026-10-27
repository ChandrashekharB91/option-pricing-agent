"""Option strategies: legs, the strategy library, and candidate generation around the forward.

A strategy is a set of legs per *unit*; the recommender sizes units in lots. Strikes are chosen in
units of the expected move to expiry (sd = F x ATM vol x sqrt(T)), so the same templates fit a
4-day weekly and a 2-month monthly, and only strikes with a live two-sided quote are used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Optional

FAMILIES = {
    "buy": "Buy options",
    "debit": "Debit spreads",
    "hedged": "Hedged selling",
    "naked": "Naked selling",
    "futures": "Futures + options",
}


@dataclass(frozen=True)
class StrategyLeg:
    instrument: str  # "call" | "put" | "future"
    side: int  # +1 buy, -1 sell
    expiry: date
    strike: Optional[float] = None  # None for a future
    ratio: int = 1  # lots of this leg per strategy unit

    def __post_init__(self):
        if self.instrument not in ("call", "put", "future"):
            raise ValueError(f"unknown instrument {self.instrument!r}")
        if self.side not in (1, -1):
            raise ValueError("side must be +1 (buy) or -1 (sell)")
        if (self.instrument == "future") != (self.strike is None):
            raise ValueError("options need a strike; futures must not have one")
        if self.ratio < 1:
            raise ValueError("ratio must be a positive whole number of lots")

    @property
    def signed(self) -> int:
        return self.side * self.ratio

    @property
    def label(self) -> str:
        verb = "Buy" if self.side > 0 else "Sell"
        qty = f"{self.ratio} x " if self.ratio > 1 else ""
        what = "future" if self.instrument == "future" else f"{self.strike:,.0f} {self.instrument}"
        return f"{verb} {qty}{what}"

    def to_dict(self) -> dict:
        return {"instrument": self.instrument, "side": "buy" if self.side > 0 else "sell",
                "strike": self.strike, "expiry": self.expiry.isoformat(), "ratio": self.ratio}

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyLeg":
        side = d.get("side", 1)
        if isinstance(side, str):
            side = {"buy": 1, "long": 1, "sell": -1, "short": -1}[side.lower()]
        exp = d["expiry"] if isinstance(d["expiry"], date) else date.fromisoformat(str(d["expiry"])[:10])
        strike = d.get("strike")
        return cls(str(d["instrument"]).lower(), int(side), exp, None if strike is None else float(strike),
                   int(d.get("ratio", 1)))


@dataclass(frozen=True)
class Strategy:
    name: str
    family: str
    view: str  # bullish | bearish | neutral | volatile | hedge
    legs: tuple[StrategyLeg, ...]
    note: str = ""

    @property
    def key(self) -> tuple:
        return tuple(sorted((l.instrument, l.side, l.strike or 0.0, l.ratio, l.expiry) for l in self.legs))

    @property
    def label(self) -> str:
        strikes = "/".join(f"{k:,.0f}" for k in sorted({l.strike for l in self.legs if l.strike is not None}))
        exp = self.legs[0].expiry.strftime("%d %b")
        return f"{self.name} {strikes} {exp}".replace("  ", " ").strip()

    @property
    def has_short(self) -> bool:
        return any(l.side < 0 or l.instrument == "future" for l in self.legs)

    def to_dict(self) -> dict:
        return {"name": self.name, "family": self.family, "family_label": FAMILIES.get(self.family, self.family),
                "view": self.view, "label": self.label, "legs": [l.to_dict() for l in self.legs], "note": self.note}

    @classmethod
    def custom(cls, legs: Iterable[StrategyLeg], name: str = "Custom strategy") -> "Strategy":
        legs = tuple(legs)
        if not legs:
            raise ValueError("a strategy needs at least one leg")
        if len({l.expiry for l in legs}) > 1:
            raise ValueError("all legs must share one expiry (calendar spreads are not supported yet)")
        return cls(name, _family_of(legs), "custom", legs)


def _family_of(legs) -> str:
    if any(l.instrument == "future" for l in legs):
        return "futures"
    shorts = sum(l.ratio for l in legs if l.side < 0)
    longs = sum(l.ratio for l in legs if l.side > 0)
    if shorts == 0:
        return "buy"
    if shorts > longs or not longs:
        return "naked"
    return "hedged" if _is_credit_shape(legs) else "debit"


def _is_credit_shape(legs) -> bool:
    # short legs nearer the money than the long legs of the same type -> premium collected
    for typ in ("call", "put"):
        s = [l.strike for l in legs if l.instrument == typ and l.side < 0]
        b = [l.strike for l in legs if l.instrument == typ and l.side > 0]
        if s and b:
            if typ == "call" and min(s) < min(b):
                return True
            if typ == "put" and max(s) > max(b):
                return True
    return False


# ------------------------------------------------------------------ candidates

@dataclass
class StrikeGrid:
    """Listed strikes with a live two-sided quote, by option type."""
    forward: float
    sd: float  # expected move to expiry in index points
    calls: list[float]
    puts: list[float]
    expiry: date
    step: float = field(init=False)

    def __post_init__(self):
        ks = sorted(set(self.calls) | set(self.puts))
        diffs = [b - a for a, b in zip(ks, ks[1:]) if b > a]
        self.step = min(diffs) if diffs else 50.0

    def near(self, x: float, typ: str) -> Optional[float]:
        ks = self.calls if typ == "call" else self.puts
        if not ks:
            return None
        k = min(ks, key=lambda s: abs(s - x))
        return k if abs(k - x) <= max(2 * self.step, 0.35 * self.sd) else None

    def at(self, z: float, typ: str) -> Optional[float]:
        return self.near(self.forward + z * self.sd, typ)


def _legs(*parts) -> Optional[tuple[StrategyLeg, ...]]:
    if any(p is None for p in parts):
        return None
    return tuple(parts)


def generate_candidates(g: StrikeGrid, families: Optional[Iterable[str]] = None) -> list[Strategy]:
    """Every template in the library, at strikes spaced in units of the expected move."""
    fams = set(families or FAMILIES)
    e = g.expiry
    out: dict[tuple, Strategy] = {}

    def C(z, side, ratio=1):
        k = g.at(z, "call")
        return None if k is None else StrategyLeg("call", side, e, k, ratio)

    def P(z, side, ratio=1):
        k = g.at(z, "put")
        return None if k is None else StrategyLeg("put", side, e, k, ratio)

    def FUT(side):
        return StrategyLeg("future", side, e)

    def add(name, family, view, legs, note=""):
        if family not in fams or legs is None:
            return
        strikes = [l.strike for l in legs if l.strike is not None]
        if len(strikes) != len(set((l.instrument, l.strike) for l in legs if l.strike is not None)):
            return  # two legs collapsed onto the same strike and type
        s = Strategy(name, family, view, legs, note)
        out.setdefault(s.key, s)

    def ordered(legs, typ, increasing=True):
        if legs is None:
            return None
        ks = [l.strike for l in legs if l.instrument == typ]
        return legs if ks == sorted(ks, reverse=not increasing) and len(set(ks)) == len(ks) else None

    # buy options
    for z in (-0.5, 0.0, 0.25, 0.5, 1.0):
        add("Long call", "buy", "bullish", _legs(C(z, 1)))
        add("Long put", "buy", "bearish", _legs(P(-z, 1)))
    add("Long straddle", "buy", "volatile", _legs(C(0, 1), P(0, 1)))
    for z in (0.5, 1.0):
        add("Long strangle", "buy", "volatile", _legs(P(-z, 1), C(z, 1)))

    # debit spreads
    for z1 in (-0.25, 0.0, 0.25, 0.5):
        for w in (0.5, 1.0, 1.5):
            add("Bull call spread", "debit", "bullish", ordered(_legs(C(z1, 1), C(z1 + w, -1)), "call"))
            add("Bear put spread", "debit", "bearish", ordered(_legs(P(-z1, 1), P(-z1 - w, -1)), "put", False))
    for c in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for w in (0.5, 1.0):
            add("Long call butterfly", "debit", "neutral" if c == 0 else ("bullish" if c > 0 else "bearish"),
                ordered(_legs(C(c - w, 1), C(c, -1, 2), C(c + w, 1)), "call"),
                "profits most if the index ends near the middle strike")

    # hedged selling
    for z in (0.0, 0.25, 0.5, 1.0):
        for w in (0.5, 1.0):
            add("Bull put spread", "hedged", "bullish", ordered(_legs(P(-z - w, 1), P(-z, -1)), "put"))
            add("Bear call spread", "hedged", "bearish", ordered(_legs(C(z, -1), C(z + w, 1)), "call"))
    for z in (0.5, 1.0, 1.5):
        for w in (0.5, 1.0):
            add("Iron condor", "hedged", "neutral",
                _legs(P(-z - w, 1), P(-z, -1), C(z, -1), C(z + w, 1)))
    for w in (0.5, 1.0, 1.5):
        add("Iron butterfly", "hedged", "neutral", _legs(P(-w, 1), P(0, -1), C(0, -1), C(w, 1)))

    # naked selling
    add("Short straddle", "naked", "neutral", _legs(C(0, -1), P(0, -1)), "unlimited loss if the index moves far")
    for z in (0.5, 1.0, 1.5):
        add("Short strangle", "naked", "neutral", _legs(P(-z, -1), C(z, -1)), "unlimited loss if the index moves far")
        add("Short put", "naked", "bullish", _legs(P(-z, -1)), "large loss if the index falls")
        add("Short call", "naked", "bearish", _legs(C(z, -1)), "unlimited loss if the index rises")

    # futures and options
    add("Long future", "futures", "bullish", (FUT(1),))
    add("Short future", "futures", "bearish", (FUT(-1),))
    for z in (0.0, 0.5, 1.0):
        add("Protective put", "futures", "bullish", _legs(FUT(1), P(-z, 1)), "long future insured by a put")
        add("Protective call", "futures", "bearish", _legs(FUT(-1), C(z, 1)), "short future insured by a call")
    for z in (0.5, 1.0):
        add("Covered call", "futures", "neutral", _legs(FUT(1), C(z, -1)), "long future, call sold for income")
        add("Collar", "futures", "bullish", _legs(FUT(1), P(-z, 1), C(z, -1)), "long future between a put and a call")
    return list(out.values())
