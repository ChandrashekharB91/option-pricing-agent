"""Option chain QA: which quotes are fit to build a volatility smile from, and why not.

Every quote is either kept or dropped with named reasons, so the validator and the
quant can see exactly what fed the surface. Rules (defaults in ``QAConfig``):

    missing_bid_or_ask   no live two-sided market
    crossed_or_locked    ask <= bid
    itm_side             in-the-money side of the strike (OTM options carry the vol information)
    below_intrinsic      mid below discounted intrinsic value (arbitrage)
    price_below_min      mid below ``min_mid``: tick size dominates the price
    wide_spread          spread above max(max_rel_spread x mid, min_spread_ticks x tick)
    outside_moneyness    |ln(K / F)| above ``max_abs_log_moneyness``

Flags (kept, down-weighted later): ``no_volume_today``.
Chain-level warnings: expiry too close, too few strikes left, stale curve.
Static arbitrage on the kept quotes: call prices must fall with strike and be convex.
Violations within one tick are ignored; up to ``arb_fail_points`` they warn (real but too
small to trade after costs); beyond that the chain fails.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .data.models import OptionChain, OptionQuote


@dataclass(frozen=True)
class QAConfig:
    otm_only: bool = True
    min_mid: float = 0.5
    max_rel_spread: float = 0.10
    min_spread_ticks: int = 2
    tick_size: float = 0.05
    max_abs_log_moneyness: float = 0.25
    min_time_to_expiry_days: float = 0.25
    min_strikes: int = 7
    max_curve_age_days: int = 10
    arb_fail_points: float = 0.5  # static arbitrage larger than this fails the chain; smaller only warns


@dataclass(frozen=True)
class QuoteDecision:
    quote: OptionQuote
    keep: bool
    reasons: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArbViolation:
    kind: str  # "monotonicity" | "convexity"
    strikes: tuple[float, ...]
    size: float  # amount of the violation in index points, beyond the bid-ask tolerance


@dataclass
class QAReport:
    expiry: str
    decisions: list[QuoteDecision]
    arbitrage: list[ArbViolation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def kept(self) -> list[OptionQuote]:
        return sorted((d.quote for d in self.decisions if d.keep), key=lambda q: q.strike)

    @property
    def dropped(self) -> list[QuoteDecision]:
        return [d for d in self.decisions if not d.keep]

    def counts(self) -> dict[str, int]:
        out = {"total": len(self.decisions), "kept": len(self.kept)}
        for d in self.dropped:
            for r in d.reasons:
                out[r] = out.get(r, 0) + 1
        for d in self.decisions:
            for f in d.flags:
                out[f"flag:{f}"] = out.get(f"flag:{f}", 0) + 1
        return out

    @property
    def status(self) -> str:
        if any(w.startswith("FAIL") for w in self.warnings):
            return "FAIL"
        return "WARN" if self.warnings else "PASS"

    def summary(self) -> str:
        c = self.counts()
        drops = ", ".join(f"{k} {v}" for k, v in c.items() if k not in ("total", "kept") and not k.startswith("flag:"))
        flags = ", ".join(f"{k[5:]} {v}" for k, v in c.items() if k.startswith("flag:"))
        lines = [f"QA {self.expiry}: kept {c['kept']} of {c['total']} quotes ({self.status})",
                 f"  dropped: {drops or 'none'}", f"  flags: {flags or 'none'}"]
        lines += [f"  arbitrage: {a.kind} at {', '.join(f'{k:g}' for k in a.strikes)} ({a.size:.2f} pts)"
                  for a in self.arbitrage]
        lines += [f"  {w}" for w in self.warnings]
        return "\n".join(lines)


def review_quote(q: OptionQuote, forward: float, discount_factor: float, cfg: QAConfig) -> QuoteDecision:
    reasons, flags = [], []
    if math.isnan(q.bid) or math.isnan(q.ask):
        reasons.append("missing_bid_or_ask")
    elif q.ask <= q.bid:
        reasons.append("crossed_or_locked")
    is_call = q.option_type == "call"
    if cfg.otm_only and ((is_call and q.strike < forward) or (not is_call and q.strike >= forward)):
        reasons.append("itm_side")
    if not reasons:
        mid = q.mid
        intrinsic = discount_factor * max((forward - q.strike) if is_call else (q.strike - forward), 0.0)
        if mid < intrinsic - 0.5 * q.spread:
            reasons.append("below_intrinsic")
        if mid < cfg.min_mid:
            reasons.append("price_below_min")
        if q.spread > max(cfg.max_rel_spread * mid, cfg.min_spread_ticks * cfg.tick_size):
            reasons.append("wide_spread")
    if abs(math.log(q.strike / forward)) > cfg.max_abs_log_moneyness:
        reasons.append("outside_moneyness")
    if q.volume == 0:
        flags.append("no_volume_today")
    return QuoteDecision(q, keep=not reasons, reasons=tuple(reasons), flags=tuple(flags))


def _undiscounted_call(q: OptionQuote, forward: float, df: float) -> tuple[float, float]:
    """Undiscounted call value from a call or put quote (via parity) and its half-spread."""
    value = q.mid / df
    if q.option_type == "put":
        value += forward - q.strike
    return value, 0.5 * q.spread / df


def static_arbitrage(quotes: list[OptionQuote], forward: float, df: float,
                     tick_size: float = 0.05) -> list[ArbViolation]:
    """Monotonicity and convexity of undiscounted call prices across kept strikes.
    A violation counts only if it exceeds the bid-ask tolerance plus one price tick
    (tick rounding of near-zero wing prices is not arbitrage)."""
    pts = sorted((q.strike, *_undiscounted_call(q, forward, df)) for q in quotes)
    tol = tick_size / df
    out = []
    for (k1, c1, h1), (k2, c2, h2) in zip(pts, pts[1:]):
        excess = (c2 - h2) - (c1 + h1)  # call price must not rise with strike
        if excess > tol:
            out.append(ArbViolation("monotonicity", (k1, k2), excess))
    for (k1, c1, h1), (k2, c2, h2), (k3, c3, h3) in zip(pts, pts[1:], pts[2:]):
        w = (k3 - k2) / (k3 - k1)
        excess = (c2 - h2) - (w * (c1 + h1) + (1 - w) * (c3 + h3))  # convexity
        if excess > tol:
            out.append(ArbViolation("convexity", (k1, k2, k3), excess))
    return out


def review_chain(chain: OptionChain, forward: float, discount_factor: float, cfg: Optional[QAConfig] = None,
                 curve_age_days: Optional[int] = None) -> QAReport:
    cfg = cfg or QAConfig()
    decisions = [review_quote(q, forward, discount_factor, cfg)
                 for q in sorted(chain.quotes, key=lambda q: (q.strike, q.option_type))]
    report = QAReport(expiry=chain.expiry.isoformat(), decisions=decisions)
    report.arbitrage = static_arbitrage(report.kept, forward, discount_factor, cfg.tick_size)
    if report.arbitrage:
        worst = max(a.size for a in report.arbitrage)
        level = "FAIL" if worst > cfg.arb_fail_points else "WARN"
        report.warnings.append(f"{level} {len(report.arbitrage)} static-arbitrage violation(s), largest "
                               f"{worst:.2f} pts (fail above {cfg.arb_fail_points} pts)")

    days = chain.time_to_expiry() * 365.0
    if days < cfg.min_time_to_expiry_days:
        report.warnings.append(f"FAIL expiry in {days * 24:.1f} hours: below the {cfg.min_time_to_expiry_days} day minimum")
    if len({q.strike for q in report.kept}) < cfg.min_strikes:
        report.warnings.append(f"FAIL only {len(report.kept)} usable strikes (minimum {cfg.min_strikes})")
    if curve_age_days is not None and curve_age_days > cfg.max_curve_age_days:
        report.warnings.append(f"WARN discount curve is {curve_age_days} days old")
    return report
