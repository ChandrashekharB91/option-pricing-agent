"""Independent checks on a priced option: the deterministic toolkit behind the validator agent.

``validate(result, surface)`` re-derives what it can from scratch and compares, never trusting the
pricer's own numbers. Each check returns PASS, WARN or FAIL with the value and the limit, using the
tolerances in the spec. Any FAIL withholds the result; WARNs are shown to the quant with the reason.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .black import black76_greeks, black76_price
from .pricing import PricingResult
from .surface import VolSurface

PRICE_TOL = lambda p: max(0.01, 1e-5 * abs(p))  # noqa: E731  QuantLib cross-check: max(Rs 0.01, 0.001%)
GREEK_RTOL = 1e-4  # 0.01% relative
BUMP_WARN = {"delta": 0.005, "vega": 0.005, "gamma": 0.02}
BUMP_FAIL = {"delta": 0.02, "vega": 0.02, "gamma": 0.05}
PARITY_TOL = 0.01  # index points
SVI_GAP_WARN = 0.0025  # 0.25 vol pt between SVI and the market near the strike


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # PASS | WARN | FAIL | SKIP
    detail: str
    value: float | None = None
    limit: str = ""


@dataclass
class ValidationReport:
    checks: list[Check]

    @property
    def status(self) -> str:
        s = {c.status for c in self.checks}
        return "FAIL" if "FAIL" in s else ("WARN" if "WARN" in s else "PASS")

    @property
    def released(self) -> bool:
        return self.status != "FAIL"

    def summary(self) -> str:
        width = max(len(c.name) for c in self.checks)
        lines = [f"Validation: {self.status} ({'released' if self.released else 'withheld'})"]
        lines += [f"  {c.status:<4}  {c.name:<{width}}  {c.detail}" for c in self.checks]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"status": self.status, "released": self.released, "checks": [asdict(c) for c in self.checks]}


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def validate(result: PricingResult, surface: VolSurface) -> ValidationReport:
    r = result
    F, K, T, rate, vol = r.forward, r.strike, r.time_to_expiry, r.rate, r.vol
    df = math.exp(-rate * T)
    is_call = r.option_type == "call"
    checks: list[Check] = []
    smile = next((s for s in surface.smiles if s.slice.expiry == r.expiry), None)

    # 1. market inputs ------------------------------------------------------------
    if smile is not None:
        qa = smile.slice.qa
        checks.append(Check("chain QA", qa.status, f"{qa.counts()['kept']} of {qa.counts()['total']} quotes kept"
                            + (f"; {qa.warnings[0]}" if qa.warnings else "")))
        fc = smile.slice.forward_check
        if fc:
            checks.append(Check("forward vs future", fc.verdict, f"gap {fc.gap:+.3%} (warn > 0.05%, fail > 0.20%)",
                                fc.gap, "0.05% / 0.20%"))
        else:
            checks.append(Check("forward vs future", "SKIP", "no future for this expiry"))
    else:
        checks.append(Check("listed expiry", "WARN", "expiry not listed: forward and vol interpolated across expiries"))
    age = r.curve_age_days
    checks.append(Check("discount curve age", "WARN" if age is not None and age > 10 else "PASS",
                        f"{r.curve_label}, {age} days old" if age is not None else r.curve_label, age, "10 days"))

    # 2. vol source and model ----------------------------------------------------------
    if smile is not None:
        fit = smile.fit
        checks.append(Check("SVI fit", fit.status, f"RMSE {fit.rmse * 100:.2f} vol pts, "
                            f"{fit.share_inside(0.0025):.0%} within bid-ask or 0.25 pt, butterfly "
                            f"{'clean' if fit.butterfly_free else 'VIOLATED'}", fit.rmse, "0.5 / 1.5 vol pts"))
        if r.vol_source in ("market", "svi_adjusted"):
            n, worst = smile.adjusted_convexity()
            checks.append(Check("smile convexity", "PASS" if worst <= 0.5 else "WARN",
                                f"{n} negative butterflies on a 5-pt grid, worst {worst:.3f} pts (quote noise)",
                                worst, "0.5 pts"))
        gap_k = smile.slice.strikes[int(np.argmin(np.abs(smile.slice.strikes - K)))]
        gap = float(fit.vol(gap_k)) - float(smile.market_vol(gap_k) or fit.vol(gap_k))
        checks.append(Check("SVI vs market", "WARN" if abs(gap) > SVI_GAP_WARN else "PASS",
                            f"{gap * 100:+.2f} vol pts at {gap_k:,.0f} (model risk if SVI were used)", gap, "0.25 vol pt"))
    src_status = {"market": "PASS", "svi_adjusted": "PASS", "svi": "WARN", "interpolated": "WARN"}[r.vol_source]
    src_note = {"market": "listed strike priced at its market vol",
                "svi_adjusted": "between listed strikes: SVI corrected to neighbouring market vols",
                "svi": "outside the quoted strikes: SVI extrapolation",
                "interpolated": "between expiries: SVI total variance interpolated in time"}[r.vol_source]
    checks.append(Check("vol source", src_status, f"{vol:.2%} ({src_note})"))
    if smile is not None and r.vol_source == "svi":
        # A further-out option can never be worth more than a nearer one the market offers:
        # compare with every listed quote of the same type, including ones QA dropped.
        dominated = [d.quote for d in smile.slice.qa.decisions
                     if d.quote.option_type == r.option_type and not math.isnan(d.quote.ask)
                     and ((is_call and d.quote.strike < K) or (not is_call and d.quote.strike > K))
                     and d.quote.ask < r.price]
        if dominated:
            q = min(dominated, key=lambda q: abs(q.strike - K))
            checks.append(Check("wing vs listed quotes", "FAIL",
                                f"model {r.price:.2f} above the {q.strike:,.0f} {r.option_type} offered at {q.ask:.2f}, "
                                "a strike nearer the money", r.price - q.ask, "price <= nearer listed ask"))
        else:
            checks.append(Check("wing vs listed quotes", "PASS", "below every nearer listed offer"))
    if surface.calendar:
        v = surface.calendar[0]
        checks.append(Check("calendar arbitrage", "FAIL", f"{v.expiries[0]} vs {v.expiries[1]} at k={v.k:+.3f}"))
    elif len(surface.smiles) > 1:
        checks.append(Check("calendar arbitrage", "PASS", f"none across {len(surface.smiles)} expiries"))

    # 3. the number itself --------------------------------------------------------------
    price = black76_price(F, K, T, rate, vol, r.option_type)
    checks.append(Check("reprice", "PASS" if abs(price - r.price) <= 1e-9 * max(1, price) else "FAIL",
                        f"independent reprice {price:.6f} vs result {r.price:.6f}"))
    call = black76_price(F, K, T, rate, vol, "call")
    put = black76_price(F, K, T, rate, vol, "put")
    parity = call - put - df * (F - K)
    checks.append(Check("put-call parity", "PASS" if abs(parity) <= PARITY_TOL else "FAIL",
                        f"C - P - DF(F - K) = {parity:+.2e}", parity, f"{PARITY_TOL} pts"))
    lower = df * max((F - K) if is_call else (K - F), 0.0)
    upper = df * (F if is_call else K)
    ok = lower - 1e-9 <= r.price <= upper + 1e-9
    checks.append(Check("price bounds", "PASS" if ok else "FAIL",
                        f"{lower:,.2f} <= {r.price:,.2f} <= {upper:,.2f}"))
    g = r.greeks
    sign_ok = (g["gamma"] >= 0 and g["vega"] >= 0 and
               (0 <= g["delta"] <= 1 if is_call else -1 <= g["delta"] <= 0))
    checks.append(Check("Greek signs", "PASS" if sign_ok else "FAIL",
                        f"delta {g['delta']:+.4f}, gamma {g['gamma']:.6f}, vega {g['vega']:.4f}"))

    # 4. independent library ------------------------------------------------------------
    try:
        import QuantLib as ql

        payoff = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, K)
        bc = ql.BlackCalculator(payoff, F, vol * math.sqrt(T), df)
        ql_vals = {"price": bc.value(), "delta": bc.deltaForward(), "gamma": bc.gammaForward(),
                   "vega": bc.vega(T) / 100.0}
        ours = {"price": r.price, "delta": g["delta"], "gamma": g["gamma"], "vega": g["vega"]}
        worst = max(_rel(ours[n], ql_vals[n]) for n in ("delta", "gamma", "vega"))
        p_ok = abs(ours["price"] - ql_vals["price"]) <= PRICE_TOL(ql_vals["price"])
        checks.append(Check("QuantLib cross-check", "PASS" if p_ok and worst <= GREEK_RTOL else "FAIL",
                            f"price diff {ours['price'] - ql_vals['price']:+.1e}, worst Greek diff {worst:.1e} rel",
                            worst, "price max(0.01, 0.001%), Greeks 0.01%"))
    except ImportError:
        checks.append(Check("QuantLib cross-check", "WARN", "QuantLib not installed"))

    # 5. analytic vs bumped Greeks ------------------------------------------------------
    hF, hv = F * 1e-4, 1e-4
    bump = {
        "delta": (black76_price(F + hF, K, T, rate, vol, r.option_type)
                  - black76_price(F - hF, K, T, rate, vol, r.option_type)) / (2 * hF),
        "gamma": (black76_price(F + hF, K, T, rate, vol, r.option_type) - 2 * price
                  + black76_price(F - hF, K, T, rate, vol, r.option_type)) / hF ** 2,
        "vega": (black76_price(F, K, T, rate, vol + hv, r.option_type)
                 - black76_price(F, K, T, rate, vol - hv, r.option_type)) / (2 * hv) / 100.0,
    }
    states, worst_txt = [], []
    for n, b in bump.items():
        e = _rel(g[n], b) if abs(b) > 1e-10 else abs(g[n] - b)
        states.append("FAIL" if e > BUMP_FAIL[n] else ("WARN" if e > BUMP_WARN[n] else "PASS"))
        worst_txt.append(f"{n} {e:.1e}")
    status = "FAIL" if "FAIL" in states else ("WARN" if "WARN" in states else "PASS")
    checks.append(Check("analytic vs bumped Greeks", status, ", ".join(worst_txt) + " rel"))

    return ValidationReport(checks)
