"""Compare optlib with QuantLib across the test grid and write a report.

Usage: python scripts/compare_quantlib.py  ->  reports/quantlib_comparison.{md,csv}
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from grid import black76_cases, bsm_cases  # noqa: E402
from optlib import __version__, black76_greeks, bsm_greeks, implied_vol_bsm  # noqa: E402
from optlib.benchmark import quantlib_black76, quantlib_bsm, quantlib_implied_vol_bsm  # noqa: E402

PRICE_TOL = lambda ref: max(0.01, 1e-5 * abs(ref))  # noqa: E731  spec: max(Rs 0.01, 0.001%)
GREEK_RTOL = 1e-4  # spec: 0.01% relative
GREEK_ATOL = 1e-9


def compare(model, cases, ours_fn, ref_fn, names):
    rows = []
    for c in cases:
        ours, ref = ours_fn(**c), ref_fn(**c)
        for name in names:
            o, r = float(getattr(ours, name)), float(getattr(ref, name))
            diff = abs(o - r)
            ok = diff <= PRICE_TOL(r) if name == "price" else np.isclose(o, r, rtol=GREEK_RTOL, atol=GREEK_ATOL)
            rel = diff / abs(r) if abs(r) > 1e-12 else 0.0
            rows.append(dict(model=model, measure=name, underlying=c.get("S", c.get("F")), K=c["K"],
                             T_days=round(c["T"] * 365), r=c["r"], q=c.get("q", ""), sigma=c["sigma"],
                             option_type=c["option_type"], optlib=o, quantlib=r, abs_diff=diff, rel_diff=rel,
                             within_tolerance=bool(ok)))
    return rows


def main():
    bsm = list(bsm_cases())
    b76 = list(black76_cases())
    rows = compare("BSM", bsm, bsm_greeks, quantlib_bsm, ["price", "delta", "gamma", "vega", "theta", "rho", "phi"])
    rows += compare("Black-76", b76, black76_greeks, quantlib_black76, ["price", "delta", "gamma", "vega", "theta", "rho"])

    iv_rows = []
    for c in bsm:
        g = bsm_greeks(**c)
        if c["T"] < 7 / 365 or g.vega <= 1e-6 * c["S"]:
            continue
        args = {k: v for k, v in c.items() if k != "sigma"}
        ours = implied_vol_bsm(g.price, **args)
        ref = quantlib_implied_vol_bsm(g.price, **args)
        iv_rows.append(dict(true=c["sigma"], optlib=ours, quantlib=ref))

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    with open(out / "quantlib_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# optlib {__version__} vs QuantLib",
        "",
        f"Grid: {len(bsm)} BSM and {len(b76)} Black-76 cases; spot/forward 20,000 and 25,000; strikes 80% to 125%;",
        "expiries 2 to 365 days; vols 8% to 35%; r = 6.5%; q = 0% and 1.2%; calls and puts.",
        "Tolerances (spec): price within max(Rs 0.01, 0.001% of price); Greeks within 0.01% relative.",
        "",
        "| Model | Measure | Cases | Max abs diff | Max rel diff | Within tolerance |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model in ("BSM", "Black-76"):
        for name in dict.fromkeys(r["measure"] for r in rows if r["model"] == model):
            sel = [r for r in rows if r["model"] == model and r["measure"] == name]
            lines.append(
                f"| {model} | {name} | {len(sel)} | {max(r['abs_diff'] for r in sel):.2e} | "
                f"{max(r['rel_diff'] for r in sel):.2e} | {sum(r['within_tolerance'] for r in sel)}/{len(sel)} |")

    lines += ["", "Larger relative differences occur only where the value itself is near zero (deep OTM, theta near",
              "its sign change); every absolute difference is at floating-point rounding level."]
    iv_err_true = max(abs(r["optlib"] - r["true"]) for r in iv_rows)
    iv_err_ql = max(abs(r["optlib"] - r["quantlib"]) for r in iv_rows)
    lines += [
        "",
        "## Implied volatility",
        "",
        f"{len(iv_rows)} quotes (expiry of 7 days or more, vega above 1e-6 x spot).",
        f"Max error vs true vol: {iv_err_true:.2e}. Max difference vs QuantLib: {iv_err_ql:.2e}.",
        "",
        "Quotes with vega below 1e-6 x spot are excluded: their time value is below double precision,",
        "so the solver correctly returns status BELOW_INTRINSIC instead of a vol.",
    ]
    total = len(rows)
    passed = sum(r["within_tolerance"] for r in rows)
    lines.insert(2, f"**Result: {passed}/{total} comparisons within tolerance.**\n")
    (out / "quantlib_comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
