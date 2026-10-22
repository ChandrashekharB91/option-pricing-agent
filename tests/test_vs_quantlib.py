"""In-house pricer vs QuantLib, using the spec's closed-form tolerances:
price within max(0.01, 0.001% of price); Greeks within 0.01% relative."""
import numpy as np
import pytest

from grid import black76_cases, bsm_cases, case_id
from optlib import black76_greeks, bsm_greeks, implied_vol_bsm

pytest.importorskip("QuantLib")
from optlib.benchmark import quantlib_black76, quantlib_bsm, quantlib_implied_vol_bsm  # noqa: E402

GREEK_RTOL = 1e-4
GREEK_ATOL = 1e-9

BSM = list(bsm_cases())
B76 = list(black76_cases())


def price_ok(ours, ref):
    return abs(ours - ref) <= max(0.01, 1e-5 * abs(ref))


def greek_ok(ours, ref):
    return np.isclose(ours, ref, rtol=GREEK_RTOL, atol=GREEK_ATOL)


@pytest.mark.parametrize("c", BSM, ids=[case_id(c) for c in BSM])
def test_bsm_matches_quantlib(c):
    ours = bsm_greeks(**c)
    ref = quantlib_bsm(**c)
    assert price_ok(ours.price, ref.price)
    for name in ("delta", "gamma", "vega", "theta", "rho", "phi"):
        assert greek_ok(getattr(ours, name), getattr(ref, name)), name


@pytest.mark.parametrize("c", B76, ids=[case_id(c) for c in B76])
def test_black76_matches_quantlib(c):
    ours = black76_greeks(**c)
    ref = quantlib_black76(**c)
    assert price_ok(ours.price, ref.price)
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert greek_ok(getattr(ours, name), getattr(ref, name)), name


# Only quotes whose time value is resolvable in double precision (vega > 1e-6 x spot);
# below that, an ITM price is numerically indistinguishable from intrinsic.
IV_CASES = [c for c in BSM
            if c["T"] >= 7 / 365 and 0.9 <= c["K"] / c["S"] <= 1.1 and bsm_greeks(**c).vega > 1e-6 * c["S"]]


@pytest.mark.parametrize("c", IV_CASES, ids=[case_id(c) for c in IV_CASES])
def test_implied_vol_matches_quantlib(c):
    price = bsm_greeks(**c).price
    args = {k: v for k, v in c.items() if k != "sigma"}
    ours = implied_vol_bsm(price, **args)
    ref = quantlib_implied_vol_bsm(price, **args)
    assert abs(ours - c["sigma"]) < 1e-8
    assert abs(ours - ref) < 1e-6
