"""Self-consistency: parity, bounds, signs, analytic vs bumped Greeks, IV round trips."""
import numpy as np
import pytest

from grid import bsm_cases, case_id
from optlib import black76_greeks, black76_price, bsm_greeks, bsm_price, implied_vol_black76, implied_vol_bsm
from optlib.iv import ABOVE_MAX, BELOW_INTRINSIC, INVALID, OK

BSM = list(bsm_cases())


def test_put_call_parity_bsm():
    for c in BSM:
        call = bsm_price(**{**c, "option_type": "call"})
        put = bsm_price(**{**c, "option_type": "put"})
        S, K, T, r, q = c["S"], c["K"], c["T"], c["r"], c["q"]
        assert abs(call - put - (S * np.exp(-q * T) - K * np.exp(-r * T))) < 1e-8 * S


def test_price_bounds_and_signs():
    for c in BSM:
        g = bsm_greeks(**c)
        S, K, T, r, q = c["S"], c["K"], c["T"], c["r"], c["q"]
        fwd_intrinsic = max((S * np.exp(-q * T) - K * np.exp(-r * T)) * (1 if c["option_type"] == "call" else -1), 0)
        assert g.price >= fwd_intrinsic - 1e-9
        assert g.price <= (S * np.exp(-q * T) if c["option_type"] == "call" else K * np.exp(-r * T))
        assert g.gamma >= 0 and g.vega >= 0
        if c["option_type"] == "call":
            assert 0 <= g.delta <= 1
        else:
            assert -1 <= g.delta <= 0


BUMP_CASES = [c for c in BSM if c["T"] >= 7 / 365 and 0.9 <= c["K"] / c["S"] <= 1.1]


@pytest.mark.parametrize("c", BUMP_CASES, ids=[case_id(c) for c in BUMP_CASES])
def test_analytic_vs_bumped(c):
    g = bsm_greeks(**c)
    S, T, sig, r, q = c["S"], c["T"], c["sigma"], c["r"], c["q"]
    hs, hv, ht, hr = S * 1e-4, 1e-5, 1e-6, 1e-6

    def price(**kw):
        return bsm_price(**{**c, **kw})

    def delta(**kw):
        return bsm_greeks(**{**c, **kw}).delta

    def gamma(**kw):
        return bsm_greeks(**{**c, **kw}).gamma

    def vega(**kw):
        return bsm_greeks(**{**c, **kw}).vega

    checks = {
        "delta": (price(S=S + hs) - price(S=S - hs)) / (2 * hs),
        "gamma": (delta(S=S + hs) - delta(S=S - hs)) / (2 * hs),
        "vega": (price(sigma=sig + hv) - price(sigma=sig - hv)) / (2 * hv),
        "theta": -(price(T=T + ht) - price(T=T - ht)) / (2 * ht),
        "rho": (price(r=r + hr) - price(r=r - hr)) / (2 * hr),
        "phi": (price(q=q + hr) - price(q=q - hr)) / (2 * hr),
        "vanna": (delta(sigma=sig + hv) - delta(sigma=sig - hv)) / (2 * hv),
        "volga": (vega(sigma=sig + hv) - vega(sigma=sig - hv)) / (2 * hv),
        "charm": -(delta(T=T + ht) - delta(T=T - ht)) / (2 * ht),
        "speed": (gamma(S=S + hs) - gamma(S=S - hs)) / (2 * hs),
        "zomma": (gamma(sigma=sig + hv) - gamma(sigma=sig - hv)) / (2 * hv),
    }
    for name, bumped in checks.items():
        analytic = getattr(g, name)
        scale = max(abs(analytic), abs(bumped))
        assert abs(analytic - bumped) <= 5e-3 * scale + 1e-9, (name, analytic, bumped)


def test_black76_equals_bsm_on_forward():
    for c in BSM:
        F = c["S"] * np.exp((c["r"] - c["q"]) * c["T"])
        b76 = black76_price(F, c["K"], c["T"], c["r"], c["sigma"], c["option_type"])
        assert abs(b76 - bsm_price(**c)) < 1e-9 * c["S"]


def test_iv_round_trip_full_grid():
    for c in BSM:
        g = bsm_greeks(**c)
        args = {k: v for k, v in c.items() if k != "sigma"}
        vol, status = implied_vol_bsm(g.price, **args, return_status=True)
        if status == OK:
            # well-conditioned quotes recover sigma; ill-conditioned ones must still reprice
            if g.vega > 1e-6 * c["S"]:
                assert abs(vol - c["sigma"]) < 1e-7, c
            assert abs(bsm_price(**{**c, "sigma": vol}) - g.price) < 1e-8 * c["S"]
        else:
            # only quotes whose value is numerically indistinguishable from intrinsic may fail
            assert g.vega < 1e-8 * c["S"], (c, status)


def test_iv_vectorised():
    F, T, r = 25000.0, 30 / 365, 0.065
    K = np.linspace(21000, 29000, 41)
    vols = 0.12 + 0.3 * ((K / F) - 1) ** 2
    calls = black76_price(F, K, T, r, vols, "call")
    puts = black76_price(F, K, T, r, vols, "put")
    assert np.allclose(implied_vol_black76(calls, F, K, T, r, "call"), vols, atol=1e-9)
    assert np.allclose(implied_vol_black76(puts, F, K, T, r, "put"), vols, atol=1e-9)


def test_iv_rejects_arbitrage_and_expired():
    F, K, T, r = 25000.0, 24000.0, 30 / 365, 0.065
    df = np.exp(-r * T)
    _, s1 = implied_vol_black76(0.9 * (F - K) * df, F, K, T, r, "call", return_status=True)
    _, s2 = implied_vol_black76(1.01 * F * df, F, K, T, r, "call", return_status=True)
    _, s3 = implied_vol_black76(1100.0, F, K, 0.0, r, "call", return_status=True)
    assert (s1, s2, s3) == (BELOW_INTRINSIC, ABOVE_MAX, INVALID)


def test_expired_and_zero_vol():
    g = bsm_greeks(25000, 24000, 0.0, 0.065, 0.0, 0.2, "call")
    assert g.price == pytest.approx(1000.0) and g.delta == 1.0 and g.gamma == 0.0
    g0 = black76_greeks(25000, 26000, 0.5, 0.065, 0.0, "put")
    assert g0.price == pytest.approx(1000.0 * np.exp(-0.065 * 0.5)) and g0.vega == 0.0


def test_vectorised_matches_scalar():
    S, K = 25000.0, np.array([23000.0, 25000.0, 27000.0])
    vec = bsm_greeks(S, K, 0.25, 0.065, 0.01, 0.15, ["put", "call", "call"]).to_dict()
    for i, (k, t) in enumerate(zip(K, ["put", "call", "call"])):
        sc = bsm_greeks(S, k, 0.25, 0.065, 0.01, 0.15, t).to_dict()
        for name, val in sc.items():
            assert vec[name][i] == pytest.approx(val, rel=1e-14, abs=1e-14)


def test_input_validation():
    with pytest.raises(ValueError):
        bsm_price(-1, 100, 1, 0.05, 0, 0.2)
    with pytest.raises(ValueError):
        bsm_price(100, 100, 1, 0.05, 0, 0.2, "straddle")
