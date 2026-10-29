"""SVI fit, surface, pricing and validator: synthetic cases with known answers plus the live snapshot."""
import math
from dataclasses import replace
from datetime import date, datetime

import numpy as np
import pytest

from helpers import FIX, fbil_curve
from optlib import black76_price
from optlib.curves import flat_curve
from optlib.data.models import IST, OptionChain, OptionQuote, year_fraction
from optlib.data.replay import snapshot_source
from optlib.pricing import price_option
from optlib.slices import build_vol_slice
from optlib.surface import ExpirySmile, VolSurface, build_surface
from optlib.svi import SVIParams, fit_slice, fit_svi
from optlib.validator import validate

AS_OF = datetime(2026, 11, 2, 10, 0, tzinfo=IST)
SPOT, R = 22500.0, 0.058
TRUE = {  # expiry -> (forward, SVI params): calendar-consistent synthetic surface
    date(2026, 11, 23): (22560.0, SVIParams(a=0.0006, b=0.030, rho=-0.45, m=0.01, s=0.06)),
    date(2026, 12, 29): (22680.0, SVIParams(a=0.0018, b=0.055, rho=-0.40, m=0.02, s=0.09)),
}


# ------------------------------------------------------------------ SVI

def test_svi_recovers_known_parameters():
    p = SVIParams(a=-0.002, b=0.08, rho=-0.3, m=0.01, s=0.15)
    T = 30 / 365
    k = np.linspace(-0.2, 0.15, 36)
    iv = np.sqrt(p.total_variance(k) / T)
    fit = fit_svi(k, iv, T)
    assert fit.rmse < 1e-6
    for name in ("a", "b", "rho", "m", "s"):
        assert getattr(fit.params, name) == pytest.approx(getattr(p, name), abs=2e-4), name
    assert fit.butterfly_free and fit.status == "PASS"


def test_svi_respects_lee_bound_and_positive_variance():
    k = np.linspace(-0.5, 0.5, 41)
    T = 0.1
    iv = 0.2 + 3.0 * np.abs(k)  # wings far steeper than Lee allows
    fit = fit_svi(k, iv, T)
    p = fit.params
    assert p.b * (1 + abs(p.rho)) <= 2 + 1e-9
    assert p.min_variance() > 0
    assert any("Lee" in w for w in fit.warnings)


def test_svi_needs_enough_points():
    with pytest.raises(ValueError):
        fit_svi([0.0, 0.1, 0.2], [0.2, 0.2, 0.2], 0.1)


def test_g_function_detects_butterfly_arbitrage():
    good = SVIParams(a=0.001, b=0.05, rho=-0.3, m=0.0, s=0.1)
    bad = SVIParams(a=0.0001, b=1.8, rho=0.0, m=0.0, s=0.02)  # extreme curvature: negative density
    k = np.linspace(-1, 1, 401)
    assert good.g(k).min() >= 0
    assert bad.g(k).min() < 0


# ------------------------------------------------------------------ synthetic surface

def synthetic_chain(expiry, F, p, spread=0.1):
    T = year_fraction(AS_OF, expiry)
    quotes = []
    for i, K in enumerate(np.arange(20000.0, 25001.0, 100.0)):
        vol = math.sqrt(float(p.total_variance(math.log(K / F))) / T)
        for kind in ("call", "put"):
            mid = black76_price(F, K, T, R, vol, kind)
            if mid < 0.6:
                continue
            quotes.append(OptionQuote(f"{expiry}{i}{kind}", expiry, float(K), kind, bid=round(mid - spread / 2, 2),
                                      ask=round(mid + spread / 2, 2), volume=100))
    return OptionChain("NIFTY", "NSE_INDEX|Nifty 50", expiry, SPOT, AS_OF, "test", tuple(quotes))


class FakeSource:
    def __init__(self, params):
        self.params = params

    def expiries(self, underlying):
        return sorted(self.params)

    def option_chain(self, underlying, expiry):
        F, p = self.params[expiry]
        return synthetic_chain(expiry, F, p)

    def futures(self, underlying):
        return []


def test_surface_recovers_synthetic_smiles_and_interpolates():
    surf = build_surface(FakeSource(TRUE), flat_curve(AS_OF.date(), R), "NIFTY")
    assert surf.status in ("PASS", "WARN") and surf.calendar == []
    for sm in surf.smiles:
        F, p = TRUE[sm.expiry]
        assert sm.forward == pytest.approx(F, abs=0.5)
        assert sm.fit.rmse < 2e-4
        k = np.linspace(-0.08, 0.06, 15)
        np.testing.assert_allclose(sm.fit.params.total_variance(k), p.total_variance(k), rtol=2e-3, atol=2e-6)
    # between expiries: total variance linear in time at fixed log-moneyness
    t1, t2 = (s.time_to_expiry for s in surf.smiles)
    T = 0.5 * (t1 + t2)
    v, src = surf.vol(22600.0, T)
    k = math.log(22600.0 / surf.forward(T))
    w = 0.5 * (surf.smiles[0].fit.params.total_variance(k) + surf.smiles[1].fit.params.total_variance(k))
    assert src == "interpolated" and v == pytest.approx(math.sqrt(w / T), rel=1e-12)
    assert surf.smiles[0].forward < surf.forward(T) < surf.smiles[1].forward


def test_calendar_violation_detected():
    flipped = {date(2026, 11, 23): TRUE[date(2026, 12, 29)], date(2026, 12, 29): TRUE[date(2026, 11, 23)]}
    surf = build_surface(FakeSource(flipped), flat_curve(AS_OF.date(), R), "NIFTY")
    assert surf.calendar and surf.status == "FAIL"


def test_smile_layers():
    surf = build_surface(FakeSource(TRUE), flat_curve(AS_OF.date(), R), "NIFTY")
    sm = surf.smiles[0]
    assert sm.vol(22600.0)[1] == "market"
    v, src = sm.vol(22650.0)
    assert src == "svi_adjusted"
    lo, hi = sm.vol(22600.0)[0], sm.vol(22700.0)[0]
    assert min(lo, hi) - 1e-4 <= v <= max(lo, hi) + 1e-4
    assert sm.vol(40000.0)[1] == "svi"


# ------------------------------------------------------------------ live snapshot: pricing and validation

@pytest.fixture(scope="module")
def live_surface():
    src, meta = snapshot_source(FIX / "upstox_live_nifty_20261023_full.json", FIX / "upstox_instruments_sample.json")
    return build_surface(src, fbil_curve(), "NIFTY", expiries=[meta["expiry"]])


def test_live_svi_quality(live_surface):
    sm = live_surface.smiles[0]
    assert sm.fit.rmse < 0.005  # spec pass level: 0.5 vol pt
    assert sm.fit.butterfly_free
    assert sm.fit.share_inside(0.0025) > 0.80
    assert abs(sm.fit.atm_gap(sm.slice.atm_vol())) < 0.003  # raw SVI ~0.2 pt low at the money on this slice
    n, worst = sm.adjusted_convexity()
    assert worst < 0.5


def test_live_price_and_validate_atm(live_surface):
    res = price_option(live_surface, 22500, date(2026, 10, 27), "call", quantity=65)
    assert res.vol_source == "market" and res.vol == pytest.approx(0.1628, abs=2e-4)
    assert res.price == pytest.approx(163.44, abs=0.05)
    assert res.position_greeks()["delta"] == pytest.approx(65 * res.greeks["delta"])
    rep = validate(res, live_surface)
    assert rep.released
    by = {c.name: c.status for c in rep.checks}
    for name in ("reprice", "put-call parity", "price bounds", "Greek signs", "QuantLib cross-check",
                 "analytic vs bumped Greeks", "forward vs future"):
        assert by[name] == "PASS", name


def test_live_off_grid_strike(live_surface):
    res = price_option(live_surface, 22525, date(2026, 10, 27), "put")
    assert res.vol_source == "svi_adjusted"
    assert validate(res, live_surface).released


def steep_wings(surface):
    """Copy of a surface whose SVI wings are twice as steep: far strikes priced well above the market."""
    import copy

    out = copy.deepcopy(surface)
    p = out.smiles[0].fit.params
    out.smiles[0].fit.params = replace(p, b=min(2 * p.b, 1.99 / (1 + abs(p.rho))))
    return out


def test_validator_withholds_wing_price_above_listed_offer(live_surface):
    # with the fitted smile the far wing is priced below every nearer offer
    res = price_option(live_surface, 27500, date(2026, 10, 27), "call")
    rep = validate(res, live_surface)
    assert res.vol_source == "svi" and rep.released
    assert any(c.name == "wing vs listed quotes" and c.status == "PASS" for c in rep.checks)
    # a smile with over-steep wings prices it above a nearer listed offer: withheld
    bad = steep_wings(live_surface)
    res = price_option(bad, 27500, date(2026, 10, 27), "call")
    rep = validate(res, bad)
    assert res.vol_source == "svi" and not rep.released
    assert any(c.name == "wing vs listed quotes" and c.status == "FAIL" for c in rep.checks)


def test_fit_range_and_four_expiry_snapshot():
    """Live 29 Oct 2026 snapshot: four expiries fit within +/-3 expected moves, no calendar arbitrage."""
    src, _ = snapshot_source(FIX / "upstox_live_nifty_4exp_20261029.json", FIX / "upstox_instruments_20261029.json")
    exps = [date(2026, 11, 3), date(2026, 11, 9), date(2026, 11, 17), date(2026, 11, 23)]
    surf = build_surface(src, flat_curve(date(2026, 10, 29), 0.058), "NIFTY", expiries=exps)
    assert surf.calendar == []
    for sm in surf.smiles:
        f = sm.fit
        sd = sm.slice.atm_vol() * math.sqrt(sm.time_to_expiry)
        assert f.status == "PASS" and f.rmse < 0.0025
        assert f.fit_range[1] == pytest.approx(3 * sd, rel=0.02) and np.all(np.abs(f.k) <= f.fit_range[1] + 1e-12)
        assert abs(float(f.vol_k(0.0)) - sm.slice.atm_vol()) < 0.0015  # ATM within 0.15 vol pt
    weekly = surf.smiles[0]
    assert weekly.fit.excluded > 40  # far-wing quotes left out of the fit ...
    far = weekly.slice.strikes[-1]
    assert weekly.vol(far)[1] == "market"  # ... but still priced at their own market vol


def test_validator_catches_tampered_result(live_surface):
    res = price_option(live_surface, 22500, date(2026, 10, 27), "call")
    bad = replace(res, price=res.price + 1.0, greeks={**res.greeks, "delta": 1.3})
    rep = validate(bad, live_surface)
    by = {c.name: c.status for c in rep.checks}
    assert by["reprice"] == "FAIL" and by["Greek signs"] == "FAIL" and by["QuantLib cross-check"] == "FAIL"
    assert not rep.released


def test_pricing_rejects_bad_inputs(live_surface):
    with pytest.raises(ValueError):
        price_option(live_surface, 22500, date(2026, 10, 27), "straddle")
    with pytest.raises(ValueError):
        price_option(live_surface, 22500, date(2026, 10, 20), "call")
