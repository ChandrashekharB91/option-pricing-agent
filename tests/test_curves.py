"""MIBOR-OIS bootstrap: inputs reprice exactly, conventions, interpolation, FBIL loader."""
import math
from datetime import date, datetime

import pytest

from helpers import PRICING_DATE, fbil_client, fbil_curve
from optlib.curves import (
    DiscountCurve,
    OISQuotes,
    add_months,
    bootstrap_mibor_ois,
    flat_curve,
    following,
    next_business_day,
    tenor_months,
)
from optlib.data.fbil import FBILError, parse_mibor_ois
from optlib.data.models import IST

QUOTES = OISQuotes(date(2026, 10, 15), 5.87, {"1M": 5.70, "2M": 5.76, "3M": 5.90, "6M": 6.16, "9M": 6.30,
                                               "1Y": 6.44, "2Y": 6.59, "3Y": 6.66, "4Y": 6.71, "5Y": 6.77})


def yf(a, b):
    return (b - a).days / 365.0


def test_date_helpers():
    assert next_business_day(date(2026, 10, 23)) == date(2026, 10, 26)  # Fri -> Mon
    assert following(date(2026, 10, 24)) == date(2026, 10, 26)
    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert [tenor_months(t) for t in ("1M", "9M", "1Y", "5Y")] == [1, 9, 12, 60]
    with pytest.raises(ValueError):
        tenor_months("2W")


def test_fbil_loader_picks_latest_complete_fixing():
    q = fbil_client().ois_quotes(PRICING_DATE)
    assert q.fixing_date == date(2026, 10, 15) and q.overnight == 5.87
    assert q.tenors == QUOTES.tenors
    q2 = fbil_client().ois_quotes(date(2026, 10, 9))
    assert q2.fixing_date == date(2026, 10, 9)
    with pytest.raises(FBILError):
        fbil_client().ois_quotes(date(2026, 9, 1), lookback_days=5)


def test_parse_mibor_ois_rejects_unknown_tenor():
    with pytest.raises(ValueError):
        parse_mibor_ois([{"processRunDate": "2026-10-15 00:00:00", "tenorName": "2W", "rate": 5.0}])


def test_bullet_tenors_reprice_exactly():
    c = bootstrap_mibor_ois(QUOTES)
    start, df_start = c.pillars["O/N"]
    assert df_start == pytest.approx(1 / (1 + 0.0587 * yf(QUOTES.fixing_date, start)), rel=1e-15)
    for t in ("1M", "2M", "3M", "6M", "9M", "1Y"):
        mat, df = c.pillars[t]
        implied = (df_start / df - 1) / yf(start, mat) * 100
        assert implied == pytest.approx(QUOTES.tenors[t], abs=1e-10), t


def test_swap_tenors_reprice_at_par():
    c = bootstrap_mibor_ois(QUOTES)
    start, df_start = c.pillars["O/N"]
    for t in ("2Y", "3Y", "4Y", "5Y"):
        n = tenor_months(t)
        pays = [following(add_months(start, 6 * i)) for i in range(1, n // 6 + 1)]
        R, fixed, prev = QUOTES.tenors[t] / 100, 0.0, start
        for d in pays:
            fixed += R * yf(prev, d) * c.df(c.time(d))
            prev = d
        assert fixed == pytest.approx(df_start - c.df(c.time(pays[-1])), abs=1e-12), t


def test_curve_shape_and_interpolation():
    c = bootstrap_mibor_ois(QUOTES)
    assert all(b < a for a, b in zip(c.dfs, c.dfs[1:]))  # discount factors fall
    t1, t2 = c.times[3], c.times[4]
    mid = 0.5 * (t1 + t2)
    # log-linear: constant forward between pillars
    assert c.forward_rate(t1, mid) == pytest.approx(c.forward_rate(mid, t2), rel=1e-10)
    assert c.df(0) == 1.0 and c.df(-1) == 1.0
    assert c.df(10.0) < c.df(c.times[-1])  # flat-forward extrapolation


def test_rolled_curve_records_age():
    c = fbil_curve()
    assert c.reference_date == PRICING_DATE and c.source_date == date(2026, 10, 15) and c.age_days() == 8
    s = datetime(2026, 10, 23, 11, 3, tzinfo=IST)
    r = c.rate_for_option(s, datetime(2026, 10, 27, 15, 30, tzinfo=IST))
    assert 0.055 < r < 0.060  # overnight-ish: between O/N and 1M MIBOR-OIS, continuous
    T = (datetime(2026, 10, 27, 15, 30, tzinfo=IST) - s).total_seconds() / (365 * 86400)
    assert math.exp(-r * T) == pytest.approx(c.discount(s, datetime(2026, 10, 27, 15, 30, tzinfo=IST)))


def test_flat_curve():
    c = flat_curve(date(2026, 1, 1), 0.065)
    assert c.zero_rate(0.37) == pytest.approx(0.065) and c.df(2.0) == pytest.approx(math.exp(-0.13))


def test_bad_inputs():
    with pytest.raises(ValueError):
        OISQuotes(date(2026, 10, 15), 0.0587, {"1M": 0.057})  # decimals instead of percent
    with pytest.raises(ValueError):
        DiscountCurve(date(2026, 1, 1), [0.0, 0.5, 0.4], [1.0, 0.99, 0.98])
    with pytest.raises(ValueError):
        fbil_curve().time(datetime(2026, 10, 23, 11, 0))  # naive datetime
