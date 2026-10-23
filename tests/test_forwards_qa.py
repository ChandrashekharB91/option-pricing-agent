"""Parity forward, chain QA and vol slice: synthetic cases with known answers, plus the live snapshot."""
import math
from dataclasses import replace
from datetime import date, datetime

import numpy as np
import pytest

from helpers import fbil_curve, live_market
from optlib import black76_price
from optlib.curves import flat_curve
from optlib.data.models import IST, OptionChain, OptionQuote, year_fraction
from optlib.forwards import check_against_future, parity_forward, parity_regression
from optlib.qa import QAConfig, review_chain, review_quote, static_arbitrage
from optlib.slices import build_vol_slice

AS_OF = datetime(2026, 11, 2, 10, 0, tzinfo=IST)
EXPIRY = date(2026, 11, 23)
F_TRUE, R = 22650.0, 0.058


def smile(k):
    x = math.log(k / F_TRUE)
    return 0.15 - 0.4 * x + 3.0 * x * x


def synthetic_chain(spread=0.10, extra=None, F=F_TRUE):
    T = year_fraction(AS_OF, EXPIRY)
    quotes = []
    for i, k in enumerate(np.arange(20000.0, 25001.0, 100.0)):
        for kind in ("call", "put"):
            mid = black76_price(F, k, T, R, smile(k), kind)
            bid, ask = max(round(mid - spread / 2, 2), 0.0), round(mid + spread / 2, 2)
            quotes.append(OptionQuote(f"K{i}{kind[0]}", EXPIRY, float(k), kind, bid=bid if bid > 0 else math.nan,
                                      ask=ask, volume=1000, oi=5000.0))
    for q in extra or []:
        quotes = [x for x in quotes if not (x.strike == q.strike and x.option_type == q.option_type)] + [q]
    return OptionChain("NIFTY", "NSE_INDEX|Nifty 50", EXPIRY, spot=22500.0, as_of=AS_OF, source="test",
                       quotes=tuple(quotes))


def test_parity_forward_recovers_truth():
    ch = synthetic_chain()
    df = math.exp(-R * ch.time_to_expiry())
    est = parity_forward(ch, df)
    assert est.forward == pytest.approx(F_TRUE, abs=0.1)
    assert est.n_strikes == 11 and est.dispersion < 0.2
    assert est.implied_carry == pytest.approx(math.log(est.forward / 22500) / ch.time_to_expiry())


def test_parity_forward_ignores_a_bad_quote():
    ch = synthetic_chain()
    good = ch.quote(22700.0, "call")
    bad = replace(good, bid=good.bid + 40, ask=good.ask + 40)  # stale/erroneous quote
    ch2 = synthetic_chain(extra=[bad])
    est = parity_forward(ch2, math.exp(-R * ch2.time_to_expiry()))
    assert est.forward == pytest.approx(F_TRUE, abs=0.2)
    assert 22700.0 not in est.strikes


def test_parity_forward_needs_pairs():
    ch = synthetic_chain()
    only_calls = replace(ch, quotes=tuple(q for q in ch.quotes if q.option_type == "call"))
    with pytest.raises(ValueError):
        parity_forward(only_calls, 0.99)


def test_parity_regression_recovers_df_for_longer_expiry():
    ch = synthetic_chain(spread=0.0001)
    F, df = parity_regression(ch, ch.strikes)
    assert F == pytest.approx(F_TRUE, abs=0.05)
    assert df == pytest.approx(math.exp(-R * ch.time_to_expiry()), abs=1e-5)


def test_future_check_verdicts():
    assert check_against_future(22500 * 1.0004, 22500).verdict == "PASS"
    assert check_against_future(22500 * 1.0010, 22500).verdict == "WARN"
    assert check_against_future(22500 * 0.9970, 22500).verdict == "FAIL"


def test_quote_rules():
    cfg = QAConfig()
    base = OptionQuote("k", EXPIRY, 23000.0, "call", bid=40.0, ask=40.5, volume=10)
    assert review_quote(base, 22650, 0.997, cfg).keep
    cases = {
        "missing_bid_or_ask": replace(base, bid=math.nan),
        "crossed_or_locked": replace(base, bid=41.0, ask=40.5),
        "itm_side": replace(base, strike=22000.0, bid=680.0, ask=681.0),
        "price_below_min": replace(base, bid=0.35, ask=0.40),
        "wide_spread": replace(base, bid=30.0, ask=40.0),
        "outside_moneyness": replace(base, strike=30000.0, bid=1.0, ask=1.05),
    }
    for reason, q in cases.items():
        d = review_quote(q, 22650, 0.997, cfg)
        assert not d.keep and reason in d.reasons, reason
    put = OptionQuote("p", EXPIRY, 22000.0, "put", bid=10.0, ask=10.5, volume=0)
    d = review_quote(put, 22650, 0.997, cfg)
    assert d.keep and d.flags == ("no_volume_today",)
    below = OptionQuote("p2", EXPIRY, 23000.0, "put", bid=300.0, ask=301.0)  # ITM put worth >= 0.997 x 350
    assert "below_intrinsic" in review_quote(below, 22650, 0.997, QAConfig(otm_only=False)).reasons


def test_static_arbitrage_detection():
    ch = synthetic_chain()
    df = math.exp(-R * ch.time_to_expiry())
    calls = [q for q in ch.calls() if q.strike >= F_TRUE]
    assert static_arbitrage(calls, F_TRUE, df) == []
    q = calls[3]
    bumped = [replace(c, bid=c.bid + 25, ask=c.ask + 25) if c is q else c for c in calls]
    kinds = {v.kind for v in static_arbitrage(bumped, F_TRUE, df)}
    assert "convexity" in kinds
    tick_noise = [replace(c, bid=c.bid + 0.03, ask=c.ask + 0.03) if c is q else c for c in calls]
    assert static_arbitrage(tick_noise, F_TRUE, df) == []


def test_small_arbitrage_warns_large_fails():
    ch = synthetic_chain()
    q = ch.quote(23500.0, "call")
    for bump, expected in ((1.8, "WARN"), (3.0, "FAIL")):  # butterfly slack here is ~1.5 pts
        bad = synthetic_chain(extra=[replace(q, bid=q.bid + bump, ask=q.ask + bump)])
        rep = review_chain(bad, F_TRUE, math.exp(-R * bad.time_to_expiry()))
        assert rep.arbitrage and rep.status == expected, bump


def test_vol_slice_synthetic_recovers_smile():
    ch = synthetic_chain(spread=0.05)
    s = build_vol_slice(ch, flat_curve(AS_OF.date(), R), future_price=F_TRUE)
    assert s.forward_check.verdict == "PASS"
    assert s.qa.status == "PASS"
    assert set(s.option_types[s.strikes < F_TRUE]) == {"put"} and set(s.option_types[s.strikes > F_TRUE]) == {"call"}
    want = np.array([smile(k) for k in s.strikes])
    liquid = s.vega > 50
    assert np.max(np.abs(s.iv_mid[liquid] - want[liquid])) < 2e-4
    assert np.all(s.iv_bid[liquid] < s.iv_mid[liquid]) and np.all(s.iv_mid[liquid] < s.iv_ask[liquid])
    assert s.atm_vol() == pytest.approx(0.15, abs=2e-3)
    assert np.all(s.weights > 0) and len(s.to_records()) == len(s.strikes)


def test_vol_slice_short_expiry_fails_qa():
    ch = replace(synthetic_chain(), as_of=datetime(2026, 11, 23, 13, 0, tzinfo=IST))
    s = build_vol_slice(ch, flat_curve(EXPIRY, R))
    assert s.qa.status == "FAIL" and any("hours" in w for w in s.qa.warnings)


# ------------------------------------------------------------------ live snapshot

def test_live_slice_end_to_end():
    chain, futures = live_market()
    curve = fbil_curve()
    s = build_vol_slice(chain, curve, future_price=futures[0].mid)
    assert 0.055 < s.rate < 0.060
    assert s.forward.forward == pytest.approx(22513.9, abs=1.0)
    assert s.forward_check.verdict == "PASS" and abs(s.forward_check.gap) < 0.0003
    assert s.forward.dispersion < 5.0
    assert s.qa.counts()["kept"] == 60 and s.qa.arbitrage == []
    assert s.qa.status == "PASS"  # curve age 8 days is within the 10-day limit
    assert s.atm_vol() == pytest.approx(0.1626, abs=0.002)
    # put-call consistency at the money: same strike, both sides, same vol off the parity forward
    from optlib import implied_vol_black76
    T, F = s.time_to_expiry, s.forward.forward
    c, p = chain.quote(22500.0, "call"), chain.quote(22500.0, "put")
    v_c = implied_vol_black76(c.mid, F, 22500.0, T, s.rate, "call")
    v_p = implied_vol_black76(p.mid, F, 22500.0, T, s.rate, "put")
    assert abs(v_c - v_p) < 0.001


def test_live_slice_flags_stale_curve():
    chain, futures = live_market()
    s = build_vol_slice(chain, fbil_curve(), config=QAConfig(max_curve_age_days=5))
    assert s.qa.status == "WARN" and any("8 days old" in w for w in s.qa.warnings)
