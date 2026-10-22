"""Independent QuantLib benchmark for the in-house pricer.

Prices the same trade with QuantLib's AnalyticEuropeanEngine on flat curves, so the
validator can compare optlib against an external library on every run.
QuantLib is an optional dependency: ``pip install optlib[dev]``.

Time convention: T in years on Actual/365 Fixed. QuantLib needs a date grid, so T is
rounded to whole days; pass T = days / 365 for an exact comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

_EVAL_YEAR, _EVAL_MONTH, _EVAL_DAY = 2026, 1, 5


def _ql():
    try:
        import QuantLib as ql  # noqa: N813
    except ImportError as exc:  # pragma: no cover
        raise ImportError("QuantLib is required for the benchmark: pip install QuantLib") from exc
    return ql


@dataclass(frozen=True)
class QLResult:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    phi: float | None


def _setup(T: float):
    ql = _ql()
    today = ql.Date(_EVAL_DAY, _EVAL_MONTH, _EVAL_YEAR)
    ql.Settings.instance().evaluationDate = today
    days = int(round(T * 365.0))
    if days <= 0:
        raise ValueError("benchmark needs at least one day to expiry")
    return ql, today, today + days


def _option(ql, maturity, K, option_type):
    kind = ql.Option.Call if str(option_type).lower() in ("call", "c") else ql.Option.Put
    return ql.EuropeanOption(ql.PlainVanillaPayoff(kind, float(K)), ql.EuropeanExercise(maturity))


def _flat(ql, today, rate):
    return ql.YieldTermStructureHandle(ql.FlatForward(today, float(rate), ql.Actual365Fixed(), ql.Continuous))


def _vol(ql, today, sigma):
    return ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), float(sigma), ql.Actual365Fixed()))


def quantlib_bsm(S, K, T, r, q, sigma, option_type="call") -> QLResult:
    """QuantLib BSM price and Greeks (spot S, continuous dividend yield q)."""
    ql, today, maturity = _setup(T)
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(float(S))), _flat(ql, today, q), _flat(ql, today, r), _vol(ql, today, sigma))
    opt = _option(ql, maturity, K, option_type)
    opt.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return QLResult(opt.NPV(), opt.delta(), opt.gamma(), opt.vega(), opt.theta(), opt.rho(), opt.dividendRho())


def quantlib_black76(F, K, T, r, sigma, option_type="call") -> QLResult:
    """QuantLib Black-76 price and Greeks on forward F. rho is the total dV/dr with F fixed."""
    ql, today, maturity = _setup(T)
    process = ql.BlackProcess(ql.QuoteHandle(ql.SimpleQuote(float(F))), _flat(ql, today, r), _vol(ql, today, sigma))
    opt = _option(ql, maturity, K, option_type)
    opt.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return QLResult(opt.NPV(), opt.delta(), opt.gamma(), opt.vega(), opt.theta(),
                    opt.rho() + opt.dividendRho(), None)


def quantlib_implied_vol_bsm(price, S, K, T, r, q, option_type="call", accuracy=1e-12) -> float:
    """QuantLib implied BSM volatility for the same conventions."""
    ql, today, maturity = _setup(T)
    vol_quote = ql.SimpleQuote(0.2)
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(float(S))), _flat(ql, today, q), _flat(ql, today, r),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(
            today, ql.NullCalendar(), ql.QuoteHandle(vol_quote), ql.Actual365Fixed())))
    opt = _option(ql, maturity, K, option_type)
    opt.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return opt.impliedVolatility(float(price), process, accuracy, 1000, 1e-7, 10.0)
