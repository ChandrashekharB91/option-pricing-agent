"""Shared test grid: index-option-like levels, NSE-style short and long expiries."""
import itertools

SPOTS = [20000.0, 25000.0]
MONEYNESS = [0.80, 0.90, 0.97, 1.00, 1.03, 1.10, 1.25]
DAYS = [2, 7, 30, 91, 365]
RATES = [0.065]
DIV_YIELDS = [0.0, 0.012]
VOLS = [0.08, 0.15, 0.35]
TYPES = ["call", "put"]


def bsm_cases():
    for S, m, d, r, q, v, t in itertools.product(SPOTS, MONEYNESS, DAYS, RATES, DIV_YIELDS, VOLS, TYPES):
        yield dict(S=S, K=round(S * m, 2), T=d / 365.0, r=r, q=q, sigma=v, option_type=t)


def black76_cases():
    for F, m, d, r, v, t in itertools.product(SPOTS, MONEYNESS, DAYS, RATES, VOLS, TYPES):
        yield dict(F=F, K=round(F * m, 2), T=d / 365.0, r=r, sigma=v, option_type=t)


def case_id(c):
    return "-".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in c.items())
