# Option Pricing Agent

In-house option pricing library (`optlib`) and, as it grows, the agent, API and dashboard built on it.
Phase 1 covers NSE index options (European, cash-settled), priced with Black-76 on the forward.

## Status

| Component | State |
| --- | --- |
| Black-76 and BSM pricer, analytic Greeks | Done, matches QuantLib |
| Implied volatility solver | Done, matches QuantLib |
| NSE calendar, MIBOR-OIS curve, forwards | Next |
| SVI surface, validator, API, agent, dashboard | Planned (phase 1) |

## Install

```bash
pip install -e ".[dev]"   # numpy, scipy; QuantLib and pytest for benchmarking
pytest -q
python scripts/compare_quantlib.py   # writes reports/quantlib_comparison.md
```

## Usage

```python
from optlib import black76_greeks, implied_vol_black76

g = black76_greeks(F=25000, K=25500, T=30/365, r=0.065, sigma=0.14, option_type="call")
g.price, g.delta, g.gamma
g.reporting_units()      # vega per vol point, theta per day, rho per bp

vol = implied_vol_black76(price=210.5, F=25000, K=25500, T=30/365, r=0.065, option_type="call")
```

All functions are vectorised: pass NumPy arrays for strikes, expiries or option types to price a whole chain.

## Conventions

- Time in years on Actual/365 Fixed; rates continuously compounded.
- Raw Greeks are per 1.00 change in the input and per year; `reporting_units()` converts to desk units.
- Theta and charm are sensitivities to calendar time passing (negative of d/dT).
- Black-76 rho is dV/dr with the forward held fixed; BSM rho holds spot and dividend yield fixed.
- `implied_vol_*` returns NaN with a status code when a quote is outside no-arbitrage bounds
  or its time value is below double precision.

## Validation

Every result is checked against QuantLib (`optlib.benchmark`) with the spec tolerances:
price within max(₹0.01, 0.001% of price), Greeks within 0.01% relative.
See `reports/quantlib_comparison.md`.
