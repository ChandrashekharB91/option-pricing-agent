# Option Pricing Agent

In-house option pricing library (`optlib`) and, as it grows, the agent, API and dashboard built on it.
Phase 1 covers NSE index options (European, cash-settled), priced with Black-76 on the forward.

## Status

| Component | State |
| --- | --- |
| Black-76 and BSM pricer, analytic Greeks | Done, matches QuantLib |
| Implied volatility solver | Done, matches QuantLib |
| Upstox market data adapter (chain, spot, futures, expiries) | Done, tested on recorded-format fixtures |
| MIBOR-OIS curve, parity forwards, chain QA, NSE bhavcopy | Next |
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

## Market data (Upstox)

The adapter uses Upstox's free, read-only **Analytics Token** (valid one year, cannot place orders).

1. Generate the token in Upstox Developer Apps → Analytics → Generate Token.
2. Copy `.env.example` to `.env` in the repo root and paste the token after `UPSTOX_ANALYTICS_TOKEN=`.
   `.env` is git-ignored; never commit it or paste the token anywhere else.
3. Check the feed end to end:

```bash
python scripts/upstox_smoke.py                       # NIFTY, nearest expiry
python scripts/upstox_smoke.py --underlying BANKNIFTY --save
```

In code:

```python
from optlib.data import UpstoxSource

src = UpstoxSource.from_env()               # reads the token from .env
expiries = src.expiries("NIFTY")            # from the exchange's contract list, holiday shifts included
chain = src.option_chain("NIFTY", expiries[0])
chain.spot, chain.time_to_expiry(), chain.quote(25700, "call").mid
rows = chain.to_records()                   # flat rows for pandas or Snowflake
futures = src.futures("NIFTY")
```

Supported underlyings: NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, NIFTYNXT50. Missing or zero bids/asks
become NaN. Upstox's own IV and delta are kept as `vendor_iv` / `vendor_delta` for comparison only.
Every tool depends on the `MarketDataSource` interface, so Bloomberg can replace Upstox later.

## Conventions

- Time in years on Actual/365 Fixed; rates continuously compounded.
- NSE index options expire at 15:30 IST; time to expiry is measured to that instant.
- Raw Greeks are per 1.00 change in the input and per year; `reporting_units()` converts to desk units.
- Theta and charm are sensitivities to calendar time passing (negative of d/dT).
- Black-76 rho is dV/dr with the forward held fixed; BSM rho holds spot and dividend yield fixed.
- `implied_vol_*` returns NaN with a status code when a quote is outside no-arbitrage bounds
  or its time value is below double precision.

## Validation

Every result is checked against QuantLib (`optlib.benchmark`) with the spec tolerances:
price within max(₹0.01, 0.001% of price), Greeks within 0.01% relative.
See `reports/quantlib_comparison.md`.
