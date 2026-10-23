# Option Pricing Agent

In-house option pricing library (`optlib`) and, as it grows, the agent, API and dashboard built on it.
Phase 1 covers NSE index options (European, cash-settled), priced with Black-76 on the forward.

## Status

| Component | State |
| --- | --- |
| Black-76 and BSM pricer, analytic Greeks | Done, matches QuantLib |
| Implied volatility solver | Done, matches QuantLib |
| Upstox market data adapter (chain, spot, futures, expiries) | Done, validated on a live NIFTY snapshot |
| FBIL MIBOR-OIS discount curve | Done, reprices FBIL inputs exactly |
| Parity forwards, futures check, chain QA, vol slice | Done, validated on a live NIFTY snapshot |
| SVI smile fit, vol surface, calendar checks | Done; on live data RMSE 0.21 vol pts |
| `price_option` and validator checks | Done; QuantLib-exact, withholds market-inconsistent prices |
| API, orchestrator and validator agents, dashboard | Next |
| SVI surface, validator, API, agent, dashboard | Planned (phase 1) |

## Walkthrough notebook

`notebooks/01_pipeline_walkthrough.ipynb` follows one NIFTY chain through every step built so far (market data,
curve, forward, QA, smile, Greeks, QuantLib check) with charts and plain-language notes. It replays the live
23 Oct 2026 snapshot by default, so it runs offline; set `LIVE = True` in the first cell for fresh data.

```bash
pip install -e ".[notebook]"
jupyter lab notebooks/01_pipeline_walkthrough.ipynb
```

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

## From chain to smile

```python
from optlib.curves import bootstrap_mibor_ois
from optlib.data.fbil import FBILClient
from optlib.slices import build_vol_slice

quotes = FBILClient().ois_quotes(chain.as_of.date())        # latest FBIL MIBOR-OIS fixing
curve = bootstrap_mibor_ois(quotes, chain.as_of.date())     # rolled to today; records its age
s = build_vol_slice(chain, curve, future_price=futures[0].mid)

s.rate, s.forward.forward, s.forward_check.verdict          # curve rate, parity forward, vs future
print(s.qa.summary())                                       # what was kept, dropped and why
s.atm_vol(), s.to_records()                                 # bid/mid/ask vols for the SVI fit
```

- **Curve:** FBIL MIBOR-OIS (bullet up to 1Y, semi-annual beyond, Act/365, T+1 start), log-linear
  discount factors. FBIL's public endpoint lags by several business days; the curve records its age and
  QA warns past 10 days. Same-day rates can be passed as `OISQuotes` directly.
- **Forward:** spread-weighted median of per-strike put-call parity forwards near the money, checked
  against the same-expiry future (warn 0.05%, fail 0.20%). Spot x carry is not used: it misses the
  futures basis (0.17% on 23 Oct 2026, enough to split ATM call and put vols by 3.7 points).
- **QA:** named drop rules (missing/crossed quotes, ITM side, below intrinsic, price < ₹0.50, wide
  spread, far moneyness), a no-volume flag, and static-arbitrage checks with a one-tick tolerance;
  arbitrage up to 0.5 points warns (too small to trade after costs), larger fails.

## Surface, pricing and validation

```python
from optlib.surface import build_surface
from optlib.pricing import price_option
from optlib.validator import validate

surf = build_surface(src, curve, "NIFTY", max_expiries=4)   # QA + SVI per expiry, calendar checks
print(surf.summary())

res = price_option(surf, 22500, expiries[0], "call", quantity=65)   # one lot
res.price, res.vol, res.vol_source, res.position_greeks()
report = validate(res, surf)
print(report.summary())          # PASS / WARN / FAIL per check; FAIL withholds the result
```

- **SVI** (raw, quasi-explicit fit): residuals in bid-ask spreads, positive variance, Roger Lee wing bounds,
  Gatheral's g(k) butterfly check. Raw SVI cannot follow a four-day NIFTY smile to within its 0.01-0.03 vol-pt
  bands (0.2 pts low at the money on 23 Oct 2026), so vanilla pricing uses **layers**: listed strikes at their
  market vol, in-between strikes at SVI corrected to their neighbours, SVI wings beyond the quotes. Across
  expiries and for exotics, pure SVI with total variance interpolated linearly in time.
- **Validator**: chain QA, forward vs future, curve age, SVI fit, smile convexity, SVI-vs-market gap, vol source,
  wing price vs nearer listed offers, calendar arbitrage, independent reprice, put-call parity, price bounds,
  Greek signs, QuantLib `BlackCalculator` at the exact time to expiry, analytic vs bumped Greeks.
- **Known limitation**: far out-of-the-money wings (beyond the last strike QA keeps, about 3,500 points out on
  23 Oct) are extrapolated a few paise above the market's tick-floor offers; the validator withholds these.

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
