# Regression report: PASS

Run 2026-10-27T14:05:36+00:00 at commit ef68452; golden values from 2026-10-27T14:02:33+00:00 (Add Phase 3 strategies section (recommender, projector, hedge finder)).

## Golden values

| Section | What it freezes | Values | Failed | Largest difference | Status |
| --- | --- | ---: | ---: | ---: | --- |
| black76 | Black-76 price and 10 Greeks on a 420-case grid | 4620 | 0 | 0.00e+00 | PASS |
| implied_vol | implied vol recovered from grid prices | 284 | 0 | 0.00e+00 | PASS |
| curve | FBIL MIBOR-OIS curve of 23 Oct 2026 | 19 | 0 | 0.00e+00 | PASS |
| live_surface | forward, QA, implied vols and SVI on the recorded NIFTY snapshot | 141 | 0 | 0.00e+00 | PASS |
| synthetic_surface | two-expiry surface: fit recovery, time interpolation, calendar check | 18 | 0 | 0.00e+00 | PASS |
| pricing | prices, Greeks and validator verdicts for 6 NIFTY options | 144 | 0 | 0.00e+00 | PASS |
| book | sample book value and Greeks | 6 | 0 | 0.00e+00 | PASS |
| risk | Phase 2: VaR, ES, backtest, stress, P&L explain on the sample book | 23 | 0 | 0.00e+00 | PASS |
| strategies | Phase 3: recommender, P&L projector and hedge finder | 30 | 0 | 0.00e+00 | PASS |

## Benchmark gates

| Gate | Limit | Result | Status |
| --- | --- | --- | --- |
| QuantLib price | max(Rs 0.01, 0.001%) per case | largest difference 9.09e-12 index points | PASS |
| QuantLib delta | 0.01% relative (abs 1e-9 near zero) | largest relative difference 9.86e-12 | PASS |
| QuantLib gamma | 0.01% relative (abs 1e-9 near zero) | largest relative difference 1.34e-13 | PASS |
| QuantLib vega | 0.01% relative (abs 1e-9 near zero) | largest relative difference 1.74e-12 | PASS |
| QuantLib theta | 0.01% relative (abs 1e-9 near zero) | largest relative difference 3.94e-10 | PASS |
| QuantLib rho | 0.01% relative (abs 1e-9 near zero) | largest relative difference 7.94e-09 | PASS |
| implied vol round trip | 1e-8 (0.000001 vol pt) | largest error 1.55e-10 over 284 cases | PASS |

## Tolerances

| Section | Values | Tolerance | Why |
| --- | --- | --- | --- |
| black76 | `*` | abs 1e-10 + rel 1e-09 | closed form: floating-point noise only |
| implied_vol | `*` | abs 1e-09 | solver stops at 1e-12 relative in price |
| curve | `*/df` | abs 1e-12 | bootstrap root-finder xtol 1e-15 |
| curve | `*` | abs 1e-10 | rates derived from the discount factors |
| live_surface | `*/forward` | abs 1e-06 | median of parity forwards: arithmetic only |
| live_surface | `*/iv_mid/*` | abs 1e-09 | implied vol solver on recorded quotes |
| live_surface | `*/svi/*` | abs 0.0001 | optimiser output: 0.01 vol point, a tenth of a typical spread |
| live_surface | `*` | abs 1e-09 | deterministic arithmetic |
| synthetic_surface | `*` | abs 0.0001 | optimiser output: 0.01 vol point |
| pricing | `*/price` | abs 0.001 | 0.001 index points, 1/50 of a tick |
| pricing | `*/vol` | abs 1e-05 | 0.001 vol point |
| pricing | `*` | abs 1e-08 + rel 0.0001 | Greeks: 0.01% relative |
| book | `*/value` | abs 0.5 | Rs 0.50 on the book |
| book | `*` | abs 1e-08 + rel 0.0001 | Greeks: 0.01% relative |
| risk | `pnl_explain/*` | abs 1 | Rs 1 per P&L component |
| strategies | `*` | abs 1 + rel 0.0001 | Rs 1 or 0.01%: projections integrate over a fitted smile |
| risk | `*` | abs 1e-06 + rel 0.0001 | 0.01% relative |
