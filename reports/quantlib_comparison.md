# optlib 0.1.0 vs QuantLib

**Result: 8400/8400 comparisons within tolerance.**

Grid: 840 BSM and 420 Black-76 cases; spot/forward 20,000 and 25,000; strikes 80% to 125%;
expiries 2 to 365 days; vols 8% to 35%; r = 6.5%; q = 0% and 1.2%; calls and puts.
Tolerances (spec): price within max(Rs 0.01, 0.001% of price); Greeks within 0.01% relative.

| Model | Measure | Cases | Max abs diff | Max rel diff | Within tolerance |
| --- | --- | --- | --- | --- | --- |
| BSM | price | 840 | 1.18e-11 | 1.39e-05 | 840/840 |
| BSM | delta | 840 | 1.37e-14 | 5.67e-08 | 840/840 |
| BSM | gamma | 840 | 1.56e-17 | 2.83e-12 | 840/840 |
| BSM | vega | 840 | 3.24e-11 | 2.79e-12 | 840/840 |
| BSM | theta | 840 | 5.74e-10 | 3.15e-03 | 840/840 |
| BSM | rho | 840 | 3.64e-11 | 3.26e-08 | 840/840 |
| BSM | phi | 840 | 3.64e-11 | 5.67e-08 | 840/840 |
| Black-76 | price | 420 | 9.09e-12 | 4.14e-07 | 420/420 |
| Black-76 | delta | 420 | 3.55e-15 | 1.96e-09 | 420/420 |
| Black-76 | gamma | 420 | 5.41e-18 | 1.70e-12 | 420/420 |
| Black-76 | vega | 420 | 2.23e-11 | 1.74e-12 | 420/420 |
| Black-76 | theta | 420 | 1.42e-10 | 9.24e-06 | 420/420 |
| Black-76 | rho | 420 | 7.28e-12 | 4.14e-07 | 420/420 |

Larger relative differences occur only where the value itself is near zero (deep OTM, theta near
its sign change); every absolute difference is at floating-point rounding level.

## Implied volatility

564 quotes (expiry of 7 days or more, vega above 1e-6 x spot).
Max error vs true vol: 1.61e-10. Max difference vs QuantLib: 1.92e-10.

Quotes with vega below 1e-6 x spot are excluded: their time value is below double precision,
so the solver correctly returns status BELOW_INTRINSIC instead of a vol.
