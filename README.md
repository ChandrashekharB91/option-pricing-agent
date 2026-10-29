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
| FastAPI backend, Python client | Done |
| Orchestrator and validator agents | Done; provider-neutral (Claude or any OpenAI-compatible API), tested with scripted models |
| Streamlit dashboard: market & smile, pricer, book risk, agent chat | Done |
| Risk: historical VaR/ES, backtest, stress tests, limits, P&L explain, HTML risk report | Done (Phase 2); limits are placeholders until risk sets them |
| Strategy lab: recommender, P&L projector, hedge finder | Done (Phase 3); margins are estimates; bhavcopy backtest pending |

## Run the app (local)

```powershell
pip install -e ".[app]"
copy .env.example .env          # then set LLM_PROVIDER / LLM_API_KEY / LLM_MODEL (and UPSTOX_ANALYTICS_TOKEN for live data)
python scripts/check_llm.py      # confirms the chosen model can run the agent

# terminal 1: the API (docs at http://127.0.0.1:8000/docs)
uvicorn optlib_app.api:app

# terminal 2: the dashboard (opens http://localhost:8501)
python -m streamlit run src/optlib_app/dashboard.py
```

No administrator rights are needed. `.streamlit/config.toml` keeps the dashboard on this machine only
(127.0.0.1), so Windows does not ask to allow it through the firewall; if a firewall prompt still appears,
cancel it and the dashboard still works at http://localhost:8501. `python -m streamlit` also works where IT
policy blocks `streamlit.exe` in the user's Scripts folder.

`DATA_MODE=replay` (default) serves a recorded NIFTY snapshot, so everything works offline and outside market
hours. `REPLAY_SNAPSHOT=20261023` (default) is one weekly expiry; `REPLAY_SNAPSHOT=20261029` is the live snapshot of
29 Oct 2026 with four expiries (3, 9, 17 and 23 Nov), for pricing across expiries. `DATA_MODE=live` uses Upstox and FBIL; surfaces are cached for 60 seconds.

| Endpoint | What it returns |
| --- | --- |
| `GET /health` | data mode, snapshot time |
| `GET /expiries/{underlying}` | expiries available |
| `GET /overview/{underlying}` | spot, curve, per-expiry forward, rate, ATM vol, QA and SVI status |
| `GET /smile/{underlying}/{expiry}` | bid/mid/ask vols by strike, SVI curve, QA drops |
| `POST /price` | price, Greeks (per unit and position), inputs, validation report |
| `POST /book` | per-position and total value and Greeks, worst validation status |
| `POST /scenarios` | spot x vol P&L grid, full revaluation |
| `POST /risk` | VaR/ES, ES contributions, backtest, stress tests, limit checks, Greeks by expiry |
| `GET /risk/eod_days` | days with NSE end-of-day prices available for P&L explain |
| `POST /risk/pnl_explain` | day-on-day P&L split into delta, gamma, vega, theta, cross, rate, unexplained |
| `POST /risk/report` | self-contained HTML risk report (charts, tables, commentary) |
| `POST /agent/ask` | agent answer, tool calls, independent validator review, release status |

From a notebook: `from optlib_app.client import PricingClient; PricingClient().price("2026-10-27", 22500, "call", lots=2)`.

**Agents.** The orchestrator calls the pricing tools and explains results; it is instructed never to state a
number that did not come from a tool, and every number in its answer is checked against the tool results
(untraceable numbers are listed and the answer is marked WARN). The validator is a separate call that sees only
the question and the tool inputs/outputs (never the orchestrator's text) and returns PASS/WARN/FAIL. Any
deterministic FAIL or validator FAIL withholds the answer.

**Which LLM.** The agents run on whatever API `.env` names; nothing in the code is tied to one vendor.

| `LLM_PROVIDER` | Endpoint | Notes |
| --- | --- | --- |
| `anthropic` | Anthropic API | Claude; default model `claude-sonnet-5-5` |
| `groq`, `gemini`, `openrouter`, `github`, `cerebras`, `mistral`, `openai` | preset OpenAI-compatible URL | free tiers available on most; set `LLM_MODEL` |
| `ollama` | `http://localhost:11434/v1` | local, free, no key; data stays on the machine |
| `custom` | `LLM_BASE_URL` | any OpenAI-compatible server (e.g. an internal gateway) |

After setting it, run `python scripts/check_llm.py`: it sends one pricing question through the full agent on
the recorded snapshot and reports whether the model called the tool correctly, kept its numbers grounded, and
what the validator said.

## Risk management (Phase 2)

All risk numbers come from full Black-76 revaluation of the book, not Greek approximations.

- **VaR and ES.** Historical simulation over the last N days (default 500) of index and India VIX closes.
  Each day's index log return and relative VIX change is applied to every position (sticky strike, one day of
  time decay). 1-day VaR at 99% and ES at 97.5% (the FRTB pair), with each position's contribution to ES.
  History stops the day before the pricing date, so there is no look-ahead.
- **Backtest.** VaR recomputed each day over the last 250 days and compared with that day's P&L:
  exception count, Kupiec (frequency) and Christoffersen (clustering) tests, Basel traffic light.
- **Stress tests.** Real market days replayed on today's book (COVID crash and rebound, 2024 election
  results, Budget 2021, Russia-Ukraine, 2019 tax cut, Aug 2015 sell-off) plus hypothetical spot and vol shocks.
- **Limits.** Net delta, gamma, vega, theta, VaR and worst stress against `config/risk_limits.json`
  (amber from 80%, red above 100%). The values shipped are placeholders for the risk team to set.
- **P&L explain.** Uses NSE F&O end-of-day bhavcopy closes on two days: each leg's vol is implied from its
  close, the forward from put-call parity on traded strikes. Actual P&L is split into delta, gamma, vega, theta,
  cross (vanna, volga), rate and unexplained; WARN above 5% unexplained, FAIL above 15%.
- **Report.** One HTML file with every chart and table, downloadable from the dashboard or `POST /risk/report`.

Data: Upstox historical daily candles (public, no token) for index and India VIX history, cached to CSV;
NSE bhavcopy files from nsearchives. Replay mode uses the recorded files in `tests/fixtures`.
Not yet included: exchange SPAN margin (needs NSE SPAN risk-parameter files).

## Strategy lab (Phase 3): recommender, P&L projector, hedge finder

Internal analysis tool, not investment advice. Every result carries this disclaimer: *This is not financial
advice. The recommendations are model output for internal analysis only. The author is not responsible for any
losses or gains from decisions based on them.*

- **Recommender.** Enter any capital amount and a market view. The engine works through the library of
  buy-options, debit-spread, hedged-selling, naked-selling and futures-plus-options strategies, roughly 90 to 100
  of them at listed strikes spaced by the expected move. It then:
  - prices each strategy at the bid/ask you would actually pay, with all charges;
  - estimates margin and sizes the trade in lots to fit the capital and a loss budget (default 50% of capital);
  - projects the P&L and ranks by expected profit per rupee at risk, expected return on capital, or chance of profit.
- **Recommendation rule.** An idea is *recommended* only if its expected P&L is above zero **and** its chance of
  profit is at least the threshold (default 50%, a slider), under the chosen view, within the capital and the loss
  budget. Otherwise the screen says there is no recommendation and lists the closest ideas with the reason each
  failed. Chance of profit = the probability, under the chosen view's distribution of NIFTY at the horizon, of
  the index levels where the strategy's P&L after all charges is above zero.
- **Why this strategy.** Every idea carries its basis (the market view it suits, its volatility stance and risk
  shape) and how it fits the inputs: the breakeven move it needs against the view's expected move, and whether
  implied vs realised volatility favours buying or selling premium.
- **Views.** The four views are:
  - *my view*: target level, horizon, vol and IV change;
  - *model*: EWMA realised vol, no drift;
  - *market-implied*: the risk-neutral density from the smile, re-centred so it reprices the quotes;
  - *blend*: half my view, half the model.

  Under the market view every strategy loses its costs on average, so any edge shown comes from the view.
- **P&L projector / strategy builder.** For any combination of calls, puts and the future it shows:
  - payoff at expiry, at the horizon and now;
  - breakevens, max profit and loss, and loss in a 3-sd move;
  - expected P&L, chance of profit and ES95 under each view;
  - Greeks, a cost breakdown, and a P&L heatmap by index level and day.
- **Hedge finder.** Searches futures, options and spreads (1 to 3 lots) that most reduce a book's ES95 within a
  cash budget.
- **Costs and margin.** Charges (STT at the 1 Apr 2026 rates, exchange, SEBI, stamp duty, GST, brokerage) are in
  `config/charges.json`. The SPAN-style margin estimate's parameters are in `config/margin.json`; confirm against
  the broker. Every recommendation and hedge search is appended to `data/recommendations.jsonl` for review.

| Endpoint | What it returns |
| --- | --- |
| `GET /strategy/meta` | families, objectives, views, expiries, disclaimer |
| `POST /strategy/recommend` | ranked, sized ideas with projections; families that do not fit the capital |
| `POST /strategy/project` | full projection of a user-built strategy, with a P&L heatmap |
| `POST /strategy/hedge` | best hedges for a book, risk before and after |

The agent has matching tools (`recommend_strategies`, `project_strategy`, `find_hedge`) and always ends such
answers with the disclaimer.

### Backtest and calibration

```powershell
python scripts/fetch_bhavcopy.py --start 2024-07-08 --end 2026-10-23   # once, about 3 minutes, ~45 MB
python scripts/backtest.py --start 2024-07-08 --end 2026-10-22          # about 5 minutes
```

The backtest rebuilds every trading day's market from NSE's end-of-day files, then runs every strategy through
the projector. Each idea is settled at NIFTY's actual expiry close after charges. It reports:

- calibration: predicted chance of profit against the share that actually won;
- predicted against realised P&L, by strategy;
- how ideas that passed the recommendation rule performed.

The output is `reports/backtest_report.html` and `reports/backtest_summary.json`; per-trade rows go to
`data/backtest/trades.csv.gz`.

## Daily email feed

A daily email for readers with the day's NIFTY option-market numbers: index level and India VIX, ATM implied
vol by expiry with its day-on-day change, 25-delta risk reversal and butterfly, the expected (1-sd) move to each
expiry, market-implied (risk-neutral) odds at the weekly expiry, reference prices with Greeks and validation
status, open interest and put/call ratio, futures basis, MIBOR-OIS rates, and the data-quality checks. It carries
no trade ideas (the strategy recommender stays on the internal dashboard) and every email ends with a
not-investment-advice disclaimer and a one-click unsubscribe link. The layout works in light and dark mail
clients and on phones.

**Sign-up.** Readers open the sign-up page at `http://127.0.0.1:8000/feed/signup` (with the API running;
`python scripts/daily_feed.py --signup` opens it), or use the dashboard's *Daily feed* tab. Readers tick a consent box; a
confirmation email goes out and the subscription starts only when they click it (double opt-in). Unsubscribe
links are personal; admins can also erase a subscriber completely. Subscribers live in `data/feed/subscribers.db`
(git-ignored).

**Try it without email.** Leave `SMTP_HOST` empty: every email is saved to `data/feed/outbox/*.eml` (double-click
to open in Outlook or Mail).

```powershell
python scripts/daily_feed.py --preview                  # writes reports/feed/feed_YYYYMMDD.html and opens it
python scripts/daily_feed.py --signup                   # opens the sign-up page (API must be running)
python scripts/daily_feed.py --add reader@company.com --name "Reader"   # sends the confirmation
python scripts/daily_feed.py --test you@company.com     # one [TEST] copy
python scripts/daily_feed.py --send                     # all confirmed subscribers, once per day
python scripts/daily_feed.py --subscribers
```

**Sending for real.** Set `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD` and `MAIL_FROM` in `.env` (examples in
`.env.example`; for Gmail use an app password, never the account password). Set `FEED_PUBLIC_URL` to an address
readers can reach, or the confirm/unsubscribe links will only work on this machine; until then readers can reply
UNSUBSCRIBE and an admin removes them in the dashboard.

**Schedule.** Quotes are live only in market hours, so send at about 15:20 IST with `DATA_MODE=live`. Windows
Task Scheduler, no administrator rights needed (the PC must be on and logged in):

```powershell
# from the repo folder; scripts\daily_feed.cmd uses .venv and logs to data\feed\task.log
schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 15:20 /TN "NIFTY options daily" /TR "$PWD\scripts\daily_feed.cmd"
schtasks /Run /TN "NIFTY options daily"       # try it once now
```

Market holidays are not detected yet: on a holiday the script sends the last available quotes, so pause the task
on NSE holidays. The send log is `data/feed/sent.jsonl`; the day-on-day changes use `data/feed/history.jsonl`.

**Compliance note.** The email is market information, not research or advice. Sending recommendations to
people outside the desk would need SEBI research-analyst or investment-adviser registration, which is why the
recommender is not part of the feed. Keep the consent record and honour unsubscribes (the DPDP Act, 2023).

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

### Regression suite (model sign-off)

`python scripts/regression.py` recomputes about 5,200 frozen outputs and compares them with
`tests/regression/golden_phase1.json`. The outputs cover:

- the Black-76 grid and implied vols;
- the FBIL curve;
- forwards, QA, implied vols and SVI on the recorded NIFTY snapshot;
- a synthetic two-expiry surface;
- prices, Greeks and validator verdicts;
- the sample book and its Phase 2 risk numbers.

It also runs the QuantLib gates. It writes `reports/regression_report.md` and exits 1 on any difference beyond
tolerance. The same checks run in `pytest` and in CI on every push.

A deliberate model change that moves numbers needs
`python scripts/regression.py --update --reason "what changed and why"`. That stamps the golden file and adds
a line to `tests/regression/CHANGELOG.md`, which the reviewer approves in the pull request.
Run the script before every demo.
