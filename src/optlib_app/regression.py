"""Regression suite for model sign-off: freeze reference outputs, fail on any unapproved change.

Two kinds of control:

* **Golden values.** Every number the model produces on a fixed set of inputs (an analytic grid,
  the recorded FBIL curve, the recorded 23 Oct 2026 NIFTY snapshot, a synthetic two-expiry surface,
  a sample book) is stored in ``tests/regression/golden_phase1.json``. A run recomputes them and
  compares within the tolerances below. Changing a golden value needs ``--update --reason``, which
  stamps the file and appends to ``tests/regression/CHANGELOG.md`` so the change is reviewed.
* **Benchmark gates.** Independent of the golden file: the pricer must match QuantLib on the grid,
  and implied vol must round-trip, within the spec tolerances.

Run:  python scripts/regression.py            (check; writes reports/regression_report.md)
      python scripts/regression.py --update --reason "why the numbers changed"
"""
from __future__ import annotations

import fnmatch
import itertools
import json
import math
import platform
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures"
GOLDEN = ROOT / "tests" / "regression" / "golden_phase1.json"
CHANGELOG = ROOT / "tests" / "regression" / "CHANGELOG.md"

SAMPLE_BOOK = [{"expiry": "2026-10-27", "strike": 22500, "option_type": "call", "lots": -2},
               {"expiry": "2026-10-27", "strike": 22500, "option_type": "put", "lots": -2},
               {"expiry": "2026-10-27", "strike": 22000, "option_type": "put", "lots": 2}]
PRICING_CASES = [  # (label, strike, type): listed, off-grid, OTM and a wing the validator must withhold
    ("ATM call", 22500, "call"), ("ATM put", 22500, "put"), ("off-grid put", 22525, "put"),
    ("OTM call", 23000, "call"), ("OTM put", 22000, "put"), ("wing call", 27500, "call")]


# ------------------------------------------------------------------ tolerances

@dataclass(frozen=True)
class Tol:
    atol: float
    rtol: float
    why: str

    def ok(self, gold: float, cur: float) -> bool:
        return abs(cur - gold) <= self.atol + self.rtol * abs(gold)

    def text(self) -> str:
        parts = [f"abs {self.atol:g}" if self.atol else "", f"rel {self.rtol:g}" if self.rtol else ""]
        return " + ".join(p for p in parts if p) or "exact"


# (section, key pattern) -> tolerance; first match wins. Text and integer values are compared exactly.
TOLERANCES: list[tuple[str, str, Tol]] = [
    ("black76", "*", Tol(1e-10, 1e-9, "closed form: floating-point noise only")),
    ("implied_vol", "*", Tol(1e-9, 0, "solver stops at 1e-12 relative in price")),
    ("curve", "*/df", Tol(1e-12, 0, "bootstrap root-finder xtol 1e-15")),
    ("curve", "*", Tol(1e-10, 0, "rates derived from the discount factors")),
    ("live_surface", "*/forward", Tol(1e-6, 0, "median of parity forwards: arithmetic only")),
    ("live_surface", "*/iv_mid/*", Tol(1e-9, 0, "implied vol solver on recorded quotes")),
    ("live_surface", "*/svi/*", Tol(1e-4, 0, "optimiser output: 0.01 vol point, a tenth of a typical spread")),
    ("live_surface", "*", Tol(1e-9, 0, "deterministic arithmetic")),
    ("synthetic_surface", "*", Tol(1e-4, 0, "optimiser output: 0.01 vol point")),
    ("pricing", "*/price", Tol(1e-3, 0, "0.001 index points, 1/50 of a tick")),
    ("pricing", "*/vol", Tol(1e-5, 0, "0.001 vol point")),
    ("pricing", "*", Tol(1e-8, 1e-4, "Greeks: 0.01% relative")),
    ("book", "*/value", Tol(0.5, 0, "Rs 0.50 on the book")),
    ("book", "*", Tol(1e-8, 1e-4, "Greeks: 0.01% relative")),
    ("risk", "pnl_explain/*", Tol(1.0, 0, "Rs 1 per P&L component")),
    ("strategies", "*", Tol(1.0, 1e-4, "Rs 1 or 0.01%: projections integrate over a fitted smile")),
    ("risk", "*", Tol(1e-6, 1e-4, "0.01% relative")),
]


def tolerance(section: str, key: str) -> Tol:
    for sec, pat, tol in TOLERANCES:
        if sec == section and fnmatch.fnmatchcase(key, pat):
            return tol
    raise KeyError(f"no tolerance for {section}/{key}")


def _num(x: Any) -> Any:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if not math.isfinite(x) else x
    return x


# ------------------------------------------------------------------ sections

def _black76() -> dict:
    from optlib import black76_greeks

    out = {}
    for c in grid_black76():
        g = black76_greeks(**c).to_dict()
        for m in ("price", "delta", "gamma", "vega", "theta", "rho", "vanna", "volga", "charm", "speed", "zomma"):
            out[f"{case_key(c)}/{m}"] = g[m]
    return out


def _implied_vol() -> dict:
    from optlib import black76_greeks
    from optlib.iv import implied_vol_black76

    out = {}
    for c in grid_black76():
        g = black76_greeks(**c)
        if c["T"] < 7 / 365 or g.vega <= 1e-6 * c["F"]:
            continue  # no information in the price
        out[f"{case_key(c)}/iv"] = float(implied_vol_black76(float(g.price), c["F"], c["K"], c["T"], c["r"],
                                                             c["option_type"]))
    return out


def _curve() -> dict:
    from optlib.curves import bootstrap_mibor_ois
    from optlib.data.models import IST, nse_expiry_instant
    from optlib.data.replay import fbil_from_files

    day = date(2026, 10, 23)
    q = fbil_from_files(FIX / "fbil_miborois_20261001_20261023.json", FIX / "fbil_ovnmibor_20261001_20261023.json")
    c = bootstrap_mibor_ois(q.ois_quotes(day), day)
    out = {f"pillar/{t}/df": df for t, (_, df) in c.pillars.items()}
    out.update({f"zero/{lab}/rate": c.zero_rate(t) for lab, t in
                (("1M", 1 / 12), ("3M", 0.25), ("6M", 0.5), ("1Y", 1.0), ("2Y", 2.0), ("3Y", 3.0), ("5Y", 5.0))})
    start = datetime(2026, 10, 23, 11, 3, tzinfo=IST)
    out["option_rate/2026-10-27/rate"] = c.rate_for_option(start, nse_expiry_instant(date(2026, 10, 27)))
    return out


def _live_surface(svc) -> dict:
    surf = svc.surface("NIFTY")
    out = {"calendar_violations": len(surf.calendar)}
    for sm in surf.smiles:
        e, s = sm.slice.expiry, sm.slice
        out[f"{e}/forward"] = sm.forward
        out[f"{e}/rate"] = s.rate
        out[f"{e}/time_to_expiry"] = sm.time_to_expiry
        if s.forward_check:
            out[f"{e}/future_gap"] = s.forward_check.gap
        for k, v in s.qa.counts().items():
            out[f"{e}/qa/{k}"] = v
        out[f"{e}/qa/status"] = s.qa.status
        out[f"{e}/qa/arbitrage"] = len(s.qa.arbitrage)
        for K, typ, iv in zip(s.strikes, s.option_types, s.iv_mid):
            out[f"{e}/iv_mid/{K:g}{typ[0].upper()}"] = iv
        for k in np.round(np.arange(-0.10, 0.1001, 0.02), 2):
            out[f"{e}/svi/vol_k{k:+.2f}"] = float(sm.fit.vol_k(k))
        out[f"{e}/svi/rmse"] = sm.fit.rmse
        out[f"{e}/fit_status"] = sm.fit.status
        out[f"{e}/butterfly_free"] = sm.fit.butterfly_free
    return out


def _synthetic_surface() -> dict:
    """Two-expiry surface from known SVI smiles: fit recovery, time interpolation, calendar check."""
    from optlib import black76_price
    from optlib.curves import flat_curve
    from optlib.data.models import IST, OptionChain, OptionQuote, year_fraction
    from optlib.surface import build_surface
    from optlib.svi import SVIParams

    as_of, spot, r = datetime(2026, 11, 2, 10, 0, tzinfo=IST), 22500.0, 0.058
    true = {date(2026, 11, 23): (22560.0, SVIParams(a=0.0006, b=0.030, rho=-0.45, m=0.01, s=0.06)),
            date(2026, 12, 29): (22680.0, SVIParams(a=0.0018, b=0.055, rho=-0.40, m=0.02, s=0.09))}

    class Source:
        def __init__(self, params):
            self.params = params

        def expiries(self, underlying):
            return sorted(self.params)

        def futures(self, underlying):
            return []

        def option_chain(self, underlying, expiry):
            F, p = self.params[expiry]
            T = year_fraction(as_of, expiry)
            quotes = []
            for i, K in enumerate(np.arange(20000.0, 25001.0, 100.0)):
                vol = math.sqrt(float(p.total_variance(math.log(K / F))) / T)
                for kind in ("call", "put"):
                    mid = black76_price(F, K, T, r, vol, kind)
                    if mid >= 0.6:
                        quotes.append(OptionQuote(f"{expiry}{i}{kind}", expiry, float(K), kind,
                                                  bid=round(mid - 0.05, 2), ask=round(mid + 0.05, 2), volume=100))
            return OptionChain("NIFTY", "NSE_INDEX|Nifty 50", expiry, spot, as_of, "synthetic", tuple(quotes))

    curve = flat_curve(as_of.date(), r)
    surf = build_surface(Source(true), curve, "NIFTY")
    out = {"calendar_violations": len(surf.calendar)}
    for sm in surf.smiles:
        out[f"{sm.slice.expiry}/forward"] = sm.forward
        for k in (-0.08, -0.04, 0.0, 0.04):
            out[f"{sm.slice.expiry}/svi/vol_k{k:+.2f}"] = float(sm.fit.vol_k(k))
    t1, t2 = (s.time_to_expiry for s in surf.smiles)
    T = 0.5 * (t1 + t2)
    for K in (21500.0, 22600.0, 23500.0):
        v, src = surf.vol(K, T)
        out[f"interpolated/{K:g}/vol"] = v
        out[f"interpolated/{K:g}/source"] = src
    flipped = {date(2026, 11, 23): true[date(2026, 12, 29)], date(2026, 12, 29): true[date(2026, 11, 23)]}
    out["flipped/calendar_violations"] = len(build_surface(Source(flipped), curve, "NIFTY").calendar)
    return out


def _pricing(svc) -> dict:
    out = {}
    for label, K, typ in PRICING_CASES:
        p = svc.price("NIFTY", "2026-10-27", K, typ, 1)
        res, val = p["result"], p["validation"]
        key = label.replace(" ", "_")
        out[f"{key}/price"] = res["price"]
        out[f"{key}/vol"] = res["vol"]
        out[f"{key}/vol_source"] = res["vol_source"]
        for g in ("delta", "gamma", "vega", "theta", "rho", "vanna", "volga"):
            out[f"{key}/greek_{g}"] = res["greeks"][g]
        out[f"{key}/released"] = val["released"]
        for c in val["checks"]:
            out[f"{key}/check/{c['name']}"] = c["status"]
    return out


def _book(svc) -> dict:
    b = svc.book(SAMPLE_BOOK)
    return {f"total/{k}": v for k, v in b["totals"].items()}


def _risk(svc) -> dict:
    r = svc.risk(SAMPLE_BOOK)
    v, bt = r["var"], r["backtest"]
    out = {"var/var99": v["var"], "var/es975": v["es"], "var/worst_date": v["worst"]["date"],
           "var/worst_pnl": v["worst"]["pnl"], "backtest/exceptions": bt["exceptions"],
           "backtest/traffic_light": bt["traffic_light"], "backtest/kupiec_p": bt["kupiec_p"],
           "backtest/christoffersen_p": bt["christoffersen_p"], "stress/worst_scenario": r["stress"][0]["scenario"],
           "stress/worst_pnl": r["stress"][0]["pnl"], "status": r["status"]}
    for c in v["es_contributions"]:
        out[f"var/es_contribution/{c['label']}"] = c["contribution"]
    e = svc.pnl_explain(SAMPLE_BOOK, "2026-10-22", "2026-10-23")
    out.update({f"pnl_explain/{k}": x for k, x in e["totals"].items()})
    out["pnl_explain_status"] = e["status"]
    return out


def _strategies(svc) -> dict:
    out = {}
    r = svc.recommend(20000, {"source": "user", "target": 22900})
    out["recommend_20k_user_22900/considered"] = r["considered"]
    out["recommend_20k_user_22900/feasible"] = r["feasible"]
    for n, i in enumerate(r["ideas"][:3], 1):
        out[f"recommend_20k_user_22900/{n}/label"] = i["strategy"]["label"]
        out[f"recommend_20k_user_22900/{n}/lots"] = i["lots"]
        out[f"recommend_20k_user_22900/{n}/capital"] = i["capital_required"]
        out[f"recommend_20k_user_22900/{n}/expected"] = i["views"]["user"]["expected"]
    legs = [{"instrument": "call", "side": "buy", "strike": 22500, "expiry": "2026-10-27"},
            {"instrument": "call", "side": "sell", "strike": 22700, "expiry": "2026-10-27"}]
    p = svc.project_strategy(legs, 1)
    for k in ("premium", "entry_charges", "capital_required", "max_profit", "max_loss"):
        out[f"bull_call_22500_22700/{k}"] = p[k]
    out["bull_call_22500_22700/margin"] = p["margin"]["total"]
    out["bull_call_22500_22700/breakeven"] = p["breakevens"][0]
    for v in ("market", "model", "user"):
        out[f"bull_call_22500_22700/{v}_expected"] = p["views"][v]["expected"]
        out[f"bull_call_22500_22700/{v}_prob_profit"] = p["views"][v]["prob_profit"]
    h = svc.hedge(SAMPLE_BOOK, 100000)
    out["hedge/before_es95"] = h["before"]["es95"]
    out["hedge/best"] = h["hedges"][0]["label"]
    out["hedge/best_es95"] = h["hedges"][0]["es95"]
    return out


SECTIONS: dict[str, tuple[str, Callable]] = {  # name -> (what it freezes, function(svc))
    "black76": ("Black-76 price and 10 Greeks on a 420-case grid", lambda svc: _black76()),
    "implied_vol": ("implied vol recovered from grid prices", lambda svc: _implied_vol()),
    "curve": ("FBIL MIBOR-OIS curve of 23 Oct 2026", lambda svc: _curve()),
    "live_surface": ("forward, QA, implied vols and SVI on the recorded NIFTY snapshot", _live_surface),
    "synthetic_surface": ("two-expiry surface: fit recovery, time interpolation, calendar check",
                          lambda svc: _synthetic_surface()),
    "pricing": ("prices, Greeks and validator verdicts for 6 NIFTY options", _pricing),
    "book": ("sample book value and Greeks", _book),
    "risk": ("Phase 2: VaR, ES, backtest, stress, P&L explain on the sample book", _risk),
    "strategies": ("Phase 3: recommender, P&L projector and hedge finder", _strategies),
}


# ------------------------------------------------------------------ grid and gates

def grid_black76():
    spots, money, days, vols = [20000.0, 25000.0], [0.80, 0.90, 0.97, 1.00, 1.03, 1.10, 1.25], [2, 7, 30, 91, 365], \
        [0.08, 0.15, 0.35]
    for F, m, d, v, t in itertools.product(spots, money, days, vols, ["call", "put"]):
        yield dict(F=F, K=round(F * m, 2), T=d / 365.0, r=0.065, sigma=v, option_type=t)


def case_key(c: dict) -> str:
    return f"F{c['F']:g}_K{c['K']:g}_{round(c['T'] * 365)}d_v{c['sigma']:g}_{c['option_type']}"


def gates() -> list[dict]:
    """Independent checks against QuantLib and the spec tolerances (not against the golden file)."""
    from optlib import black76_greeks
    from optlib.benchmark import quantlib_black76
    from optlib.iv import implied_vol_black76

    names = ("delta", "gamma", "vega", "theta", "rho")
    price_excess, greek_rel, greek_ok = 0.0, dict.fromkeys(names, 0.0), dict.fromkeys(names, True)
    worst_price, iv_err, n = 0.0, 0.0, 0
    for c in grid_black76():
        ours, ref = black76_greeks(**c), quantlib_black76(**c)
        d = abs(float(ours.price) - ref.price)
        worst_price = max(worst_price, d)
        price_excess = max(price_excess, d / max(0.01, 1e-5 * abs(ref.price)))
        for g in names:
            o, q = float(getattr(ours, g)), float(getattr(ref, g))
            greek_ok[g] &= abs(o - q) <= 1e-9 + 1e-4 * abs(q)  # same rule as scripts/compare_quantlib.py
            if abs(q) > 1e-6:
                greek_rel[g] = max(greek_rel[g], abs(o - q) / abs(q))
        if c["T"] >= 7 / 365 and ours.vega > 1e-6 * c["F"]:
            iv = float(implied_vol_black76(float(ours.price), c["F"], c["K"], c["T"], c["r"], c["option_type"]))
            iv_err = max(iv_err, abs(iv - c["sigma"]))
            n += 1
    rows = [dict(gate="QuantLib price", value=worst_price, limit="max(Rs 0.01, 0.001%) per case",
                 passed=price_excess <= 1.0, detail=f"largest difference {worst_price:.2e} index points")]
    rows += [dict(gate=f"QuantLib {g}", value=v, limit="0.01% relative (abs 1e-9 near zero)", passed=greek_ok[g],
                  detail=f"largest relative difference {v:.2e}") for g, v in greek_rel.items()]
    rows.append(dict(gate="implied vol round trip", value=iv_err, limit="1e-8 (0.000001 vol pt)", passed=iv_err <= 1e-8,
                     detail=f"largest error {iv_err:.2e} over {n} cases"))
    return rows


# ------------------------------------------------------------------ run and compare

def compute(sections: Optional[list[str]] = None, svc=None) -> dict[str, dict]:
    if svc is None:
        from .config import Settings
        from .service import PricingService

        svc = PricingService(Settings(data_mode="replay"))
    return {name: {k: _num(v) for k, v in SECTIONS[name][1](svc).items()} for name in (sections or SECTIONS)}


def compare(golden: dict, current: dict) -> list[dict]:
    rows = []
    for section, cur in current.items():
        gold = golden.get(section, {})
        for key in sorted(set(gold) | set(cur)):
            g, c = gold.get(key, "<missing>"), cur.get(key, "<missing>")
            row = dict(section=section, key=key, golden=g, current=c, diff=None, tolerance="exact")
            if key not in gold:
                row["status"] = "NEW"
            elif key not in cur:
                row["status"] = "MISSING"
            elif isinstance(g, float) and isinstance(c, (int, float)) and not isinstance(c, bool):
                tol = tolerance(section, key)
                row.update(diff=abs(c - g), tolerance=tol.text(), status="PASS" if tol.ok(g, c) else "FAIL")
            else:
                row["status"] = "PASS" if g == c else "FAIL"
            rows.append(row)
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for section in dict.fromkeys(r["section"] for r in rows):
        rs = [r for r in rows if r["section"] == section]
        diffs = [r["diff"] for r in rs if r["diff"] is not None]
        bad = [r for r in rs if r["status"] != "PASS"]
        out.append(dict(section=section, what=SECTIONS[section][0], values=len(rs), failed=len(bad),
                        max_diff=max(diffs) if diffs else 0.0, status="PASS" if not bad else "FAIL"))
    return out


def load_golden(path: Path = GOLDEN) -> dict:
    return json.loads(path.read_text())


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def write_golden(current: dict, reason: str, path: Path = GOLDEN) -> list[dict]:
    """Replace the golden values; returns the rows that changed. The reason goes in the file and the changelog."""
    import scipy

    old = load_golden(path) if path.exists() else {"values": {}}
    changed = [r for r in compare(old["values"], current) if r["status"] != "PASS"]
    meta = dict(updated=datetime.now(timezone.utc).isoformat(timespec="seconds"), reason=reason,
                git_commit=_git("rev-parse", "--short", "HEAD"), author=_git("config", "user.name"),
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform(terse=True), counts={s: len(v) for s, v in current.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "values": current}, indent=1, sort_keys=True) + "\n")
    line = (f"| {meta['updated'][:10]} | {meta['git_commit'] or '-'} | {meta['author'] or '-'} | {reason} | "
            f"{len(changed)} values changed or added |\n")
    if not CHANGELOG.exists():
        CHANGELOG.write_text("# Golden value changes\n\nEvery change to `golden_phase1.json` needs a reason and a "
                             "reviewer. Approve it in the pull request.\n\n| Date | Commit | Author | Reason | Effect |\n"
                             "| --- | --- | --- | --- | --- |\n")
    with CHANGELOG.open("a") as f:
        f.write(line)
    return changed


def report_markdown(rows: list[dict], gate_rows: list[dict], meta: dict) -> str:
    summary = summarise(rows)
    ok = all(s["status"] == "PASS" for s in summary) and all(g["passed"] for g in gate_rows)
    lines = [f"# Regression report: {'PASS' if ok else 'FAIL'}", "",
             f"Run {datetime.now(timezone.utc).isoformat(timespec='seconds')} at commit "
             f"{_git('rev-parse', '--short', 'HEAD') or '-'}; golden values from {meta.get('updated', '?')} "
             f"({meta.get('reason', '')}).", "", "## Golden values", "",
             "| Section | What it freezes | Values | Failed | Largest difference | Status |",
             "| --- | --- | ---: | ---: | ---: | --- |"]
    lines += [f"| {s['section']} | {s['what']} | {s['values']} | {s['failed']} | {s['max_diff']:.2e} | {s['status']} |"
              for s in summary]
    lines += ["", "## Benchmark gates", "", "| Gate | Limit | Result | Status |", "| --- | --- | --- | --- |"]
    lines += [f"| {g['gate']} | {g['limit']} | {g['detail']} | {'PASS' if g['passed'] else 'FAIL'} |" for g in gate_rows]
    bad = [r for r in rows if r["status"] != "PASS"]
    if bad:
        lines += ["", "## Differences", "", "| Section | Key | Golden | Current | Difference | Tolerance | Status |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in bad[:200]:
            diff = "" if r["diff"] is None else f"{r['diff']:.2e}"
            lines.append(f"| {r['section']} | {r['key']} | {r['golden']} | {r['current']} | {diff} | "
                         f"{r['tolerance']} | {r['status']} |")
        if len(bad) > 200:
            lines.append(f"| ... | {len(bad) - 200} more | | | | | |")
    lines += ["", "## Tolerances", "", "| Section | Values | Tolerance | Why |", "| --- | --- | --- | --- |"]
    lines += [f"| {s} | `{p}` | {t.text()} | {t.why} |" for s, p, t in TOLERANCES]
    return "\n".join(lines) + "\n"
