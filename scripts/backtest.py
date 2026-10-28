"""Backtest and calibration of the strategy engine on NSE end-of-day history (NIFTY weekly options).

    python scripts/fetch_bhavcopy.py --start 2024-07-08 --end 2026-10-23     # once: download the data
    python scripts/backtest.py --start 2024-07-08 --end 2026-10-22

Writes data/backtest/trades.csv.gz (one row per day and strategy), reports/backtest_summary.json and
reports/backtest_report.html (calibration, predicted vs realised P&L by strategy, the recommendation rule).
See optlib.backtest for the method and assumptions.
"""
import argparse
import csv
import gzip
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optlib import backtest as bt  # noqa: E402
from optlib.charges import Charges  # noqa: E402
from optlib.data.bhavcopy import load_symbol_rows  # noqa: E402
from optlib.data.history import History  # noqa: E402
from optlib.margin import MarginParams  # noqa: E402
from optlib_app.config import Settings  # noqa: E402

_G = {}


def _init(history_path, cache, charges_path, margin_path, cfg):
    h = History.from_csv(history_path)
    _G.update(h=h, cache=cache, ch=Charges.load(charges_path), mp=MarginParams.load(margin_path), cfg=cfg,
              settle=bt.settle_from_history(h, cfg.symbol))


def _one(day: date):
    try:
        rows = load_symbol_rows(day, _G["cache"], _G["cfg"].symbol)
    except Exception:  # noqa: BLE001
        return day, []
    return day, bt.run_day(rows, day, _G["h"], _G["settle"], _G["cfg"], _G["ch"], _G["mp"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2024-07-08")
    ap.add_argument("--end", default="2026-10-22")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-prob", type=float, default=0.5)
    ap.add_argument("--half-spread", type=float, default=0.0025, help="each side of the close, share of price")
    a = ap.parse_args()
    s = Settings()
    h = History.from_csv(s.history_path)
    days = [d for d in h.dates if date.fromisoformat(a.start) <= d <= date.fromisoformat(a.end)]
    cfg = bt.BacktestConfig(min_prob_profit=a.min_prob, half_spread_rel=a.half_spread)
    cache = s.cache_dir / "bhav"
    t0, rows, used = time.time(), [], 0
    with ProcessPoolExecutor(a.workers, initializer=_init,
                             initargs=(s.history_path, cache, s.charges_path, s.margin_path, cfg)) as ex:
        for day, out in ex.map(_one, days, chunksize=4):
            if out:
                used += 1
                rows.extend(out)
    print(f"{used} of {len(days)} days usable, {len(rows):,} strategy ideas, {time.time() - t0:.0f}s")
    if not rows:
        print("No data: run scripts/fetch_bhavcopy.py first.")
        return 1

    out_dir = ROOT / "data" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [k for k in rows[0] if k != "breakevens"]
    with gzip.open(out_dir / "trades.csv.gz", "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = {
        "period": [a.start, a.end], "days": len(days), "days_used": used, "ideas": len(rows),
        "assumptions": {"rate": cfg.rate, "half_spread_rel": cfg.half_spread_rel, "min_prob_profit": cfg.min_prob_profit,
                        "expiry_window_days": [cfg.min_days, cfg.max_days], "charges": "config/charges.json (current)",
                        "margin": "config/margin.json (estimate)"},
        "data_checks": bt.data_checks(rows, h),
        "views": {},
    }
    for v in cfg.views:
        summary["views"][v] = {"brier": bt.brier(rows, v), "calibration": bt.calibration(rows, v),
                               "by_strategy": bt.by_strategy(rows, v), "rule": bt.rule_performance(rows, v)}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "backtest_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    from optlib_app.backtest_report import render

    (ROOT / "reports" / "backtest_report.html").write_text(render(summary), encoding="utf-8")
    for v, d in summary["views"].items():
        r = d["rule"]
        print(f"[{v}] Brier {d['brier']:.3f}; rule: {r['ideas']:,} ideas, predicted win {r.get('predicted_pop', 0):.0%}"
              f" vs realised {r.get('realised_win_rate', 0):.0%}; mean P&L predicted {r.get('predicted_pnl_pts', 0):+.1f}"
              f" vs realised {r.get('realised_pnl_pts', 0):+.1f} pts per unit")
    print("Report: reports/backtest_report.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
