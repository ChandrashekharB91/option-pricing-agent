"""Download NSE F&O end-of-day files (NIFTY rows only, gzip) for a date range into data/cache/bhav.

    python scripts/fetch_bhavcopy.py --start 2024-07-08 --end 2026-10-22

Days come from the index history (trading days only). Already-cached days are skipped, so it can be re-run.
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optlib.data.bhavcopy import load_symbol_rows  # noqa: E402
from optlib.data.history import History  # noqa: E402
from optlib_app.config import Settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-07-08")
    ap.add_argument("--end", default="2026-10-22")
    ap.add_argument("--symbol", default="NIFTY")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    s = Settings()
    days = [d for d in History.from_csv(s.history_path).dates
            if date.fromisoformat(a.start) <= d <= date.fromisoformat(a.end)]
    out = s.cache_dir / "bhav"
    t0, ok, bad = time.time(), 0, []

    def one(d):
        try:
            load_symbol_rows(d, out, a.symbol)
            return d, None
        except Exception as exc:  # noqa: BLE001
            return d, str(exc)[:120]

    with ThreadPoolExecutor(a.workers) as ex:
        for d, err in ex.map(one, days):
            if err:
                bad.append((d, err))
            else:
                ok += 1
    print(f"{ok} of {len(days)} days cached in {time.time() - t0:.0f}s at {out}")
    for d, err in bad[:20]:
        print(f"  missing {d}: {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
