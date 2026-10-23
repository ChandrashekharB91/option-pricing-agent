"""End-to-end check of the Upstox feed on your machine.

    python scripts/upstox_smoke.py                 # NIFTY, nearest expiry
    python scripts/upstox_smoke.py --underlying BANKNIFTY --expiry 2026-11-23
    python scripts/upstox_smoke.py --save          # also write a snapshot to data/snapshots/

What it checks:
  1. The token in .env is accepted (read-only Analytics Token).
  2. Expiries, index spot, futures and the full option chain download and parse.
  3. A parity-implied forward from near-ATM calls and puts, compared with the future
     of the same expiry (spec tolerance: warn 0.05%, fail 0.20%).
  4. optlib implied vols on the OTM side vs the vols Upstox reports.

The rate is a flat placeholder (default 6.5%) until the MIBOR-OIS curve is built.
The token is read from .env and never printed or saved.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optlib import implied_vol_black76  # noqa: E402
from optlib.data import IST, UpstoxAuthError, UpstoxClient, UpstoxError, UpstoxSource  # noqa: E402
from optlib.data.env import mask  # noqa: E402
from optlib.data.upstox import TOKEN_ENV_VAR  # noqa: E402

FWD_WARN, FWD_FAIL = 0.0005, 0.0020


class RecordingSession:
    """Wraps requests.Session and keeps response bodies (never headers) for --save."""

    def __init__(self):
        import requests

        self._s = requests.Session()
        self.records = []

    def get(self, url, params=None, headers=None, timeout=None):
        resp = self._s.get(url, params=params, headers=headers, timeout=timeout)
        try:
            body = resp.json()
        except ValueError:
            body = None
        self.records.append({"url": url, "params": params, "status": resp.status_code, "body": body})
        return resp


def line(char="-"):
    print(char * 72)


def price_of(q):
    """Mid when two-sided; last traded otherwise (e.g. outside market hours)."""
    return (q.mid, "mid") if q.two_sided else (q.last, "last")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--underlying", default="NIFTY")
    ap.add_argument("--expiry", help="YYYY-MM-DD (default: nearest listed)")
    ap.add_argument("--rate", type=float, default=0.065, help="flat placeholder rate until the curve is built")
    ap.add_argument("--strikes", type=int, default=5, help="strikes each side of ATM in the IV table")
    ap.add_argument("--save", action="store_true", help="write the snapshot to data/snapshots/")
    args = ap.parse_args()

    session = RecordingSession()
    try:
        client = UpstoxClient.from_env([ROOT, Path.cwd()], session=session)
    except KeyError as exc:
        print(f"FAIL: {exc.args[0]}")
        return 1
    print(f"Token loaded from {TOKEN_ENV_VAR}: {mask(client._token)}")  # masked, never the full value
    src = UpstoxSource(client, cache_dir=ROOT / "data" / "cache")

    try:
        symbol, key = src.resolve(args.underlying)
        expiries = src.expiries(symbol)
        if not expiries:
            print(f"FAIL: no listed expiries for {symbol}")
            return 1
        expiry = date.fromisoformat(args.expiry) if args.expiry else expiries[0]
        line("=")
        print(f"{symbol} ({key})   lot size {src.instruments.lot_size(symbol)}")
        print("Next expiries:", ", ".join(f"{d:%a %d %b %Y}" for d in expiries[:6]))

        spot = src.spot(symbol)
        chain = src.option_chain(symbol, expiry)
        futures = src.futures(symbol)
    except UpstoxAuthError as exc:
        print(f"FAIL (token): {exc}")
        return 1
    except UpstoxError as exc:
        print(f"FAIL (Upstox): {exc}")
        return 1

    T = chain.time_to_expiry()
    df = math.exp(-args.rate * T)
    two_sided = sum(q.two_sided for q in chain.quotes)
    line()
    print(f"Snapshot      {chain.as_of:%Y-%m-%d %H:%M:%S} IST")
    print(f"Spot          {spot:,.2f} (quotes API)   {chain.spot:,.2f} (chain)")
    print(f"Expiry        {chain.expiry:%a %d %b %Y} 15:30 IST   T = {T * 365:.3f} days")
    print(f"Chain         {len(chain.strikes)} strikes, {len(chain.quotes)} options, {two_sided} two-sided")
    if two_sided == 0:
        print("              No live bids/asks (market closed?): using last traded prices below.")
    for f in futures:
        print(f"Future        {f.expiry:%d %b %Y}  last {f.last:,.2f}  bid/ask {f.bid:,.2f}/{f.ask:,.2f}  "
              f"OI {f.oi:,.0f}")

    # ---------------------------------------------------- parity-implied forward
    near = sorted(chain.strikes, key=lambda k: abs(k - chain.spot))[:9]
    fwds = []
    for k in near:
        c, p = chain.quote(k, "call"), chain.quote(k, "put")
        if not c or not p:
            continue
        (cp, _), (pp, _) = price_of(c), price_of(p)
        if math.isnan(cp) or math.isnan(pp):
            continue
        fwds.append(k + (cp - pp) / df)
    line()
    status = "OK"
    if len(fwds) >= 3:
        F = statistics.median(fwds)
        print(f"Parity forward {F:,.2f}  (median of {len(fwds)} strikes, range {min(fwds):,.2f} to {max(fwds):,.2f}, "
              f"r = {args.rate:.2%} placeholder)")
        same = [f for f in futures if f.expiry == chain.expiry]
        if same:
            fut_px = same[0].mid if not math.isnan(same[0].mid) else same[0].last
            gap = (F - fut_px) / fut_px
            verdict = "PASS" if abs(gap) <= FWD_WARN else ("WARN" if abs(gap) <= FWD_FAIL else "FAIL")
            status = verdict if verdict != "PASS" else status
            print(f"vs future      {fut_px:,.2f}  gap {gap:+.3%}  -> {verdict} (warn > 0.05%, fail > 0.20%)")
        else:
            print("vs future      no future for this expiry (weekly); check runs on monthly expiries")
    else:
        F = chain.spot * math.exp(args.rate * T)
        print(f"Too few call/put pairs priced; using spot carried at r: F = {F:,.2f}")
        status = "WARN"

    # ---------------------------------------------------- IV comparison
    line()
    atm = min(chain.strikes, key=lambda k: abs(k - F))
    i = chain.strikes.index(atm)
    window = chain.strikes[max(0, i - args.strikes): i + args.strikes + 1]
    print(f"{'Strike':>9} {'Side':>5} {'Price':>9} {'Src':>4} {'optlib IV':>10} {'Upstox IV':>10} {'Diff (pts)':>11}")
    diffs = []
    for k in window:
        side = "call" if k >= F else "put"
        q = chain.quote(k, side)
        if not q:
            continue
        px, how = price_of(q)
        vol = implied_vol_black76(px, F, k, T, args.rate, side) if not math.isnan(px) else math.nan
        ours = vol * 100
        theirs = q.vendor_iv
        diff = ours - theirs
        if not math.isnan(diff):
            diffs.append(diff)
        print(f"{k:>9,.0f} {side:>5} {px:>9,.2f} {how:>4} {ours:>9.2f}% {theirs:>9.2f}% {diff:>+11.2f}")
    if diffs:
        print(f"Mean difference vs Upstox: {statistics.mean(diffs):+.2f} vol pts "
              "(expected: Upstox uses its own forward, rate and expiry time)")

    # ---------------------------------------------------- save
    if args.save:
        out = ROOT / "data" / "snapshots"
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(IST).strftime("%Y%m%dT%H%M%S")
        path = out / f"upstox_{symbol}_{chain.expiry.isoformat()}_{stamp}.json"
        path.write_text(json.dumps({"as_of": chain.as_of.isoformat(), "underlying": symbol,
                                    "expiry": chain.expiry.isoformat(), "responses": session.records},
                                   indent=1, default=str), encoding="utf-8")
        print(f"Saved raw responses (no token) to {path.relative_to(ROOT)}")

    line("=")
    print(f"RESULT: {status}" + ("" if status == "OK" else "  (feed works; see the forward check above)"))
    return 2 if status == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
