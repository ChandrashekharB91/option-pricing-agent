"""End-to-end check of the live pricing inputs on your machine.

    python scripts/upstox_smoke.py                 # NIFTY, nearest expiry
    python scripts/upstox_smoke.py --underlying BANKNIFTY --expiry 2026-11-23
    python scripts/upstox_smoke.py --save          # also write a snapshot to data/snapshots/

What it runs:
  1. Upstox: token check, expiries, index spot, futures, full option chain.
  2. FBIL: latest MIBOR-OIS fixing -> discount curve (falls back to --rate if FBIL is unreachable).
  3. Parity-implied forward, checked against the same-expiry future (warn 0.05%, fail 0.20%).
  4. Chain QA: which quotes are kept for the smile and why others are dropped.
  5. The smile: optlib implied vols (bid/mid/ask) next to the vols Upstox reports.

The Upstox token is read from .env and never printed or saved.
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

from optlib.curves import bootstrap_mibor_ois, flat_curve  # noqa: E402
from optlib.data import IST, UpstoxAuthError, UpstoxClient, UpstoxError, UpstoxSource  # noqa: E402
from optlib.data.env import mask  # noqa: E402
from optlib.data.fbil import FBILClient  # noqa: E402
from optlib.data.upstox import TOKEN_ENV_VAR  # noqa: E402
from optlib.slices import build_vol_slice  # noqa: E402


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
    print(char * 78)


def load_curve(pricing_date: date, fallback_rate: float, fbil: FBILClient | None = None):
    try:
        quotes = (fbil or FBILClient(cache_dir=ROOT / "data" / "cache")).ois_quotes(pricing_date)
        curve = bootstrap_mibor_ois(quotes, pricing_date)
        pts = ", ".join(f"{t} {r:.2f}%" for t, r in list(quotes.tenors.items())[:6])
        return curve, f"FBIL MIBOR-OIS fixed {quotes.fixing_date:%d %b %Y} ({curve.age_days()} days old): O/N " \
                      f"{quotes.overnight:.2f}%, {pts}"
    except Exception as exc:  # FBIL down or blocked: keep going with a flat rate, clearly labelled
        return flat_curve(pricing_date, fallback_rate), f"FBIL unavailable ({type(exc).__name__}); " \
                                                       f"FLAT {fallback_rate:.2%} placeholder"


def main(argv=None, fbil: FBILClient | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--underlying", default="NIFTY")
    ap.add_argument("--expiry", help="YYYY-MM-DD (default: nearest listed)")
    ap.add_argument("--rate", type=float, default=0.06, help="fallback flat rate if FBIL is unreachable")
    ap.add_argument("--strikes", type=int, default=6, help="strikes each side of ATM in the smile table")
    ap.add_argument("--save", action="store_true", help="write the raw responses to data/snapshots/")
    args = ap.parse_args(argv)

    session = RecordingSession()
    try:
        client = UpstoxClient.from_env([ROOT, Path.cwd()], session=session)
    except KeyError as exc:
        print(f"FAIL: {exc.args[0]}")
        return 1
    print(f"Token loaded from {TOKEN_ENV_VAR}: {mask(client._token)}")
    src = UpstoxSource(client, cache_dir=ROOT / "data" / "cache")

    try:
        symbol, key = src.resolve(args.underlying)
        expiries = src.expiries(symbol)
        if not expiries:
            print(f"FAIL: no listed expiries for {symbol}")
            return 1
        expiry = date.fromisoformat(args.expiry) if args.expiry else expiries[0]
        spot = src.spot(symbol)
        chain = src.option_chain(symbol, expiry)
        futures = src.futures(symbol)
    except UpstoxAuthError as exc:
        print(f"FAIL (token): {exc}")
        return 1
    except UpstoxError as exc:
        print(f"FAIL (Upstox): {exc}")
        return 1

    curve, curve_note = load_curve(chain.as_of.date(), args.rate, fbil)
    same = [f for f in futures if f.expiry == chain.expiry]
    fut_px = None
    if same:
        fut_px = same[0].mid if not math.isnan(same[0].mid) else same[0].last

    line("=")
    print(f"{symbol} ({key})   lot size {src.instruments.lot_size(symbol)}")
    print("Next expiries:", ", ".join(f"{d:%a %d %b}" for d in expiries[:6]))
    line()
    print(f"Snapshot      {chain.as_of:%Y-%m-%d %H:%M:%S} IST")
    print(f"Spot          {spot:,.2f}")
    print(f"Expiry        {chain.expiry:%a %d %b %Y} 15:30 IST   T = {chain.time_to_expiry() * 365:.3f} days")
    print(f"Chain         {len(chain.strikes)} strikes, {len(chain.quotes)} options, "
          f"{sum(q.two_sided for q in chain.quotes)} with live bid and ask")
    for f in futures:
        print(f"Future        {f.expiry:%d %b %Y}  bid/ask {f.bid:,.2f}/{f.ask:,.2f}  last {f.last:,.2f}")
    print(f"Curve         {curve_note}")

    try:
        s = build_vol_slice(chain, curve, future_price=fut_px)
    except ValueError as exc:
        print(f"FAIL: {exc} (market closed? bids and asks are needed for the forward)")
        return 2

    # ---------------------------------------------------- forward
    line()
    fw = s.forward
    print(f"Rate          {s.rate:.3%} continuous to expiry   discount factor {s.discount_factor:.6f}")
    print(f"Forward       {fw.forward:,.2f}  from {fw.n_strikes} strikes {min(fw.strikes):,.0f}-{max(fw.strikes):,.0f}, "
          f"spread of estimates {fw.dispersion:.2f} pts")
    naive = chain.spot * math.exp(s.rate * s.time_to_expiry)
    print(f"Spot x carry  {naive:,.2f}  ({(naive - fw.forward) / fw.forward:+.3%}; implied carry "
          f"{fw.implied_carry:.1%} p.a. -> pricing uses the parity forward)")
    if s.forward_check:
        c = s.forward_check
        print(f"vs future     {c.future:,.2f}  gap {c.gap:+.3%} -> {c.verdict} (warn > 0.05%, fail > 0.20%)")
    else:
        print("vs future     no future for this expiry (weekly); checked on monthly expiries")

    # ---------------------------------------------------- QA
    line()
    print(s.qa.summary())

    # ---------------------------------------------------- smile
    line()
    print(f"ATM vol {s.atm_vol():.2%}")
    print(f"{'Strike':>8} {'Side':>5} {'Bid':>8} {'Ask':>8} {'IV bid':>8} {'IV mid':>8} {'IV ask':>8} "
          f"{'Upstox':>8} {'Diff':>6}")
    atm_i = int(min(range(len(s.strikes)), key=lambda i: abs(s.strikes[i] - fw.forward)))
    diffs = []
    for i in range(max(0, atm_i - args.strikes), min(len(s.strikes), atm_i + args.strikes + 1)):
        q = chain.quote(float(s.strikes[i]), str(s.option_types[i]))
        vendor = q.vendor_iv if q else math.nan
        diff = s.iv_mid[i] * 100 - vendor
        if not math.isnan(diff):
            diffs.append(diff)
        print(f"{s.strikes[i]:>8,.0f} {s.option_types[i]:>5} {s.bid[i]:>8.2f} {s.ask[i]:>8.2f} "
              f"{s.iv_bid[i]:>8.2%} {s.iv_mid[i]:>8.2%} {s.iv_ask[i]:>8.2%} {vendor:>7.2f}% {diff:>+6.2f}")
    if diffs:
        print(f"Mean optlib - Upstox: {statistics.mean(diffs):+.2f} vol pts (Upstox uses its own forward and timing)")

    if args.save:
        out = ROOT / "data" / "snapshots"
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(IST).strftime("%Y%m%dT%H%M%S")
        path = out / f"upstox_{symbol}_{chain.expiry.isoformat()}_{stamp}.json"
        path.write_text(json.dumps({"as_of": chain.as_of.isoformat(), "underlying": symbol,
                                    "expiry": chain.expiry.isoformat(), "responses": session.records},
                                   indent=1, default=str), encoding="utf-8")
        print(f"Saved raw responses (no token) to {path.relative_to(ROOT)}")

    statuses = [s.qa.status] + ([s.forward_check.verdict] if s.forward_check else [])
    result = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")
    line("=")
    print(f"RESULT: {result}")
    return 2 if result == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
