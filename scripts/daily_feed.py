"""Daily NIFTY options feed: build today's numbers, preview them, and email them to confirmed subscribers.

    python scripts/daily_feed.py --preview                 # writes reports/feed/feed_YYYYMMDD.html; sends nothing
    python scripts/daily_feed.py --test you@example.com    # sends one copy marked [TEST] to that address
    python scripts/daily_feed.py --send                    # sends to every active subscriber (once per day)
    python scripts/daily_feed.py --subscribers             # lists subscribers (no tokens)
    python scripts/daily_feed.py --add a@b.com --name "A B"  # signs someone up (they still confirm by email)

Live data needs DATA_MODE=live and the Upstox token in .env; quotes are only live in market hours, so schedule
--send for about 15:20 IST on trading days (Windows Task Scheduler: see README, "Daily email feed").
Email settings (SMTP_HOST etc.) are described in src/optlib_app/mailer.py. Without SMTP_HOST every email is saved
to data/feed/outbox as an .eml file instead of being sent.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optlib_app.feed_jobs import FeedJobs  # noqa: E402
from optlib_app.service import PricingService  # noqa: E402
from optlib_app.subscribers import SubscriptionError  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preview", action="store_true", help="write the HTML email to reports/feed/ and stop")
    g.add_argument("--test", nargs="+", metavar="EMAIL", help="send a [TEST] copy to these addresses only")
    g.add_argument("--send", action="store_true", help="send to all active subscribers")
    g.add_argument("--subscribers", action="store_true", help="list subscribers")
    g.add_argument("--add", metavar="EMAIL", help="sign up an address (a confirmation email goes out)")
    ap.add_argument("--name", default="", help="with --add: the reader's name")
    ap.add_argument("--force", action="store_true", help="send even if today's feed already went out, or on a weekend")
    ap.add_argument("--underlying", default="NIFTY")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles and Task Scheduler logs are not UTF-8 by default
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    svc = PricingService()
    jobs = FeedJobs(svc)
    print(f"Data: {svc.mode} · mail: {jobs.mail.describe()}")

    if a.subscribers:
        subs = jobs.store.all()
        for s in subs:
            print(f"  {s.status:<13} {s.email:<40} {s.name:<24} since {s.confirmed_at or s.created_at}")
        print(f"{len(subs)} total: {jobs.store.counts()}")
        return 0
    if a.add:
        try:
            print(jobs.subscribe(a.add, a.name, consent=True, source="script")["message"])
        except SubscriptionError as exc:
            print(f"Not added: {exc}")
            return 1
        return 0

    now = datetime.now(IST)
    if a.send and svc.mode == "live" and now.weekday() >= 5 and not a.force:
        print("Weekend: no trading today, nothing sent (use --force to override).")
        return 0

    feed = jobs.build(a.underlying)
    print(f"Feed for {feed['date']} (snapshot {feed['as_of'][11:16]} IST): {feed['underlying']} {feed['spot']:,.2f}, "
          f"surface {feed['surface_status']}")
    for h in feed["highlights"]:
        print(f"  - {h}")

    if a.preview:
        out = ROOT / "reports" / "feed" / f"feed_{feed['date'].replace('-', '')}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(jobs.preview_html(feed), encoding="utf-8")
        print(f"Preview: {out}  (open it in a browser)")
        return 0

    try:
        res = jobs.send(to=a.test, force=a.force, feed=feed)
    except SubscriptionError as exc:
        print(f"Not sent: {exc}")
        return 1
    print(res["message"])
    for f in res.get("files", []):
        print(f"  saved {f}")
    for e in res.get("errors", []):
        print(f"  FAILED {e['to']}: {e['error']}")
    return 1 if res.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
