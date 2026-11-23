"""Daily NIFTY options feed: build today's numbers, preview them, and email them to confirmed subscribers.

    python scripts/daily_feed.py                           # same as --preview
    python scripts/daily_feed.py --preview                 # writes reports/feed/feed_YYYYMMDD.html, opens it; sends nothing
    python scripts/daily_feed.py --signup                  # opens the reader website: sign up, log in (the API must be running)
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
import webbrowser
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optlib_app.feed_jobs import FeedJobs  # noqa: E402
from optlib_app.service import PricingService  # noqa: E402
from optlib_app.subscribers import SubscriptionError  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")


def open_signup(base: str) -> int:
    """Open the sign-up page served by the API; explain how to start the API if it is not running."""
    import requests

    url = f"{base}/"
    try:
        requests.get(f"{base}/health", timeout=3).raise_for_status()
    except requests.RequestException:
        print(f"The API is not running at {base}. Start it in another terminal with\n"
              f"    python -m uvicorn optlib_app.api:app\nthen run this again (or open {url} yourself).")
        return 1
    print(f"Opening {url}")
    webbrowser.open(url)
    return 0


def explain_outbox() -> None:
    """Say why email is off, by .env line number, without showing any value."""
    from optlib.data.env import env_key_lines

    env = ROOT / ".env"
    keys = ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "MAIL_FROM")
    if not env.is_file():
        print(f"  Email is off: there is no .env file at {env}")
        return
    lines = env_key_lines(env, keys)
    print(f"  Email is off. In {env}:")
    for k in keys:
        where = lines[k]
        if not where:
            print(f"    {k:<14} missing")
        else:
            print(f"    {k:<14} " + ", ".join(f"line {n} {'(set)' if ok else '(EMPTY)'}" for n, ok in where))
    others = [p.name for p in ROOT.glob(".env*") if p.name not in (".env", ".env.example")]
    if others:
        print(f"    Other files that look like .env: {', '.join(others)} (only '.env' is read; Notepad can save "
              f"'.env.txt' by mistake)")


def nse_holiday(day) -> str:
    """Name of today's NSE F&O trading holiday from Upstox's public calendar, or '' (also '' if unreachable)."""
    import requests

    try:
        rows = requests.get(f"https://api.upstox.com/v2/market/holidays/{day.isoformat()}", timeout=20).json()["data"]
    except Exception:  # noqa: BLE001 - if the calendar is down, try to send; quotes will tell
        return ""
    for h in rows or []:
        if h.get("holiday_type") == "TRADING_HOLIDAY" and "NFO" in (h.get("closed_exchanges") or []):
            return h.get("description") or "trading holiday"
    return ""


def wait_until(hhmm: str, max_wait_min: int = 45) -> None:
    import time

    hh, mm = (int(x) for x in hhmm.split(":"))
    now = datetime.now(IST)
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    secs = (target - now).total_seconds()
    if 0 < secs <= max_wait_min * 60:
        print(f"Waiting until {hhmm} IST ({secs / 60:.1f} min) ...", flush=True)
        time.sleep(secs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--preview", action="store_true", help="write the HTML email to reports/feed/, open it, and stop "
                                                          "(the default)")
    g.add_argument("--signup", action="store_true", help="open the sign-up page in the browser (API must be running)")
    g.add_argument("--test", nargs="+", metavar="EMAIL", help="send a [TEST] copy to these addresses only")
    g.add_argument("--send", action="store_true", help="send to all active subscribers")
    g.add_argument("--subscribers", action="store_true", help="list subscribers")
    g.add_argument("--add", metavar="EMAIL", help="sign up an address (a confirmation email goes out)")
    ap.add_argument("--name", default="", help="with --add: the reader's name")
    ap.add_argument("--force", action="store_true",
                    help="send even if today's feed already went out, or on a weekend or NSE holiday")
    ap.add_argument("--at", metavar="HH:MM", help="wait until this IST time before fetching quotes (the cloud "
                                                  "scheduler can start a few minutes early)")
    ap.add_argument("--underlying", default="NIFTY")
    ap.add_argument("--no-open", action="store_true", help="with --preview: do not open the browser")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles and Task Scheduler logs are not UTF-8 by default
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not (a.test or a.send or a.subscribers or a.add or a.signup):
        a.preview = True
    svc = PricingService()
    jobs = FeedJobs(svc)
    if a.signup:
        return open_signup(jobs.public_url)
    print(f"Data: {svc.mode} · mail: {jobs.mail.describe()}")
    print(f"Subscriber list: {jobs.store.where}")
    if jobs.mail.mode == "outbox":
        explain_outbox()

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
    if a.send and svc.mode == "live" and not a.force:
        if now.weekday() >= 5:
            print("Weekend: no trading today, nothing sent (use --force to override).")
            return 0
        holiday = nse_holiday(now.date())
        if holiday:
            print(f"NSE holiday today ({holiday}): nothing sent (use --force to override).")
            return 0
    if a.at:
        wait_until(a.at)

    feed = jobs.build(a.underlying)
    print(f"Feed for {feed['date']} (snapshot {feed['as_of'][11:16]} IST): {feed['underlying']} {feed['spot']:,.2f}, "
          f"surface {feed['surface_status']}")
    for h in feed["highlights"]:
        print(f"  - {h}")

    if a.preview:
        out = ROOT / "reports" / "feed" / f"feed_{feed['date'].replace('-', '')}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(jobs.preview_html(feed), encoding="utf-8")
        print(f"Preview: {out}")
        if not a.no_open:
            webbrowser.open(out.as_uri())
        print("To subscribe readers: python scripts/daily_feed.py --signup  (or the dashboard's Daily feed tab)")
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
