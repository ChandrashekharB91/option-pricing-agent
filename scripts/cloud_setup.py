"""One-time move to the cloud: run on the computer that has your .env.

    python scripts/cloud_setup.py

1. checks CLOUDFLARE_API_TOKEN (in .env) reaches the D1 database "option-pricing-agent";
2. copies the local subscriber list (data/feed/subscribers.db: accounts, subscriptions) into D1, skipping
   addresses already there;
3. writes cloud_secrets.toml (git-ignored): paste it into Streamlit Community Cloud > your app > Settings >
   Secrets;
4. sets the GitHub Actions secrets and variables with the GitHub CLI if it is installed and logged in; otherwise
   prints exactly what to add at GitHub > Settings > Secrets and variables > Actions.

Nothing secret is printed.
"""
import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optlib.data.env import read_env_file  # noqa: E402
from optlib_app.d1 import D1Error, connect_from_settings  # noqa: E402
from optlib_app.subscribers import SubscriberStore  # noqa: E402

LOCAL_ONLY = {"REPLAY_SNAPSHOT", "PRICING_API_URL", "FEED_PUBLIC_URL", "FEED_LINKS", "FEED_AUTO_SEND",
              "FEED_SEND_TIME", "EMBEDDED_API"}
GH_SECRETS = ("UPSTOX_ANALYTICS_TOKEN", "SMTP_PASSWORD", "CLOUDFLARE_API_TOKEN")


def migrate(local_db: Path, store: SubscriberStore) -> tuple[int, int]:
    if not local_db.is_file():
        return 0, 0
    src = sqlite3.connect(local_db)
    src.row_factory = sqlite3.Row
    added = skipped = 0
    for r in src.execute("SELECT * FROM subscribers"):
        row = dict(r)
        cols = [c for c in ("email", "name", "status", "token", "created_at", "confirmed_at", "unsubscribed_at",
                            "consent", "source", "password_hash", "reset_token", "reset_expires") if c in row]
        with store._db() as db:
            cur = db.execute(f"INSERT OR IGNORE INTO subscribers ({', '.join(cols)}) VALUES "
                             f"({', '.join('?' for _ in cols)})", [row[c] for c in cols])
        added += cur.rowcount
        skipped += 1 - cur.rowcount
    src.close()
    return added, skipped


def toml_value(v: str) -> str:
    return json.dumps(v)  # a JSON string is a valid TOML basic string


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app-url", default="https://nifty-options-agent.streamlit.app",
                    help="the address you give the app on Streamlit Community Cloud")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    env = read_env_file(ROOT / ".env")
    get = lambda k, default="": env.get(k, "") or default  # noqa: E731

    print("1. Cloudflare D1")
    if not env.get("CLOUDFLARE_API_TOKEN"):
        print("   CLOUDFLARE_API_TOKEN is not in .env. Create one at https://dash.cloudflare.com/profile/api-tokens\n"
              "   (Create Token > Custom token > Permissions: Account / D1 / Edit), add the line\n"
              "   CLOUDFLARE_API_TOKEN=... to .env and run this again.")
        return 1
    try:
        conn = connect_from_settings(get)
        store = SubscriberStore(ROOT / "data" / "feed" / "unused.db", remote=conn)
    except D1Error as exc:
        print(f"   {exc}")
        return 1
    print(f"   connected: {store.counts()}")

    print("2. Subscriber list")
    added, skipped = migrate(ROOT / "data" / "feed" / "subscribers.db", store)
    print(f"   copied {added} from this computer ({skipped} already in D1); now {store.counts()}")

    print("3. Streamlit Community Cloud secrets")
    values = {k: v for k, v in env.items() if v and k not in LOCAL_ONLY}
    values.update({"DATA_MODE": "live", "FEED_LINKS": "app", "FEED_PUBLIC_URL": a.app_url.rstrip("/"),
                   "FEED_AUTO_SEND": "off",  # GitHub Actions sends the daily email
                   "D1_DATABASE_NAME": env.get("D1_DATABASE_NAME", "option-pricing-agent")})
    out = ROOT / "cloud_secrets.toml"
    out.write_text("# Paste into Streamlit Community Cloud > app > Settings > Secrets. Do not commit.\n"
                   + "".join(f"{k} = {toml_value(v)}\n" for k, v in sorted(values.items())), encoding="utf-8")
    print(f"   wrote {out} ({len(values)} settings)")

    print("4. GitHub Actions (the 09:30 daily email)")
    variables = {"SMTP_USER": env.get("SMTP_USER", ""), "MAIL_FROM": env.get("MAIL_FROM", "") or env.get("SMTP_USER", ""),
                 "FEED_PUBLIC_URL": a.app_url.rstrip("/"), "FEED_SEND_TIME": env.get("FEED_SEND_TIME", "09:30")}
    missing = [k for k in GH_SECRETS if not env.get(k)]
    if missing:
        print(f"   missing in .env: {', '.join(missing)}")
    gh = shutil.which("gh")
    if gh and subprocess.run([gh, "auth", "status"], capture_output=True).returncode == 0:
        for k in GH_SECRETS:
            if env.get(k):
                subprocess.run([gh, "secret", "set", k, "--body", env[k]], cwd=ROOT, check=True, capture_output=True)
                print(f"   secret   {k}: set")
        for k, v in variables.items():
            if v:
                subprocess.run([gh, "variable", "set", k, "--body", v], cwd=ROOT, check=True, capture_output=True)
                print(f"   variable {k}: {v}")
    else:
        print("   GitHub CLI not available. On github.com: your repo > Settings > Secrets and variables > Actions:")
        print("   Secrets tab, New repository secret, one each (values are in your .env):")
        for k in GH_SECRETS:
            print(f"     {k}")
        print("   Variables tab, New repository variable:")
        for k, v in variables.items():
            print(f"     {k} = {v}")
    print("\nDone. Next: deploy the app on Streamlit Community Cloud (README, 'Running in the cloud').")
    return 0


if __name__ == "__main__":
    sys.exit(main())
