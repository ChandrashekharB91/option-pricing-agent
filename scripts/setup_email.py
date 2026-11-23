"""Connect a Gmail account for the daily email, without editing .env by hand.

    python scripts/setup_email.py

Asks for the Gmail address and the 16-character app password (typed invisibly), writes the SMTP lines into the
repo's .env (replacing any earlier email lines, keeping everything else), checks the login with Gmail, and sends
you a short test email. The password goes only into .env, which git ignores; it is never printed.

Create the app password first at https://myaccount.google.com/apppasswords (needs 2-Step Verification).
"""
import getpass
import re
import smtplib
import ssl
import sys
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".env"
KEYS = ("SMTP_HOST", "SMTP_PORT", "SMTP_SECURITY", "SMTP_USER", "SMTP_PASSWORD", "MAIL_FROM", "ADMIN_EMAILS")


def write_env(values: dict) -> None:
    lines = ENV.read_text(encoding="utf-8-sig").splitlines() if ENV.exists() else []
    keep = [ln for ln in lines if ln.split("=", 1)[0].strip().removeprefix("export ").strip() not in values]
    while keep and not keep[-1].strip():
        keep.pop()
    block = ["", "# Daily email (written by scripts/setup_email.py)"] + [f"{k}={v}" for k, v in values.items()]
    ENV.write_text("\n".join(keep + block) + "\n", encoding="utf-8")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(__doc__.split("\n\n")[0].strip() + "\n")
    address = input("Gmail address (e.g. you@gmail.com): ").strip()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", address, re.I):
        print("That does not look like an email address; nothing changed.")
        return 1
    pw = getpass.getpass("App password (16 letters; typing is hidden, spaces are fine): ")
    pw = re.sub(r"\s+", "", pw)
    if len(pw) != 16 or not pw.isalpha():
        print(f"Google app passwords are 16 letters; this has {len(pw)} characters. Nothing changed.\n"
              "Create one at https://myaccount.google.com/apppasswords and run this again.")
        return 1

    print("Checking the login with Gmail ...")
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as s:
            s.starttls(context=ssl.create_default_context())
            s.login(address, pw)
            msg = EmailMessage()
            msg["Subject"], msg["From"], msg["To"] = "Option Pricing Agent: email is connected", address, address
            msg.set_content("Your Option Pricing Agent can now send emails: verification links, password resets "
                            "and the daily NIFTY options feed.")
            s.send_message(msg)
    except smtplib.SMTPAuthenticationError:
        print("Gmail refused the login: the app password is wrong or was deleted. Create a new one at\n"
              "https://myaccount.google.com/apppasswords and run this again. Nothing changed.")
        return 1
    except (smtplib.SMTPException, OSError) as exc:
        print(f"Could not reach Gmail ({exc}). If you are on the office network, port 587 may be blocked.\n"
              "Nothing changed.")
        return 1

    write_env({"SMTP_HOST": "smtp.gmail.com", "SMTP_PORT": "587", "SMTP_SECURITY": "starttls",
               "SMTP_USER": address, "SMTP_PASSWORD": pw, "MAIL_FROM": address, "ADMIN_EMAILS": address})
    print(f"Done. Login works, a test email is on its way to {address}, and the settings are saved in\n  {ENV}\n"
          "Next: restart  python scripts/share.py  (its start-up box should say the daily email is automatic).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
