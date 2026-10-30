"""Share the app: one command starts everything and prints a public link anyone can open and sign up at.

    python scripts/share.py              # public link through a Cloudflare quick tunnel
    python scripts/share.py --local      # this computer only (http://127.0.0.1:8501), no tunnel

What it does:
  1. gets cloudflared (downloads the official single file from Cloudflare's GitHub releases into tools/ the first
     time; no installer, no administrator rights);
  2. opens a quick tunnel to the dashboard and reads the https://....trycloudflare.com address it is given;
  3. starts the pricing API and the dashboard with that address, so links in verification, password-reset and
     daily emails point to it;
  4. prints the link and keeps everything running until Ctrl+C.

The quick-tunnel address changes every time this starts, and the app is reachable only while this computer and
this window stay on. Email links sent before a restart stop working. For a fixed address, add a domain to
Cloudflare and use a named tunnel (README, "Sharing the app").
"""
import argparse
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RELEASE = "https://github.com/cloudflare/cloudflared/releases/latest/download/"
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
API_PORT, APP_PORT = 8000, 8501


def cloudflared() -> str:
    """Path to cloudflared: on PATH, in tools/, or downloaded into tools/."""
    exe = "cloudflared.exe" if os.name == "nt" else "cloudflared"
    found = shutil.which("cloudflared") or (str(TOOLS / exe) if (TOOLS / exe).exists() else None)
    if found:
        return found
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
    if os.name == "nt":
        asset = f"cloudflared-windows-{arch}.exe"
    elif sys.platform == "darwin":
        sys.exit("On macOS install cloudflared with: brew install cloudflared")
    else:
        asset = f"cloudflared-linux-{arch}"
    TOOLS.mkdir(exist_ok=True)
    print(f"Downloading cloudflared ({asset}) from Cloudflare's GitHub releases ...")
    with urllib.request.urlopen(RELEASE + asset, timeout=120) as r, open(TOOLS / exe, "wb") as f:
        shutil.copyfileobj(r, f)
    if os.name != "nt":
        (TOOLS / exe).chmod(0o755)
    return str(TOOLS / exe)


def port_free(port: int) -> bool:
    import socket

    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_http(url: str, seconds: float = 90) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:  # noqa: BLE001 - not up yet
            time.sleep(1)
    return False


TUNNEL_LOG: list[str] = []


def tunnel_errors(n: int = 12) -> str:
    lines = [x for x in TUNNEL_LOG if " ERR " in x or "error" in x.lower()] or TUNNEL_LOG
    return "\n  ".join(lines[-n:])


def start_tunnel(binary: str, protocol: str = "auto") -> tuple[subprocess.Popen, str]:
    """Start a quick tunnel and return once Cloudflare has registered a connection to this computer."""
    proc = subprocess.Popen([binary, "tunnel", "--no-autoupdate", "--protocol", protocol,
                             "--url", f"http://127.0.0.1:{APP_PORT}"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace")
    found: list[str] = []
    connected = threading.Event()

    def read():
        for line in proc.stdout:  # keep draining so cloudflared never blocks on a full pipe
            TUNNEL_LOG.append(line.rstrip())
            m = URL_RE.search(line)
            if m and not found:
                found.append(m.group(0))
            if "Registered tunnel connection" in line:
                connected.set()

    threading.Thread(target=read, daemon=True).start()
    print("Connecting to Cloudflare ...")
    end = time.time() + 90
    while time.time() < end and not connected.is_set() and proc.poll() is None:
        time.sleep(0.3)
    if not (found and connected.is_set()):
        proc.terminate()
        hint = ("\nIf the messages mention port 7844, QUIC or timeouts, the network blocks Cloudflare's tunnel "
                "port: try\n  python scripts/share.py --protocol http2\nor ask IT to allow outbound TCP/UDP 7844 "
                "to Cloudflare.")
        sys.exit("Cloudflare did not connect the tunnel. cloudflared said:\n  " + tunnel_errors() + hint)
    return proc, found[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="store_true", help="no tunnel: this computer only")
    ap.add_argument("--no-browser", action="store_true", help="do not open the link in the browser")
    ap.add_argument("--protocol", default="auto", choices=["auto", "http2", "quic"],
                    help="tunnel transport; http2 helps on networks that block QUIC")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    for port, what in ((API_PORT, "pricing API"), (APP_PORT, "dashboard")):
        if not port_free(port):
            print(f"Port {port} is in use: an earlier {what} is still running. Close that window (or run\n"
                  f"  Stop-Process -Id (Get-NetTCPConnection -LocalPort {port} -State Listen).OwningProcess\n"
                  f"in PowerShell), then start this again.")
            return 1

    procs: list[subprocess.Popen] = []
    if a.local:
        public = f"http://127.0.0.1:{APP_PORT}"
    else:
        tunnel, public = start_tunnel(cloudflared(), a.protocol)
        procs.append(tunnel)

    env = {**os.environ, "FEED_PUBLIC_URL": public, "FEED_LINKS": "app", "PYTHONIOENCODING": "utf-8"}
    py = sys.executable
    procs.append(subprocess.Popen([py, "-m", "uvicorn", "optlib_app.api:app", "--host", "127.0.0.1",
                                   "--port", str(API_PORT)], cwd=ROOT, env=env))
    if not wait_http(f"http://127.0.0.1:{API_PORT}/health"):
        print("The pricing API did not start; see the messages above.")
        for p in procs:
            p.terminate()
        return 1
    procs.append(subprocess.Popen([py, "-m", "streamlit", "run", str(ROOT / "src" / "optlib_app" / "dashboard.py"),
                                   "--server.headless", "true", "--server.address", "127.0.0.1",
                                   "--server.port", str(APP_PORT)], cwd=ROOT, env=env))
    wait_http(f"http://127.0.0.1:{APP_PORT}", 60)

    bar = "=" * 72
    print(f"\n{bar}\n  Share this link:  {public}\n{bar}")
    print("  People open it, create an account (email + password) and use the app.")
    print("  Keep this window open; Ctrl+C stops the app and the link.")
    if not a.local:
        print("  The link changes each time you start this.")
    print(bar + "\n")
    if not a.no_browser:
        webbrowser.open(public)

    def stop(*_):
        for p in reversed(procs):
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop)
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(2)
    except KeyboardInterrupt:
        stop()
    dead = [p.args[0] if isinstance(p.args, list) else p.args for p in procs if p.poll() is not None]
    print(f"A part of the app stopped ({', '.join(map(str, dead))}); shutting down the rest.")
    if TUNNEL_LOG and procs[0].poll() is not None and not a.local:
        print("cloudflared said:\n  " + tunnel_errors())
    stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
