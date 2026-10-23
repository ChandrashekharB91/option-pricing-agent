"""Small HTTP client for the pricing API, for notebooks and the dashboard.

    from optlib_app.client import PricingClient
    api = PricingClient()                      # http://127.0.0.1:8000 by default
    api.price("2026-10-27", 22500, "call", lots=2)["result"]["price"]
"""
from __future__ import annotations

from typing import Optional

import requests

from .config import Settings


class PricingAPIError(Exception):
    def __init__(self, status: int, detail: str):
        self.status, self.detail = status, detail
        super().__init__(f"HTTP {status}: {detail}")


class PricingClient:
    def __init__(self, base_url: Optional[str] = None, timeout: float = 120.0, session=None):
        self.base_url = (base_url or Settings().api_url).rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def _req(self, method: str, path: str, **kw):
        resp = self.session.request(method, f"{self.base_url}{path}", timeout=self.timeout, **kw)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise PricingAPIError(resp.status_code, str(detail))
        return resp.json()

    def health(self):
        return self._req("GET", "/health")

    def expiries(self, underlying: str = "NIFTY"):
        return self._req("GET", f"/expiries/{underlying}")["expiries"]

    def overview(self, underlying: str = "NIFTY"):
        return self._req("GET", f"/overview/{underlying}")

    def smile(self, expiry: str, underlying: str = "NIFTY"):
        return self._req("GET", f"/smile/{underlying}/{expiry}")

    def price(self, expiry: str, strike: float, option_type: str, lots: float = 1, underlying: str = "NIFTY"):
        return self._req("POST", "/price", json={"underlying": underlying, "expiry": expiry, "strike": strike,
                                                 "option_type": option_type, "lots": lots})

    def book(self, positions: list[dict]):
        return self._req("POST", "/book", json={"positions": positions})

    def scenarios(self, positions: list[dict], spot_shocks=None, vol_shocks=None):
        return self._req("POST", "/scenarios", json={"positions": positions, "spot_shocks": spot_shocks,
                                                     "vol_shocks": vol_shocks})

    def ask(self, question: str, history: Optional[list] = None):
        return self._req("POST", "/agent/ask", json={"question": question, "history": history or []})
