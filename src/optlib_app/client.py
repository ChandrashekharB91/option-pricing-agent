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

    def risk(self, positions: list[dict], window: int = 500):
        return self._req("POST", "/risk", json={"positions": positions, "window": window})

    def eod_days(self):
        return self._req("GET", "/risk/eod_days")["days"]

    def pnl_explain(self, positions: list[dict], day0: str, day1: str):
        return self._req("POST", "/risk/pnl_explain", json={"positions": positions, "day0": day0, "day1": day1})

    def risk_report(self, positions: list[dict], day0: Optional[str] = None, day1: Optional[str] = None) -> str:
        resp = self.session.post(f"{self.base_url}/risk/report", json={"positions": positions, "day0": day0,
                                                                         "day1": day1}, timeout=self.timeout)
        if resp.status_code >= 400:
            raise PricingAPIError(resp.status_code, resp.text[:300])
        return resp.text

    def strategy_meta(self, underlying: str = "NIFTY"):
        return self._req("GET", "/strategy/meta", params={"underlying": underlying})

    def recommend(self, capital: float, view: Optional[dict] = None, expiry: Optional[str] = None,
                  families: Optional[list] = None, risk_budget: float = 0.5, objective: str = "risk_adjusted",
                  top: int = 8, underlying: str = "NIFTY"):
        return self._req("POST", "/strategy/recommend", json={
            "capital": capital, "view": view or {}, "expiry": expiry, "families": families, "risk_budget": risk_budget,
            "objective": objective, "top": top, "underlying": underlying})

    def project_strategy(self, legs: list[dict], lots: int = 1, view: Optional[dict] = None, underlying: str = "NIFTY",
                         name: str = "Custom strategy"):
        return self._req("POST", "/strategy/project", json={"legs": legs, "lots": lots, "view": view or {},
                                                            "underlying": underlying, "name": name})

    def hedge(self, positions: list[dict], budget: float, view: Optional[dict] = None, rank_by: str = "worst_case",
              top: int = 3):
        return self._req("POST", "/strategy/hedge", json={"positions": positions, "budget": budget, "top": top,
                                                          "view": view or {"source": "model"}, "rank_by": rank_by})

    def ask(self, question: str, history: Optional[list] = None):
        return self._req("POST", "/agent/ask", json={"question": question, "history": history or []})
