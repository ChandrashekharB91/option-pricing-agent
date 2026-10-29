"""The pricing service: one object that the API, the agents and the dashboard all call,
so a number is the same wherever a quant sees it.

All maths happens in ``optlib``; this layer fetches data, caches surfaces, and shapes results
into plain JSON-ready dicts.
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import numpy as np

from optlib import black76_price
from optlib.curves import bootstrap_mibor_ois
from optlib.data.fbil import FBILClient
from optlib.data.models import IST
from optlib.data.replay import fbil_from_files, snapshot_source
from optlib.data.upstox import UNDERLYINGS, UpstoxSource
from optlib.pricing import price_option
from optlib.surface import VolSurface, build_surface
from optlib.validator import validate
from optlib.data.bhavcopy import load_bhavcopy, parse_rows
from optlib.data.history import History, load_history
from optlib.pnl_explain import explain
from optlib.charges import Charges
from optlib.margin import MarginParams
from optlib.projector import NotQuoted
from optlib.recommender import (DISCLAIMER, View, find_hedges, log_event, market_state, project_custom,
                                recommend)
from optlib.risk import (Leg, Limits, backtest, check_limits, greeks_by_expiry, historical_scenarios, stress_tests,
                         var_es)

from .config import Settings

DEFAULT_SPOT_SHOCKS = [-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05]
DEFAULT_VOL_SHOCKS = [-0.05, -0.02, 0.0, 0.02, 0.05, 0.10]


class ServiceError(Exception):
    """A request the service cannot fulfil (bad input, missing data); shown to the user as is."""


@dataclass
class Position:
    underlying: str
    expiry: date
    strike: float
    option_type: str
    lots: float  # signed: + long, - short

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        try:
            return cls(underlying=str(d.get("underlying", "NIFTY")).upper(),
                       expiry=_to_date(d["expiry"]), strike=float(d["strike"]),
                       option_type=str(d["option_type"]).lower(), lots=float(d["lots"]))
        except (KeyError, ValueError, TypeError) as exc:
            raise ServiceError(f"bad position {d}: {exc}") from exc


def _to_date(x) -> date:
    return x if isinstance(x, date) else date.fromisoformat(str(x)[:10])


def _clean(x):
    """Make numpy / NaN values JSON-safe."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (date, datetime)):
        return x.isoformat()
    return x


class PricingService:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self._lock = threading.RLock()
        self._source = None
        self._fbil = None
        self._curves: dict = {}
        self._surfaces: dict = {}
        self._history = None
        self._bhav: dict = {}

    # data ---------------------------------------------------------------------
    @property
    def mode(self) -> str:
        return self.settings.data_mode

    def source(self):
        with self._lock:
            if self._source is None:
                s = self.settings
                if s.data_mode == "replay":
                    self._source, self._meta = snapshot_source(s.snapshot_path, s.instruments_path)
                else:
                    self._source = UpstoxSource.from_env(cache_dir=s.cache_dir)
                    self._meta = {}
            return self._source

    def fbil(self):
        with self._lock:
            if self._fbil is None:
                s = self.settings
                self._fbil = (fbil_from_files(s.fbil_ois_path, s.fbil_overnight_path) if s.data_mode == "replay"
                              else FBILClient(cache_dir=s.cache_dir))
            return self._fbil

    def curve(self, pricing_date: date):
        with self._lock:
            if pricing_date not in self._curves:
                quotes = self.fbil().ois_quotes(pricing_date)
                self._curves[pricing_date] = bootstrap_mibor_ois(quotes, pricing_date)
            return self._curves[pricing_date]

    def _recorded_expiries(self) -> list[date]:
        snap = json.loads(self.settings.snapshot_path.read_text(encoding="utf-8"))
        return sorted({date.fromisoformat(r["params"]["expiry_date"]) for r in snap["responses"]
                       if r["url"].endswith("/option/chain")})

    def expiries(self, underlying: str) -> list[date]:
        underlying = self._check_underlying(underlying)
        if self.mode == "replay":
            return self._recorded_expiries() if underlying == "NIFTY" else []
        return self.source().expiries(underlying)

    def now(self) -> datetime:
        if self.mode == "replay":
            self.source()
            return datetime.fromisoformat(self._meta["as_of"])
        return datetime.now(IST)

    def surface(self, underlying: str, max_expiries: int = 4) -> VolSurface:
        underlying = self._check_underlying(underlying)
        key = (underlying, max_expiries)
        with self._lock:
            hit = self._surfaces.get(key)
            fresh = hit and (self.mode == "replay" or time.time() - hit[0] < self.settings.surface_ttl_seconds)
            if fresh:
                return hit[1]
            exps = self.expiries(underlying)[:max_expiries]
            if not exps:
                raise ServiceError(f"no data for {underlying} in {self.mode} mode")
            surf = build_surface(self.source(), self.curve(self.now().date()), underlying, expiries=exps)
            self._surfaces[key] = (time.time(), surf)
            return surf

    @staticmethod
    def _check_underlying(u: str) -> str:
        u = (u or "").strip().upper()
        if u not in UNDERLYINGS:
            raise ServiceError(f"unknown underlying {u!r}; supported: {', '.join(UNDERLYINGS)}")
        return u

    def _smile(self, surf: VolSurface, expiry: date):
        sm = next((s for s in surf.smiles if s.expiry == expiry), None)
        if sm is None:
            raise ServiceError(f"expiry {expiry} not available; available: "
                               f"{', '.join(s.slice.expiry for s in surf.smiles)}")
        return sm

    def lot_size(self, underlying: str) -> int:
        return self.source().instruments.lot_size(self._check_underlying(underlying))

    # views ----------------------------------------------------------------------
    def health(self) -> dict:
        from .config import llm_status

        return {"status": "ok", "data_mode": self.mode, "as_of": self.now().isoformat(),
                "underlyings": list(UNDERLYINGS), "llm": llm_status()}

    def overview(self, underlying: str = "NIFTY") -> dict:
        surf = self.surface(underlying)
        futures = self.source().futures(surf.underlying)
        c = surf.curve
        return _clean({
            "underlying": surf.underlying, "as_of": surf.as_of, "spot": surf.spot, "data_mode": self.mode,
            "lot_size": self.lot_size(surf.underlying), "surface_status": surf.status,
            "curve": {"label": c.label, "age_days": c.age_days(),
                      "pillars": {t: {"maturity": d, "discount_factor": df, "zero_rate": c.zero_rate(c.time(d))}
                                  for t, (d, df) in c.pillars.items()}},
            "futures": [{"expiry": f.expiry, "bid": f.bid, "ask": f.ask, "last": f.last, "oi": f.oi} for f in futures],
            "expiries": [{
                "expiry": s.slice.expiry, "days": s.time_to_expiry * 365, "forward": s.forward,
                "rate": s.slice.rate, "atm_vol": s.slice.atm_vol(),
                "forward_check": None if not s.slice.forward_check else
                {"future": s.slice.forward_check.future, "gap": s.slice.forward_check.gap,
                 "verdict": s.slice.forward_check.verdict},
                "qa_status": s.slice.qa.status, "qa_summary": s.slice.qa.summary(),
                "svi_status": s.fit.status, "svi_rmse_vol_pts": s.fit.rmse * 100,
                "svi_atm_gap_vol_pts": s.fit.atm_gap(s.slice.atm_vol()) * 100,
                "svi_params": s.fit.params.as_dict(),
            } for s in surf.smiles],
            "calendar_violations": [v.__dict__ for v in surf.calendar],
        })

    def smile(self, underlying: str, expiry) -> dict:
        surf = self.surface(underlying)
        sm = self._smile(surf, _to_date(expiry))
        s = sm.slice
        F = s.forward.forward
        lo, hi = sm.fit.fit_range
        k_lo = max(float(s.strikes.min()), F * math.exp(max(lo, -5.0)))
        k_hi = min(float(s.strikes.max()), F * math.exp(min(hi, 5.0)))
        grid = np.linspace(k_lo, k_hi, 200)  # SVI is drawn where it was fitted; beyond, prices use market vols
        return _clean({
            "underlying": surf.underlying, "expiry": s.expiry, "forward": F, "atm_vol": s.atm_vol(),
            "quotes": s.to_records(), "fit_range": [k_lo, k_hi], "excluded": sm.fit.excluded,
            "svi_curve": [{"strike": k, "vol": float(sm.fit.vol(k))} for k in grid],
            "qa_summary": s.qa.summary(),
            "dropped": [{"strike": d.quote.strike, "type": d.quote.option_type, "reasons": list(d.reasons)}
                        for d in s.qa.dropped if d.reasons != ("itm_side",)],
        })

    def _surface_for(self, underlying: str, expiry: date) -> VolSurface:
        """The cached surface, extended to include ``expiry`` if it is listed further out."""
        exps = self.expiries(underlying)
        n = max(4, exps.index(expiry) + 1) if expiry in exps else 4
        return self.surface(underlying, max_expiries=n)

    def price(self, underlying: str, expiry, strike: float, option_type: str, lots: float = 1.0) -> dict:
        surf = self._surface_for(underlying, _to_date(expiry))
        lot = self.lot_size(surf.underlying)
        try:
            res = price_option(surf, float(strike), _to_date(expiry), option_type, quantity=float(lots) * lot)
        except ValueError as exc:
            raise ServiceError(str(exc)) from exc
        rep = validate(res, surf)
        return _clean({"result": res.to_dict(), "lots": lots, "lot_size": lot, "validation": rep.to_dict(),
                       "validation_summary": rep.summary()})

    def book(self, positions: list[dict]) -> dict:
        if not positions:
            raise ServiceError("the book is empty")
        rows, total = [], {"value": 0.0, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        statuses = []
        for p in map(Position.from_dict, positions):
            out = self.price(p.underlying, p.expiry, p.strike, p.option_type, p.lots)
            r = out["result"]
            pg = r["position_greeks"]
            rows.append({"underlying": p.underlying, "expiry": p.expiry.isoformat(), "strike": p.strike,
                         "option_type": p.option_type, "lots": p.lots, "quantity": r["quantity"],
                         "price": r["price"], "vol": r["vol"], "vol_source": r["vol_source"], "value": r["value"],
                         **{g: pg[g] for g in ("delta", "gamma", "vega", "theta", "rho")},
                         "validation": out["validation"]["status"]})
            statuses.append(out["validation"]["status"])
            total["value"] += r["value"]
            for g in ("delta", "gamma", "vega", "theta", "rho"):
                total[g] += pg[g]
        status = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")
        return _clean({"positions": rows, "totals": total, "status": status,
                       "units": {"value": "Rs", "delta": "index units (vs forward)", "gamma": "per index point",
                                 "vega": "Rs per vol point", "theta": "Rs per calendar day", "rho": "Rs per bp"}})

    def scenarios(self, positions: list[dict], spot_shocks: Optional[list[float]] = None,
                  vol_shocks: Optional[list[float]] = None) -> dict:
        """Full revaluation P&L of the book on a spot x vol grid (sticky strike, time held fixed)."""
        spot_shocks = spot_shocks or DEFAULT_SPOT_SHOCKS
        vol_shocks = vol_shocks or DEFAULT_VOL_SHOCKS
        legs = []
        for p in map(Position.from_dict, positions):
            r = self.price(p.underlying, p.expiry, p.strike, p.option_type, p.lots)["result"]
            legs.append(r)
        if not legs:
            raise ServiceError("the book is empty")
        grid = np.zeros((len(vol_shocks), len(spot_shocks)))
        for i, dv in enumerate(vol_shocks):
            for j, ds in enumerate(spot_shocks):
                pnl = 0.0
                for r in legs:
                    new = black76_price(r["forward"] * (1 + ds), r["strike"], r["time_to_expiry"], r["rate"],
                                        max(r["vol"] + dv, 0.005), r["option_type"])
                    pnl += (float(new) - r["price"]) * r["quantity"]
                grid[i, j] = pnl
        return _clean({"spot_shocks": spot_shocks, "vol_shocks": vol_shocks, "pnl": grid.tolist(),
                       "method": "full revaluation, sticky strike, time held fixed; P&L in Rs",
                       "worst": float(grid.min()), "best": float(grid.max())})

    # ------------------------------------------------------------------ risk (phase 2)
    def history(self) -> History:
        with self._lock:
            if self._history is None:
                s = self.settings
                self._history = (History.from_csv(s.history_path) if self.mode == "replay"
                                 else load_history(s.cache_dir, end=datetime.now(IST).date()))
            return self._history

    def eod_days(self) -> list[str]:
        """Trading days with end-of-day F&O data on hand (replay) for P&L explain."""
        return sorted(self.settings.bhavcopy_files) if self.mode == "replay" else []

    def bhavcopy(self, day: date):
        key = day.isoformat()
        with self._lock:
            if key not in self._bhav:
                if self.mode == "replay":
                    path = self.settings.bhavcopy_files.get(key)
                    if path is None:
                        raise ServiceError(f"replay data has end-of-day files for {', '.join(self.eod_days())} only")
                    self._bhav[key] = parse_rows(path.read_text(encoding="utf-8"))
                else:
                    try:
                        self._bhav[key] = load_bhavcopy(day, self.settings.cache_dir)
                    except RuntimeError as exc:
                        raise ServiceError(str(exc)) from exc
            return self._bhav[key]

    def _results(self, positions: list[dict]) -> list[dict]:
        out = []
        for p in map(Position.from_dict, positions):
            out.append(self.price(p.underlying, p.expiry, p.strike, p.option_type, p.lots)["result"])
        return out

    def risk(self, positions: list[dict], window: int = 500, var_level: float = 0.99, es_level: float = 0.975,
             test_days: int = 250) -> dict:
        """Book Greeks by expiry, VaR/ES with contributions, backtest, stress tests and limit utilisation."""
        book = self.book(positions)
        legs = [Leg.from_result(r) for r in self._results(positions)]
        hist = self.history().before(self.now().date())  # no look-ahead: history up to the previous close
        try:
            scen = historical_scenarios(hist, [l.underlying for l in legs], window)
        except (ValueError, KeyError) as exc:
            raise ServiceError(f"history: {exc}") from exc
        v = var_es(legs, scen, var_level, es_level)
        try:
            bt = backtest(legs, hist, window, test_days, var_level).to_dict()
        except ValueError as exc:
            bt = {"error": str(exc)}
        st = stress_tests(legs, hist)
        limits = check_limits(book["totals"], v.var, min(r["pnl"] for r in st), Limits.load(self.settings.limits_path))
        states = [book["status"]] + [l["status"] for l in limits]
        light = bt.get("traffic_light")
        states.append({"green": "PASS", "amber": "WARN", "red": "FAIL"}.get(light, "WARN"))
        status = "FAIL" if "FAIL" in states else ("WARN" if "WARN" in states else "PASS")
        return _clean({"as_of": self.now(), "data_mode": self.mode, "status": status, "book": book,
                       "by_expiry": greeks_by_expiry(book["positions"]), "var": v.to_dict(), "backtest": bt,
                       "stress": st, "limits": limits, "history_last_date": hist.dates[-1]})

    def pnl_explain(self, positions: list[dict], day0, day1) -> dict:
        d0, d1 = _to_date(day0), _to_date(day1)
        try:
            e = explain(positions, self.bhavcopy(d0), self.bhavcopy(d1), self.curve(d0), self.curve(d1))
        except ValueError as exc:
            raise ServiceError(str(exc)) from exc
        return _clean(e.to_dict())

    def risk_report(self, positions: list[dict], day0=None, day1=None) -> str:
        from .report import render_risk_report

        risk = self.risk(positions)
        pnl = None
        if day0 and day1:
            try:
                pnl = self.pnl_explain(positions, day0, day1)
            except ServiceError as exc:
                pnl = {"error": str(exc)}
        return render_risk_report(risk, pnl)

    # ------------------------------------------------------------------ strategies (phase 3)
    def market_state(self, underlying: str = "NIFTY", expiry=None):
        """Forward, smile, live bid/ask, futures and realised vol for one expiry, as the projector needs them."""
        underlying = self._check_underlying(underlying)
        exps = self.expiries(underlying)
        exp = _to_date(expiry) if expiry else (exps[0] if exps else None)
        if exp is None:
            raise ServiceError(f"no expiries for {underlying} in {self.mode} mode")
        surf = self._surface_for(underlying, exp)
        sm = self._smile(surf, exp)
        try:
            hist = self.history()
        except (OSError, ValueError, KeyError):
            hist = None
        chain = self.source().option_chain(underlying, exp)
        return market_state(surf, sm, chain, self.source().futures(underlying), self.lot_size(underlying), hist)

    def _costs(self):
        return Charges.load(self.settings.charges_path), MarginParams.load(self.settings.margin_path)

    @staticmethod
    def _view(view: Optional[dict]) -> View:
        v = dict(view or {})
        if v.get("horizon"):
            v["horizon"] = _to_date(v["horizon"])
        try:
            return View(**{k: v[k] for k in ("source", "target", "horizon", "vol", "iv_shift") if v.get(k) is not None})
        except (TypeError, ValueError) as exc:
            raise ServiceError(f"view: {exc}") from exc

    def recommend(self, capital: float, view: Optional[dict] = None, underlying: str = "NIFTY", expiry=None,
                  families: Optional[list] = None, risk_budget: float = 0.5, objective: str = "risk_adjusted",
                  top: int = 8, min_prob_profit: float = 0.5) -> dict:
        state = self.market_state(underlying, expiry)
        v = self._view(view)
        if v.horizon and not (state.as_of.date() <= v.horizon <= state.expiry):
            raise ServiceError(f"horizon must be between {state.as_of.date()} and the expiry {state.expiry}")
        charges, margin = self._costs()
        try:
            out = recommend(state, float(capital), v, charges, margin, families, risk_budget, objective, top,
                            min_prob_profit=float(min_prob_profit))
        except ValueError as exc:
            raise ServiceError(str(exc)) from exc
        out["data_mode"] = self.mode
        self._audit("recommend", {"capital": capital, "view": view, "underlying": underlying, "expiry": str(expiry),
                                  "families": families, "risk_budget": risk_budget, "objective": objective,
                                  "min_prob_profit": min_prob_profit}, out)
        return _clean(out)

    def project_strategy(self, legs: list[dict], lots: int = 1, view: Optional[dict] = None,
                         underlying: str = "NIFTY", name: str = "Custom strategy") -> dict:
        if not legs:
            raise ServiceError("add at least one leg")
        exp = legs[0].get("expiry")
        state = self.market_state(underlying, exp)
        charges, margin = self._costs()
        try:
            return _clean(project_custom(state, legs, int(lots), self._view(view), charges, margin, name))
        except (ValueError, KeyError) as exc:
            raise ServiceError(str(exc)) from exc

    def hedge(self, positions: list[dict], budget: float, view: Optional[dict] = None, underlying: str = "NIFTY",
              top: int = 3, rank_by: str = "worst_case") -> dict:
        if not positions:
            raise ServiceError("the book is empty")
        state = self.market_state(underlying, positions[0].get("expiry"))
        charges, margin = self._costs()
        try:
            out = find_hedges(state, positions, self._view(view or {"source": "model"}), charges, margin,
                              float(budget), top, rank_by)
        except (ValueError, NotQuoted) as exc:
            raise ServiceError(str(exc)) from exc
        self._audit("hedge", {"positions": positions, "budget": budget, "view": view, "rank_by": rank_by}, out)
        return _clean(out)

    def _audit(self, kind: str, request: dict, result: dict) -> None:
        try:
            log_event(self.settings.audit_log_path, kind, request, result)
        except OSError:
            pass  # the audit log must never break a request
