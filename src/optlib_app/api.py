"""FastAPI backend: every pricing tool as a typed endpoint.

Run:  uvicorn optlib_app.api:app --reload        (then open http://127.0.0.1:8000/docs)

The dashboard, notebooks (via optlib_app.client) and the agents all go through the same
PricingService, so a number is identical wherever it is shown.
"""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from optlib.data.upstox import UpstoxError

from .service import PricingService, ServiceError

app = FastAPI(title="Option Pricing Agent API", version="0.1.0",
              description="In-house option pricing: market data, smiles, prices, Greeks, validation, book risk.")

_service: Optional[PricingService] = None


def get_service() -> PricingService:
    global _service
    if _service is None:
        _service = PricingService()
    return _service


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UpstoxError as exc:
        raise HTTPException(status_code=502, detail=f"market data: {exc}") from exc
    except KeyError as exc:  # missing credentials
        raise HTTPException(status_code=503, detail=str(exc.args[0] if exc.args else exc)) from exc


class PriceRequest(BaseModel):
    underlying: str = "NIFTY"
    expiry: date
    strike: float = Field(gt=0)
    option_type: Literal["call", "put"]
    lots: float = 1.0


class PositionIn(BaseModel):
    underlying: str = "NIFTY"
    expiry: date
    strike: float = Field(gt=0)
    option_type: Literal["call", "put"]
    lots: float = Field(description="signed: positive long, negative short")


class BookRequest(BaseModel):
    positions: list[PositionIn] = Field(min_length=1)


class ScenarioRequest(BookRequest):
    spot_shocks: Optional[list[float]] = Field(default=None, description="relative, e.g. -0.02 for -2%")
    vol_shocks: Optional[list[float]] = Field(default=None, description="absolute, e.g. 0.02 for +2 vol points")


class RiskRequest(BookRequest):
    window: int = Field(default=500, ge=100, le=2500, description="historical days for VaR")
    var_level: float = Field(default=0.99, gt=0.5, lt=1)
    es_level: float = Field(default=0.975, gt=0.5, lt=1)


class ExplainRequest(BookRequest):
    day0: date
    day1: date


class ReportRequest(BookRequest):
    day0: Optional[date] = None
    day1: Optional[date] = None


class ViewIn(BaseModel):
    source: Literal["user", "market", "model", "blend"] = "blend"
    target: Optional[float] = Field(default=None, gt=0, description="index level expected at the horizon")
    horizon: Optional[date] = Field(default=None, description="default: expiry")
    vol: Optional[float] = Field(default=None, gt=0, lt=3, description="volatility of the move, decimal")
    iv_shift: float = Field(default=0.0, gt=-0.5, lt=0.5, description="implied vol change by the horizon, decimal")


class RecommendRequest(BaseModel):
    capital: float = Field(gt=0, description="rupees available; any amount")
    view: ViewIn = Field(default_factory=ViewIn)
    underlying: str = "NIFTY"
    expiry: Optional[date] = None
    families: Optional[list[Literal["buy", "debit", "hedged", "naked", "futures"]]] = None
    risk_budget: float = Field(default=0.5, gt=0, le=1, description="max loss as a share of capital")
    objective: Literal["risk_adjusted", "expected_return", "probability"] = "risk_adjusted"
    top: int = Field(default=8, ge=1, le=20)


class LegIn(BaseModel):
    instrument: Literal["call", "put", "future"]
    side: Literal["buy", "sell"]
    expiry: date
    strike: Optional[float] = Field(default=None, gt=0)
    ratio: int = Field(default=1, ge=1, le=10)


class ProjectRequest(BaseModel):
    legs: list[LegIn] = Field(min_length=1, max_length=8)
    lots: int = Field(default=1, ge=1, le=50)
    view: ViewIn = Field(default_factory=ViewIn)
    underlying: str = "NIFTY"
    name: str = "Custom strategy"


class HedgeRequest(BookRequest):
    budget: float = Field(gt=0, description="rupees for premium plus extra margin")
    view: ViewIn = Field(default_factory=lambda: ViewIn(source="model"))
    rank_by: Literal["worst_case", "es95", "per_rupee"] = "worst_case"
    top: int = Field(default=3, ge=1, le=10)


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    history: list[dict] = Field(default_factory=list)


def _positions(req: BookRequest) -> list[dict]:
    return [p.model_dump() for p in req.positions]


@app.get("/health")
def health(svc: PricingService = Depends(get_service)):
    return _call(svc.health)


@app.get("/expiries/{underlying}")
def expiries(underlying: str, svc: PricingService = Depends(get_service)):
    return {"underlying": underlying.upper(), "expiries": [d.isoformat() for d in _call(svc.expiries, underlying)]}


@app.get("/overview/{underlying}")
def overview(underlying: str, svc: PricingService = Depends(get_service)):
    return _call(svc.overview, underlying)


@app.get("/smile/{underlying}/{expiry}")
def smile(underlying: str, expiry: date, svc: PricingService = Depends(get_service)):
    return _call(svc.smile, underlying, expiry)


@app.post("/price")
def price(req: PriceRequest, svc: PricingService = Depends(get_service)):
    return _call(svc.price, req.underlying, req.expiry, req.strike, req.option_type, req.lots)


@app.post("/book")
def book(req: BookRequest, svc: PricingService = Depends(get_service)):
    return _call(svc.book, _positions(req))


@app.post("/scenarios")
def scenarios(req: ScenarioRequest, svc: PricingService = Depends(get_service)):
    return _call(svc.scenarios, _positions(req), req.spot_shocks, req.vol_shocks)


@app.post("/risk")
def risk(req: RiskRequest, svc: PricingService = Depends(get_service)):
    return _call(svc.risk, _positions(req), req.window, req.var_level, req.es_level)


@app.get("/risk/eod_days")
def eod_days(svc: PricingService = Depends(get_service)):
    return {"days": svc.eod_days(), "data_mode": svc.mode}


@app.post("/risk/pnl_explain")
def pnl_explain(req: ExplainRequest, svc: PricingService = Depends(get_service)):
    return _call(svc.pnl_explain, _positions(req), req.day0, req.day1)


@app.post("/risk/report", response_class=HTMLResponse)
def risk_report(req: ReportRequest, svc: PricingService = Depends(get_service)):
    return HTMLResponse(_call(svc.risk_report, _positions(req), req.day0, req.day1))


@app.get("/strategy/meta")
def strategy_meta(underlying: str = "NIFTY", svc: PricingService = Depends(get_service)):
    from optlib.recommender import DISCLAIMER, OBJECTIVES, VIEW_SOURCES
    from optlib.strategies import FAMILIES

    from optlib.recommender import HEDGE_RANKS

    return {"families": FAMILIES, "objectives": OBJECTIVES, "view_sources": VIEW_SOURCES, "disclaimer": DISCLAIMER,
            "hedge_ranks": HEDGE_RANKS,
            "expiries": [d.isoformat() for d in _call(svc.expiries, underlying)]}


@app.post("/strategy/recommend")
def strategy_recommend(req: RecommendRequest, svc: PricingService = Depends(get_service)):
    return _call(svc.recommend, req.capital, req.view.model_dump(), req.underlying, req.expiry, req.families,
                 req.risk_budget, req.objective, req.top)


@app.post("/strategy/project")
def strategy_project(req: ProjectRequest, svc: PricingService = Depends(get_service)):
    return _call(svc.project_strategy, [l.model_dump() for l in req.legs], req.lots, req.view.model_dump(),
                 req.underlying, req.name)


@app.post("/strategy/hedge")
def strategy_hedge(req: HedgeRequest, svc: PricingService = Depends(get_service)):
    return _call(svc.hedge, _positions(req), req.budget, req.view.model_dump(), "NIFTY", req.top, req.rank_by)


@app.post("/agent/ask")
def ask(req: AskRequest, svc: PricingService = Depends(get_service)):
    from .agent import AgentUnavailable, OptionPricingAgent

    try:
        agent = OptionPricingAgent(svc)
    except AgentUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _call(lambda: agent.ask(req.question, req.history).to_dict())
