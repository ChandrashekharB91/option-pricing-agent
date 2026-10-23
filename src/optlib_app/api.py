"""FastAPI backend: every pricing tool as a typed endpoint.

Run:  uvicorn optlib_app.api:app --reload        (then open http://127.0.0.1:8000/docs)

The dashboard, notebooks (via optlib_app.client) and the agents all go through the same
PricingService, so a number is identical wherever it is shown.
"""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from fastapi import Depends, FastAPI, HTTPException
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


@app.post("/agent/ask")
def ask(req: AskRequest, svc: PricingService = Depends(get_service)):
    from .agent import AgentUnavailable, OptionPricingAgent

    try:
        agent = OptionPricingAgent(svc)
    except AgentUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _call(lambda: agent.ask(req.question, req.history).to_dict())
