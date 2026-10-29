"""FastAPI backend: every pricing tool as a typed endpoint.

Run:  uvicorn optlib_app.api:app --reload        (then open http://127.0.0.1:8000/docs)

The dashboard, notebooks (via optlib_app.client) and the agents all go through the same
PricingService, so a number is identical wherever it is shown.
"""
from __future__ import annotations

import secrets
from datetime import date
from typing import Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
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
    min_prob_profit: float = Field(default=0.5, ge=0, lt=1,
                                   description="recommend only ideas with at least this chance of profit "
                                               "(and expected P&L above zero)")
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
                 req.risk_budget, req.objective, req.top, req.min_prob_profit)


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


# ------------------------------------------------------------------ daily feed (email digest)
_feed = None


def get_feed(svc: PricingService = Depends(get_service)):
    from .feed_jobs import FeedJobs

    global _feed
    if _feed is None or _feed.svc is not svc:
        _feed = FeedJobs(svc)
    return _feed


def require_admin(request: Request, x_admin_key: Optional[str] = Header(default=None), feed=Depends(get_feed)):
    """Admin actions: FEED_ADMIN_KEY must match when it is set; without a key only this machine may call."""
    if feed.admin_key:
        if not x_admin_key or not secrets.compare_digest(x_admin_key, feed.admin_key):
            raise HTTPException(status_code=403, detail="admin key required (X-Admin-Key header)")
        return
    host = request.client.host if request.client else ""
    if host not in ("127.0.0.1", "::1", "localhost", "testclient"):
        raise HTTPException(status_code=403, detail="set FEED_ADMIN_KEY to allow admin actions from other machines")


class SubscribeRequest(BaseModel):
    email: str = Field(..., max_length=254)
    name: str = Field("", max_length=80)
    consent: bool = False


class SendRequest(BaseModel):
    to: Optional[list[str]] = Field(None, description="test addresses; omit to send to all active subscribers")
    force: bool = False


@app.post("/feed/subscribe")
def feed_subscribe(req: SubscribeRequest, feed=Depends(get_feed)):
    from .subscribers import SubscriptionError

    try:
        return feed.subscribe(req.email, req.name, req.consent, source="api")
    except SubscriptionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/feed/signup", response_class=HTMLResponse)
def feed_signup_form():
    """Sign-up page for readers: open http://<api address>/feed/signup in a browser."""
    from .feed_jobs import signup_page

    return signup_page()


@app.post("/feed/signup", response_class=HTMLResponse)
async def feed_signup_submit(request: Request, feed=Depends(get_feed)):
    from urllib.parse import parse_qs

    from .feed_jobs import signup_page
    from .subscribers import SubscriptionError

    form = {k: v[0] for k, v in parse_qs((await request.body()).decode("utf-8", "replace")).items()}
    email, name = form.get("email", ""), form.get("name", "")
    if form.get("website"):  # hidden field only bots fill in: answer as if it worked, send nothing
        return signup_page("Thanks. Check your inbox for the confirmation link.")
    try:
        r = feed.subscribe(email, name, form.get("consent") == "yes", source="signup page")
    except SubscriptionError as exc:
        return HTMLResponse(signup_page(str(exc).capitalize() + ".", ok=False, email=email, name=name), status_code=400)
    return signup_page(r["message"], ok=r["status"] != "error")


@app.get("/feed/confirm", response_class=HTMLResponse)
def feed_confirm(token: str = "", feed=Depends(get_feed)):
    from .feed_jobs import page

    s = feed.confirm(token)
    if not s:
        return HTMLResponse(page("Link not recognised", "This confirmation link is not valid. Please sign up again.",
                                 ok=False), status_code=404)
    if s.status == "unsubscribed":
        return page("You are unsubscribed", "This address unsubscribed earlier. Sign up again to restart the feed.",
                    ok=False)
    return page("Subscription confirmed", f"{s.email} will receive the NIFTY options daily feed after each "
                                          "trading day. Every email has an unsubscribe link.")


@app.get("/feed/unsubscribe", response_class=HTMLResponse)
@app.post("/feed/unsubscribe", response_class=HTMLResponse)
def feed_unsubscribe(token: str = "", feed=Depends(get_feed)):
    from .feed_jobs import page

    s = feed.unsubscribe(token)
    if not s:
        return HTMLResponse(page("Link not recognised", "This unsubscribe link is not valid (or is from a test or "
                                 "preview email). Reply to any feed email with UNSUBSCRIBE instead.", ok=False),
                            status_code=404)
    return page("You are unsubscribed", f"{s.email} will not receive any more feed emails.")


@app.get("/feed/preview", response_class=HTMLResponse)
def feed_preview(name: str = "", feed=Depends(get_feed)):
    return _call(feed.preview_html, None, name)


@app.get("/feed/status", dependencies=[Depends(require_admin)])
def feed_status(feed=Depends(get_feed)):
    return feed.status()


@app.get("/feed/subscribers", dependencies=[Depends(require_admin)])
def feed_subscribers(feed=Depends(get_feed)):
    return {"subscribers": [s.public() for s in feed.store.all()], "counts": feed.store.counts()}


@app.delete("/feed/subscribers/{email}", dependencies=[Depends(require_admin)])
def feed_delete(email: str, feed=Depends(get_feed)):
    return {"deleted": feed.store.delete(email)}


@app.post("/feed/send", dependencies=[Depends(require_admin)])
def feed_send(req: SendRequest, feed=Depends(get_feed)):
    from .subscribers import SubscriptionError

    try:
        return _call(feed.send, req.to, req.force)
    except SubscriptionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
