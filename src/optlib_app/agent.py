"""Orchestrator and validator agents, on whichever LLM API is configured (see optlib_app.llm).

Orchestrator: plans the request, calls the pricing tools, explains the result. It never computes a
number itself; every figure in its answer must come from a tool result.

Validator: a separate model call that sees only the user's question and the tool calls (inputs and
outputs, including the deterministic validation reports). It never sees the orchestrator's reasoning
or draft. It can raise severity but never clear a deterministic FAIL.

Number grounding: every number in the final answer is checked against the tool results (allowing
for rounding and percent formatting). Numbers that cannot be traced are listed and the answer is
marked WARN. This matters most on smaller free models, which are likelier to invent figures.

Release rule: any deterministic FAIL, or a validator FAIL, withholds the answer.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .config import get_setting
from .llm import LLMClient, LLMError, LLMNotConfigured, make_llm
from .service import PricingService, ServiceError

MAX_TURNS = 8

ORCHESTRATOR_PROMPT = """You are the option pricing assistant for a quantitative analytics team.
You price NSE index options (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, NIFTYNXT50; European, cash-settled)
using the team's in-house library through the tools provided.

Rules:
1. Every number you state (price, vol, Greek, forward, rate, P&L) must be copied from a tool result in this
   conversation. Never compute, estimate, add up or invent figures yourself; if a total is needed, call a tool
   that returns it (book_risk returns totals).
2. If the trade is ambiguous (missing expiry, strike, call/put or size), ask one short question instead of guessing.
   Lots are signed: positive long, negative short. Quote sizes in lots and state the lot size from the tools.
3. Use market_overview first when you need the available expiries, spot, forward or rates.
4. Report every validation status. If a result has WARN checks, say what they mean in plain words.
   If any check FAILs, say the result is withheld and why; do not present the failed number as usable.
5. Explain briefly: the key drivers (forward, vol and where it came from, time to expiry), and where the
   model is least reliable (e.g. SVI extrapolation beyond quoted strikes).
6. Be concise. Use desk units: vega per vol point, theta per calendar day, rho per basis point, values in Rs.
"""

VALIDATOR_PROMPT = """You are an independent model validator for an option pricing system.
You receive a user's question and the exact tool calls made to answer it: inputs, outputs, and the
deterministic validation report for each priced result. You do NOT see the assistant's reasoning or answer.

Check:
- the tool inputs match the question (underlying, expiry, strike, call/put, sign and size of position);
- results are internally consistent (price within bounds, Greek signs, position Greeks = per-unit x quantity);
- every FAIL and WARN in the validation reports, and whether it affects what the user asked for;
- anything missing that the user asked for.

Reply in at most 120 words, then a final line exactly of the form:
VERDICT: PASS | WARN | FAIL
Use FAIL if inputs do not match the question or a result must not be used; WARN for usable results with caveats."""

TOOLS = [
    {
        "name": "market_overview",
        "description": "Spot, available expiries with forward, rate, ATM vol, chain-QA and SVI fit status, futures, "
                       "and the discount curve for one underlying. Call first to learn which expiries exist.",
        "input_schema": {"type": "object", "properties": {
            "underlying": {"type": "string", "description": "e.g. NIFTY"}}, "required": ["underlying"]},
    },
    {
        "name": "get_smile",
        "description": "Market implied vols (bid/mid/ask) by strike for one expiry, near the money, plus QA summary.",
        "input_schema": {"type": "object", "properties": {
            "underlying": {"type": "string"},
            "expiry": {"type": "string", "description": "YYYY-MM-DD"},
            "strikes_each_side": {"type": "integer", "description": "strikes either side of the forward, default 8"}},
            "required": ["underlying", "expiry"]},
    },
    {
        "name": "price_option",
        "description": "Price one option position and run the validation checks. Returns price per unit, value, "
                       "vol and its source, per-unit and position Greeks (desk units), inputs, and validation.",
        "input_schema": {"type": "object", "properties": {
            "underlying": {"type": "string"},
            "expiry": {"type": "string", "description": "YYYY-MM-DD"},
            "strike": {"type": "number"},
            "option_type": {"type": "string", "enum": ["call", "put"]},
            "lots": {"type": "number", "description": "signed: + long, - short; default 1"}},
            "required": ["underlying", "expiry", "strike", "option_type"]},
    },
    {
        "name": "book_risk",
        "description": "Price a book of positions: per-position values and Greeks, totals, and validation status.",
        "input_schema": {"type": "object", "properties": {"positions": {"type": "array", "items": {
            "type": "object", "properties": {
                "underlying": {"type": "string"}, "expiry": {"type": "string"}, "strike": {"type": "number"},
                "option_type": {"type": "string", "enum": ["call", "put"]}, "lots": {"type": "number"}},
            "required": ["expiry", "strike", "option_type", "lots"]}}}, "required": ["positions"]},
    },
    {
        "name": "scenario_grid",
        "description": "Full-revaluation P&L of a book on a spot x vol grid (sticky strike, time fixed). "
                       "spot_shocks are relative (-0.02 = -2%), vol_shocks absolute (0.02 = +2 vol points).",
        "input_schema": {"type": "object", "properties": {
            "positions": {"type": "array", "items": {"type": "object"}},
            "spot_shocks": {"type": "array", "items": {"type": "number"}},
            "vol_shocks": {"type": "array", "items": {"type": "number"}}}, "required": ["positions"]},
    },
]


class AgentUnavailable(Exception):
    pass


@dataclass
class ToolRecord:
    name: str
    input: dict
    output: Any
    error: Optional[str] = None


@dataclass
class AgentReply:
    question: str
    answer: str
    status: str  # PASS | WARN | FAIL
    released: bool
    validator_review: str
    validator_verdict: str
    deterministic_status: str
    ungrounded_numbers: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    withheld_draft: Optional[str] = None
    provider: str = ""
    model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _worst(*statuses: str) -> str:
    s = set(statuses)
    return "FAIL" if "FAIL" in s else ("WARN" if "WARN" in s else "PASS")


def _round(x, nd=6):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v, nd) for v in x]
    return x


# ------------------------------------------------------------------ number grounding

_NUM = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{2,3})+(?:\.\d+)?(?![\d,])|(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}",
                   re.IGNORECASE)


def _numbers(text: str) -> list[tuple[str, float, int]]:
    """(raw, value, decimals) for each number in text, ignoring dates and years."""
    text = _DATE.sub(" ", text)
    out = []
    for m in _NUM.finditer(text):
        raw = m.group(0)
        clean = raw.replace(",", "")
        try:
            v = float(clean)
        except ValueError:
            continue
        if raw.isdigit() and 1900 <= v <= 2100:
            continue
        out.append((raw, v, len(clean.split(".")[1]) if "." in clean else 0))
    return out


def _tool_numbers(records: list["ToolRecord"]) -> list[float]:
    vals: list[float] = []

    def walk(x):
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            if math.isfinite(x):
                vals.extend((float(x), float(x) * 100))  # fractions shown as percentages
        elif isinstance(x, str):
            vals.extend(v for _, v, _ in _numbers(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    for r in records:
        walk(r.input)
        walk(r.output)
    return vals


def ungrounded_numbers(answer: str, question: str, records: list["ToolRecord"]) -> list[str]:
    """Numbers in the answer that no tool result (or the question) supports, allowing for rounding."""
    pool = _tool_numbers(records) + [v for _, v, _ in _numbers(question)]
    missing = []
    for raw, v, d in _numbers(answer):
        if d == 0 and abs(v) <= 10:  # small counts ("2 lots", "3 checks") carry no pricing information
            continue
        tol = 0.5 * 10 ** (-d) + 1e-9 + 1e-6 * abs(v)
        if not any(abs(abs(v) - abs(t)) <= tol for t in pool):
            missing.append(raw)
    return missing


# ------------------------------------------------------------------ agent

class OptionPricingAgent:
    def __init__(self, service: PricingService, llm: Optional[LLMClient] = None,
                 validator_llm: Optional[LLMClient] = None, max_turns: int = MAX_TURNS):
        self.service = service
        try:
            self.llm = llm or make_llm(get_setting, "orchestrator")
            self.validator_llm = validator_llm or (make_llm(get_setting, "validator") if llm is None else llm)
        except LLMNotConfigured as exc:
            raise AgentUnavailable(str(exc)) from exc
        self.max_turns = max_turns

    # tools ----------------------------------------------------------------------------
    def _run_tool(self, name: str, args: dict) -> Any:
        svc = self.service
        if name == "market_overview":
            o = svc.overview(args.get("underlying", "NIFTY"))
            o["curve"].pop("pillars", None)
            for e in o["expiries"]:
                e.pop("svi_params", None)
            return o
        if name == "get_smile":
            s = svc.smile(args.get("underlying", "NIFTY"), args["expiry"])
            n = int(args.get("strikes_each_side", 8))
            q = s["quotes"]
            i = min(range(len(q)), key=lambda j: abs(q[j]["strike"] - s["forward"]))
            return {"expiry": s["expiry"], "forward": s["forward"], "atm_vol": s["atm_vol"],
                    "quotes": [{k: r[k] for k in ("strike", "type", "bid", "ask", "iv_bid", "iv_mid", "iv_ask")}
                               for r in q[max(0, i - n): i + n + 1]],
                    "qa_summary": s["qa_summary"]}
        if name == "price_option":
            out = svc.price(args.get("underlying", "NIFTY"), args["expiry"], args["strike"], args["option_type"],
                            args.get("lots", 1))
            r = out["result"]
            return {"underlying": r["underlying"], "expiry": r["expiry"], "strike": r["strike"],
                    "option_type": r["option_type"], "lots": out["lots"], "lot_size": out["lot_size"],
                    "quantity_units": r["quantity"], "price_per_unit": r["price"], "value_rs": r["value"],
                    "vol": r["vol"], "vol_source": r["vol_source"], "forward": r["forward"], "spot": r["spot"],
                    "rate": r["rate"], "days_to_expiry": r["time_to_expiry"] * 365, "as_of": r["as_of"],
                    "greeks_per_unit": r["greeks"], "position_greeks": r["position_greeks"],
                    "validation": out["validation"]}
        if name == "book_risk":
            return svc.book(args["positions"])
        if name == "scenario_grid":
            return svc.scenarios(args["positions"], args.get("spot_shocks"), args.get("vol_shocks"))
        raise ServiceError(f"unknown tool {name}")

    @staticmethod
    def _statuses(records: list[ToolRecord]) -> list[str]:
        out = []
        for r in records:
            if r.error or not isinstance(r.output, dict):
                continue
            if "validation" in r.output:
                out.append(r.output["validation"]["status"])
            elif r.name == "book_risk":
                out.append(r.output.get("status", "PASS"))
        return out

    # main loop ---------------------------------------------------------------------------
    def ask(self, question: str, history: Optional[list] = None) -> AgentReply:
        messages = [{"role": m["role"], "content": m["content"]} for m in (history or [])
                    if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
        messages.append({"role": "user", "content": question})
        records: list[ToolRecord] = []
        answer = ""
        try:
            for _ in range(self.max_turns):
                reply = self.llm.chat(ORCHESTRATOR_PROMPT, messages, TOOLS)
                messages.append({"role": "assistant", "content": reply.text, "tool_calls": reply.tool_calls})
                if not reply.tool_calls:
                    answer = reply.text
                    break
                for tc in reply.tool_calls:
                    if tc.parse_error:
                        out, err = None, tc.parse_error
                    else:
                        try:
                            out, err = _round(self._run_tool(tc.name, dict(tc.arguments))), None
                        except (ServiceError, KeyError, ValueError, TypeError) as exc:
                            out, err = None, str(exc)
                    records.append(ToolRecord(tc.name, dict(tc.arguments), out, err))
                    messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                     "content": json.dumps(out) if err is None else err, "is_error": err is not None})
            else:
                answer = "I could not finish within the step limit; please narrow the request."
            review, verdict = self._validate(question, records)
        except LLMError as exc:
            return AgentReply(question=question, answer=f"The LLM API failed: {exc}", status="FAIL", released=False,
                              validator_review="", validator_verdict="", deterministic_status="",
                              tool_calls=[asdict(r) for r in records], provider=self.llm.provider,
                              model=self.llm.model)

        deterministic = _worst(*self._statuses(records)) if records else "PASS"
        loose = ungrounded_numbers(answer, question, records)
        status = _worst(deterministic, verdict, "WARN" if loose else "PASS")
        released = status != "FAIL"
        reply = AgentReply(question=question, answer=answer, status=status, released=released,
                           validator_review=review, validator_verdict=verdict, deterministic_status=deterministic,
                           ungrounded_numbers=loose, tool_calls=[asdict(r) for r in records],
                           provider=self.llm.provider, model=self.llm.model)
        if not released:
            reply.withheld_draft = answer
            reasons = self._fail_reasons(records) or [f"independent validator: {review}"]
            reply.answer = "Result withheld by validation:\n" + "\n".join(f"- {r}" for r in reasons)
        return reply

    @staticmethod
    def _fail_reasons(records: list[ToolRecord]) -> list[str]:
        out = []
        for r in records:
            v = r.output.get("validation") if isinstance(r.output, dict) else None
            if v:
                out += [f"{r.input.get('strike')} {r.input.get('option_type')}: {c['name']}: {c['detail']}"
                        for c in v["checks"] if c["status"] == "FAIL"]
        return out

    def _validate(self, question: str, records: list[ToolRecord]) -> tuple[str, str]:
        if not records:
            return "No tool calls were made, so there is nothing to validate.", "PASS"
        payload = {"question": question, "tool_calls": [asdict(r) for r in records]}
        reply = self.validator_llm.chat(VALIDATOR_PROMPT, [{"role": "user", "content": json.dumps(payload, default=str)}],
                                        tools=None, max_tokens=600)
        m = re.search(r"VERDICT:\s*\**\s*(PASS|WARN|FAIL)", reply.text, re.IGNORECASE)
        verdict = m.group(1).upper() if m else "WARN"  # an unparseable review is not a clean pass
        review = re.sub(r"\n?\**VERDICT:.*$", "", reply.text, flags=re.S | re.IGNORECASE).strip()
        return review, verdict
