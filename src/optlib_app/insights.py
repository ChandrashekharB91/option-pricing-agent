"""AI-written market note: a few plain sentences on the day's NIFTY options market, for the email and the app.

The model gets only the day's computed numbers and the news headlines. Its text is published only if
  * every number in it appears in that input (the agent's grounding check, rounding allowed), and
  * it gives no advice (no buy/sell/should-style wording).
Otherwise it is retried once with the problem named, then dropped: the email goes out without a note rather than
with an unchecked one.

Model: INSIGHTS_PROVIDER / INSIGHTS_API_KEY / INSIGHTS_MODEL if set (e.g. gemini); else GEMINI_API_KEY (+ optional
GEMINI_MODEL) uses Google Gemini; else the agent's LLM (LLM_PROVIDER ...).
"""
from __future__ import annotations

import json
import logging
import re
from types import SimpleNamespace
from typing import Callable, Optional

from .agent import ungrounded_numbers
from .llm import DEFAULT_MODELS, LLMError, LLMNotConfigured, make_llm

ADVICE = re.compile(r"\b(buy|sell|short|go long|should|must|recommend\w*|consider (buying|selling)|"
                    r"opportunit\w*|target price|stop[- ]loss|will (rise|fall|rally|crash))\b", re.I)
SYSTEM = """You are a markets analyst writing the opening note of a daily NIFTY options email for professionals.
Write 3 to 4 short sentences (at most 90 words) describing today's options market: index move, implied versus
realised volatility, skew or term structure, expected move and positioning (put/call ratio), and, if relevant, one
or two news themes from the headlines.
Rules: use ONLY numbers that appear in the data given, written the same way (you may round to fewer decimals);
do not compute new numbers; describe, do not predict or advise (never say buy, sell, should, opportunity, target);
plain English, no markdown, no bullet points, no greeting."""
# room for "thinking" models (Gemini 3.x Flash counts its reasoning tokens against this limit)
MAX_TOKENS = 4096
log = logging.getLogger(__name__)
LAST_PROBLEM = {"text": ""}  # why the last note was not published (shown by scripts/check_ai_note.py)


def insights_llm(get: Callable[[str, str], str]):
    """The model for notes: INSIGHTS_* settings, else Gemini via GEMINI_API_KEY, else the agent's model."""
    if get("INSIGHTS_PROVIDER", ""):
        mapping = {"LLM_PROVIDER": "INSIGHTS_PROVIDER", "LLM_API_KEY": "INSIGHTS_API_KEY",
                   "LLM_MODEL": "INSIGHTS_MODEL", "LLM_BASE_URL": "INSIGHTS_BASE_URL"}
        return make_llm(lambda k, d="": get(mapping[k], d) if k in mapping else get(k, d), "insights")
    if get("GEMINI_API_KEY", ""):
        gem = {"LLM_PROVIDER": "gemini", "LLM_API_KEY": get("GEMINI_API_KEY", ""),
               "LLM_MODEL": get("GEMINI_MODEL", "") or DEFAULT_MODELS["gemini"], "LLM_BASE_URL": ""}
        return make_llm(lambda k, d="": gem.get(k, d) if k in gem else get(k, d), "insights")
    return make_llm(get, "insights")


def facts(feed: dict) -> dict:
    """The numbers the model may use, rounded the way a reader sees them."""
    e = feed["expiries"]
    oi = feed.get("open_interest") or {}
    f = {
        "date": feed["date"], "index": feed["underlying"], "spot": round(feed["spot"], 2),
        "change_points": round(feed["change"], 2) if feed.get("change") is not None else None,
        "change_pct": round(feed["change_pct"] * 100, 2) if feed.get("change_pct") is not None else None,
        "india_vix_pct": round(feed["hist_vix"] * 100, 2) if feed.get("hist_vix") else None,
        "realised_vol_20d_pct": round(feed["hist_rv20"] * 100, 2) if feed.get("hist_rv20") else None,
        "implied_minus_realised_vol_pts": round(feed["vrp"] * 100, 2) if feed.get("vrp") is not None else None,
        "put_call_oi_ratio": round(oi["pcr"], 2) if oi.get("pcr") else None,
        "put_call_oi_ratio_prev": round(oi["pcr_prev"], 2) if oi.get("pcr_prev") else None,
        "expiries": [{
            "expiry": x["expiry"], "days": round(x["days"], 1), "atm_vol_pct": round(x["atm_vol"] * 100, 2),
            "atm_vol_change_pts": (round(feed["changes"]["atm_vol"][x["expiry"]] * 100, 2)
                                   if x["expiry"] in feed["changes"].get("atm_vol", {}) else None),
            "expected_move_points": round(x["expected_move"]), "expected_move_pct": round(x["expected_move_pct"] * 100, 2),
            "range_low": round(x["range_lo"]), "range_high": round(x["range_hi"]),
            "risk_reversal_25d_pts": round(x["rr25"] * 100, 2), "butterfly_25d_pts": round(x["bf25"] * 100, 2),
        } for x in e],
    }
    if feed.get("iv_rank", {}).get("available"):
        r = feed["iv_rank"]
        f["vix_1y_rank_pct"] = round(r["rank"] * 100)
        f["vix_1y_percentile_pct"] = round(r["percentile"] * 100)
    if oi.get("max_pain"):
        f["max_pain"] = oi["max_pain"]
    return f


def market_note(feed: dict, headlines: Optional[list[dict]] = None, llm=None,
                get: Optional[Callable[[str, str], str]] = None) -> Optional[dict]:
    """{"text", "model", "checks"} or None when no model is configured or the text fails the checks."""
    LAST_PROBLEM["text"] = ""
    try:
        llm = llm or insights_llm(get)
    except LLMNotConfigured as exc:
        LAST_PROBLEM["text"] = f"no model configured: {exc}"
        return None
    data = facts(feed)
    heads = [h["title"] for h in (headlines or [])[:8]]
    user = ("DATA (JSON):\n" + json.dumps(data, indent=1) + "\n\nHEADLINES:\n" + "\n".join(f"- {t}" for t in heads))
    record = SimpleNamespace(input={}, output={"data": data, "headlines": heads})
    messages = [{"role": "user", "content": user}]
    for attempt in range(2):
        try:
            reply = llm.chat(SYSTEM, messages, tools=None, max_tokens=MAX_TOKENS)
        except LLMError as exc:
            LAST_PROBLEM["text"] = f"model call failed: {exc}"
            log.warning("AI note: %s", LAST_PROBLEM["text"])
            return None
        text = re.sub(r"\s+", " ", (reply.text or "").replace("*", "")).strip()
        loose = ungrounded_numbers(text, user, [record])
        advice = ADVICE.findall(text)
        if text and not loose and not advice and len(text) <= 900:
            return {"text": text, "model": getattr(llm, "model", ""), "provider": getattr(llm, "provider", ""),
                    "checks": "every number matched to the data; no advice wording"}
        problems = []
        if loose:
            problems.append(f"these numbers are not in the data: {', '.join(loose[:6])}")
        if advice:
            problems.append(f"remove advice wording: {', '.join(sorted({a if isinstance(a, str) else a[0] for a in advice}))}")
        if not text:
            problems.append("the note was empty")
        LAST_PROBLEM["text"] = f"attempt {attempt + 1}: " + "; ".join(problems) + (f" | text: {text[:300]}" if text else "")
        log.warning("AI note rejected, %s", LAST_PROBLEM["text"])
        messages += [{"role": "assistant", "content": text or "(empty)"},
                     {"role": "user", "content": "Rewrite the note. " + "; ".join(problems) + "."}]
    return None
