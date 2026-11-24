"""Plain-English strategy builder: "I have 2 lakh, I think NIFTY reaches 24,500 by next week, low risk" -> ideas.

How it reads the text
  * Numbers (capital, target level, range, % move, dates) are read by a deterministic parser, never by a model, so
    a figure in the result is always one the user wrote (or the market data, stated as an assumption).
  * Wording (direction, risk appetite, objective, volatility view, which strategy families) is read by keyword rules;
    if a language model is configured (Gemini / the agent's model) it classifies only what the rules left open.
  * Everything the builder assumed is listed back to the user next to the result, and the request is the ordinary
    recommender call, so the ideas, costs and checks are the same as in the Strategy lab.

This is analysis, not investment advice.
"""
from __future__ import annotations

import json
import math
import re
from datetime import date, timedelta
from typing import Callable, Optional

WORDS = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "l": 1e5, "crore": 1e7, "crores": 1e7, "cr": 1e7,
         "k": 1e3, "thousand": 1e3}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                           "nov", "dec"])}
DIRECTION = [
    ("volatile", r"\b(big move|large move|either (way|direction)|breakout|volatile|swing either|sharp move)\b"),
    ("neutral", r"\b(sideways|range[- ]?bound|flat|neutral|stay (in|within|between)|consolidat\w*|not move much)\b"),
    ("up", r"\b(bullish|rally|rise|rises|rising|go up|goes up|move up|upside|higher|gain|climb\w*|break above|up)\b"),
    ("down", r"\b(bearish|fall|falls|falling|drop\w*|decline\w*|go down|goes down|downside|lower|correct\w*|crash\w*|"
             r"down|slide|slip\w*)\b"),
]
RISK = [("low", r"\b(low[- ]risk|conservative|safe|cautious|limited (loss|risk)|small loss|protect\w*|defined risk)\b"),
        ("high", r"\b(high[- ]risk|aggressive|risky|max(imum)? (profit|return)|leverage\w*)\b")]
OBJECTIVE = [("probability", r"\b(high(er)? (probability|chance)|likely|consistent|win rate|most probable)\b"),
             ("expected_return", r"\b(max(imum|imise|imize)? (return|profit)|highest return|biggest (gain|profit))\b")]
VOL = [(0.02, r"\b(vol(atility)?|iv|vix)\b[^.]{0,25}\b(rise|rises|up|increase\w*|spike\w*|higher|jump\w*)\b"),
       (-0.02, r"\b(vol(atility)?|iv|vix)\b[^.]{0,25}\b(fall|falls|down|decrease\w*|drop\w*|lower|crush\w*|cool\w*)\b"),
       (-0.02, r"\b(iv crush|vol crush)\b")]
FAMILY_RULES = [
    (["buy"], r"\b(only|just) (buy|long) options?\b|\bbuy (calls?|puts?) only\b|\bno selling\b"),
    (["debit"], r"\b(spreads? only|only spreads?|debit spread\w*)\b"),
    (["hedged", "naked"], r"\b(sell(ing)? (options|premium)|collect premium|theta|option writing)\b"),
]
SYSTEM = """Classify a trader's description of a NIFTY options idea. Reply with JSON only, no prose:
{"direction": "up|down|neutral|volatile|none", "risk": "low|medium|high|none", "objective":
"risk_adjusted|probability|expected_return|none", "vol_view": "up|down|none"}
Use "none" when the text does not say. Do not invent numbers."""


class IntentError(ValueError):
    """The text could not be turned into a request (message is shown to the user)."""


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def parse_capital(text: str) -> Optional[float]:
    t = text.lower().replace("₹", " rs ")
    m = re.search(r"(?:rs\.?|inr|capital(?: of)?|budget(?: of)?|have|with|invest(?:ing)?|put in|use)\s*"
                  r"([\d,]+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|cr|k|thousand|l)?\b", t)
    if not m:
        m = re.search(r"\b([\d,]+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|cr|k|thousand)\b", t)
    if not m:
        return None
    value = _num(m.group(1)) * WORDS.get((m.group(2) or "").strip(), 1)
    return value if value >= 1000 else None


def _levels(text: str, spot: float) -> list[tuple[int, float]]:
    """(position, value) of numbers that look like index levels (within 40% of spot)."""
    out = []
    for m in re.finditer(r"(?<![\d.])(\d{1,3}(?:,\d{2,3})+|\d{4,6})(?:\.\d+)?(?![\d%]|\s*(?:lakh|lac|k\b|cr|crore|"
                         r"thousand|rs|rupees))", text.lower()):
        v = _num(m.group(1))
        if 0.6 * spot <= v <= 1.4 * spot:
            out.append((m.start(), v))
    return out


def parse_levels(text: str, spot: float) -> dict:
    t = text.lower()
    lv = _levels(text, spot)
    m = re.search(r"(?:between|range(?: of)?|within)\s*([\d,]{4,})\s*(?:and|to|-|–)\s*([\d,]{4,})", t)
    if m:
        lo, hi = sorted([_num(m.group(1)), _num(m.group(2))])
        if 0.6 * spot <= lo < hi <= 1.4 * spot:
            return {"range": [lo, hi]}
    m = re.search(r"([\d,]{4,})\s*(?:-|–)\s*([\d,]{4,})", t)
    if m:
        lo, hi = sorted([_num(m.group(1)), _num(m.group(2))])
        if 0.6 * spot <= lo < hi <= 1.4 * spot and hi - lo < 0.2 * spot:
            return {"range": [lo, hi]}
    m = re.search(r"\b(up|rise|rally|gain|fall|drop|down|decline|correct)\w*\s*(?:by\s*)?(\d+(?:\.\d+)?)\s*%", t) \
        or re.search(r"(\d+(?:\.\d+)?)\s*%\s*(up|rise|rally|gain|higher|fall|drop|down|decline|lower|correction)", t)
    if m:
        a, b = m.groups()
        word, pct = (a, b) if not a[0].isdigit() else (b, a)
        sign = -1 if re.match(r"fall|drop|down|declin|correct|lower", word) else 1
        return {"pct": sign * float(pct) / 100}
    if lv:
        return {"target": lv[-1][1]}
    return {}


def parse_horizon(text: str, today: date) -> tuple[Optional[date], str]:
    t = text.lower()
    m = re.search(r"\bin\s*(\d{1,2})\s*(trading )?days?\b", t)
    if m:
        return today + timedelta(days=int(m.group(1)) + (int(m.group(1)) // 5) * 2), f"in {m.group(1)} days"
    m = re.search(r"\bin\s*(\d)\s*weeks?\b", t)
    if m:
        return today + timedelta(weeks=int(m.group(1))), f"in {m.group(1)} weeks"
    if re.search(r"\b(today|intraday|by close)\b", t):
        return today, "today"
    if re.search(r"\btomorrow\b", t):
        return today + timedelta(days=1), "tomorrow"
    if re.search(r"\b(this|end of( the)?) week\b|\bweek ?end\b", t):
        return today + timedelta(days=(4 - today.weekday()) % 7), "end of this week"
    if re.search(r"\bnext week\b", t):
        return today + timedelta(days=(4 - today.weekday()) % 7 + 7), "end of next week"
    if re.search(r"\b(this|end of( the)?) month\b|\bmonth[- ]end\b", t):
        nxt = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
        return nxt - timedelta(days=1), "end of this month"
    if re.search(r"\bnext month\b", t):
        nxt = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
        return (nxt.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1), "end of next month"
    for i, d in enumerate(WEEKDAYS[:5]):
        if re.search(rf"\b(by|on|till|until|before)?\s*(next\s+)?{d}\b", t):
            ahead = (i - today.weekday()) % 7 or 7
            if re.search(rf"\bnext\s+{d}\b", t) and ahead < 7:
                ahead += 7
            return today + timedelta(days=ahead), d.capitalize()
    m = re.search(r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*", t) or \
        re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s*(\d{1,2})\b", t)
    if m:
        a, b = m.groups()
        day, mon = (int(a), MONTHS[b]) if a.isdigit() else (int(b), MONTHS[a])
        try:
            d = date(today.year, mon, day)
        except ValueError:
            return None, ""
        if d < today:
            d = date(today.year + 1, mon, day)
        return d, d.strftime("%d %b")
    if re.search(r"\b(by|till|until|at|on) (the )?expiry\b|\bthis expiry\b", t):
        return None, "expiry"
    return None, ""


def _first(rules, text: str):
    for value, pat in rules:
        if re.search(pat, text, re.I):
            return value
    return None


def classify(text: str, llm=None) -> dict:
    """Wording -> direction / risk / objective / vol view / families. Rules first, then the model for the gaps."""
    t = text.lower()
    no_vol = t
    for _, pat in VOL:  # "volatility will rise" is a vol view, not an index direction
        no_vol = re.sub(pat, " ", no_vol)
    out = {"direction": _first(DIRECTION, no_vol), "risk": _first(RISK, t), "objective": _first(OBJECTIVE, t),
           "vol_shift": _first(VOL, t), "families": _first(FAMILY_RULES, t), "read_by": "rules"}
    if re.search(r"\bno (naked|unlimited|uncovered)\b|\bwithout naked\b", t):
        out["no_naked"] = True
    if llm is not None and (out["direction"] is None or out["risk"] is None):
        try:
            reply = llm.chat(SYSTEM, [{"role": "user", "content": text}], tools=None, max_tokens=200)
            m = re.search(r"\{.*\}", reply.text or "", re.S)
            got = json.loads(m.group(0)) if m else {}
        except Exception:  # noqa: BLE001 - the model is optional; the rules still give an answer
            got = {}
        filled = []
        if out["direction"] is None and got.get("direction") in ("up", "down", "neutral", "volatile"):
            out["direction"] = got["direction"]; filled.append("direction")
        if out["risk"] is None and got.get("risk") in ("low", "medium", "high"):
            out["risk"] = got["risk"]; filled.append("risk")
        if out["objective"] is None and got.get("objective") in ("risk_adjusted", "probability", "expected_return"):
            out["objective"] = got["objective"]; filled.append("objective")
        if out["vol_shift"] is None and got.get("vol_view") in ("up", "down"):
            out["vol_shift"] = 0.02 if got["vol_view"] == "up" else -0.02; filled.append("vol view")
        if filled:
            out["read_by"] = f"rules + {getattr(llm, 'model', 'model')} ({', '.join(filled)})"
    return out


def build_request(text: str, market: dict, llm=None, default_capital: Optional[float] = None) -> dict:
    """Turn the text into a recommender request.

    ``market``: {"spot", "forward", "atm_vol", "as_of" (date), "expiries" [date, ...]} for the underlying.
    Returns {"request": {...recommend kwargs...}, "interpretation": {...}, "assumptions": [...]}.
    """
    text = (text or "").strip()
    if len(text) < 8:
        raise IntentError("Describe the idea in a sentence, e.g. \"I have 2 lakh and expect NIFTY to reach 24,500 "
                          "by next Friday, low risk\".")
    spot, fwd, atm = float(market["spot"]), float(market["forward"]), float(market["atm_vol"])
    today = market["as_of"]
    exps = sorted(e for e in market["expiries"] if e > today) or sorted(market["expiries"])
    assumptions: list[str] = []

    capital = parse_capital(text)
    levels = parse_levels(text, spot)
    horizon, horizon_text = parse_horizon(text, today)
    words = classify(text, llm)
    if capital is None and not levels and not horizon_text and not any(
            words[k] is not None for k in ("direction", "risk", "objective", "vol_shift", "families")):
        raise IntentError("Couldn't find a capital, a level, a direction, a date or a risk level in that. Try e.g. "
                          "\"2 lakh, NIFTY to 24,500 by next Friday, low risk\".")
    if capital is None:
        capital = default_capital or 200000.0
        assumptions.append(f"No capital given: used ₹{capital:,.0f}.")

    # expiry: the first one on or after the horizon (or the nearest weekly)
    if horizon:
        if horizon > exps[-1]:
            assumptions.append(f"The horizon {horizon:%d %b} is beyond the listed expiries: used {exps[-1]:%d %b}.")
            horizon = exps[-1]
        expiry = next(e for e in exps if e >= horizon)
        if horizon <= today:
            horizon = None
            assumptions.append("A same-day view can't be modelled to a horizon: used the expiry.")
    else:
        expiry = exps[0]
        assumptions.append(f"No date given: used the nearest expiry, {expiry:%d %b %Y}.")
    years = max(((horizon or expiry) - today).days, 1) / 365.0
    sd = fwd * atm * math.sqrt(years)

    direction = words["direction"]
    view = {"source": "blend", "iv_shift": words["vol_shift"] or 0.0}
    if horizon and horizon != expiry:
        view["horizon"] = horizon.isoformat()
    families = words["families"]
    min_prob = 0.5
    if "range" in levels:
        lo, hi = levels["range"]
        view.update(source="user", target=(lo + hi) / 2,
                    vol=max((hi - lo) / 2 / (fwd * math.sqrt(years)), 0.02))
        direction = "neutral"
        assumptions.append(f"Range {lo:,.0f}–{hi:,.0f} read as a one-standard-deviation band around {(lo + hi) / 2:,.0f}.")
    elif "target" in levels or "pct" in levels:
        target = levels.get("target") or spot * (1 + levels["pct"])
        if "pct" in levels:
            assumptions.append(f"A {levels['pct'] * 100:+.1f}% move from spot {spot:,.0f} is a target of {target:,.0f}.")
        view.update(source="user", target=round(target, 2))
        direction = direction or ("up" if target > fwd else "down")
        min_prob = 0.4
    elif direction in ("up", "down"):
        target = fwd + (0.75 if direction == "up" else -0.75) * sd
        view.update(source="user", target=round(target, 2))
        min_prob = 0.4
        assumptions.append(f"No level given: a {direction} view is read as three-quarters of an expected move, "
                           f"to {target:,.0f}.")
    elif direction == "neutral":
        view.update(source="user", target=round(fwd, 2), vol=round(atm * 0.7, 4))
        assumptions.append("A sideways view is read as the index near the forward, moving 30% less than implied.")
    elif direction == "volatile":
        view.update(source="user", target=round(fwd, 2), vol=round(atm * 1.3, 4))
        families = families or ["buy", "debit"]
        min_prob = 0.3
        assumptions.append("A big-move view is read as volatility 30% above implied, no direction.")
    else:
        assumptions.append("No direction given: used the blended market/model view.")

    risk = words["risk"] or "medium"
    if words["risk"] is None:
        assumptions.append("No risk level given: medium (max loss up to half the capital).")
    risk_budget = {"low": 0.25, "medium": 0.5, "high": 0.8}[risk]
    if families is None:
        families = ["buy", "debit", "hedged", "futures"] + (["naked"] if risk == "high" else [])
        if risk != "high":
            assumptions.append("Naked option selling left out (only with a high-risk request).")
    if words.get("no_naked") and "naked" in families:
        families = [f for f in families if f != "naked"]
    if view["iv_shift"]:
        assumptions.append(f"Implied volatility assumed to move {view['iv_shift'] * 100:+.0f} points by the horizon.")
    objective = words["objective"] or "risk_adjusted"

    request = {"capital": round(capital, 2), "view": view, "expiry": expiry.isoformat(), "families": families,
               "risk_budget": risk_budget, "objective": objective, "min_prob_profit": min_prob, "top": 6}
    interpretation = {
        "capital": capital, "direction": direction or "none", "target": view.get("target"),
        "range": levels.get("range"), "horizon": (horizon or expiry).isoformat(), "horizon_text": horizon_text,
        "expiry": expiry.isoformat(), "risk": risk, "objective": objective, "vol_shift": view["iv_shift"],
        "families": families, "read_by": words["read_by"], "spot": spot, "expected_move": sd,
    }
    return {"request": request, "interpretation": interpretation, "assumptions": assumptions}


def strategy_from_text(svc, text: str, underlying: str = "NIFTY", llm=None,
                       get: Optional[Callable[[str, str], str]] = None) -> dict:
    """Parse the text and run the recommender. If nothing passes, retry once without the probability floor."""
    surf = svc.surface(underlying, skip_today=True)
    sm = surf.smiles[0]
    market = {"spot": surf.spot, "forward": sm.forward, "atm_vol": sm.slice.atm_vol(), "as_of": surf.as_of.date(),
              "expiries": list(svc.expiries(underlying))}
    if llm is None and get is not None:
        try:
            from .insights import insights_llm
            llm = insights_llm(get)
        except Exception:  # noqa: BLE001 - rules alone are enough
            llm = None
    built = build_request(text, market, llm)
    r = built["request"]
    result = svc.recommend(r["capital"], r["view"], underlying, r["expiry"], r["families"], r["risk_budget"],
                           r["objective"], r["top"], r["min_prob_profit"])
    if not result.get("ideas") or not any(i.get("recommended") for i in result["ideas"]):
        if r["min_prob_profit"] > 0:
            r["min_prob_profit"] = 0.0
            built["assumptions"].append("Nothing met the chance-of-profit floor, so it was removed: "
                                        "read each idea's probability carefully.")
            result = svc.recommend(r["capital"], r["view"], underlying, r["expiry"], r["families"],
                                   r["risk_budget"], r["objective"], r["top"], 0.0)
    built["result"] = result
    built["text"] = text
    return built
