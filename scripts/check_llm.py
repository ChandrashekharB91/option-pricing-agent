"""Check that the LLM configured in .env can run the agent.

    python scripts/check_llm.py
    python scripts/check_llm.py --list-models     (which model ids your key can use)
    python scripts/check_llm.py --question "Price 2 lots of the 22500 put expiring 2026-10-27"

Runs one real question through the full agent on the recorded 23 Oct 2026 snapshot (no market data
needed) and reports: did the model call the pricing tool with the right arguments, did it answer, are all
its numbers traceable to tool results, and what did the validator say. Use it to qualify a model before
the team relies on it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optlib_app.agent import AgentUnavailable, OptionPricingAgent  # noqa: E402
from optlib_app.config import Settings, get_setting, llm_status  # noqa: E402
from optlib_app.llm import LLMError, make_llm  # noqa: E402
from optlib_app.service import PricingService  # noqa: E402

DEFAULT_Q = "Price 1 lot of the NIFTY 22500 call expiring 2026-10-27 and give me delta, gamma, vega and theta."


def _print_models() -> bool:
    try:
        llm = make_llm(get_setting)
        lister = getattr(llm, "list_models", None)
        if lister is None:
            print("  (this provider has no model listing here; see its console for model ids)")
            return False
        ids = lister()
    except LLMError as exc:
        print(f"  could not list models: {exc}")
        return False
    for m in ids:
        print(f"  {m}")
    return bool(ids)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--question", default=DEFAULT_Q)
    ap.add_argument("--list-models", action="store_true", help="list the model ids your key can use, then stop")
    args = ap.parse_args(argv)

    st = llm_status()
    if not st["configured"]:
        print(f"NOT CONFIGURED: {st['message']}")
        return 1
    print(f"LLM: {st['provider']} / {st['model']}")
    if args.list_models:
        return 0 if _print_models() else 1
    try:
        agent = OptionPricingAgent(PricingService(Settings(data_mode="replay")))
    except AgentUnavailable as exc:
        print(f"NOT CONFIGURED: {exc}")
        return 1

    t0 = time.time()
    reply = agent.ask(args.question)
    secs = time.time() - t0
    if reply.answer.startswith("The LLM API failed"):
        print(f"API ERROR after {secs:.1f} s: {reply.answer[len('The LLM API failed: '):]}")
        print("Check LLM_API_KEY, LLM_MODEL and LLM_BASE_URL (and for Ollama, that the server is running).")
        if "404" in reply.answer or "does not exist" in reply.answer:
            print("The model id was not found for this key. Models this key can use:")
            _print_models()
            print("Put one of these in .env as LLM_MODEL (prefer a large instruction model that supports tool use).")
        return 1
    calls = reply.tool_calls
    priced = [c for c in calls if c["name"] == "price_option" and not c["error"]]

    print("-" * 78)
    print(f"Q: {args.question}")
    print(f"A: {reply.answer}")
    print("-" * 78)
    print(f"Time                {secs:.1f} s")
    print(f"Tool calls          {len(calls)}: " + ", ".join(
        f"{c['name']}({'error' if c['error'] else 'ok'})" for c in calls))
    checks = {
        "called price_option": bool(priced),
        "right arguments": any(c["input"].get("strike") == 22500 and c["input"].get("option_type") == "call"
                               and str(c["input"].get("expiry", "")).startswith("2026-10-27") for c in priced)
        if args.question == DEFAULT_Q else bool(priced),
        "answered": bool(reply.answer.strip()) and not reply.answer.startswith("The LLM API failed"),
        "numbers grounded": not reply.ungrounded_numbers,
        "validator verdict parsed": reply.validator_verdict in ("PASS", "WARN", "FAIL"),
    }
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL':<5} {name}")
    if reply.ungrounded_numbers:
        print(f"      numbers not found in any tool result: {', '.join(reply.ungrounded_numbers)}")
    print(f"Validator           {reply.validator_verdict}: {reply.validator_review[:300]}")
    print(f"Overall status      {reply.status} ({'released' if reply.released else 'withheld'})")
    ok = all(checks.values())
    print("=" * 78)
    print("This model can run the agent." if ok else
          "This model struggled; try a larger tool-capable model or another provider.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
