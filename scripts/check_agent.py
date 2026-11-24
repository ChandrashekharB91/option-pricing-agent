"""Run the AI analyst once from the command line and print each step, the model used and the checks.

    python scripts/check_agent.py                    # the morning desk brief
    python scripts/check_agent.py "Price the ATM straddle on the weekly expiry"

Uses the same settings as the app (.env / environment). Never prints API keys.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optlib_app.agent import AgentUnavailable, OptionPricingAgent, agent_llm  # noqa: E402
from optlib_app.service import PricingService  # noqa: E402

BRIEF = ("Prepare today's NIFTY options desk brief. Check the market data first and say if anything is unreliable. "
         "Then cover: where implied volatility sits against the past year and against realised vol, the expected "
         "move to the nearest weekly expiry, positioning from open interest, the price and Greeks of 1 lot of the "
         "at-the-money straddle on that expiry, and the main news themes. Finish with strategy ideas for Rs 2 lakh "
         "if I expect NIFTY to stay in a range until that expiry.")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    task = " ".join(sys.argv[1:]) or BRIEF
    try:
        llm = agent_llm()
    except Exception as exc:  # noqa: BLE001
        print(f"No model: {exc}")
        return 1
    chain = getattr(llm, "models", None) or [getattr(llm, "model", "?")]
    print(f"Model chain: {getattr(llm, 'provider', '?')} {' -> '.join(chain)}")
    t0 = time.time()

    def show(e):
        k = e["type"]
        if k == "plan":
            print(f"[{time.time() - t0:5.1f}s] round {e['turn']} via {e.get('model', '')}: {', '.join(e['tools'])}")
        elif k == "tool_done":
            print(f"          {e['name']}: {e.get('summary') or ''} {e.get('error') or ''} [{e['status']}]")
        elif k == "validated":
            print(f"[{time.time() - t0:5.1f}s] validator: {e['verdict']}")

    try:
        reply = OptionPricingAgent(PricingService()).ask(task, on_event=show)
    except AgentUnavailable as exc:
        print(f"Agent unavailable: {exc}")
        return 1
    print(f"\nDone in {time.time() - t0:.0f} s · answered by {reply.model} · status {reply.status} · "
          f"numbers not traced: {reply.ungrounded_numbers or 'none'}\n")
    print(reply.answer)
    return 0 if reply.status != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
