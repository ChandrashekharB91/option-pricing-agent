"""Check the AI market note end to end: which model is used, whether it answers, and whether the note passes.

    python scripts/check_ai_note.py

Reads the same settings as the app (.env / environment). Never prints the API key.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optlib_app import insights  # noqa: E402
from optlib_app.config import get_setting  # noqa: E402
from optlib_app.feed import build_feed  # noqa: E402
from optlib_app.llm import LLMNotConfigured  # noqa: E402
from optlib_app.news import headlines  # noqa: E402
from optlib_app.service import PricingService  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    key = get_setting("GEMINI_API_KEY", "")
    print(f"GEMINI_API_KEY: {'set (' + str(len(key)) + ' characters)' if key else 'not set'}")
    if key:
        shape = []
        if any(ch.isspace() for ch in key):
            shape.append("contains a space: remove it")
        if "#" in key:
            shape.append("contains '#': remove any comment after the key")
        if key[:1] in "'\"" or key[-1:] in "'\"":
            shape.append("has a stray quote")
        print("  key format: " + ("; ".join(shape) if shape else
                                  f"starts with {key[:4]!r}, no spaces or quotes"))
    try:
        llm = insights.insights_llm(get_setting)
    except LLMNotConfigured as exc:
        print(f"No model for the note: {exc}")
        return 1
    print(f"Model: {getattr(llm, 'provider', '?')} · {getattr(llm, 'model', '?')}")
    try:
        reply = llm.chat("Reply with the single word OK.", [{"role": "user", "content": "ping"}], tools=None,
                         max_tokens=insights.MAX_TOKENS)
        print(f"Connection: OK (model replied {reply.text[:40]!r})")
    except Exception as exc:  # noqa: BLE001
        print(f"Connection FAILED: {exc}")
        return 1
    print("Building today's numbers…")
    feed = build_feed(PricingService())
    news = headlines(8, get=get_setting)
    print(f"Headlines: {len(news)}")
    note = insights.market_note(feed, news, llm=llm)
    if note:
        print("\nAI NOTE (passed the checks):\n" + note["text"])
        return 0
    print(f"\nNo note published. Reason: {insights.LAST_PROBLEM['text'] or 'unknown'}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
