"""Verify the .env is complete and that the APIs actually respond.

    python scripts/check_setup.py

Key values are never printed - only whether they load, masked.
"""

import os
import sys

import logging

from dotenv import load_dotenv

logging.getLogger("google_genai").setLevel(logging.ERROR)
load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "google").strip().lower()
LLM_KEY = {"google": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY"}.get(PROVIDER)


def show(name: str) -> bool:
    value = os.getenv(name, "")
    if not value or value.endswith("..."):
        print(f"  {name:18} MISSING - still a placeholder")
        return False
    print(f"  {name:18} loaded  ({value[:4]}...{value[-2:]}, {len(value)} chars)")
    return True


print(f"provider: {PROVIDER}\n")

if LLM_KEY is None:
    sys.exit(
        "LLM_PROVIDER in your .env is not a supported value "
        f"({len(PROVIDER)} chars). It must be exactly 'google' or 'openai' - "
        "it is the provider name, not a key."
    )

ok = show(LLM_KEY) & show("TAVILY_API_KEY")
if not ok:
    sys.exit("\nFill in the missing key(s) in .env, then run this again.")

print(f"\ntesting {PROVIDER} ...", end=" ", flush=True)
try:
    sys.path.insert(0, "src")
    from research_agent.llm import get_llm

    reply = get_llm().invoke("Reply with the single word: OK")
    print(f"WORKS  -> {reply.text.strip()[:40]!r}")
except Exception as exc:
    print(f"FAILED\n   {type(exc).__name__}: {str(exc)[:400]}")
    sys.exit(1)

print("testing tavily ...", end=" ", flush=True)
try:
    from research_agent.tools import search

    hits = search("what is LangGraph")
    print(f"WORKS  -> {len(hits)} result(s), first: {hits[0]['title'][:50]!r}")
except Exception as exc:
    print(f"FAILED\n   {type(exc).__name__}: {str(exc)[:400]}")
    sys.exit(1)

from research_agent.tracing import status as tracing_status

print(f"\n{tracing_status()}")
print("\nEverything is working. You are ready to run the agent.")
