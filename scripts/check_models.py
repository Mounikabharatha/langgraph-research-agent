"""Check every model in the fallback chain actually works.

    python scripts/check_models.py

Google retires models without much warning, and a model appearing in the API's
own list is not proof it can be called - `gemini-2.5-flash` was listed and
returned 404. Run this if answers start failing, and prune `GEMINI_FALLBACKS`
in llm.py accordingly.

Each check costs one request against that model's daily quota.
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("google_genai").setLevel(logging.ERROR)

from dotenv import load_dotenv

load_dotenv()

from research_agent.llm import model_names
from research_agent.nodes import Plan

from langchain_google_genai import ChatGoogleGenerativeAI

print("Checking the fallback chain. Each model is asked for structured output,\n"
      "which is what the planner and critic need.\n")

working, broken, exhausted = [], [], []
for name in model_names():
    try:
        plan = ChatGoogleGenerativeAI(model=name).with_structured_output(Plan).invoke(
            "Break this into search queries: what does a PS5 cost?"
        )
        print(f"  {name:24} OK          ({len(plan.sub_questions)} queries)")
        working.append(name)
    except Exception as exc:
        text = str(exc)
        if "RESOURCE_EXHAUSTED" in text or "429" in text:
            print(f"  {name:24} quota spent (today)")
            exhausted.append(name)
        else:
            print(f"  {name:24} BROKEN      {type(exc).__name__}: {text[:60]}")
            broken.append(name)

print(f"\n  {len(working)} working, {len(exhausted)} exhausted, {len(broken)} broken")
if broken:
    print("  Remove the broken ones from GEMINI_FALLBACKS in src/research_agent/llm.py")
sys.exit(1 if broken else 0)
