"""The agent's one real tool: web search.

`search()` is deliberately a plain function returning plain dicts. That keeps
the nodes testable - a test can monkeypatch this without touching LangGraph.
"""

from __future__ import annotations

import os
from functools import lru_cache

from .state import Finding

MAX_RESULTS_PER_QUERY = 3


@lru_cache(maxsize=1)
def _tavily():
    from langchain_tavily import TavilySearch

    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Get a free key at https://tavily.com "
            "and put it in your .env file."
        )
    return TavilySearch(max_results=MAX_RESULTS_PER_QUERY)


def search(sub_question: str) -> list[Finding]:
    """Search the web for one sub-question and normalise the results."""
    raw = _tavily().invoke({"query": sub_question})

    # TavilySearch does NOT raise on API errors - it returns {"error": ...}.
    # Swallowing that would make a bad API key look like "no results found",
    # which is the worst kind of failure: silent and misleading.
    if isinstance(raw, dict) and raw.get("error"):
        raise RuntimeError(f"Tavily search failed: {raw['error']}")

    # Recent versions return {"results": [...]}, older ones a bare list.
    # Handle both so a dependency bump does not break us.
    results = raw.get("results", []) if isinstance(raw, dict) else raw

    return [
        Finding(
            sub_question=sub_question,
            title=item.get("title", "Untitled"),
            url=item.get("url", ""),
            snippet=(item.get("content") or "")[:600],
        )
        for item in results
    ]
