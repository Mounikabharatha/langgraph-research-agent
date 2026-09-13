"""Long-term memory: what the agent remembers *between* runs.

The checkpointer (see `graph.py`) is short-term memory - it remembers one
thread so a paused run can resume. This module is the other half the roadmap
PDF describes: a Store holding finished research, searchable semantically, so
a question asked next week can benefit from research done today.

    checkpointer -> "where was I in THIS run?"       (thread-scoped)
    store        -> "have I researched this before?" (across all runs)
"""

from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Iterator, TypedDict

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

NAMESPACE = ("research", "reports")
DEFAULT_STORE_DB = "memory.db"
DEFAULT_EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIMS = 3072

# Cosine similarity runs high even for unrelated text. Measured with
# gemini-embedding-001: "explain reducers in langgraph" scores 0.874 against a
# stored LangGraph question and 0.697 against "how to bake sourdough bread".
# Without a floor, every past report counts as relevant.
MIN_RELEVANCE = 0.80


class Memory(TypedDict):
    question: str
    report: str
    score: float


def memory_enabled() -> bool:
    """Memory is opt-out, but it needs the same key the embeddings use."""
    if os.getenv("MEMORY_ENABLED", "true").strip().lower() in {"0", "false", "no"}:
        return False
    return bool(os.getenv("GOOGLE_API_KEY"))


def min_relevance() -> float:
    try:
        return float(os.getenv("MEMORY_MIN_RELEVANCE", MIN_RELEVANCE))
    except ValueError:
        return MIN_RELEVANCE


def get_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)
    )


@contextmanager
def open_store(db_path: str | None = None) -> Iterator["BaseStore | None"]:
    """Yield a semantic store, or None if memory is off or unavailable.

    Yielding None rather than raising is deliberate: losing long-term memory
    should degrade the agent, not stop it. The run still works, it just starts
    cold.
    """
    if not memory_enabled():
        yield None
        return

    from langgraph.store.sqlite import SqliteStore

    path = db_path or os.getenv("MEMORY_DB", DEFAULT_STORE_DB)
    index = {"dims": EMBED_DIMS, "embed": get_embeddings(), "fields": ["question"]}

    # Only the SETUP is guarded. Wrapping the yield in try/except would catch
    # exceptions thrown in by the caller's block and then yield a second time,
    # which raises "generator didn't stop after throw()" and hides the real error.
    stack = ExitStack()
    try:
        store = stack.enter_context(SqliteStore.from_conn_string(path, index=index))
        store.setup()
    except Exception:
        stack.close()
        yield None
        return

    with stack:
        yield store


def recall(store: "BaseStore | None", question: str, limit: int = 3) -> list[Memory]:
    """Find past reports semantically close to this question."""
    if store is None or not question.strip():
        return []

    floor = min_relevance()
    try:
        hits = store.search(NAMESPACE, query=question, limit=limit)
    except Exception:
        return []  # a dead embedding API must not stop the research

    return [
        Memory(
            question=hit.value.get("question", ""),
            report=hit.value.get("report", ""),
            score=round(hit.score, 3),
        )
        for hit in hits
        if hit.score is not None and hit.score >= floor
    ]


def remember(
    store: "BaseStore | None", key: str, question: str, report: str
) -> bool:
    """Save a finished report. Returns whether it was actually stored."""
    if store is None or not report.strip():
        return False
    try:
        store.put(NAMESPACE, key, {"question": question, "report": report})
        return True
    except Exception:
        return False


def status() -> str:
    if not memory_enabled():
        return "long-term memory OFF (set MEMORY_ENABLED=true and GOOGLE_API_KEY)"
    return (
        f"long-term memory ON -> {os.getenv('MEMORY_DB', DEFAULT_STORE_DB)} "
        f"(relevance floor {min_relevance()})"
    )
