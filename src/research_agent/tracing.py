"""LangSmith tracing.

There is no code required to *enable* tracing - LangChain picks it up from the
environment. (The roadmap PDF's `Runtime.invoke_with_observability()` does not
exist.) What this module adds is the part that makes traces actually useful:
a name, tags and metadata on every run, so a trace can be tied back to the
thread that produced it.

Env vars, read by langsmith in this order (`LANGSMITH_*` wins over
`LANGCHAIN_*`):

    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=lsv2_...
    LANGSMITH_PROJECT=research-agent

Note that langsmith caches these reads, so `.env` must be loaded before the
first graph call - both entry points do that at startup.
"""

from __future__ import annotations

import os

PROJECT_DEFAULT = "research-agent"


def tracing_enabled() -> bool:
    """True if LangSmith tracing is switched on and has a key to send with."""
    from langsmith.utils import tracing_is_enabled

    return bool(tracing_is_enabled()) and bool(api_key())


def api_key() -> str | None:
    return os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")


def project() -> str:
    return (
        os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or PROJECT_DEFAULT
    )


def status() -> str:
    """One human-readable line about the current tracing setup."""
    from langsmith.utils import tracing_is_enabled

    if not tracing_is_enabled():
        return "tracing OFF (set LANGSMITH_TRACING=true to enable)"
    if not api_key():
        return "tracing requested but LANGSMITH_API_KEY is missing - nothing will be sent"
    return f"tracing ON -> project {project()!r}"


def graph_config(thread_id: str, *, run_name: str = "research", **metadata) -> dict:
    """The config every graph call should be given.

    `configurable.thread_id` is what the checkpointer keys on. The rest is
    purely for observability: without a run_name every trace in LangSmith is
    called "LangGraph", and without the thread_id in metadata there is no way
    to line a trace up with a row in checkpoints.db.
    """
    config: dict = {"configurable": {"thread_id": thread_id}}

    if not tracing_enabled():
        return config

    config["run_name"] = run_name
    config["tags"] = ["research-agent", os.getenv("LLM_PROVIDER", "google")]
    config["metadata"] = {
        "thread_id": thread_id,
        "model": os.getenv("GEMINI_MODEL", os.getenv("OPENAI_MODEL", "unknown")),
        **{k: v for k, v in metadata.items() if v is not None},
    }
    return config
