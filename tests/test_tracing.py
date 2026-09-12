"""Tracing config tests. No LangSmith account needed.

langsmith caches env var reads with lru_cache, so every test here clears that
cache first - otherwise the second test in the file reads the first one's
environment and passes for the wrong reason.
"""

from __future__ import annotations

import pytest

from research_agent import tracing

TRACING_VARS = [
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGCHAIN_PROJECT",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    from langsmith.utils import get_env_var

    for var in TRACING_VARS:
        monkeypatch.delenv(var, raising=False)
    get_env_var.cache_clear()
    yield
    get_env_var.cache_clear()


def enable(monkeypatch, **extra):
    from langsmith.utils import get_env_var

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_fake")
    for k, v in extra.items():
        monkeypatch.setenv(k, v)
    get_env_var.cache_clear()


# --------------------------------------------------------------------------
# graph_config
# --------------------------------------------------------------------------
def test_config_is_minimal_when_tracing_is_off():
    """thread_id is all the checkpointer needs; no observability cruft."""
    assert tracing.graph_config("t1") == {"configurable": {"thread_id": "t1"}}


def test_config_carries_thread_id_when_tracing_is_on(monkeypatch):
    enable(monkeypatch)
    config = tracing.graph_config("t1", question="why?")

    assert config["configurable"]["thread_id"] == "t1"
    assert config["run_name"] == "research"
    assert "research-agent" in config["tags"]
    # Without this you cannot line a LangSmith trace up with checkpoints.db.
    assert config["metadata"]["thread_id"] == "t1"
    assert config["metadata"]["question"] == "why?"


def test_config_drops_none_metadata(monkeypatch):
    enable(monkeypatch)
    config = tracing.graph_config("t1", question=None)
    assert "question" not in config["metadata"]


def test_tracing_needs_a_key_not_just_the_flag(monkeypatch):
    """LANGSMITH_TRACING=true with no key sends nothing - do not pretend."""
    from langsmith.utils import get_env_var

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    get_env_var.cache_clear()

    assert tracing.tracing_enabled() is False
    assert "missing" in tracing.status()


# --------------------------------------------------------------------------
# status / project
# --------------------------------------------------------------------------
def test_status_off_by_default():
    assert "OFF" in tracing.status()


def test_status_names_the_project(monkeypatch):
    enable(monkeypatch, LANGSMITH_PROJECT="my-project")
    assert "my-project" in tracing.status()


def test_langchain_prefixed_vars_still_work(monkeypatch):
    """langsmith falls back from LANGSMITH_* to LANGCHAIN_*; so must we."""
    from langsmith.utils import get_env_var

    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_fake")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "legacy-project")
    get_env_var.cache_clear()

    assert tracing.tracing_enabled() is True
    assert tracing.project() == "legacy-project"
