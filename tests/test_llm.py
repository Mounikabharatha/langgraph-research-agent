"""Credential handling in the LLM factory."""

import os


def test_clean_secret_strips_invisible_characters():
    """A key copied from a web page can arrive with an invisible separator.

    This is not hypothetical: it cost a recording session. The key looks
    correct in every editor and 401s on every call.
    """
    from research_agent.llm import clean_secret

    assert clean_secret("AQ.abc123 ") == "AQ.abc123"
    assert clean_secret("﻿AQ.abc123") == "AQ.abc123"
    assert clean_secret("AQ.abc123 ") == "AQ.abc123"
    assert clean_secret("  AQ.abc123\n") == "AQ.abc123"
    # A clean key must come back untouched.
    assert clean_secret("AQ.abc123") == "AQ.abc123"


def test_require_repairs_the_environment(monkeypatch):
    """Cleaning our own copy is not enough - the SDKs read os.environ."""
    from research_agent.llm import _require

    monkeypatch.setenv("GOOGLE_API_KEY", "AQ.abc123 ")
    _require("GOOGLE_API_KEY", "hint")
    assert os.environ["GOOGLE_API_KEY"] == "AQ.abc123"
