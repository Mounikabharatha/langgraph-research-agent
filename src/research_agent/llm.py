"""Single place where the LLM is constructed, so it is easy to swap or fake."""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """Return the chat model configured via environment variables.

    Swapping providers is a one-line change here - nothing else in the graph
    needs to know which model is behind it.
    """
    from langchain_openai import ChatOpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=temperature,
    )
