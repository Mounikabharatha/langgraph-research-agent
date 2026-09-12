"""Single place where the LLM is constructed, so it is easy to swap providers.

Set LLM_PROVIDER in .env to choose. Everything else in the graph is unaware of
which model is behind it - nodes just call `get_llm()`.
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

DEFAULT_PROVIDER = "google"

DEFAULT_MODELS = {
    "google": "gemini-3.1-flash-lite",
    "openai": "gpt-4o-mini",
}


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """Return the chat model configured via environment variables."""
    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()

    if provider == "google":
        return _google(temperature)
    if provider == "openai":
        return _openai(temperature)

    raise ValueError(
        f"Unknown LLM_PROVIDER {provider!r}. Supported: {', '.join(DEFAULT_MODELS)}."
    )


def _google(temperature: float) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    _require("GOOGLE_API_KEY", "Create a free key at https://aistudio.google.com/apikey")

    model = os.getenv("GEMINI_MODEL", DEFAULT_MODELS["google"])

    # Gemini 3.x uses fixed sampling defaults and warns on every call if you
    # pass temperature. Only send it to models that actually honour it.
    kwargs = {} if model.startswith("gemini-3") else {"temperature": temperature}

    return ChatGoogleGenerativeAI(model=model, **kwargs)


def _openai(temperature: float) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    _require("OPENAI_API_KEY", "Add credit at https://platform.openai.com/settings/organization/billing")

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODELS["openai"]),
        temperature=temperature,
    )


def _require(var: str, hint: str) -> None:
    value = os.getenv(var, "")
    if not value or value.endswith("..."):
        raise RuntimeError(f"{var} is not set in your .env file. {hint}")
