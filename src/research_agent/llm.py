"""Single place where the LLM is constructed, so it is easy to swap providers.

Set LLM_PROVIDER in .env to choose. Everything else in the graph is unaware of
which model is behind it - nodes just call `get_llm()`.

Gemini's free tier is counted PER MODEL PER DAY, so when one model is
exhausted another still has a full allowance. `get_llm()` therefore returns a
chain: if the first model returns 429 the call is retried on the next one,
automatically. For a deployed demo this is the difference between the first
visitor using up the day and twenty of them getting an answer.
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

DEFAULT_PROVIDER = "google"

DEFAULT_MODELS = {
    "google": "gemini-3.1-flash-lite",
    "openai": "gpt-4o-mini",
}

# Tried in order when the one before it runs out of free quota. Each entry
# carries its own separate daily allowance.
GEMINI_FALLBACKS = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
]


class OutOfQuota(RuntimeError):
    """Every configured model refused the call, almost always 429."""


def get_llm(temperature: float = 0.0, schema: type | None = None) -> Runnable:
    """A chat model, or a chain of them that fails over on exhausted quota.

    Pass `schema` to get validated structured output. It must be applied to
    each model BEFORE the fallbacks are attached, otherwise only the first
    model in the chain would be structured.
    """
    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()

    if provider == "google":
        names = [os.getenv("GEMINI_MODEL", DEFAULT_MODELS["google"])]
        names += [n for n in GEMINI_FALLBACKS if n not in names]
        runnables = [_google(name, temperature) for name in names]
    elif provider == "openai":
        runnables = [_openai(temperature)]
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER {provider!r}. Supported: {', '.join(DEFAULT_MODELS)}."
        )

    if schema is not None:
        runnables = [r.with_structured_output(schema) for r in runnables]

    first, *rest = runnables
    return first.with_fallbacks(rest) if rest else first


def model_names() -> list[str]:
    """The models that will be tried, in order. Used for error messages."""
    if os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower() != "google":
        return [os.getenv("OPENAI_MODEL", DEFAULT_MODELS["openai"])]
    first = os.getenv("GEMINI_MODEL", DEFAULT_MODELS["google"])
    return [first] + [n for n in GEMINI_FALLBACKS if n != first]


def _google(model: str, temperature: float) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    _require("GOOGLE_API_KEY", "Create a free key at https://aistudio.google.com/apikey")

    # Gemini 3.x uses fixed sampling defaults and warns on every call if you
    # pass temperature. Only send it to models that actually honour it.
    kwargs = {} if model.startswith("gemini-3") else {"temperature": temperature}
    return ChatGoogleGenerativeAI(model=model, **kwargs)


def _openai(temperature: float) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    _require(
        "OPENAI_API_KEY",
        "Add credit at https://platform.openai.com/settings/organization/billing",
    )
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODELS["openai"]),
        temperature=temperature,
    )


def _require(var: str, hint: str) -> None:
    value = os.getenv(var, "")
    if not value or value.endswith("..."):
        raise RuntimeError(f"{var} is not set in your .env file. {hint}")
