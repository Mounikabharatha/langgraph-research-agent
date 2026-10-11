"""The nodes. Each one is a pure-ish function: state in, partial update out.

This is the heart of the project. Read it top to bottom and you understand
the whole agent:

    recall -> plan -> research -> synthesize -> critique -+-> human_review
                          ^                                |
                          +------ (gaps found, retry) ------+
                                                           |
                                       finalize -> remember +
"""

from __future__ import annotations

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from .llm import get_llm
from .memory import memory_key, recall, remember
from .state import Finding, Memory, ResearchState
from .tools import search

# Maximum number of CRITIC PASSES, not retries: the first pass always
# happens, so a value of 2 permits one trip back to research. The UI calls
# this "Max critic passes" so the setting and the metric shown afterwards
# are the same number.
DEFAULT_MAX_REVISIONS = 2

# What counts as "I'm happy with this" at the review gate.
APPROVALS = {"", "approve", "approved", "ok", "yes", "looks good", "lgtm"}

# Above this similarity a recalled memory is effectively the SAME question
# asked again, not merely a related one. The distinction matters: telling the
# planner to avoid ground an earlier answer covered is right for a related
# question and actively wrong for a repeat, where it would steer the search
# away from the very thing being asked.
SAME_QUESTION_SIMILARITY = 0.93

# How much of a carried-over report to show the planner. Enough to resolve
# "it" / "that one", and no more: a longer excerpt pulls the plan back
# towards the earlier topic instead of the question actually being asked.
PARENT_EXCERPT_CHARS = 400


# --------------------------------------------------------------------------
# Structured-output schemas. `with_structured_output` makes the LLM return
# validated objects instead of a blob of text we would have to regex.
# --------------------------------------------------------------------------
class Plan(BaseModel):
    sub_questions: list[str] = Field(
        description=(
            "Search queries. Use as few as the question needs: 1-2 for a "
            "narrow factual question, 3-5 only for a genuinely broad one."
        ),
        min_length=1,
        max_length=6,
    )


class Critique(BaseModel):
    is_sufficient: bool = Field(description="True if the draft fully answers the question")
    reasoning: str = Field(description="One or two sentences explaining the verdict")
    gaps: list[str] = Field(
        default_factory=list,
        description="Follow-up search queries that would close the gaps. Empty if sufficient.",
    )


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
def recall_node(state: ResearchState, *, store: Optional[BaseStore]) -> dict:
    """Look for research we already did on a similar question.

    `store` is injected by LangGraph, which decides by resolving this
    annotation. It must be spelled `Optional[BaseStore]`, NOT
    `BaseStore | None`: with `from __future__ import annotations` the PEP 604
    union stays a string that LangGraph cannot resolve, so it injects nothing
    and the call fails with "missing keyword-only argument". Verified against
    langgraph 1.2 - `BaseStore`, `Optional[BaseStore]` and `BaseStore = None`
    all work; only the `|` form does not.

    When the graph is compiled without a store, LangGraph passes None.
    """
    # A follow-up run is seeded with its parent report already in `recalled`.
    # Keep it, and add anything semantic search turns up that is not a repeat.
    carried = list(state.get("recalled") or [])
    seen = {m.get("question") for m in carried}
    found = [m for m in recall(store, state["question"]) if m.get("question") not in seen]
    return {"recalled": carried + found}


def _recalled_context(recalled: list[Memory]) -> str:
    if not recalled:
        return ""
    parent = [m for m in recalled if m.get("carried")]
    similar = [m for m in recalled if not m.get("carried")]

    parts = []
    if parent:
        # The earlier ANSWER matters as much as the earlier question: without it
        # a follow-up like "what does it cost?" has no idea what "it" is, and
        # the planner falls back to generic searches.
        blocks = "\n\n".join(
            f"Earlier question: {m.get('question', '')}\n"
            f"Earlier answer (extract):\n{(m.get('report') or '')[:PARENT_EXCERPT_CHARS]}"
            for m in parent
        )
        parts.append(
            f"CONTEXT ONLY - an earlier question and answer:\n\n{blocks}\n\n"
            "Use this for ONE purpose: to work out what the new question "
            "refers to. If it says \"it\" or \"that one\", substitute the "
            "actual name into your search queries.\n"
            "Do NOT plan searches about the earlier topic. The new question "
            "governs entirely - if it asks for a definition, search for that "
            "definition and nothing else."
        )
    repeats = [
        m for m in similar
        if (m.get("score") or 0) >= SAME_QUESTION_SIMILARITY
    ]
    related = [m for m in similar if m not in repeats]

    if repeats:
        parts.append(
            "You have answered essentially this same question before:\n"
            + "\n".join(f"- {m.get('question', '')}" for m in repeats)
            + "\nThe reader is asking again, so answer it properly and in full. "
            "Prefer queries that surface CURRENT information, since the earlier "
            "answer may be out of date. Do not avoid the topic."
        )
    if related:
        parts.append(
            "You have also researched these related questions before:\n"
            + "\n".join(
                f"- {m.get('question', '')} (similarity {m.get('score')})" for m in related
            )
            + "\nBias your sub-questions towards what those would NOT already cover."
        )
    return "\n\n" + "\n\n".join(parts) if parts else ""


def plan_node(state: ResearchState) -> dict:
    """Decompose the user's question into concrete searchable sub-questions."""
    llm = get_llm().with_structured_output(Plan)
    plan: Plan = llm.invoke(
        [
            SystemMessage(
                "You are a research planner. Turn the user's question into the "
                "FEWEST self-contained web search queries that would answer it. "
                "A narrow factual question ('what does X cost?') needs 1-2. "
                "Only a genuinely broad question needs 4-5. Over-planning a "
                "small question wastes searches and produces a bloated answer. "
                "Do not answer the question yourself."
            ),
            HumanMessage(
                state["question"] + _recalled_context(state.get("recalled") or [])
            ),
        ]
    )
    return {
        "plan": plan.sub_questions,
        "revision": 0,
        "max_revisions": state.get("max_revisions", DEFAULT_MAX_REVISIONS),
    }


def research_node(state: ResearchState) -> dict:
    """Run a web search for every open sub-question.

    On the first pass this uses `plan`. On a retry it uses `gaps` - the
    follow-up queries the critic asked for.

    One failing query degrades the answer. EVERY query failing means the
    search API is down or the key is wrong, and that is not something to
    write a report about - see the guard at the bottom.
    """
    queries = state.get("gaps") or state["plan"]

    findings: list[Finding] = []
    failures: list[str] = []

    for query in queries:
        try:
            findings.extend(search(query))
        except Exception as exc:  # one dead query must not kill the whole run
            failures.append(f"{query!r} -> {type(exc).__name__}: {exc}")
            findings.append(
                Finding(
                    sub_question=query,
                    title="Search failed",
                    url="",
                    snippet=f"{type(exc).__name__}: {exc}",
                )
            )

    # Total failure is infrastructure, not a thin topic. Carrying on would
    # spend several more LLM calls to produce a confident "no information
    # available" report and never mention that the search API was dead.
    if failures and len(failures) == len(queries):
        raise RuntimeError(
            f"All {len(queries)} searches failed - the search API is "
            "unreachable or TAVILY_API_KEY is invalid. Stopping rather than "
            "writing a report from nothing.\n  " + "\n  ".join(failures)
        )

    # `findings` has an `operator.add` reducer, so this APPENDS to whatever
    # earlier passes already gathered.
    return {"findings": findings}


def _format_findings(findings: list[Finding]) -> str:
    return "\n\n".join(
        f"[{i}] {f['title']}\nURL: {f['url']}\n{f['snippet']}"
        for i, f in enumerate(findings, start=1)
    )


QUESTION_OPENERS = (
    "what", "why", "how", "when", "where", "which", "who", "is", "are",
    "can", "could", "does", "do", "should", "will", "tell me",
)


def looks_like_a_question(text: str) -> bool:
    """True if revision notes read as a new question rather than an edit.

    A revision can only reshape the text already written - it has no way to
    search for anything new. Typing a question here quietly gets you a
    reworded version of the old answer, which is confusing, so the UI warns.
    """
    stripped = text.strip().lower()
    if not stripped:
        return False
    if stripped.endswith("?"):
        return True
    return stripped.startswith(QUESTION_OPENERS)


def _length_rule(state: ResearchState) -> str:
    """How long the answer should be. Without this the writer always writes an
    essay, even when asked what a chair costs."""
    choice = (state.get("length") or "auto").strip().lower()
    if choice == "brief":
        return (
            "Answer in at most 3 sentences, or 3 short bullets. No preamble, no "
            "background, no headings. Lead with the direct answer."
        )
    if choice == "detailed":
        return "Give a thorough answer with headings where they help."
    return (
        "DEFAULT TO SHORT. Most questions are answered well in 1-3 sentences "
        "or 3-5 bullets, and that is what the reader wants. Lead with the "
        "answer itself, not with context. Earn every extra paragraph: only a "
        "question with several genuinely distinct parts justifies headings. "
        "Never pad an answer to look thorough."
    )


def _followup_rule(state: ResearchState) -> str:
    """Stop the writer re-explaining what the reader already read."""
    parent = [m for m in (state.get("recalled") or []) if m.get("carried")]
    if not parent:
        return ""
    return (
        "\n\nThe reader has ALREADY READ an answer to: "
        + "; ".join(m.get("question", "") for m in parent)
        + ".\nAnswer ONLY the question asked now. Do not summarise, compare "
        "against, or reintroduce that earlier topic. Assume the reader "
        "remembers it. If the question asks what something means, define that "
        "one thing and stop."
    )


def synthesize_node(state: ResearchState) -> dict:
    """Write a cited answer from everything gathered so far."""
    llm = get_llm(temperature=0.2)
    response = llm.invoke(
        [
            SystemMessage(
                "Write an answer using ONLY the sources given. "
                "Cite sources inline as [1], [2] matching their numbers. "
                "If the sources do not support a claim, say so rather than guessing. "
                "End with a 'Sources' list of the numbered URLs.\n\n"
                + _length_rule(state)
                + _followup_rule(state)
            ),
            HumanMessage(
                f"Question: {state['question']}\n\n"
                f"Sources:\n{_format_findings(state['findings'])}"
            ),
        ]
    )
    return {"draft": response.text}


def critique_node(state: ResearchState) -> dict:
    """A second LLM acts as a critic. This is what makes the loop worthwhile."""
    llm = get_llm().with_structured_output(Critique)
    verdict: Critique = llm.invoke(
        [
            SystemMessage(
                "You are a strict research reviewer. Decide whether the draft fully "
                "and accurately answers the question using the cited sources. "
                "If not, list specific follow-up search queries that would close "
                "the gaps."
            ),
            HumanMessage(
                f"Question: {state['question']}\n\nDraft:\n{state['draft']}"
            ),
        ]
    )
    return {
        "critique": verdict.reasoning,
        "gaps": [] if verdict.is_sufficient else verdict.gaps,
        "revision": state.get("revision", 0) + 1,
    }


def human_review_node(state: ResearchState) -> dict:
    """Pause the graph and hand control back to a human.

    `interrupt()` raises out of the run and checkpoints everything. The caller
    resumes later with `Command(resume=...)` and execution picks up right here,
    even in a different process. This is LangGraph's real superpower.
    """
    feedback = interrupt(
        {
            "draft": state["draft"],
            "critique": state.get("critique", ""),
            "instructions": "Reply 'approve' to accept, or type revision notes.",
        }
    )
    return {"human_feedback": str(feedback)}


def finalize_node(state: ResearchState) -> dict:
    """Apply the human's notes, if any, and emit the report.

    A rewrite goes back to the review gate so the human can look at it and
    either accept it or ask for more. Only an approval ends the run.
    """
    feedback = (state.get("human_feedback") or "").strip()

    if feedback.lower() in APPROVALS:
        return {"final_report": state["draft"], "approved": True}

    llm = get_llm(temperature=0.2)
    response = llm.invoke(
        [
            SystemMessage(
                "Rewrite the draft to do exactly what the reviewer asked.\n"
                "- If they asked for something SHORTER, CUT material. Removing "
                "whole points is correct; reformatting the same content into "
                "bullets is NOT shorter and does not satisfy the request.\n"
                "- Keep the inline [n] citations on whatever you keep. Dropping "
                "a claim drops its citation with it - that is fine.\n"
                "- Do not add anything the draft did not already contain: you "
                "have no sources beyond it."
            ),
            HumanMessage(f"Draft:\n{state['draft']}\n\nReviewer notes:\n{feedback}"),
        ]
    )
    rewritten = response.text
    # Write the rewrite into `draft` too, so the review gate shows the NEW
    # text rather than the version the reviewer just asked to change.
    return {
        "draft": rewritten,
        "final_report": rewritten,
        "approved": False,
        "edits": state.get("edits", 0) + 1,
    }


def remember_node(state: ResearchState, *, store: Optional[BaseStore]) -> dict:
    """Save the finished report so future runs can recall it.

    As with `recall_node`, `store` must not have a default - see the note
    there.
    """
    remember(
        store,
        # Key on the question, so re-researching it updates the memory
        # instead of piling up near-duplicate entries.
        key=memory_key(state["question"]),
        question=state["question"],
        report=state.get("final_report", ""),
    )
    return {}


# --------------------------------------------------------------------------
# Conditional edge
# --------------------------------------------------------------------------
def route_after_critique(state: ResearchState) -> str:
    """Decide whether to loop back for more research or move on.

    NOTE: a routing function returns the NAME of the next node. It does not
    return a bool, and the graph method is `add_conditional_edges` (plural).
    The roadmap PDF gets both of these wrong.
    """
    gaps = state.get("gaps") or []
    revision = state.get("revision", 0)
    max_revisions = state.get("max_revisions", DEFAULT_MAX_REVISIONS)

    # The step limit is a real safety guard, not decoration: without it a
    # picky critic will loop forever and burn your API budget.
    if gaps and revision < max_revisions:
        return "research"
    return "human_review"


def route_after_finalize(state: ResearchState) -> str:
    """Approved -> save and finish. Revised -> back to the reviewer.

    Unlike the critic loop above this one needs no step limit. Every pass
    requires a human to type something and press a button, so it cannot run
    away on its own; the person doing the reviewing is the guard.
    """
    return "remember" if state.get("approved") else "human_review"
