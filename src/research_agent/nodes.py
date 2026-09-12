"""The nodes. Each one is a pure-ish function: state in, partial update out.

This is the heart of the project. Read it top to bottom and you understand
the whole agent:

    plan  ->  research  ->  synthesize  ->  critique  -+-> human_review -> finalize
                  ^                                    |
                  +-------- (gaps found, retry) -------+
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from .llm import get_llm
from .state import Finding, ResearchState
from .tools import search

DEFAULT_MAX_REVISIONS = 2


# --------------------------------------------------------------------------
# Structured-output schemas. `with_structured_output` makes the LLM return
# validated objects instead of a blob of text we would have to regex.
# --------------------------------------------------------------------------
class Plan(BaseModel):
    sub_questions: list[str] = Field(
        description="3 to 5 specific, self-contained search queries",
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
def plan_node(state: ResearchState) -> dict:
    """Decompose the user's question into concrete searchable sub-questions."""
    llm = get_llm().with_structured_output(Plan)
    plan: Plan = llm.invoke(
        [
            SystemMessage(
                "You are a research planner. Break the user's question into 3-5 "
                "specific, self-contained web search queries that together would "
                "answer it. Do not answer the question yourself."
            ),
            HumanMessage(state["question"]),
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


def synthesize_node(state: ResearchState) -> dict:
    """Write a cited answer from everything gathered so far."""
    llm = get_llm(temperature=0.2)
    response = llm.invoke(
        [
            SystemMessage(
                "Write a clear, well-organised answer using ONLY the sources given. "
                "Cite sources inline as [1], [2] matching their numbers. "
                "If the sources do not support a claim, say so rather than guessing. "
                "End with a 'Sources' list of the numbered URLs."
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
    """Apply the human's notes, if any, and emit the final report."""
    feedback = (state.get("human_feedback") or "").strip()

    if feedback.lower() in {"", "approve", "approved", "ok", "yes"}:
        return {"final_report": state["draft"]}

    llm = get_llm(temperature=0.2)
    response = llm.invoke(
        [
            SystemMessage(
                "Revise the draft according to the reviewer's notes. "
                "Keep all inline citations intact."
            ),
            HumanMessage(f"Draft:\n{state['draft']}\n\nReviewer notes:\n{feedback}"),
        ]
    )
    return {"final_report": response.text}


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
