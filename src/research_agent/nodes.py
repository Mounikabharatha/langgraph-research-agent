from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .llm import get_llm
from .state import Finding, ResearchState
from .tools import search



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
    return {"plan": plan.sub_questions}


def research_node(state: ResearchState) -> dict:
    """Run a web search for every sub-question in the plan."""
    queries = state["plan"]

    findings: list[Finding] = []
    for query in queries:
        try:
            findings.extend(search(query))
        except Exception as exc:  # a dead tool must not kill the whole run
            findings.append(
                Finding(
                    sub_question=query,
                    title="Search failed",
                    url="",
                    snippet=f"{type(exc).__name__}: {exc}",
                )
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
    return {"draft": response.content}
