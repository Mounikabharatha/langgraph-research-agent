"""The shared State object that flows through every node of the graph.

In LangGraph a node is just `state -> partial_state_update`. The graph merges
each update back into this TypedDict. Fields marked with `Annotated[..., add]`
use a *reducer*: instead of overwriting, new values are appended. That is what
lets the research node accumulate findings across several loop iterations.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class Finding(TypedDict):
    """One piece of evidence gathered for one sub-question."""

    sub_question: str
    title: str
    url: str
    snippet: str


class ResearchState(TypedDict, total=False):
    # --- input ---
    question: str

    # --- planning ---
    plan: list[str]

    # Reducer: findings ACCUMULATE instead of being replaced, so a second
    # research pass adds to the first rather than wiping it out.
    findings: Annotated[list[Finding], operator.add]

    # --- writing / self-correction ---
    draft: str
    critique: str
    gaps: list[str]
    revision: int
    max_revisions: int

    # --- human-in-the-loop ---
    human_feedback: str

    # --- output ---
    final_report: str
