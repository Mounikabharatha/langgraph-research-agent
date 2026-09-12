"""Tests that run with no API keys - which means they run in CI for free.

The trick is to test the parts that carry the real logic (routing, reducers,
graph wiring) and to fake the parts that cost money (LLM and search calls).
"""

from __future__ import annotations

import pytest

from research_agent import nodes
from research_agent.graph import build_graph
from research_agent.nodes import route_after_critique
from research_agent.state import Finding


# --------------------------------------------------------------------------
# Routing - the conditional edge is where agents most often go wrong
# --------------------------------------------------------------------------
def test_router_moves_on_when_critic_is_satisfied():
    state = {"gaps": [], "revision": 1, "max_revisions": 2}
    assert route_after_critique(state) == "human_review"


def test_router_loops_back_when_gaps_remain():
    state = {"gaps": ["what about X?"], "revision": 1, "max_revisions": 2}
    assert route_after_critique(state) == "research"


def test_router_stops_looping_at_the_revision_limit():
    """The runaway-loop guard. Without this an agent can bill you forever."""
    state = {"gaps": ["still not happy"], "revision": 2, "max_revisions": 2}
    assert route_after_critique(state) == "human_review"


# --------------------------------------------------------------------------
# Graph wiring
# --------------------------------------------------------------------------
def test_graph_compiles_and_has_every_node():
    compiled = build_graph().compile()
    node_names = set(compiled.get_graph().nodes)
    for expected in ("plan", "research", "synthesize", "critique", "human_review", "finalize"):
        assert expected in node_names


def test_graph_renders_a_diagram():
    assert "critique" in build_graph().compile().get_graph().draw_mermaid()


# --------------------------------------------------------------------------
# Node behaviour, with the paid calls faked out
# --------------------------------------------------------------------------
def test_research_node_accumulates_findings(monkeypatch):
    def fake_search(query: str) -> list[Finding]:
        return [Finding(sub_question=query, title="t", url="u", snippet="s")]

    monkeypatch.setattr(nodes, "search", fake_search)

    result = nodes.research_node({"plan": ["q1", "q2", "q3"]})
    assert len(result["findings"]) == 3


def test_research_node_survives_a_failing_tool(monkeypatch):
    """A dead search API should degrade the answer, not crash the agent."""

    def exploding_search(query: str):
        raise ConnectionError("tavily is down")

    monkeypatch.setattr(nodes, "search", exploding_search)

    result = nodes.research_node({"plan": ["q1"]})
    assert len(result["findings"]) == 1
    assert "ConnectionError" in result["findings"][0]["snippet"]


def test_research_node_prefers_gaps_over_the_original_plan(monkeypatch):
    """On a retry the agent should chase the gaps, not redo the same searches."""
    seen: list[str] = []

    def recording_search(query: str) -> list[Finding]:
        seen.append(query)
        return []

    monkeypatch.setattr(nodes, "search", recording_search)

    nodes.research_node({"plan": ["original"], "gaps": ["follow-up"]})
    assert seen == ["follow-up"]


@pytest.mark.parametrize("feedback", ["approve", "OK", "yes", ""])
def test_finalize_passes_the_draft_through_on_approval(feedback):
    """No LLM should be called when the human just says yes."""
    result = nodes.finalize_node({"draft": "the answer", "human_feedback": feedback})
    assert result["final_report"] == "the answer"
