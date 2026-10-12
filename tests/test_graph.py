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


def test_research_node_survives_one_failing_query(monkeypatch):
    """A single dead query should degrade the answer, not crash the agent."""

    def flaky_search(query: str) -> list[Finding]:
        if query == "bad":
            raise ConnectionError("tavily hiccup")
        return [Finding(sub_question=query, title="t", url="u", snippet="s")]

    monkeypatch.setattr(nodes, "search", flaky_search)

    result = nodes.research_node({"plan": ["good", "bad"]})
    assert len(result["findings"]) == 2
    failed = [f for f in result["findings"] if f["title"] == "Search failed"]
    assert len(failed) == 1
    assert "ConnectionError" in failed[0]["snippet"]


def test_research_node_raises_when_every_query_fails(monkeypatch):
    """Total failure is infrastructure, not a thin topic.

    Continuing would spend several more LLM calls producing a confident
    "no information available" report that never mentions the API was dead.
    """

    def exploding_search(query: str):
        raise ConnectionError("tavily is down")

    monkeypatch.setattr(nodes, "search", exploding_search)

    with pytest.raises(RuntimeError, match="All 3 searches failed"):
        nodes.research_node({"plan": ["q1", "q2", "q3"]})


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


def test_search_raises_instead_of_silently_returning_nothing(monkeypatch):
    """Tavily returns {"error": ...} rather than raising. If we swallowed that,
    a dead API key would be indistinguishable from an empty web."""
    from research_agent import tools

    class FakeTavily:
        def invoke(self, _payload):
            return {"error": ValueError("Error 401: Unauthorized")}

    monkeypatch.setattr(tools, "_tavily", lambda: FakeTavily())

    with pytest.raises(RuntimeError, match="Tavily search failed"):
        tools.search("anything")


# --------------------------------------------------------------------------
# The review gate loops until the human approves
# --------------------------------------------------------------------------
def test_approval_ends_the_run():
    from research_agent.nodes import route_after_finalize

    assert route_after_finalize({"approved": True}) == "remember"


def test_a_rewrite_goes_back_to_the_reviewer():
    """Previously 'Request changes' worked exactly once and the run ended.
    A rewrite now returns to the gate so it can be reviewed in turn."""
    from research_agent.nodes import route_after_finalize

    assert route_after_finalize({"approved": False}) == "human_review"


def test_an_unset_approval_is_not_treated_as_approved():
    from research_agent.nodes import route_after_finalize

    assert route_after_finalize({}) == "human_review"


@pytest.mark.parametrize("word", ["approve", "Approved", "  OK  ", "yes", ""])
def test_finalize_recognises_approval_and_makes_no_llm_call(word):
    result = nodes.finalize_node({"draft": "the answer", "human_feedback": word})
    assert result["final_report"] == "the answer"
    assert result["approved"] is True


def test_finalize_rewrite_updates_the_draft_too(monkeypatch):
    """The gate shows `draft`, so a rewrite must land there as well - otherwise
    the reviewer is shown the version they just asked to change."""

    class FakeLLM:
        def invoke(self, _messages):
            return type("R", (), {"text": "the shorter answer"})()

    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: FakeLLM())

    result = nodes.finalize_node(
        {"draft": "the long answer", "human_feedback": "make it shorter", "edits": 0}
    )

    assert result["draft"] == "the shorter answer"
    assert result["final_report"] == "the shorter answer"
    assert result["approved"] is False
    assert result["edits"] == 1


# --------------------------------------------------------------------------
# Answer length - a one-fact question should not get an essay
# --------------------------------------------------------------------------
def test_auto_lets_the_question_set_the_length():
    rule = nodes._length_rule({})
    assert "LET THE QUESTION SET THE LENGTH" in rule
    assert "first sentence" in rule
    assert "do not truncate" in rule


def test_auto_and_short_ask_for_different_things():
    assert nodes._length_rule({}) != nodes._length_rule({"length": "short"})


def test_short_compresses_rather_than_truncates():
    rule = nodes._length_rule({"length": "short"})
    assert "compressed to its essentials" in rule
    assert "misleading without" in rule


def test_medium_does_not_fix_the_number_of_bullets():
    rule = nodes._length_rule({"length": "medium"})
    assert "as many bullets as the question actually needs" in rule
    assert "Never pad to reach a number" in rule


def test_long_asks_for_depth():
    assert "thorough" in nodes._length_rule({"length": "long"})


@pytest.mark.parametrize("setting", ["auto", "short", "medium", "long"])
def test_no_length_rule_hard_codes_a_count(setting):
    """Length should come from the question. A fixed sentence or bullet count
    makes a small question padded and a large one truncated."""
    import re

    assert not re.search(r"\b\d+\b", nodes._length_rule({"length": setting}))


@pytest.mark.parametrize("old,new", [("brief", "short"), ("detailed", "long")])
def test_runs_recorded_under_the_old_names_still_work(old, new):
    """Sessions saved before the rename must keep rendering and re-running."""
    assert nodes._length_rule({"length": old}) == nodes._length_rule({"length": new})


def test_an_unknown_length_falls_back_to_auto():
    assert nodes._length_rule({"length": "nonsense"}) == nodes._length_rule({})


def test_a_followup_is_told_not_to_repeat_the_background():
    """Without this the writer re-explains everything the reader just read."""
    rule = nodes._followup_rule(
        {"recalled": [{"question": "best gaming laptop", "carried": True}]}
    )
    assert "ALREADY READ" in rule
    assert "Answer ONLY the question asked now" in rule
    assert "best gaming laptop" in rule


def test_a_first_question_gets_no_followup_rule():
    assert nodes._followup_rule({}) == ""


def test_a_merely_similar_memory_is_not_treated_as_a_followup():
    """Only a carried parent means 'the reader already read this'. A memory
    found by similarity does not."""
    rule = nodes._followup_rule({"recalled": [{"question": "q", "score": 0.9}]})
    assert rule == ""


# --------------------------------------------------------------------------
# A revision cannot answer a new question - warn rather than silently reword
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "what is meant by hybrid exactly",
    "Which is better for city driving?",
    "how does regenerative braking work",
    "is it cheaper to run?",
    "Tell me about charging times",
])
def test_question_shaped_notes_are_recognised(text):
    assert nodes.looks_like_a_question(text) is True


@pytest.mark.parametrize("text", [
    "make it much shorter",
    "drop the last section",
    "put it in a table",
    "add a one-line summary at the top",
    "",
])
def test_genuine_edit_instructions_are_not_flagged(text):
    assert nodes.looks_like_a_question(text) is False


def test_the_revision_prompt_permits_cutting(monkeypatch):
    """'Keep ALL inline citations intact' forced the model to retain every
    claim, so 'make it shorter' could only reformat. It must be allowed to
    drop material."""
    captured = {}

    class FakeLLM:
        def invoke(self, messages):
            captured["system"] = messages[0].content
            return type("R", (), {"text": "short"})()

    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: FakeLLM())
    nodes.finalize_node({"draft": "long", "human_feedback": "make it shorter"})

    assert "CUT material" in captured["system"]
    assert "Keep all inline citations intact" not in captured["system"]


# --------------------------------------------------------------------------
# Model fallback - the free tier is counted per model, per day
# --------------------------------------------------------------------------
def test_several_models_are_tried_in_order(monkeypatch):
    from research_agent import llm

    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

    names = llm.model_names()
    assert names[0] == "gemini-3.1-flash-lite"
    assert len(names) > 1, "a single model means one visitor can exhaust the day"
    assert len(names) == len(set(names)), "no duplicates - they share no quota"


def test_the_chosen_model_is_never_duplicated_in_the_chain(monkeypatch):
    from research_agent import llm

    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    # Pick one that is also in the fallback list.
    monkeypatch.setenv("GEMINI_MODEL", llm.GEMINI_FALLBACKS[0])

    names = llm.model_names()
    assert names.count(llm.GEMINI_FALLBACKS[0]) == 1


def test_openai_provider_has_no_gemini_fallbacks(monkeypatch):
    from research_agent import llm

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    assert llm.model_names() == ["gpt-4o-mini"]
