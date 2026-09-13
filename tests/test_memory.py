"""Long-term memory tests. No embeddings API needed - the store is faked."""

from __future__ import annotations

import pytest

from research_agent import memory, nodes


class FakeHit:
    def __init__(self, question, report, score):
        self.value = {"question": question, "report": report}
        self.score = score


class FakeStore:
    """Minimal stand-in for BaseStore."""

    def __init__(self, hits=(), explode=False):
        self._hits = list(hits)
        self.explode = explode
        self.puts: list[tuple] = []

    def search(self, namespace, *, query=None, limit=10):
        if self.explode:
            raise ConnectionError("embeddings API down")
        return self._hits[:limit]

    def put(self, namespace, key, value):
        if self.explode:
            raise ConnectionError("store is down")
        self.puts.append((namespace, key, value))


@pytest.fixture(autouse=True)
def default_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    monkeypatch.delenv("MEMORY_ENABLED", raising=False)
    monkeypatch.delenv("MEMORY_MIN_RELEVANCE", raising=False)


# --------------------------------------------------------------------------
# The relevance floor - the whole feature hinges on this
# --------------------------------------------------------------------------
def test_recall_drops_hits_below_the_relevance_floor():
    """Unrelated text still scores ~0.70 with these embeddings. Without a
    floor, every past report would be treated as relevant."""
    store = FakeStore([
        FakeHit("what is a langgraph reducer", "R1", 0.874),
        FakeHit("how to bake sourdough bread", "R2", 0.697),
    ])

    got = memory.recall(store, "explain reducers in langgraph")

    assert [m["question"] for m in got] == ["what is a langgraph reducer"]


def test_relevance_floor_is_configurable(monkeypatch):
    monkeypatch.setenv("MEMORY_MIN_RELEVANCE", "0.60")
    store = FakeStore([FakeHit("q", "R", 0.697)])
    assert len(memory.recall(store, "anything")) == 1


def test_a_junk_relevance_setting_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("MEMORY_MIN_RELEVANCE", "not-a-number")
    assert memory.min_relevance() == memory.MIN_RELEVANCE


# --------------------------------------------------------------------------
# Degrading rather than failing
# --------------------------------------------------------------------------
def test_recall_without_a_store_returns_nothing():
    assert memory.recall(None, "anything") == []


def test_recall_survives_a_dead_embeddings_api():
    """Losing memory should start the run cold, not stop it."""
    assert memory.recall(FakeStore(explode=True), "anything") == []


def test_remember_reports_whether_it_saved():
    store = FakeStore()
    assert memory.remember(store, "k", "q", "the report") is True
    assert memory.remember(None, "k", "q", "the report") is False
    assert memory.remember(store, "k", "q", "   ") is False
    assert memory.remember(FakeStore(explode=True), "k", "q", "r") is False


def test_memory_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("MEMORY_ENABLED", "false")
    assert memory.memory_enabled() is False
    assert "OFF" in memory.status()


def test_memory_needs_a_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert memory.memory_enabled() is False


# --------------------------------------------------------------------------
# The nodes
# --------------------------------------------------------------------------
def test_recall_node_is_a_no_op_without_a_store():
    assert nodes.recall_node({"question": "q"}, store=None) == {"recalled": []}


def test_remember_node_keys_on_the_question(monkeypatch):
    """The same question asked twice should update one entry, not make two."""
    store = FakeStore()
    state = {"question": "  What Is A Reducer? ", "final_report": "R"}

    nodes.remember_node(state, store=store)
    nodes.remember_node({"question": "what is a reducer?", "final_report": "R2"}, store=store)

    keys = {put[1] for put in store.puts}
    assert len(keys) == 1, "case and whitespace should not create a second entry"


def test_planner_prompt_mentions_prior_research():
    context = nodes._recalled_context(
        [{"question": "what is a reducer", "report": "R", "score": 0.9}]
    )
    assert "already researched" in context
    assert "what is a reducer" in context


def test_planner_context_is_empty_with_no_memories():
    assert nodes._recalled_context([]) == ""


# --------------------------------------------------------------------------
# Regression guard: does LangGraph actually inject the store?
# --------------------------------------------------------------------------
def _one_node_graph(node):
    """Compile a graph containing just this node, so injection is exercised."""
    from langgraph.graph import END, START, StateGraph

    from research_agent.state import ResearchState

    builder = StateGraph(ResearchState)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    return builder


def test_store_is_really_injected_into_recall_node():
    """The annotation on `store` decides whether LangGraph injects it.

    `Optional[BaseStore]` works; `BaseStore | None` does NOT, because with
    `from __future__ import annotations` the PEP 604 union stays an
    unresolvable string. That failure is invisible until runtime, so this
    test compiles a real graph rather than inspecting the signature.
    """
    store = FakeStore([FakeHit("a question from before", "R", 0.95)])
    app = _one_node_graph(nodes.recall_node).compile(store=store)

    out = app.invoke({"question": "something similar"})

    assert out["recalled"], "store was not injected - check the `store` annotation"
    assert out["recalled"][0]["question"] == "a question from before"


def test_store_is_really_injected_into_remember_node():
    store = FakeStore()
    app = _one_node_graph(nodes.remember_node).compile(store=store)

    app.invoke({"question": "q", "final_report": "the report"})

    assert store.puts, "store was not injected - check the `store` annotation"
    assert store.puts[0][2]["report"] == "the report"


def test_graph_still_runs_with_no_store_at_all():
    """Memory off must degrade, not crash."""
    app = _one_node_graph(nodes.recall_node).compile()
    assert app.invoke({"question": "q"})["recalled"] == []
