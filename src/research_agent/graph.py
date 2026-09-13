"""Wire the nodes into a StateGraph and compile it.

Compare this file with the code samples in the roadmap PDF. The real API is:

    builder = StateGraph(MyState)
    builder.add_node("name", fn)
    builder.add_edge(START, "name")
    app = builder.compile()          # <- compile() RETURNS the runnable
    app.invoke(state)                # <- there is no create_runtime()
"""

from __future__ import annotations

from contextlib import contextmanager

from langgraph.graph import END, START, StateGraph

from .memory import open_store
from .nodes import (
    critique_node,
    finalize_node,
    human_review_node,
    plan_node,
    recall_node,
    remember_node,
    research_node,
    route_after_critique,
    synthesize_node,
)
from .state import ResearchState

DEFAULT_DB = "checkpoints.db"


def build_graph() -> StateGraph:
    """Assemble the graph. Kept separate from compile() so tests can inspect it."""
    builder = StateGraph(ResearchState)

    builder.add_node("recall", recall_node)
    builder.add_node("plan", plan_node)
    builder.add_node("research", research_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("critique", critique_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalize", finalize_node)
    builder.add_node("remember", remember_node)

    builder.add_edge(START, "recall")
    builder.add_edge("recall", "plan")
    builder.add_edge("plan", "research")
    builder.add_edge("research", "synthesize")
    builder.add_edge("synthesize", "critique")

    # The self-correction loop. `path_map` lists every node the router may
    # return, which lets LangGraph draw and validate the graph.
    builder.add_conditional_edges(
        "critique",
        route_after_critique,
        path_map={"research": "research", "human_review": "human_review"},
    )

    builder.add_edge("human_review", "finalize")
    builder.add_edge("finalize", "remember")
    builder.add_edge("remember", END)

    return builder


@contextmanager
def compiled_graph(db_path: str = DEFAULT_DB):
    """Compile the graph with SQLite checkpointing.

    Two kinds of memory are attached here, and they do different jobs:

    - `checkpointer` is short-term. It saves every state mutation for one
      thread, so an interrupted run resumes in a later process and any past
      step can be replayed by thread_id.
    - `store` is long-term. It holds finished reports from *every* run,
      searchable semantically, so a new question can benefit from research
      done weeks ago. It is None when memory is switched off.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    with SqliteSaver.from_conn_string(db_path) as checkpointer, open_store() as store:
        yield build_graph().compile(checkpointer=checkpointer, store=store)


def draw_mermaid() -> str:
    """Return a Mermaid diagram of the graph - handy for your README."""
    return build_graph().compile().get_graph().draw_mermaid()
