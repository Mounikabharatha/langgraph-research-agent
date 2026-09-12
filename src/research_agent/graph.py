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

from .nodes import (
    critique_node,
    finalize_node,
    human_review_node,
    plan_node,
    research_node,
    route_after_critique,
    synthesize_node,
)
from .state import ResearchState

DEFAULT_DB = "checkpoints.db"


def build_graph() -> StateGraph:
    """Assemble the graph. Kept separate from compile() so tests can inspect it."""
    builder = StateGraph(ResearchState)

    builder.add_node("plan", plan_node)
    builder.add_node("research", research_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("critique", critique_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "plan")
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
    builder.add_edge("finalize", END)

    return builder


@contextmanager
def compiled_graph(db_path: str = DEFAULT_DB):
    """Compile the graph with SQLite checkpointing.

    The checkpointer is what makes the agent *durable*: every state mutation is
    saved, so an interrupted run can resume in a later process, and you can
    replay or inspect any past step by thread_id.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        yield build_graph().compile(checkpointer=checkpointer)


def draw_mermaid() -> str:
    """Return a Mermaid diagram of the graph - handy for your README."""
    return build_graph().compile().get_graph().draw_mermaid()
