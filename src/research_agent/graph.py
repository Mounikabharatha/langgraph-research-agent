"""Wire the nodes into a StateGraph and compile it.

The real LangGraph API, for the record:

    builder = StateGraph(MyState)
    builder.add_node("name", fn)
    builder.add_edge(START, "name")
    app = builder.compile()          # <- compile() RETURNS the runnable
    app.invoke(state)                # <- there is no create_runtime()
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import plan_node, research_node, synthesize_node
from .state import ResearchState


def build_graph() -> StateGraph:
    """Assemble the graph. Kept separate from compile() so tests can inspect it."""
    builder = StateGraph(ResearchState)

    builder.add_node("plan", plan_node)
    builder.add_node("research", research_node)
    builder.add_node("synthesize", synthesize_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "research")
    builder.add_edge("research", "synthesize")
    builder.add_edge("synthesize", END)

    return builder


def draw_mermaid() -> str:
    """Return a Mermaid diagram of the graph - handy for your README."""
    return build_graph().compile().get_graph().draw_mermaid()
