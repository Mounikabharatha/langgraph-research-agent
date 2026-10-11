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
from pathlib import Path

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
    route_after_finalize,
    synthesize_node,
)
from .state import ResearchState

DEFAULT_DB = "checkpoints.db"

# Databases shipped in the repo so a fresh deploy has something to show.
# Hosted free tiers wipe the filesystem whenever the app sleeps, so without
# this a visitor arriving at a cold app finds an empty sidebar.
DEMO_DIR = Path("demo")


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

    # A rewrite returns to the reviewer; only an approval moves on.
    builder.add_conditional_edges(
        "finalize",
        route_after_finalize,
        path_map={"human_review": "human_review", "remember": "remember"},
    )
    builder.add_edge("remember", END)

    return builder


def seed_demo_data(*databases: str) -> list[str]:
    """Copy the shipped demo databases into place if none exist yet.

    Only ever fills a gap - an existing database is never touched, so this is
    a no-op on a machine that has already been used.
    """
    import shutil

    seeded = []
    for name in databases:
        target, source = Path(name), DEMO_DIR / Path(name).name
        if target.exists() or not source.exists():
            continue
        shutil.copy2(source, target)
        seeded.append(name)
    return seeded


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


def report_timeline(app, config: dict) -> list[dict]:
    """Reconstruct the whole back-and-forth for one thread, oldest first.

    Returns `[{"text": ..., "instruction": ...}]` where `instruction` is the
    revision note that produced that version, or None for the first draft.

    Nothing is ever overwritten in the checkpointer - each revision is its own
    checkpoint - so the full conversation is recoverable after the fact.
    """
    from .nodes import APPROVALS

    timeline: list[dict] = []
    seen: set[str] = set()

    for snapshot in reversed(list(app.get_state_history(config))):
        values = snapshot.values
        feedback = (values.get("human_feedback") or "").strip()
        instruction = feedback if feedback.lower() not in APPROVALS else None

        for field in ("draft", "final_report"):
            text = (values.get(field) or "").strip()
            if text and text not in seen:
                seen.add(text)
                timeline.append({"text": text, "instruction": instruction})

    return timeline


def draft_versions(app, config: dict) -> list[str]:
    """Just the distinct report versions, oldest first."""
    return [entry["text"] for entry in report_timeline(app, config)]


def draw_mermaid() -> str:
    """Return a Mermaid diagram of the graph - handy for your README."""
    return build_graph().compile().get_graph().draw_mermaid()
