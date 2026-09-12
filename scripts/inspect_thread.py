"""Show the saved checkpoint history for a thread.

    python scripts/inspect_thread.py <thread_id>

Proves the agent's state really is persisted step by step on disk.
"""

import sys

from research_agent.graph import DEFAULT_DB, compiled_graph

if len(sys.argv) < 2:
    sys.exit("usage: python scripts/inspect_thread.py <thread_id>")

config = {"configurable": {"thread_id": sys.argv[1]}}

with compiled_graph(DEFAULT_DB) as app:
    snapshots = list(app.get_state_history(config))
    if not snapshots:
        sys.exit("No checkpoints found for that thread id.")

    print(f"{len(snapshots)} checkpoints saved, newest first:\n")
    for snap in snapshots:
        step = snap.metadata.get("step", "?")
        source = snap.metadata.get("source", "?")
        nxt = ", ".join(snap.next) if snap.next else "END"
        keys = sorted(k for k, v in snap.values.items() if v not in (None, "", [], 0))
        print(f"  step {str(step):>3}  [{source:>6}]  next: {nxt:<14} state: {', '.join(keys) or '-'}")
