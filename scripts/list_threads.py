"""List every saved thread and whether it is finished or paused.

    python scripts/list_threads.py
"""

import sqlite3

from research_agent.graph import DEFAULT_DB, compiled_graph

con = sqlite3.connect(DEFAULT_DB)
ids = [r[0] for r in con.execute("select distinct thread_id from checkpoints")]
con.close()

if not ids:
    raise SystemExit("No threads saved yet.")

with compiled_graph(DEFAULT_DB) as app:
    print(f"{len(ids)} thread(s) in {DEFAULT_DB}:\n")
    for tid in ids:
        snap = app.get_state({"configurable": {"thread_id": tid}})
        question = (snap.values.get("question") or "?")[:44]
        if snap.next:
            status = f"PAUSED at {snap.next[0]}  <-- resumable"
        else:
            status = "finished"
        print(f"  {tid}")
        print(f"      {status}")
        print(f"      q: {question}\n")
