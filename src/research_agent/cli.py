"""Command-line entry point.

    python -m research_agent.cli "your question here"
    python -m research_agent.cli --graph            # print the Mermaid diagram
    python -m research_agent.cli --thread <id>      # resume a paused run
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid

from dotenv import load_dotenv
from langgraph.types import Command

from .graph import DEFAULT_DB, compiled_graph, draw_mermaid
from .tracing import graph_config, status as tracing_status

RULE = "=" * 70
APPROVALS = {"", "approve", "approved", "ok", "yes"}


def main(argv: list[str] | None = None) -> int:
    logging.getLogger("google_genai").setLevel(logging.ERROR)
    load_dotenv()

    parser = argparse.ArgumentParser(description="LangGraph research assistant")
    parser.add_argument("question", nargs="?", help="the question to research")
    parser.add_argument("--graph", action="store_true", help="print the Mermaid diagram and exit")
    parser.add_argument("--thread", default=None, help="resume an existing thread id")
    parser.add_argument("--db", default=DEFAULT_DB, help="checkpoint database path")
    parser.add_argument("--max-revisions", type=int, default=2)
    args = parser.parse_args(argv)

    if args.graph:
        print(draw_mermaid())
        return 0

    if not args.question and not args.thread:
        parser.error("a question is required (or --thread to resume, or --graph)")

    thread_id = args.thread or str(uuid.uuid4())
    config = graph_config(thread_id, question=args.question)

    with compiled_graph(args.db) as app:
        print(f"thread_id: {thread_id}")
        print(f"{tracing_status()}\n")

        state = app.get_state(config)
        paused = bool(state.next)

        if paused:
            # Already stopped at the review gate - do NOT feed new input, that
            # would restart the graph and throw away the saved work.
            print("resuming a paused run - the research was already done\n")
        elif state.created_at and not state.next:
            if args.thread and not args.question:
                print("this thread already finished. Its report:\n")
                print(state.values.get("final_report", "(none)"))
                return 0
            print("starting a fresh run on this thread\n")

        if not paused:
            if not args.question:
                parser.error("this thread is not paused; pass a question to start a new run")
            _stream(app, {"question": args.question, "max_revisions": args.max_revisions}, config)
            state = app.get_state(config)

        # Stopped at the human review gate.
        if state.next:
            print(f"\n{RULE}\nDRAFT FOR REVIEW\n{RULE}")
            print(state.values.get("draft", ""))
            print(RULE)
            print(f"Critic said: {state.values.get('critique', '-')}\n")

            try:
                feedback = input("Type 'approve', or your revision notes: ").strip()
            except (EOFError, KeyboardInterrupt):
                print(
                    f"\n\nStopped without deciding. Nothing is lost - every step is "
                    f"checkpointed.\nResume any time with:\n\n"
                    f"    python -m research_agent.cli --thread {thread_id}"
                    + (f" --db {args.db}" if args.db != DEFAULT_DB else "")
                    + "\n"
                )
                return 130

            _stream(app, Command(resume=feedback), config)

        final = app.get_state(config).values.get("final_report", "")
        print(f"\n{RULE}\nFINAL REPORT\n{RULE}")
        print(final)

    return 0


def _stream(app, payload, config) -> None:
    for chunk in app.stream(payload, config, stream_mode="updates"):
        for node, update in chunk.items():
            if node != "__interrupt__":
                print(f"  [{node}] {_summarise(update)}")


def _summarise(update: object) -> str:
    if not isinstance(update, dict):
        return str(update)[:100]
    parts = []
    for key, value in update.items():
        if isinstance(value, list):
            parts.append(f"{key}={len(value)} items")
        elif isinstance(value, str):
            parts.append(f"{key}={value[:60]!r}" if len(value) > 60 else f"{key}={value!r}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
