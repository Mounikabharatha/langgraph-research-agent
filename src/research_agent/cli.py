"""Command-line entry point.

    python -m research_agent.cli "your question here"
    python -m research_agent.cli --graph          # print the Mermaid diagram
"""

from __future__ import annotations

import argparse
import sys
import uuid

from dotenv import load_dotenv
from langgraph.types import Command

from .graph import DEFAULT_DB, compiled_graph, draw_mermaid


def main(argv: list[str] | None = None) -> int:
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

    if not args.question:
        parser.error("a question is required (or use --graph)")

    thread_id = args.thread or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    with compiled_graph(args.db) as app:
        print(f"thread_id: {thread_id}\n")

        # Stream so you can watch each node fire instead of staring at a
        # blank terminal for 30 seconds.
        for chunk in app.stream(
            {"question": args.question, "max_revisions": args.max_revisions},
            config,
            stream_mode="updates",
        ):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    continue
                print(f"  [{node}] {_summarise(update)}")

        # The run stopped at human_review. Show the draft and ask.
        state = app.get_state(config)
        if state.next:
            draft = state.values.get("draft", "")
            print("\n" + "=" * 70)
            print("DRAFT FOR REVIEW")
            print("=" * 70)
            print(draft)
            print("=" * 70)
            print(f"Critic said: {state.values.get('critique', '-')}\n")

            feedback = input("Type 'approve', or your revision notes: ").strip()

            # Resume exactly where interrupt() left off.
            for chunk in app.stream(Command(resume=feedback), config, stream_mode="updates"):
                for node, update in chunk.items():
                    if node != "__interrupt__":
                        print(f"  [{node}] {_summarise(update)}")

        final = app.get_state(config).values.get("final_report", "")
        print("\n" + "=" * 70)
        print("FINAL REPORT")
        print("=" * 70)
        print(final)

    return 0


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
