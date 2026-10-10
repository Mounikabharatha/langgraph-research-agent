"""Streamlit front end for the research agent.

    streamlit run app.py

The graph is the same one the CLI drives - this file adds no agent logic.
It opens the checkpointer per action because Streamlit re-executes this whole
script on every interaction, so nothing may be held open across reruns.
"""

from __future__ import annotations

import logging
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from research_agent.graph import DEFAULT_DB, compiled_graph, report_timeline
from research_agent.memory import describe, forget, open_store, status as memory_status
from research_agent.tracing import graph_config, status as tracing_status

logging.getLogger("google_genai").setLevel(logging.ERROR)
load_dotenv()

st.set_page_config(
    page_title="Research Agent",
    page_icon="🔎",
    layout="centered",
    # "auto" hides the sidebar on narrow viewports, which silently loses the
    # saved-runs list on smaller windows. Be explicit.
    initial_sidebar_state="expanded",
)

APPROVALS = {"approve", "approved", "ok", "yes"}


# --------------------------------------------------------------------------
# Graph access. Each helper opens and closes its own connection.
# --------------------------------------------------------------------------
def read_state(thread_id: str):
    with compiled_graph(DEFAULT_DB) as app:
        return app.get_state(graph_config(thread_id))


def read_timeline(thread_id: str) -> list[dict]:
    with compiled_graph(DEFAULT_DB) as app:
        return report_timeline(app, graph_config(thread_id))


def drive(payload, thread_id: str, status) -> None:
    """Stream the graph forward, writing each node update into `status`."""
    config = graph_config(thread_id)
    with compiled_graph(DEFAULT_DB) as app:
        for chunk in app.stream(payload, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    continue
                status.write(f"**{node}** — {summarise(update)}")


def summarise(update) -> str:
    if not isinstance(update, dict):
        return str(update)[:120]
    bits = []
    for key, value in update.items():
        if isinstance(value, list):
            bits.append(f"{key}: {len(value)}")
        elif isinstance(value, str):
            bits.append(f"{key}: {value[:70]}…" if len(value) > 70 else f"{key}: {value}")
        else:
            bits.append(f"{key}: {value}")
    return " · ".join(bits)


REVIEW_GATE = "human_review"


def list_threads() -> list[dict]:
    """Every saved thread, with what it followed up on.

    `next_node` is None when the run finished. Otherwise it names the node the
    run stopped before - which is only "human_review" for a genuine pause. Any
    other value means the run died there, and must not be offered as a draft
    to approve.
    """
    import sqlite3

    try:
        con = sqlite3.connect(DEFAULT_DB)
        ids = [r[0] for r in con.execute("select distinct thread_id from checkpoints")]
        con.close()
    except sqlite3.OperationalError:
        return []

    out = []
    with compiled_graph(DEFAULT_DB) as app:
        for tid in ids:
            snap = app.get_state(graph_config(tid))
            carried = next(
                (m for m in (snap.values.get("recalled") or []) if m.get("carried")),
                None,
            )
            out.append({
                "id": tid,
                "question": snap.values.get("question", "?"),
                "next": snap.next[0] if snap.next else None,
                "parent": snap.values.get("parent_thread"),
                # Runs made before parent_thread existed still know which
                # question they followed up on, so fall back to matching that.
                "parent_question": (carried or {}).get("question"),
            })

    by_question = {item["question"]: item["id"] for item in out}
    for item in out:
        if not item["parent"] and item["parent_question"]:
            match = by_question.get(item["parent_question"])
            if match and match != item["id"]:
                item["parent"] = match
    return out


def render_conversation(values: dict, timeline: list[dict]) -> None:
    """Show the whole exchange: question, answer, every revision asked for.

    Reconstructed from the checkpoints, so it survives a reload and shows
    earlier versions that a rewrite replaced on screen.
    """
    with st.chat_message("user"):
        st.markdown(values.get("question", ""))

    for i, entry in enumerate(timeline):
        if entry["instruction"]:
            with st.chat_message("user"):
                st.markdown(f"*{entry['instruction']}*")

        latest = i == len(timeline) - 1
        with st.chat_message("assistant"):
            if latest:
                st.markdown(entry["text"])
            else:
                # Superseded, but kept. Say WHY it was replaced: a version with
                # no instruction behind it came from the critic's own retry
                # loop, not from anything the reader asked for.
                why = (
                    "you asked for a change"
                    if timeline[i + 1]["instruction"]
                    else "the critic sent it back for more research"
                )
                with st.expander(
                    f"Earlier version · {len(entry['text'])} characters · "
                    f"replaced because {why}"
                ):
                    st.markdown(entry["text"])


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
st.session_state.setdefault("thread_id", None)
st.session_state.setdefault("error", None)


def start(
    question: str,
    max_revisions: int,
    parent: dict | None = None,
    length: str = "auto",
    parent_thread: str | None = None,
) -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.error = None
    payload = {"question": question, "max_revisions": max_revisions, "length": length}
    if parent:
        # Hand the previous report over deliberately. recall_node keeps it and
        # adds anything semantic search finds on top.
        payload["recalled"] = [parent]
    if parent_thread:
        payload["parent_thread"] = parent_thread
    with st.status("Researching…", expanded=True) as status:
        try:
            drive(payload, st.session_state.thread_id, status)
            status.update(label="Ready for review", state="complete")
        except Exception as exc:
            st.session_state.error = f"{type(exc).__name__}: {exc}"
            status.update(label="Stopped", state="error")


def delete_run(thread_id: str, question: str, also_forget: bool) -> None:
    """Remove a saved run, and optionally the memory it left behind."""
    with compiled_graph(DEFAULT_DB) as app:
        app.checkpointer.delete_thread(thread_id)
    if also_forget:
        with open_store() as store:
            forget(store, question)
    st.session_state.thread_id = None


def resume(feedback: str) -> None:
    st.session_state.error = None
    with st.status("Finalising…", expanded=True) as status:
        try:
            drive(Command(resume=feedback), st.session_state.thread_id, status)
            status.update(label="Done", state="complete")
        except Exception as exc:
            st.session_state.error = f"{type(exc).__name__}: {exc}"
            status.update(label="Stopped", state="error")


# --------------------------------------------------------------------------
# Sidebar. This must come before the main content - a `with st.sidebar` block
# placed after it renders nothing at all. Staleness after a new run is handled
# by st.rerun() once the run finishes, not by moving this.
# --------------------------------------------------------------------------
with st.sidebar:
    if st.button("➕  New question", use_container_width=True, type="primary"):
        st.session_state.thread_id = None
        st.session_state.error = None
        st.session_state.pop("question_box", None)
        st.rerun()

    st.subheader("Saved runs")
    threads = list_threads()
    if not threads:
        st.caption("Nothing yet — ask something.")

    by_id = {t["id"]: t for t in threads}
    children: dict[str, list[dict]] = {}
    roots: list[dict] = []
    for item in threads:
        # A follow-up whose parent was deleted is shown as a root, not lost.
        if item["parent"] and item["parent"] in by_id:
            children.setdefault(item["parent"], []).append(item)
        else:
            roots.append(item)

    def icon(item: dict) -> str:
        if item["next"] is None:
            return "✓"
        return "⏸" if item["next"] == REVIEW_GATE else "⚠️"

    def entry(item: dict, depth: int = 0) -> None:
        # Streamlit buttons cannot be indented with CSS here, so use
        # figure-spaces to make the depth of a follow-up chain visible.
        prefix = ("\u2007" * 2 * depth + "↳ ") if depth else ""
        width = max(16, 30 - 2 * depth)
        label = f"{icon(item)} {prefix}{item['question'][:width]}"
        if st.button(label, key=f"t_{item['id']}", use_container_width=True):
            st.session_state.thread_id = item["id"]
            st.session_state.error = None
            st.rerun()
        for child in children.get(item["id"], []):
            entry(child, depth + 1)

    for root in reversed(roots):
        entry(root)

    st.divider()
    st.caption(tracing_status())
    st.caption(memory_status())
    st.caption(
        "Gemini's free tier allows ~20 requests per day **per model**. "
        "One run costs 3–5. Change `GEMINI_MODEL` in `.env` for a fresh "
        "allowance."
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("🔎 Research Agent")
st.caption(
    "Plans sub-questions → searches the web → writes a cited answer → "
    "critiques itself and retries → waits for your approval."
)

with st.form("ask"):
    question = st.text_area(
        "What do you want researched?",
        placeholder="How do LangGraph checkpointers differ from LangChain memory?",
        height=90,
        # Without a key the box empties on submit and the question vanishes.
        key="question_box",
    )
    opts, retries = st.columns([3, 1])
    length = opts.radio(
        "Answer length",
        ["auto", "brief", "detailed"],
        horizontal=True,
        help=(
            "auto: match the question — a one-fact question gets a one-line "
            "answer. brief: always 3 sentences or fewer. detailed: always "
            "thorough."
        ),
    )
    max_revisions = retries.number_input("Max retries", 0, 5, 2)
    submitted = st.form_submit_button("Research", type="primary", use_container_width=True)

if submitted:
    if question.strip():
        start(question.strip(), int(max_revisions), length=length)
        # The stream above was the live feedback; rerun now so the review UI
        # and the sidebar both reflect the thread that was just created.
        st.rerun()
    else:
        st.warning("Type a question first.")

if st.session_state.error:
    st.error(st.session_state.error)

# --------------------------------------------------------------------------
# Whatever thread is selected, render its current state
# --------------------------------------------------------------------------
if st.session_state.thread_id:
    snap = read_state(st.session_state.thread_id)
    values = snap.values
    st.divider()

    # The question itself was never shown here - only truncated in the
    # sidebar - so it was easy to lose track of what was being answered.
    st.markdown(f"### {values.get('question', '(no question)')}")

    parent = next(
        (m for m in (values.get("recalled") or []) if m.get("carried")), None
    )
    if parent:
        st.caption(f"↳ follow-up to: *{parent.get('question', '')}*")

    a, b, c = st.columns(3)
    a.metric("Sub-questions", len(values.get("plan", [])))
    b.metric("Sources found", len(values.get("findings", [])))
    c.metric("Critic passes", values.get("revision", 0))

    if values.get("recalled"):
        with st.expander(f"♻️ Recalled from earlier research ({len(values['recalled'])})"):
            st.caption(
                "Earlier reports given to the planner so it does not redo work "
                "you already have. They are not cited as sources."
            )
            for m in values["recalled"]:
                st.markdown(f"**{m.get('question', '?')}**  ·  {describe(m)}")

    if values.get("plan"):
        with st.expander("What it searched for"):
            for q in values["plan"]:
                st.markdown(f"- {q}")
            if values.get("gaps"):
                st.markdown("**Follow-ups the critic asked for:**")
                for g in values["gaps"]:
                    st.markdown(f"- {g}")

    if values.get("findings"):
        with st.expander(f"Sources ({len(values['findings'])})"):
            for i, f in enumerate(values["findings"], start=1):
                if f["url"]:
                    st.markdown(f"{i}. [{f['title']}]({f['url']})")
                else:
                    st.markdown(f"{i}. ⚠️ {f['title']} — {f['snippet'][:120]}")

    report = values.get("final_report")
    draft = values.get("draft")

    stopped_at = snap.next[0] if snap.next else None

    if stopped_at and stopped_at != REVIEW_GATE:
        # Not a pause - the run failed at this node and has no draft.
        st.warning(
            f"This run stopped at **{stopped_at}** before producing a draft — "
            "usually an API key or quota problem at the time. There is nothing "
            "to approve. Ask the question again to start a fresh run."
        )

    elif stopped_at == REVIEW_GATE:
        edits = values.get("edits", 0)
        render_conversation(values, read_timeline(st.session_state.thread_id))
        if values.get("critique") and not edits:
            st.info(f"**Critic:** {values['critique']}")

        st.divider()
        st.markdown("#### Your turn")
        if edits:
            st.caption(
                f"You have asked for {edits} change{'s' if edits > 1 else ''}. "
                "Keep going, or approve when you are happy — there is no limit."
            )
        notes = st.text_area(
            "Revision notes — how should the draft above be rewritten?",
            placeholder="Make it shorter. Drop the last section. Add trade-offs.",
            height=80,
            # Key on the edit count so the box empties after each round
            # instead of re-submitting the previous instruction.
            key=f"notes_{st.session_state.thread_id}_{values.get('edits', 0)}",
            help=(
                "This rewrites the draft using the sources already gathered. "
                "It cannot look anything new up — for that, ask a new question."
            ),
        )
        ok, redo = st.columns(2)

        if ok.button("✅ Approve", type="primary", use_container_width=True):
            if notes.strip():
                # Approving would silently discard what they typed.
                st.warning(
                    "You have written revision notes. Press **Request changes** "
                    "to apply them, or clear the box to approve the draft as-is."
                )
            else:
                resume("approve")
                st.rerun()

        if redo.button("✏️ Request changes", use_container_width=True):
            if notes.strip():
                resume(notes.strip())
                st.rerun()
            else:
                st.warning("Write what you want changed, or press Approve.")

    elif report:
        render_conversation(values, read_timeline(st.session_state.thread_id))
        st.success(
            "✅ Approved — this question is closed. Nothing more will be "
            "researched for it."
        )

        st.download_button(
            "Download as Markdown",
            report,
            file_name="research-report.md",
            mime="text/markdown",
        )

        # ---- Ask a follow-up ----
        st.divider()
        st.markdown("#### Start a follow-up question")
        st.caption(
            "This closed question stays as it is. A follow-up opens a **new** "
            "one underneath it in the sidebar, carrying this report along so "
            "you can say \"it\" without repeating yourself."
        )
        follow_up = st.text_input(
            "Follow-up question",
            placeholder="What does it cost? How does it compare to the alternatives?",
            label_visibility="collapsed",
            key=f"fu_{st.session_state.thread_id}",
        )
        if st.button("🔎 Research follow-up", use_container_width=True):
            if follow_up.strip():
                start(
                    follow_up.strip(),
                    int(max_revisions),
                    length=length,
                    parent_thread=st.session_state.thread_id,
                    parent={
                        "question": values.get("question", ""),
                        "report": report,
                        "carried": True,
                    },
                )
                st.rerun()
            else:
                st.warning("Type a follow-up question first.")

    # ---- Delete ----
    st.divider()
    with st.expander("🗑️ Delete this run"):
        st.caption(
            "Removes the run and its checkpoints. Cannot be undone. The report "
            "it saved to long-term memory is separate — tick below to drop that too."
        )
        also_forget = st.checkbox("Also forget it from long-term memory")
        if st.button("Delete permanently", type="secondary"):
            delete_run(st.session_state.thread_id, values.get("question", ""), also_forget)
            st.rerun()

    st.caption(f"thread `{st.session_state.thread_id}`")
