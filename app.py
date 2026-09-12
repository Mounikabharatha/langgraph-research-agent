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

from research_agent.graph import DEFAULT_DB, compiled_graph

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
        return app.get_state({"configurable": {"thread_id": thread_id}})


def drive(payload, thread_id: str, status) -> None:
    """Stream the graph forward, writing each node update into `status`."""
    config = {"configurable": {"thread_id": thread_id}}
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


def list_threads() -> list[tuple[str, str, bool]]:
    """Return (thread_id, question, is_paused) for every saved thread."""
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
            snap = app.get_state({"configurable": {"thread_id": tid}})
            out.append((tid, snap.values.get("question", "?"), bool(snap.next)))
    return out


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
st.session_state.setdefault("thread_id", None)
st.session_state.setdefault("error", None)


def start(question: str, max_revisions: int) -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.error = None
    with st.status("Researching…", expanded=True) as status:
        try:
            drive(
                {"question": question, "max_revisions": max_revisions},
                st.session_state.thread_id,
                status,
            )
            status.update(label="Ready for review", state="complete")
        except Exception as exc:
            st.session_state.error = f"{type(exc).__name__}: {exc}"
            status.update(label="Stopped", state="error")


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
    st.subheader("Saved runs")
    threads = list_threads()
    if not threads:
        st.caption("Nothing yet — run something.")
    for tid, question, paused in reversed(threads):
        label = ("⏸ " if paused else "✓ ") + (question[:34] or "?")
        if st.button(label, key=f"t_{tid}", use_container_width=True):
            st.session_state.thread_id = tid
            st.session_state.error = None
            st.rerun()

    st.divider()
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
    )
    left, right = st.columns([3, 1])
    max_revisions = right.number_input("Max retries", 0, 5, 2)
    submitted = left.form_submit_button("Research", type="primary", use_container_width=True)

if submitted:
    if question.strip():
        start(question.strip(), int(max_revisions))
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

    a, b, c = st.columns(3)
    a.metric("Sub-questions", len(values.get("plan", [])))
    b.metric("Sources found", len(values.get("findings", [])))
    c.metric("Critic passes", values.get("revision", 0))

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

    if snap.next:  # paused at the review gate
        st.subheader("Draft for review")
        if values.get("critique"):
            st.info(f"**Critic:** {values['critique']}")
        st.markdown(draft or "")

        st.divider()
        notes = st.text_area(
            "Revision notes (leave empty to approve as-is)",
            placeholder="Make it shorter. Add a section on trade-offs.",
            height=80,
        )
        ok, redo = st.columns(2)
        if ok.button("✅ Approve", type="primary", use_container_width=True):
            resume("approve")
            st.rerun()
        if redo.button("✏️ Request changes", use_container_width=True):
            if notes.strip():
                resume(notes.strip())
                st.rerun()
            else:
                st.warning("Write what you want changed, or press Approve.")

    elif report:
        st.subheader("Final report")
        st.markdown(report)
        st.download_button(
            "Download as Markdown",
            report,
            file_name="research-report.md",
            mime="text/markdown",
        )

    st.caption(f"thread `{st.session_state.thread_id}`")
