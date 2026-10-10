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

    The opening question is NOT repeated here - it is already the heading
    above - so the thread starts with the answer and any later user bubbles
    are the revisions that were actually asked for.
    """
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


def delete_runs(runs: list[tuple[str, str]], also_forget: bool) -> None:
    """Remove saved runs, and optionally the memories they left behind.

    `runs` is [(thread_id, question)]. Deleting a parent takes its whole
    follow-up chain with it, so this handles several at once.
    """
    with compiled_graph(DEFAULT_DB) as app:
        for thread_id, _ in runs:
            app.checkpointer.delete_thread(thread_id)
    if also_forget:
        with open_store() as store:
            for _, question in runs:
                forget(store, question)
    if st.session_state.thread_id in {tid for tid, _ in runs}:
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

    threads = list_threads()
    st.subheader("Saved runs")
    delete_mode = st.toggle(
        "🗑  Delete mode",
        help="Shows a bin beside each run. Deleting a question also removes "
             "the follow-ups nested under it.",
    )
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

    def subtree(item: dict) -> list[dict]:
        """This run plus every follow-up beneath it, at any depth."""
        out = [item]
        for child in children.get(item["id"], []):
            out.extend(subtree(child))
        return out

    def icon(item: dict) -> str:
        if item["next"] is None:
            return "✓"
        return "⏸" if item["next"] == REVIEW_GATE else "⚠️"

    def entry(item: dict, depth: int = 0) -> None:
        # Streamlit buttons cannot be indented with CSS here, so use
        # figure-spaces to make the depth of a follow-up chain visible.
        prefix = ("\u2007" * 2 * depth + "↳ ") if depth else ""
        width = max(14, 28 - 2 * depth)
        label = f"{icon(item)} {prefix}{item['question'][:width]}"

        if delete_mode:
            open_col, bin_col = st.columns([5, 1])
            if open_col.button(label, key=f"t_{item['id']}", use_container_width=True):
                st.session_state.thread_id = item["id"]
                st.session_state.error = None
                st.rerun()
            if bin_col.button("🗑", key=f"d_{item['id']}", help="Delete this run"):
                st.session_state.pending_delete = item["id"]
                st.rerun()
        else:
            if st.button(label, key=f"t_{item['id']}", use_container_width=True):
                st.session_state.thread_id = item["id"]
                st.session_state.error = None
                st.rerun()

        if st.session_state.get("pending_delete") == item["id"]:
            doomed = subtree(item)
            extra = len(doomed) - 1
            st.warning(
                f"Delete this run{f' and its {extra} follow-up' if extra else ''}"
                f"{'s' if extra > 1 else ''}? This cannot be undone."
            )
            also_forget = st.checkbox(
                "Also forget from long-term memory", key=f"f_{item['id']}"
            )
            yes, no = st.columns(2)
            if yes.button("Delete", key=f"y_{item['id']}", use_container_width=True):
                delete_runs([(d["id"], d["question"]) for d in doomed], also_forget)
                st.session_state.pending_delete = None
                st.rerun()
            if no.button("Cancel", key=f"n_{item['id']}", use_container_width=True):
                st.session_state.pending_delete = None
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
# Main. Two modes: the home page for asking, and the view of one run.
# Keeping the ask form off the run view means the answer is the first
# thing on screen rather than something you scroll to.
# --------------------------------------------------------------------------
st.session_state.setdefault("length", "auto")
st.session_state.setdefault("max_revisions", 2)

STEPS = [
    ("🧠", "Plan", "splits your question up"),
    ("🌐", "Search", "reads the live web"),
    ("✍️", "Write", "answers, with sources"),
    ("🔍", "Critique", "finds its own gaps"),
    ("🔁", "Retry", "searches again if thin"),
    ("🙋", "You decide", "approve or revise"),
]

EXAMPLES = [
    "What is the current AQI in Delhi?",
    "Compare electric and hybrid cars for Indian drivers",
    "What skills do junior AI engineers need in 2026?",
]

if st.session_state.error:
    st.error(st.session_state.error)


# --------------------------------------------------------------------------
# Home
# --------------------------------------------------------------------------
if not st.session_state.thread_id:
    st.title("🔎 Research Agent")
    st.caption(
        "Ask anything. It plans, searches the live web, writes a cited answer, "
        "checks its own work, and waits for your approval."
    )

    with st.form("ask"):
        question = st.text_area(
            "What do you want researched?",
            placeholder="How do LangGraph checkpointers differ from LangChain memory?",
            height=100,
            # Without a key the box empties on submit and the question vanishes.
            key="question_box",
        )
        opts, retries = st.columns([3, 1])
        length = opts.radio(
            "Answer length",
            ["auto", "brief", "detailed"],
            horizontal=True,
            help=(
                "auto: match the question — one fact in, one line out. "
                "brief: always 3 sentences or fewer. detailed: always thorough."
            ),
        )
        max_revisions = retries.number_input(
            "Max retries", 0, 5, 2,
            help="A ceiling on how often the critic may send the work back. "
                 "Not a target — a satisfied critic uses none.",
        )
        submitted = st.form_submit_button(
            "🔎  Research", type="primary", use_container_width=True
        )

    if submitted:
        if question.strip():
            st.session_state.length = length
            st.session_state.max_revisions = int(max_revisions)
            start(question.strip(), int(max_revisions), length=length)
            st.rerun()
        else:
            st.warning("Type a question first.")

    st.caption("Or try one of these:")
    for col, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        if col.button(example, use_container_width=True, key=f"ex_{example[:12]}"):
            st.session_state.question_box = example
            st.rerun()

    st.divider()
    st.caption("**How it works** — every run goes through these six steps")
    for col, (emoji, name, what) in zip(st.columns(len(STEPS)), STEPS):
        col.markdown(f"<div style='text-align:center'>{emoji}<br><b>{name}</b>"
                     f"<br><span style='font-size:0.78em;opacity:0.7'>{what}</span></div>",
                     unsafe_allow_html=True)


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------
else:
    snap = read_state(st.session_state.thread_id)
    values = snap.values
    report = values.get("final_report")
    stopped_at = snap.next[0] if snap.next else None

    if stopped_at is None:
        badge = "✅ Approved · closed"
    elif stopped_at == REVIEW_GATE:
        badge = "⏸ Waiting for you"
    else:
        badge = f"⚠️ Stopped at {stopped_at}"

    back, state_col = st.columns([1, 3])
    if back.button("← All runs", use_container_width=True):
        st.session_state.thread_id = None
        st.rerun()
    state_col.markdown(f"<div style='text-align:right;padding-top:0.5rem;opacity:0.75'>"
                       f"{badge}</div>", unsafe_allow_html=True)

    st.markdown(f"## {values.get('question', '(no question)')}")

    parent = next(
        (m for m in (values.get("recalled") or []) if m.get("carried")), None
    )
    if parent:
        st.caption(f"↳ follow-up to: *{parent.get('question', '')}*")

    # ---- the answer, first ----
    if stopped_at and stopped_at != REVIEW_GATE:
        st.warning(
            f"This run stopped at **{stopped_at}** before producing a draft — "
            "usually an API key or quota problem at the time. There is nothing "
            "to approve. Ask the question again to start a fresh run."
        )
    else:
        render_conversation(values, read_timeline(st.session_state.thread_id))

    # ---- what you can do next ----
    if stopped_at == REVIEW_GATE:
        edits = values.get("edits", 0)
        if values.get("critique") and not edits:
            st.info(f"**The critic said:** {values['critique']}")

        st.divider()
        st.markdown("#### Your turn")
        st.caption(
            f"You have asked for {edits} change{'s' if edits > 1 else ''}. "
            "Keep going, or approve when you are happy — there is no limit."
            if edits else
            "Approve it as it stands, or say what to change. A revision "
            "rewrites the text; it does not search for anything new."
        )
        notes = st.text_area(
            "What should change?",
            placeholder="Make it shorter. Drop the last section. Add trade-offs.",
            height=80,
            # Key on the edit count so the box empties after each round
            # instead of re-submitting the previous instruction.
            key=f"notes_{st.session_state.thread_id}_{edits}",
            label_visibility="collapsed",
        )
        ok, redo = st.columns(2)
        if ok.button("✅  Approve", type="primary", use_container_width=True):
            if notes.strip():
                st.warning(
                    "You have written revision notes. Press **Request changes** "
                    "to apply them, or clear the box to approve as-is."
                )
            else:
                resume("approve")
                st.rerun()
        if redo.button("✏️  Request changes", use_container_width=True):
            if notes.strip():
                resume(notes.strip())
                st.rerun()
            else:
                st.warning("Write what you want changed, or press Approve.")

    elif report:
        st.success("Approved — this question is closed and saved to memory.")
        st.download_button(
            "⬇️  Download as Markdown",
            report,
            file_name="research-report.md",
            mime="text/markdown",
            use_container_width=True,
        )

        st.divider()
        st.markdown("#### Ask a follow-up")
        st.caption(
            "Opens a **new** question nested under this one, carrying this "
            "report along — so you can say \"it\" without repeating yourself. "
            "Unlike a revision, a follow-up searches the web again."
        )
        follow_up = st.text_input(
            "Follow-up",
            placeholder="What does it cost? How does it compare?",
            label_visibility="collapsed",
            key=f"fu_{st.session_state.thread_id}",
        )
        fu_len, fu_go = st.columns([2, 1])
        length = fu_len.radio(
            "Answer length", ["auto", "brief", "detailed"],
            horizontal=True, label_visibility="collapsed",
            index=["auto", "brief", "detailed"].index(st.session_state.length),
            key=f"fulen_{st.session_state.thread_id}",
        )
        if fu_go.button("🔎  Research", use_container_width=True, type="primary"):
            if follow_up.strip():
                st.session_state.length = length
                start(
                    follow_up.strip(),
                    st.session_state.max_revisions,
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

    # ---- how it got there ----
    st.divider()
    st.caption("**Under the hood**")
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
        with st.expander(f"🧠 What it searched for ({len(values['plan'])})"):
            for q in values["plan"]:
                st.markdown(f"- {q}")
            if values.get("gaps"):
                st.markdown("**Extra searches the critic asked for:**")
                for g in values["gaps"]:
                    st.markdown(f"- {g}")

    if values.get("findings"):
        with st.expander(f"🌐 Sources ({len(values['findings'])})"):
            for i, f in enumerate(values["findings"], start=1):
                if f["url"]:
                    st.markdown(f"{i}. [{f['title']}]({f['url']})")
                else:
                    st.markdown(f"{i}. ⚠️ {f['title']} — {f['snippet'][:120]}")

    with st.expander("🗑️ Delete this run"):
        followups = [
            item for item in list_threads()
            if item["parent"] == st.session_state.thread_id
        ]
        st.caption(
            "Removes the run and its checkpoints. Cannot be undone. The report "
            "it saved to long-term memory is separate — tick below to drop that too."
        )
        if followups:
            st.caption(
                f"This question has **{len(followups)} follow-up"
                f"{'s' if len(followups) > 1 else ''}**. Use the 🗑 Delete toggle "
                "in the sidebar to remove the whole chain at once."
            )
        also_forget = st.checkbox("Also forget it from long-term memory")
        if st.button("Delete permanently", type="secondary"):
            delete_runs(
                [(st.session_state.thread_id, values.get("question", ""))], also_forget
            )
            st.rerun()

    st.caption(f"thread `{st.session_state.thread_id}`")
