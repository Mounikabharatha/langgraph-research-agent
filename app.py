"""Streamlit front end for the research agent.

    streamlit run app.py

The graph is the same one the CLI drives - this file adds no agent logic.
It opens the checkpointer per action because Streamlit re-executes this whole
script on every interaction, so nothing may be held open across reruns.
"""

from __future__ import annotations

import logging
import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from research_agent.graph import (
    DEFAULT_DB,
    compiled_graph,
    report_timeline,
    seed_demo_data,
)
from research_agent.memory import describe, forget, open_store
from research_agent.llm import clean_secret, model_names
from research_agent.nodes import looks_like_a_question
from research_agent.tracing import graph_config

logging.getLogger("google_genai").setLevel(logging.ERROR)

# Locally the keys come from .env; on Streamlit Community Cloud they come
# from the Secrets panel. Copy those into the environment so every module
# can keep reading os.getenv and knows nothing about Streamlit.
# `clean_secret` matters here: a key pasted into the Secrets panel can carry
# an invisible character, which fails every API call while looking perfect.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, clean_secret(str(_value)))
except Exception:
    pass  # no secrets file locally, which is normal

load_dotenv()

# A hosted free tier wipes the disk when the app sleeps. Restore the demo
# sessions so a visitor never arrives at an empty app.
seed_demo_data("checkpoints.db", "memory.db")

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
                "created": snap.created_at or "",
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


def chain_for(thread_id: str, threads: list[dict]) -> list[dict]:
    """Every run in the same session as `thread_id`, oldest first.

    A follow-up is its own run - its own sources, its own approval - but a
    question and its follow-ups are one line of enquiry, so they are shown
    and listed together.
    """
    by_id = {item["id"]: item for item in threads}
    if thread_id not in by_id:
        return []

    root = by_id[thread_id]
    seen = {root["id"]}
    while root.get("parent") in by_id and root["parent"] not in seen:
        seen.add(root["parent"])
        root = by_id[root["parent"]]

    children: dict[str, list[dict]] = {}
    for item in threads:
        if item.get("parent") in by_id:
            children.setdefault(item["parent"], []).append(item)

    ordered: list[dict] = []

    def walk(node: dict) -> None:
        ordered.append(node)
        for child in sorted(children.get(node["id"], []), key=lambda c: c["created"]):
            walk(child)

    walk(root)
    return ordered


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
                st.markdown(f"**✏️ Request changes:** {entry['instruction']}")
                st.caption(
                    "Rewrote the answer above from the sources already "
                    "gathered — no new search."
                )

        latest = i == len(timeline) - 1
        previous = i == len(timeline) - 2
        with st.chat_message("assistant"):
            if latest:
                st.markdown(entry["text"])
            else:
                # Superseded, but kept. Say WHY it was replaced: a version with
                # no instruction behind it came from the critic's own retry
                # loop, not from anything the reader asked for.
                asked_for = bool(timeline[i + 1]["instruction"])
                why = (
                    "you asked for a change" if asked_for
                    else "the critic sent it back for more research"
                )
                # Open the one directly before the current answer, so a
                # revision can be compared against what it replaced.
                with st.expander(
                    f"Earlier version · {len(entry['text'])} characters · "
                    f"replaced because {why}",
                    expanded=previous and asked_for,
                ):
                    st.markdown(entry["text"])


def render_followup_box(values: dict, report: str, approve_first: bool,
                        thread_id: str | None = None) -> None:
    """The 'ask something new' box.

    Offered both after approval and at the review gate. At the gate the
    current question is approved first, so a follow-up never leaves an
    unapproved run dangling behind it.
    """
    st.caption(
        "A **new** question, nested under this one and carrying this answer "
        "along — so you can say \"it\". Searches the web again."
        + (" Closes this answer first." if approve_first else "")
    )
    thread_id = thread_id or st.session_state.thread_id
    follow_up = st.text_input(
        "Follow-up",
        placeholder="What does it cost? What is a hybrid, exactly?",
        label_visibility="collapsed",
        key=f"fu_{thread_id}",
    )
    length = st.radio(
        "Answer length", LENGTHS,
        horizontal=True, label_visibility="collapsed",
        index=length_index(st.session_state.length),
        key=f"fulen_{thread_id}",
    )
    label = "🔎  Research follow-up"
    if st.button(label, use_container_width=True, key=f"fugo_{thread_id}"):
        if not follow_up.strip():
            st.warning("Type a follow-up question first.")
            return
        st.session_state.length = length
        parent_report = report
        if approve_first:
            resume("approve", thread_id)
            parent_report = read_state(thread_id).values.get("final_report", report)
        start(
            follow_up.strip(),
            st.session_state.max_revisions,
            length=length,
            parent_thread=thread_id,
            parent={
                "question": values.get("question", ""),
                "report": parent_report,
                "carried": True,
            },
        )
        st.rerun()


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
            st.session_state.error = friendly_error(exc)
            status.update(label="Stopped", state="error")


def friendly_error(exc: Exception) -> str:
    """Turn a provider exception into something a visitor can act on.

    A stack trace is the right thing for a developer and the wrong thing for
    somebody who followed a link from a CV.
    """
    text = str(exc)
    if "RESOURCE_EXHAUSTED" in text or "429" in text or "quota" in text.lower():
        tried = ", ".join(f"`{n}`" for n in model_names())
        return (
            "**The daily free quota is used up.** This demo runs on Google "
            "Gemini's free tier, which allows roughly 20 requests per day per "
            f"model. It tried {tried} and all of them are spent.\n\n"
            "The quota resets at midnight Pacific time. In the meantime, open "
            "any saved session in the sidebar — those are real past runs, with "
            "their sources and revision history, and reading them costs "
            "nothing."
        )
    if "401" in text or "API key" in text or "API_KEY_INVALID" in text:
        return (
            "**A search or model API key is not working.** If you are running "
            "this locally, check `.env`; if this is the hosted demo, the keys "
            "need renewing."
        )
    if "Tavily" in text or "search" in text.lower():
        return (
            "**Web search is unavailable right now**, so there is nothing to "
            "write an answer from. Try again in a moment."
        )
    return f"Something went wrong: {type(exc).__name__}: {text[:300]}"


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


def continue_run(thread_id: str) -> None:
    """Carry on from the last checkpoint after an interrupted run.

    Streaming `None` resumes rather than restarting, so the searching and
    drafting already done is kept. This is the checkpointer earning its keep:
    a run killed by a restart, a crash or a dropped connection is not lost.
    """
    st.session_state.error = None
    with st.status("Picking up where it stopped…", expanded=True) as status:
        try:
            drive(None, thread_id, status)
            status.update(label="Ready for review", state="complete")
        except Exception as exc:
            st.session_state.error = friendly_error(exc)
            status.update(label="Stopped again", state="error")


def resume(feedback: str, thread_id: str) -> None:
    """Answer the review gate of ONE run.

    The thread must be passed explicitly. Falling back to
    st.session_state.thread_id resumed whichever run was selected in the
    sidebar - the root of the session - rather than the live one at the end
    of the chain. Immediately after creating a follow-up those are the same
    thread, so the bug only appeared on returning to a session later: the
    approve hit an already-closed run, did nothing, and reported success.
    """
    st.session_state.error = None
    with st.status("Finalising…", expanded=True) as status:
        try:
            drive(Command(resume=feedback), thread_id, status)
            status.update(label="Done", state="complete")
        except Exception as exc:
            st.session_state.error = friendly_error(exc)
            status.update(label="Stopped", state="error")


# --------------------------------------------------------------------------
# Sidebar. This must come before the main content - a `with st.sidebar` block
# placed after it renders nothing at all. Staleness after a new run is handled
# by st.rerun() once the run finishes, not by moving this.
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔎 Research Agent")
    st.caption("Plans · searches · cites · checks itself")
    st.divider()

    if st.button("➕  New question", use_container_width=True, type="primary"):
        st.session_state.thread_id = None
        st.session_state.error = None
        st.session_state.pop("question_box", None)
        st.rerun()

    threads = list_threads()
    st.subheader("Saved runs")
    delete_mode = st.toggle("🗑️  Delete mode")
    if not threads:
        st.caption("Nothing yet — ask something.")

    by_id = {item["id"]: item for item in threads}
    roots = [t for t in threads if t.get("parent") not in by_id]
    kids: dict[str, int] = {}
    for item in threads:
        node = item
        seen = {node["id"]}
        while node.get("parent") in by_id and node["parent"] not in seen:
            seen.add(node["parent"])
            node = by_id[node["parent"]]
        if node["id"] != item["id"]:
            kids[node["id"]] = kids.get(node["id"], 0) + 1

    def icon(item: dict) -> str:
        chain = chain_for(item["id"], threads)
        last = chain[-1] if chain else item
        if last["next"] is None:
            return "✓"
        return "⏸️" if last["next"] == REVIEW_GATE else "⚠️"

    def entry(item: dict) -> None:
        extra = kids.get(item["id"], 0)
        suffix = f"  +{extra}" if extra else ""
        label = f"{icon(item)} {item['question'][:28]}{suffix}"
        selected = st.session_state.thread_id in {
            c["id"] for c in chain_for(item["id"], threads)
        }

        if delete_mode:
            open_col, bin_col = st.columns([4, 1])
            if open_col.button(label, key=f"t_{item['id']}", use_container_width=True,
                               type="primary" if selected else "secondary"):
                st.session_state.thread_id = item["id"]
                st.session_state.error = None
                st.rerun()
            if bin_col.button("🗑️", key=f"d_{item['id']}",
                              use_container_width=True,
                              help="Delete this session"):
                st.session_state.pending_delete = item["id"]
                st.rerun()
        else:
            if st.button(label, key=f"t_{item['id']}", use_container_width=True,
                         type="primary" if selected else "secondary"):
                st.session_state.thread_id = item["id"]
                st.session_state.error = None
                st.rerun()

        if st.session_state.get("pending_delete") == item["id"]:
            doomed = chain_for(item["id"], threads)
            extra_n = len(doomed) - 1
            st.warning(
                f"Delete this session{f' and its {extra_n} follow-up' if extra_n else ''}"
                f"{'s' if extra_n > 1 else ''}? This cannot be undone."
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

    for root in sorted(roots, key=lambda r: r["created"], reverse=True):
        entry(root)


# --------------------------------------------------------------------------
# Main. Two modes: the home page for asking, and the view of one run.
# Keeping the ask form off the run view means the answer is the first
# thing on screen rather than something you scroll to.
# --------------------------------------------------------------------------
st.session_state.setdefault("length", "auto")
st.session_state.setdefault("max_revisions", 2)

LENGTHS = ["auto", "short", "medium", "long"]


def length_index(value: str) -> int:
    """Position in LENGTHS, tolerating the old brief/detailed names."""
    from research_agent.nodes import LENGTH_ALIASES

    name = LENGTH_ALIASES.get((value or "auto").lower(), (value or "auto").lower())
    return LENGTHS.index(name) if name in LENGTHS else 0


STEPS = [
    ("🧠", "Plan", "splits your question up"),
    ("🌐", "Search", "reads the live web"),
    ("✍️", "Write", "answers, with sources"),
    ("🔍", "Critique", "finds its own gaps"),
    ("🔁", "Retry", "searches again if thin"),
    ("🙋", "You decide", "approve or revise"),
]

EXAMPLES = [
    "What is the Model Context Protocol?",
    "Compare RAG and fine-tuning for internal company documents",
    "What causes hallucination in large language models?",
]

st.markdown(
    "<style>"
    "  [data-testid='stMainBlockContainer'] { padding-top: 2.5rem; }"
    "  [data-testid='stSidebarUserContent'] { padding-top: 1.5rem; }"
    "</style>",
    unsafe_allow_html=True,
)

if st.session_state.error:
    st.error(st.session_state.error)


# --------------------------------------------------------------------------
# Home
# --------------------------------------------------------------------------
if not st.session_state.thread_id:
    # An example button cannot write to question_box directly: Streamlit
    # forbids touching a widget's key once that widget has been created.
    # Stage it instead and apply it here, before the text area exists.
    if "prefill" in st.session_state:
        st.session_state.question_box = st.session_state.pop("prefill")

    # Home gets the full title; on a run view the sidebar carries the name
    # and the question itself should be the biggest thing on the page.
    st.title("🔎 Research Agent")
    st.caption(
        "Ask anything. It plans, searches the live web, writes a cited answer, "
        "checks its own work, and waits for your approval."
    )

    closed = st.session_state.pop("just_closed", None)
    if closed:
        st.success(f"✅ Saved — **{closed['question']}**")
        back, _ = st.columns([2, 3])
        if back.button(
            "↩️  Reopen to ask a follow-up", use_container_width=True
        ):
            st.session_state.thread_id = closed["id"]
            st.rerun()

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
            LENGTHS,
            horizontal=True,
            help=(
                "auto — the agent decides from the question.\n"
                "short — one or two sentences, always.\n"
                "medium — 3 to 5 bullets.\n"
                "long — thorough, with headings."
            ),
        )
        max_revisions = retries.number_input(
            "Max critic passes", 1, 5, 2,
            help="How many times the critic may review the answer. The first "
                 "pass always happens; each one after it means the critic was "
                 "unsatisfied and sent the work back for more searching. A "
                 "ceiling, not a target — most questions use just one.",
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
            st.session_state.prefill = example
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
    all_threads = list_threads()
    chain = chain_for(st.session_state.thread_id, all_threads)
    # Act on the newest run in the session - that is the live one.
    active_id = chain[-1]["id"] if chain else st.session_state.thread_id

    snap = read_state(active_id)
    values = snap.values
    report = values.get("final_report")
    stopped_at = snap.next[0] if snap.next else None

    if stopped_at is None:
        badge = "✅ Approved · closed"
    elif stopped_at == REVIEW_GATE:
        badge = "⏸️ Waiting for you"
    else:
        badge = f"⚠️ Stopped at {stopped_at}"

    back, state_col = st.columns([1, 3])
    if back.button("← All sessions", use_container_width=True):
        st.session_state.thread_id = None
        st.rerun()
    state_col.markdown(
        f"<div style='text-align:right;padding-top:0.5rem;opacity:0.75'>{badge}</div>",
        unsafe_allow_html=True,
    )

    # ---- the whole session, oldest question first ----
    for position, link in enumerate(chain):
        link_state = snap if link["id"] == active_id else read_state(link["id"])
        link_values = link_state.values

        if position:
            st.divider()
            st.markdown(f"### ↳ {link_values.get('question', '')}")
            asked_via = "🔎 Research follow-up"
        else:
            st.markdown(f"## {link_values.get('question', '')}")
            asked_via = "➕ New question"

        # Say how this question was asked and what was chosen for it, so a
        # reader can tell a follow-up from a revision without guessing.
        chosen = (link_values.get("length") or "auto").lower()
        st.caption(
            f"{asked_via}  ·  answer length: **{chosen}**  ·  "
            f"up to **{link_values.get('max_revisions', 2)}** critic passes"
        )

        if link["id"] == active_id and stopped_at and stopped_at != REVIEW_GATE:
            partial = link_values.get("draft") or ""
            st.warning(
                f"This run was interrupted at the **{stopped_at}** step — a "
                "restart, a dropped connection, or an API error at the time. "
                "Nothing is lost: every step was checkpointed."
            )
            st.caption(
                f"It already has {len(link_values.get('plan') or [])} "
                f"sub-questions and {len(link_values.get('findings') or [])} "
                f"sources"
                + (f", and a {len(partial)}-character draft." if partial else ".")
            )
            if st.button("▶️  Resume from where it stopped", type="primary"):
                continue_run(link["id"])
                st.rerun()
            if partial:
                with st.expander("What it had written so far"):
                    st.markdown(partial)
        else:
            render_conversation(link_values, read_timeline(link["id"]))

        with st.expander(
            f"Under the hood · {len(link_values.get('plan', []))} sub-questions · "
            f"{len(link_values.get('findings', []))} sources · "
            f"{link_values.get('revision', 0)} of "
            f"{link_values.get('max_revisions', 2)} critic passes"
        ):
            if link_values.get("recalled"):
                st.markdown("**♻️ Recalled from earlier research**")
                for m in link_values["recalled"]:
                    st.markdown(f"- {m.get('question', '?')} · {describe(m)}")
            if link_values.get("plan"):
                st.markdown("**🧠 What it searched for**")
                for q in link_values["plan"]:
                    st.markdown(f"- {q}")
            if link_values.get("findings"):
                st.markdown(f"**🌐 Sources ({len(link_values['findings'])})**")
                for i, f in enumerate(link_values["findings"], start=1):
                    if f["url"]:
                        st.markdown(f"{i}. [{f['title']}]({f['url']})")
                    else:
                        st.markdown(f"{i}. ⚠️ {f['title']} — {f['snippet'][:120]}")

    # ---- what you can do next ----
    if stopped_at == REVIEW_GATE:
        edits = values.get("edits", 0)
        if values.get("critique") and not edits:
            st.info(f"**The critic said:** {values['critique']}")

        st.divider()
        st.markdown("#### Your turn")
        st.caption(
            "Type once, then choose what to do with it.\n\n"
            "• **✏️ Request changes** — rewrites the answer above from the "
            "sources it already has. Stays here.\n"
            "• **🔎 Research follow-up** — treats it as a new question and "
            "searches again. **Continues this session**, nested below.\n"
            "• **✅ Approve & close** — you are happy. **Ends the session** "
            "and returns you to a blank one."
            + (f"\n\n{edits} change{'s' if edits > 1 else ''} so far."
               if edits else "")
        )

        text = st.text_area(
            "Your turn",
            placeholder=(
                "Make it shorter…        (then press Request changes)\n"
                "What does it cost?      (then press Research follow-up)"
            ),
            height=110,
            # Key on the edit count so the box empties after each round
            # instead of re-submitting the previous instruction.
            key=f"turn_{active_id}_{edits}",
            label_visibility="collapsed",
        )

        length = st.radio(
            "Answer length for a follow-up",
            LENGTHS,
            horizontal=True,
            index=length_index(st.session_state.length),
            key=f"len_{active_id}_{edits}",
            help="Only applies to a follow-up — a revision keeps the length "
                 "you ask for in the box.",
        )

        change_col, follow_col = st.columns(2)

        if change_col.button("✏️  Request changes", use_container_width=True):
            if text.strip():
                resume(text.strip(), active_id)
                st.rerun()
            else:
                st.warning("Write what you want changed.")

        if follow_col.button("🔎  Research follow-up", use_container_width=True):
            if text.strip():
                st.session_state.length = length
                parent_thread = active_id
                resume("approve", active_id)
                parent_report = read_state(parent_thread).values.get("final_report", "")
                start(
                    text.strip(),
                    st.session_state.max_revisions,
                    length=length,
                    parent_thread=parent_thread,
                    parent={
                        "question": values.get("question", ""),
                        "report": parent_report,
                        "carried": True,
                    },
                )
                st.rerun()
            else:
                st.warning("Type a follow-up question first.")

        if looks_like_a_question(text):
            st.info(
                "That reads like a **new question** — press **Research "
                "follow-up** rather than Request changes, so it actually gets "
                "looked up."
            )

        _, mid, _ = st.columns([1, 2, 1])
        if mid.button("✅  Approve & close", type="primary", use_container_width=True):
            if text.strip():
                st.warning(
                    "There is text in the box. Press **Request changes** or "
                    "**Research follow-up**, or clear it to approve as-is."
                )
            else:
                resume("approve", active_id)
                # Closing returns you to a blank session, but remember what
                # was closed so home can offer one click back - changing your
                # mind about a follow-up should not mean hunting the sidebar.
                st.session_state.just_closed = {
                    "id": chain[0]["id"],
                    "question": chain[0]["question"],
                }
                st.session_state.thread_id = None
                st.rerun()

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
        render_followup_box(values, report, approve_first=False, thread_id=active_id)

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

    st.divider()
    with st.expander("🗑️ Delete this session"):
        extra = len(chain) - 1
        st.caption(
            f"Removes this question{f' and its {extra} follow-up' if extra else ''}"
            f"{'s' if extra > 1 else ''} along with every checkpoint. Cannot be "
            "undone. Reports saved to long-term memory are separate — tick "
            "below to drop those too."
        )
        also_forget = st.checkbox("Also forget from long-term memory")
        if st.button("Delete permanently", type="secondary"):
            delete_runs([(c["id"], c["question"]) for c in chain], also_forget)
            st.rerun()

    st.caption(
        f"{len(chain)} question{'s' if len(chain) > 1 else ''} in this session"
    )
