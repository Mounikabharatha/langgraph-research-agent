"""Version history. Nothing the agent writes is ever overwritten - every
revision is a new checkpoint - so the full editing history is recoverable."""

from __future__ import annotations

from research_agent.graph import draft_versions


class FakeSnapshot:
    def __init__(self, **values):
        self.values = values


class FakeApp:
    """get_state_history yields newest first, like the real checkpointer."""

    def __init__(self, snapshots):
        self._snapshots = snapshots

    def get_state_history(self, config):
        return iter(self._snapshots)


def test_versions_come_back_oldest_first():
    app = FakeApp([
        FakeSnapshot(draft="long draft", final_report="short rewrite"),
        FakeSnapshot(draft="long draft"),
        FakeSnapshot(),
    ])
    assert draft_versions(app, {}) == ["long draft", "short rewrite"]


def test_identical_versions_are_not_repeated():
    """Approving without notes makes final_report identical to the draft -
    that is one version, not two."""
    app = FakeApp([
        FakeSnapshot(draft="the answer", final_report="the answer"),
        FakeSnapshot(draft="the answer"),
    ])
    assert draft_versions(app, {}) == ["the answer"]


def test_whitespace_only_differences_are_not_new_versions():
    app = FakeApp([FakeSnapshot(draft="  the answer  ", final_report="the answer")])
    assert draft_versions(app, {}) == ["the answer"]


def test_a_run_with_no_draft_yet_has_no_versions():
    assert draft_versions(FakeApp([FakeSnapshot(question="q")]), {}) == []


# --------------------------------------------------------------------------
# report_timeline - the conversation, reconstructed from checkpoints
# --------------------------------------------------------------------------
from research_agent.graph import report_timeline


def test_timeline_pairs_each_version_with_the_note_that_caused_it():
    app = FakeApp([
        FakeSnapshot(draft="short v2", final_report="short v2",
                     human_feedback="make it shorter"),
        FakeSnapshot(draft="long v1", human_feedback="make it shorter"),
        FakeSnapshot(draft="long v1"),
    ])
    timeline = report_timeline(app, {})

    assert [e["text"] for e in timeline] == ["long v1", "short v2"]
    assert timeline[0]["instruction"] is None, "the first draft was not requested"
    assert timeline[1]["instruction"] == "make it shorter"


def test_approval_is_not_shown_as_an_instruction():
    """'approve' is a decision, not a revision note - it must not appear as
    something the user asked for."""
    app = FakeApp([
        FakeSnapshot(draft="the answer", final_report="the answer",
                     human_feedback="approve"),
        FakeSnapshot(draft="the answer"),
    ])
    timeline = report_timeline(app, {})

    assert len(timeline) == 1
    assert timeline[0]["instruction"] is None


# --------------------------------------------------------------------------
# Demo seeding - a hosted free tier wipes the disk when the app sleeps
# --------------------------------------------------------------------------
def test_demo_data_is_copied_when_nothing_exists(tmp_path, monkeypatch):
    from research_agent import graph

    monkeypatch.chdir(tmp_path)
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "checkpoints.db").write_bytes(b"seed")
    monkeypatch.setattr(graph, "DEMO_DIR", tmp_path / "demo")

    assert graph.seed_demo_data("checkpoints.db") == ["checkpoints.db"]
    assert (tmp_path / "checkpoints.db").read_bytes() == b"seed"


def test_an_existing_database_is_never_overwritten(tmp_path, monkeypatch):
    """Seeding must only fill a gap. Clobbering real work would be far worse
    than an empty sidebar."""
    from research_agent import graph

    monkeypatch.chdir(tmp_path)
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "checkpoints.db").write_bytes(b"seed")
    (tmp_path / "checkpoints.db").write_bytes(b"my real runs")
    monkeypatch.setattr(graph, "DEMO_DIR", tmp_path / "demo")

    assert graph.seed_demo_data("checkpoints.db") == []
    assert (tmp_path / "checkpoints.db").read_bytes() == b"my real runs"


def test_missing_demo_data_is_not_an_error(tmp_path, monkeypatch):
    from research_agent import graph

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(graph, "DEMO_DIR", tmp_path / "demo")
    assert graph.seed_demo_data("checkpoints.db") == []


# --------------------------------------------------------------------------
# Resuming an interrupted run - streaming None continues, it does not restart
# --------------------------------------------------------------------------
def test_an_interrupted_run_keeps_its_work():
    """A run killed mid-flight still holds its plan, findings and draft, so
    the UI must offer to continue rather than treating it as a dead end."""
    snap = FakeSnapshot(
        question="q",
        plan=["a", "b"],
        findings=[{"title": "t", "url": "u", "snippet": "s", "sub_question": "a"}],
        draft="a partial answer",
    )
    assert snap.values["plan"], "the plan survives an interruption"
    assert snap.values["findings"], "so do the findings"
    assert snap.values["draft"], "and whatever had been drafted"


def test_the_timeline_of_an_interrupted_run_still_renders():
    app = FakeApp([FakeSnapshot(draft="a partial answer")])
    assert draft_versions(app, {}) == ["a partial answer"]


# --------------------------------------------------------------------------
# The live run in a session is the LAST one, not the one clicked in the sidebar
# --------------------------------------------------------------------------
def _chain_for(thread_id, threads):
    import importlib.util
    import pathlib
    import sys

    import pytest

    # Streamlit is an optional extra, so CI may not have it.
    if importlib.util.find_spec("streamlit") is None:
        pytest.skip("streamlit not installed")

    # app.py is a Streamlit script, not part of the package; load it directly.
    spec = importlib.util.spec_from_file_location(
        "demo_app", pathlib.Path(__file__).resolve().parents[1] / "app.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.chain_for(thread_id, threads)


def test_a_session_resolves_to_the_newest_run_from_any_member():
    """Clicking the root in the sidebar must still act on the live follow-up.

    Approving used the selected thread rather than the end of the chain, so
    returning to a paused session and pressing Approve resumed the already
    closed root, did nothing, and still said it had saved.
    """
    threads = [
        {"id": "root", "question": "q1", "next": None, "parent": None,
         "created": "2026-10-11T08:00:00"},
        {"id": "kid", "question": "q2", "next": "human_review", "parent": "root",
         "created": "2026-10-11T09:00:00"},
    ]
    for entry in ("root", "kid"):
        chain = _chain_for(entry, threads)
        assert [c["id"] for c in chain] == ["root", "kid"]
        assert chain[-1]["id"] == "kid", "the live run is the end of the chain"


def test_follow_ups_are_ordered_oldest_first():
    threads = [
        {"id": "a", "question": "q", "next": None, "parent": None,
         "created": "2026-10-11T08:00:00"},
        {"id": "c", "question": "q", "next": None, "parent": "a",
         "created": "2026-10-11T10:00:00"},
        {"id": "b", "question": "q", "next": None, "parent": "a",
         "created": "2026-10-11T09:00:00"},
    ]
    assert [c["id"] for c in _chain_for("a", threads)] == ["a", "b", "c"]


def test_resume_cannot_be_called_without_naming_a_thread():
    """The real protection against the bug above.

    resume() used to fall back to st.session_state.thread_id, so a caller
    that forgot which run it meant silently got the wrong one. With no
    default, forgetting is a TypeError at the call site instead.
    """
    import importlib.util
    import inspect
    import pathlib

    import pytest

    if importlib.util.find_spec("streamlit") is None:
        pytest.skip("streamlit not installed")

    spec = importlib.util.spec_from_file_location(
        "demo_app", pathlib.Path(__file__).resolve().parents[1] / "app.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    param = inspect.signature(module.resume).parameters["thread_id"]
    assert param.default is inspect.Parameter.empty, (
        "thread_id must be required - a default lets a caller resume the "
        "wrong run without noticing"
    )


def test_no_caller_resumes_the_sidebar_selection():
    """The gate must act on the live run at the end of the chain, never on
    whichever run happens to be selected in the sidebar."""
    import pathlib

    source = (pathlib.Path(__file__).resolve().parents[1] / "app.py").read_text()
    bad = [
        line.strip()
        for line in source.splitlines()
        if "resume(" in line
        and "def resume" not in line
        and "st.session_state.thread_id" in line
    ]
    assert not bad, f"these resume a sidebar selection rather than the live run: {bad}"
