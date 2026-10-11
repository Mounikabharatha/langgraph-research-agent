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
