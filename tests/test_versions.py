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
