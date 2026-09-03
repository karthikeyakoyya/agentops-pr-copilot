import json

from agentops.core import governance


def test_is_allowed_unrestricted_action():
    assert governance.is_allowed("retrieve_context", "viewer") is True


def test_is_allowed_restricted_action_denies_viewer():
    assert governance.is_allowed("apply_fix", "viewer") is False


def test_is_allowed_restricted_action_allows_maintainer():
    assert governance.is_allowed("apply_fix", "maintainer") is True


def test_audit_writes_jsonl(tmp_path, monkeypatch):
    log_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(governance, "AUDIT_LOG_PATH", str(log_path))

    governance.audit("test_event", foo="bar")

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "test_event"
    assert record["foo"] == "bar"
    assert "ts" in record


def test_audit_appends_multiple_events(tmp_path, monkeypatch):
    log_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(governance, "AUDIT_LOG_PATH", str(log_path))

    governance.audit("first")
    governance.audit("second")

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
