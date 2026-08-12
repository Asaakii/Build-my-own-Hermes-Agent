"""验证最小化审计事件、参数脱敏与 SQLite 持久化。"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_lite.sqlite_state_store as state_store_module
from hermes_lite.audit_log import (
    AuditEvent,
    AuditEventType,
    InMemoryAuditLog,
    summarize_tool_arguments,
)
from hermes_lite.domain import ToolCall
from hermes_lite.sqlite_state_store import (
    SQLiteStateStore,
    SQLiteStateStoreError,
    load_sqlite_state_config,
)


def make_tool_call() -> ToolCall:
    """创建包含敏感文本的调用，用于验证摘要不会复制原始参数。"""
    return ToolCall(
        call_id="call-audit-1",
        tool_name="write_file",
        arguments={
            "path": "notes/private.txt",
            "content": "PRIVATE_CONTENT_9X7",
            "overwrite": False,
            "retries": 2,
        },
    )


def test_argument_summary_keeps_only_name_type_and_text_length() -> None:
    """审计摘要不能保存文件路径、内容或其他原始参数值。"""
    call = make_tool_call()

    summary = summarize_tool_arguments(call.arguments)
    event = AuditEvent.for_tool_call(
        session_id="session-audit",
        task_id="task-audit",
        event_type=AuditEventType.TOOL_REQUESTED,
        tool_call=call,
    )

    assert summary == (
        ("content", "string(length=19)"),
        ("overwrite", "boolean"),
        ("path", "string(length=17)"),
        ("retries", "integer"),
    )
    assert event.argument_summary == summary
    assert "PRIVATE_CONTENT_9X7" not in repr(event)
    assert "notes/private.txt" not in repr(event)


def test_in_memory_log_keeps_only_validated_events() -> None:
    """默认进程内接收器只接受已验证的 AuditEvent。"""
    log = InMemoryAuditLog()
    event = AuditEvent(
        session_id="session-audit",
        task_id="task-audit",
        event_type=AuditEventType.CONFIRMATION_REJECTED,
        reason_code="invalid_or_unusable_token",
    )

    log.record_audit_event(event)

    assert log.events == [event]
    with pytest.raises(ValueError, match="event 必须是 AuditEvent"):
        log.record_audit_event("not-an-event")  # type: ignore[arg-type]


@pytest.fixture
def store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> SQLiteStateStore:
    """使用临时项目数据库验证审计记录能够跨接收器保留。"""
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(state_store_module, "PROJECT_ROOT", project_root)
    return SQLiteStateStore(load_sqlite_state_config({}))


def test_sqlite_audit_events_persist_without_raw_tool_arguments(
    store: SQLiteStateStore,
) -> None:
    """SQLite 仅保存脱敏摘要，并按会话隔离读取审计事件。"""
    call = make_tool_call()
    event = AuditEvent.for_tool_call(
        session_id="session-audit",
        task_id="task-audit",
        event_type=AuditEventType.TOOL_EXECUTED,
        tool_call=call,
        reason_code="success",
    )
    other_event = AuditEvent(
        session_id="other-session",
        task_id="other-task",
        event_type=AuditEventType.CONFIRMATION_REJECTED,
        reason_code="malformed_command",
    )

    store.record_audit_event(event)
    store.record_audit_event(other_event)
    restored = store.list_audit_events("session-audit")

    assert restored == (event,)
    assert "PRIVATE_CONTENT_9X7" not in repr(restored)
    assert "notes/private.txt" not in repr(restored)
    assert store.list_audit_events("other-session") == (other_event,)


def test_sqlite_audit_events_reject_invalid_read_limit(
    store: SQLiteStateStore,
) -> None:
    """读取上限必须是正整数，避免调用方无界读取审计历史。"""
    with pytest.raises(SQLiteStateStoreError, match="max_results 必须是正整数"):
        store.list_audit_events("session-audit", 0)
