"""验证 ToolAgent 运行时写入脱敏审计事件。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

import hermes_lite.sqlite_state_store as state_store_module

from hermes_lite.audit_log import AuditEvent, AuditEventType, InMemoryAuditLog
from hermes_lite.confirmation_policy import ConfirmationManager
from hermes_lite.domain import Message, MessageRole, Session, TaskStatus, ToolCall
from hermes_lite.tool_agent_loop import ToolAgent
from hermes_lite.sqlite_state_store import SQLiteStateStore, load_sqlite_state_config
from hermes_lite.tool_registry import ToolDefinition, ToolRegistry, ToolRiskLevel


class ScriptedModel:
    """以固定消息模拟工具感知模型。"""

    def __init__(self, responses: list[Message]) -> None:
        self._responses = responses
        self.call_count = 0

    def respond(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]],
    ) -> Message:
        """返回下一条预设模型消息。"""
        del messages, tools
        self.call_count += 1
        return self._responses.pop(0)


def make_registry(handler_calls: list[dict[str, object]]) -> ToolRegistry:
    """创建一项只读和一项写入测试工具。"""
    registry = ToolRegistry()

    def handler(arguments: dict[str, object]) -> str:
        handler_calls.append(arguments)
        return "工具已处理"

    for name, risk_level in (
        ("read_text", ToolRiskLevel.READ_ONLY),
        ("write_text", ToolRiskLevel.WRITE),
    ):
        registry.register(
            ToolDefinition(
                name=name,
                description=f"{name} 的测试说明。",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "测试文本。",
                        },
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
                risk_level=risk_level,
                handler=handler,
            )
        )

    return registry


def make_tool_request(tool_name: str = "write_text") -> Message:
    """构造包含不应写入审计日志的原始文本的工具请求。"""
    return Message(
        role=MessageRole.ASSISTANT,
        content=None,
        tool_calls=(
            ToolCall(
                call_id="call-audit-runtime",
                tool_name=tool_name,
                arguments={"text": "PRIVATE_RUNTIME_TEXT"},
            ),
        ),
    )


def test_high_risk_lifecycle_is_audited_without_raw_text_or_token() -> None:
    """高风险调用应记录完整生命周期，但不记录原始参数或确认令牌。"""
    handler_calls: list[dict[str, object]] = []
    audit_log = InMemoryAuditLog()
    agent = ToolAgent(
        ScriptedModel([make_tool_request()]),
        make_registry(handler_calls),
        confirmation_manager=ConfirmationManager(
            token_factory=lambda: "PRIVATE_CONFIRM_TOKEN",
        ),
        audit_recorder=audit_log,
    )
    session = Session(session_id="audit-session")

    waiting_turn = agent.run_turn(session, "执行写入。", task_id="task-wait")

    assert waiting_turn.task.status is TaskStatus.BLOCKED
    assert [event.event_type for event in audit_log.events] == [
        AuditEventType.TOOL_REQUESTED,
        AuditEventType.CONFIRMATION_REQUIRED,
    ]
    assert "PRIVATE_RUNTIME_TEXT" not in repr(audit_log.events)
    assert "PRIVATE_CONFIRM_TOKEN" not in repr(audit_log.events)
    assert handler_calls == []

    completed_turn = agent.run_turn(
        session,
        "/confirm PRIVATE_CONFIRM_TOKEN",
        task_id="task-confirm",
    )

    assert completed_turn.task.status is TaskStatus.COMPLETED
    assert [event.event_type for event in audit_log.events] == [
        AuditEventType.TOOL_REQUESTED,
        AuditEventType.CONFIRMATION_REQUIRED,
        AuditEventType.CONFIRMATION_ACCEPTED,
        AuditEventType.TOOL_EXECUTED,
    ]
    assert handler_calls == [{"text": "PRIVATE_RUNTIME_TEXT"}]
    assert "PRIVATE_RUNTIME_TEXT" not in repr(audit_log.events)
    assert "PRIVATE_CONFIRM_TOKEN" not in repr(audit_log.events)


def test_invalid_confirmation_records_rejection_without_token() -> None:
    """无效令牌只记录拒绝类别，不记录用户提交的令牌。"""
    audit_log = InMemoryAuditLog()
    agent = ToolAgent(
        ScriptedModel([]),
        make_registry([]),
        audit_recorder=audit_log,
    )

    turn = agent.run_turn(
        Session(session_id="audit-session"),
        "/confirm SHOULD_NOT_APPEAR_IN_AUDIT",
        task_id="task-rejected",
    )

    assert turn.task.status is TaskStatus.FAILED
    assert [event.event_type for event in audit_log.events] == [
        AuditEventType.CONFIRMATION_REJECTED,
    ]
    assert audit_log.events[0].reason_code == "invalid_or_unusable_token"
    assert "SHOULD_NOT_APPEAR_IN_AUDIT" not in repr(audit_log.events)


def test_unknown_tool_is_audited_as_registry_rejection() -> None:
    """未知工具不能执行，审计应区分请求和注册表拒绝。"""
    audit_log = InMemoryAuditLog()
    model = ScriptedModel(
        [
            make_tool_request("unknown_tool"),
            Message(role=MessageRole.ASSISTANT, content="已停止。"),
        ]
    )
    agent = ToolAgent(model, make_registry([]), audit_recorder=audit_log)

    turn = agent.run_turn(
        Session(session_id="audit-session"),
        "尝试未知工具。",
        task_id="task-unknown-tool",
    )

    assert turn.task.status is TaskStatus.COMPLETED
    assert [event.event_type for event in audit_log.events] == [
        AuditEventType.TOOL_REQUESTED,
        AuditEventType.TOOL_REJECTED,
    ]
    assert audit_log.events[-1].reason_code == "rejected_by_registry"


def test_tool_agent_can_persist_audit_events_to_sqlite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """运行装配传入 SQLite 接收器后，审计记录可跨对象读取。"""
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(state_store_module, "PROJECT_ROOT", project_root)
    store = SQLiteStateStore(load_sqlite_state_config({}))
    agent = ToolAgent(
        ScriptedModel([make_tool_request()]),
        make_registry([]),
        confirmation_manager=ConfirmationManager(token_factory=lambda: "token"),
        audit_recorder=store,
    )

    turn = agent.run_turn(
        Session(session_id="sqlite-audit-session"),
        "执行写入。",
        task_id="sqlite-audit-task",
    )
    restored = store.list_audit_events("sqlite-audit-session")

    assert turn.task.status is TaskStatus.BLOCKED
    assert [event.event_type for event in restored] == [
        AuditEventType.TOOL_REQUESTED,
        AuditEventType.CONFIRMATION_REQUIRED,
    ]
    assert "PRIVATE_RUNTIME_TEXT" not in repr(restored)


class FailingAuditLog:
    """模拟不可用审计接收器，验证高风险操作不会先于审计执行。"""

    def record_audit_event(self, event: AuditEvent) -> None:
        """任何记录尝试都失败。"""
        del event
        raise RuntimeError("storage unavailable")


def test_audit_failure_blocks_high_risk_tool_before_execution() -> None:
    """审计不可用时，高风险工具不得被签发或执行。"""
    handler_calls: list[dict[str, object]] = []
    agent = ToolAgent(
        ScriptedModel([make_tool_request()]),
        make_registry(handler_calls),
        audit_recorder=FailingAuditLog(),
    )

    turn = agent.run_turn(
        Session(session_id="audit-session"),
        "执行写入。",
        task_id="task-audit-failure",
    )

    assert turn.task.status is TaskStatus.FAILED
    assert turn.error_message == "审计事件记录失败，已停止工具操作"
    assert handler_calls == []
