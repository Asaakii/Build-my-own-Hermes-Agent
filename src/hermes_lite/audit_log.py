"""HermesLite 的最小化工具审计事件与参数脱敏。"""

from __future__ import annotations

from typing import Protocol
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import runtime_checkable

from hermes_lite.domain import ToolCall, require_tool_name


class AuditLogError(ValueError):
    """表示审计事件或审计接收器不符合最小化记录规则。"""


class AuditEventType(str, Enum):
    """可追溯的工具生命周期事件类型。"""

    TOOL_REQUESTED = "tool_requested"
    CONFIRMATION_REQUIRED = "confirmation_required"
    CONFIRMATION_ACCEPTED = "confirmation_accepted"
    CONFIRMATION_REJECTED = "confirmation_rejected"
    TOOL_EXECUTED = "tool_executed"
    TOOL_REJECTED = "tool_rejected"


def _require_text(value: object, field_name: str) -> str:
    """验证非空文本并去除首尾空白。"""
    if not isinstance(value, str):
        raise AuditLogError(f"{field_name} 必须是文本")

    cleaned_value = value.strip()
    if not cleaned_value:
        raise AuditLogError(f"{field_name} 不能为空")

    return cleaned_value


def _utc_now() -> datetime:
    """为事件生成带 UTC 时区的创建时间。"""
    return datetime.now(UTC)


def _require_utc_time(value: object) -> datetime:
    """验证并规范化事件时间，避免保存无时区时间。"""
    if not isinstance(value, datetime):
        raise AuditLogError("created_at 必须是 datetime")

    if value.tzinfo is None or value.utcoffset() is None:
        raise AuditLogError("created_at 必须包含时区")

    return value.astimezone(UTC)


def summarize_tool_arguments(arguments: object) -> tuple[tuple[str, str], ...]:
    """只保存参数名、基础类型和文本长度，绝不保存原始参数值。"""
    if not isinstance(arguments, dict):
        raise AuditLogError("arguments 必须是字典")

    summary: list[tuple[str, str]] = []
    for raw_name, value in arguments.items():
        parameter_name = _require_text(raw_name, "parameter_name")

        if isinstance(value, str):
            descriptor = f"string(length={len(value)})"
        elif isinstance(value, bool):
            descriptor = "boolean"
        elif isinstance(value, int):
            descriptor = "integer"
        elif isinstance(value, float):
            descriptor = "number"
        elif value is None:
            descriptor = "null"
        else:
            descriptor = "non_primitive"

        summary.append((parameter_name, descriptor))

    return tuple(sorted(summary))


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """不携带原始工具参数或工具输出的最小审计记录。"""

    session_id: str
    task_id: str
    event_type: AuditEventType
    tool_name: str | None = None
    call_id: str | None = None
    argument_summary: tuple[tuple[str, str], ...] = ()
    reason_code: str | None = None
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        """验证可追溯身份、固定事件类型与安全摘要格式。"""
        object.__setattr__(self, "session_id", _require_text(self.session_id, "session_id"))
        object.__setattr__(self, "task_id", _require_text(self.task_id, "task_id"))

        if not isinstance(self.event_type, AuditEventType):
            raise AuditLogError("event_type 必须是 AuditEventType")

        if self.tool_name is not None:
            object.__setattr__(self, "tool_name", require_tool_name(self.tool_name))

        if self.call_id is not None:
            object.__setattr__(self, "call_id", _require_text(self.call_id, "call_id"))

        if not isinstance(self.argument_summary, tuple):
            raise AuditLogError("argument_summary 必须是元组")

        normalized_summary: list[tuple[str, str]] = []
        for item in self.argument_summary:
            if not isinstance(item, tuple) or len(item) != 2:
                raise AuditLogError("argument_summary 项必须是二元组")
            normalized_summary.append(
                (_require_text(item[0], "parameter_name"), _require_text(item[1], "parameter_summary"))
            )
        object.__setattr__(self, "argument_summary", tuple(normalized_summary))

        if self.reason_code is not None:
            object.__setattr__(
                self,
                "reason_code",
                _require_text(self.reason_code, "reason_code"),
            )

        object.__setattr__(self, "created_at", _require_utc_time(self.created_at))

    @classmethod
    def for_tool_call(
        cls,
        *,
        session_id: str,
        task_id: str,
        event_type: AuditEventType,
        tool_call: ToolCall,
        reason_code: str | None = None,
    ) -> AuditEvent:
        """从 ToolCall 创建不含原始参数值的审计事件。"""
        if not isinstance(tool_call, ToolCall):
            raise AuditLogError("tool_call 必须是 ToolCall")

        return cls(
            session_id=session_id,
            task_id=task_id,
            event_type=event_type,
            tool_name=tool_call.tool_name,
            call_id=tool_call.call_id,
            argument_summary=summarize_tool_arguments(tool_call.arguments),
            reason_code=reason_code,
        )


@runtime_checkable
class AuditEventRecorder(Protocol):
    """ToolAgent 依赖的最小审计写入接口。"""

    def record_audit_event(self, event: AuditEvent) -> None:
        """持久或暂存一条已脱敏审计事件。"""


class InMemoryAuditLog:
    """用于默认运行和测试的进程内审计接收器。"""

    def __init__(self) -> None:
        """创建为空的事件列表。"""
        self.events: list[AuditEvent] = []

    def record_audit_event(self, event: AuditEvent) -> None:
        """追加一条已验证事件，不修改其内容。"""
        if not isinstance(event, AuditEvent):
            raise AuditLogError("event 必须是 AuditEvent")

        self.events.append(event)
