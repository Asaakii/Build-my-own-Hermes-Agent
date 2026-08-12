"""HermesLite 的任务关联、安全结构化运行日志。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
import json
import logging
import re


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9:_-]+$")
_TASK_ID_CONTEXT: ContextVar[str | None] = ContextVar(
    "hermes_lite_task_id",
    default=None,
)


class RuntimeLogError(ValueError):
    """表示运行日志的事件或安全字段不符合约束。"""


class RuntimeEvent(str, Enum):
    """运行日志允许记录的固定事件名称。"""

    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_BLOCKED = "task_blocked"
    MODEL_REQUEST_STARTED = "model_request_started"
    MODEL_REQUEST_RETRY = "model_request_retry"
    MODEL_REQUEST_FAILED = "model_request_failed"
    MODEL_REQUEST_COMPLETED = "model_request_completed"
    TOOL_REQUESTED = "tool_requested"
    TOOL_FINISHED = "tool_finished"


def _require_identifier(value: object, field_name: str) -> str:
    """验证任务与工具标识只包含可安全输出的固定字符。"""
    if not isinstance(value, str):
        raise RuntimeLogError(f"{field_name} 必须是文本")

    normalized_value = value.strip()
    if not normalized_value or not _IDENTIFIER_PATTERN.fullmatch(normalized_value):
        raise RuntimeLogError(f"{field_name} 格式无效")

    return normalized_value


def _require_positive_int(value: object, field_name: str) -> int:
    """验证轮次和尝试次数是非布尔值的正整数。"""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeLogError(f"{field_name} 必须是正整数")

    return value


@contextmanager
def task_log_context(task_id: object) -> Iterator[None]:
    """在当前调用链中临时绑定任务标识，供模型客户端关联日志。"""
    normalized_task_id = _require_identifier(task_id, "task_id")
    token = _TASK_ID_CONTEXT.set(normalized_task_id)
    try:
        yield
    finally:
        _TASK_ID_CONTEXT.reset(token)


def emit_runtime_event(
    logger: logging.Logger,
    event: RuntimeEvent,
    *,
    task_id: object | None = None,
    tool_name: object | None = None,
    round_number: object | None = None,
    error_kind: object | None = None,
    attempt: object | None = None,
    max_attempts: object | None = None,
    task_status: object | None = None,
    level: int = logging.INFO,
) -> None:
    """写入固定字段日志；不接受提示词、参数、输出或异常原文。"""
    if not isinstance(logger, logging.Logger):
        raise RuntimeLogError("logger 必须是 logging.Logger")
    if not isinstance(event, RuntimeEvent):
        raise RuntimeLogError("event 必须是 RuntimeEvent")
    if isinstance(level, bool) or not isinstance(level, int):
        raise RuntimeLogError("level 必须是整数")

    effective_task_id = task_id if task_id is not None else _TASK_ID_CONTEXT.get()
    normalized_task_id = (
        _require_identifier(effective_task_id, "task_id")
        if effective_task_id is not None
        else None
    )
    normalized_tool_name = (
        _require_identifier(tool_name, "tool_name")
        if tool_name is not None
        else None
    )
    normalized_round_number = (
        _require_positive_int(round_number, "round_number")
        if round_number is not None
        else None
    )
    normalized_error_kind = (
        _require_identifier(error_kind, "error_kind")
        if error_kind is not None
        else None
    )
    normalized_attempt = (
        _require_positive_int(attempt, "attempt")
        if attempt is not None
        else None
    )
    normalized_max_attempts = (
        _require_positive_int(max_attempts, "max_attempts")
        if max_attempts is not None
        else None
    )
    normalized_task_status = (
        _require_identifier(task_status, "task_status")
        if task_status is not None
        else None
    )

    logger.log(
        level,
        "runtime_event=%s",
        event.value,
        extra={
            "event": event.value,
            "task_id": normalized_task_id,
            "tool_name": normalized_tool_name,
            "round_number": normalized_round_number,
            "error_kind": normalized_error_kind,
            "attempt": normalized_attempt,
            "max_attempts": normalized_max_attempts,
            "task_status": normalized_task_status,
        },
    )


class SafeJsonFormatter(logging.Formatter):
    """只将固定运行字段序列化为 JSON，不输出日志调用方的任意属性。"""

    def format(self, record: logging.LogRecord) -> str:
        """从 LogRecord 白名单字段构造一条可查询的 JSON 日志。"""
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", None),
            "task_id": getattr(record, "task_id", None),
            "tool_name": getattr(record, "tool_name", None),
            "round_number": getattr(record, "round_number", None),
            "error_kind": getattr(record, "error_kind", None),
            "attempt": getattr(record, "attempt", None),
            "max_attempts": getattr(record, "max_attempts", None),
            "task_status": getattr(record, "task_status", None),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
