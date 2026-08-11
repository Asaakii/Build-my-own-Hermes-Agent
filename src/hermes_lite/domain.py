"""HermesLite 的核心领域数据模型。

这些模型只描述“系统中有什么数据”，暂不处理模型调用、工具执行或数据库存储。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re


_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _require_text(value: object, field_name: str) -> str:
    """验证必填文本，并返回去除首尾空白后的值。"""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} 必须是文本")

    cleaned_value = value.strip()
    if not cleaned_value:
        raise ValueError(f"{field_name} 不能为空")

    return cleaned_value

def require_tool_name(value: object) -> str:
    """验证工具名称不会携带路径或特殊字符。"""
    tool_name = _require_text(value, "tool_name")

    if not _TOOL_NAME_PATTERN.fullmatch(tool_name):
        raise ValueError("tool_name 只能包含小写字母、数字和下划线")

    return tool_name

class MessageRole(str, Enum):
    """会话消息允许使用的角色。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TaskStatus(str, Enum):
    """任务在 Agent Loop 中的生命周期状态。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class Message:
    """一条会话消息，支持文本、工具请求与工具结果。"""

    role: MessageRole
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        """验证不同消息角色允许使用的字段组合。"""
        if not isinstance(self.role, MessageRole):
            raise ValueError("role 必须是 MessageRole")

        if not isinstance(self.tool_calls, tuple):
            raise ValueError("tool_calls 必须是元组")

        if not all(isinstance(call, ToolCall) for call in self.tool_calls):
            raise ValueError("tool_calls 中的元素必须是 ToolCall")

        if self.role in {MessageRole.SYSTEM, MessageRole.USER}:
            if self.tool_calls:
                raise ValueError("系统和用户消息不能包含工具调用")

            if self.tool_call_id is not None:
                raise ValueError("系统和用户消息不能包含 tool_call_id")

            object.__setattr__(
                self,
                "content",
                _require_text(self.content, "content"),
            )
            return

        if self.role is MessageRole.ASSISTANT:
            if self.tool_call_id is not None:
                raise ValueError("助手消息不能包含 tool_call_id")

            if self.tool_calls:
                if self.content is not None:
                    raise ValueError("助手工具请求不能包含文本内容")
                return

            object.__setattr__(
                self,
                "content",
                _require_text(self.content, "content"),
            )
            return

        if self.tool_calls:
            raise ValueError("工具结果消息不能包含工具调用")

        object.__setattr__(
            self,
            "content",
            _require_text(self.content, "content"),
        )
        object.__setattr__(
            self,
            "tool_call_id",
            _require_text(self.tool_call_id, "tool_call_id"),
        )

    @classmethod
    def from_tool_result(cls, result: ToolResult) -> Message:
        """把受控工具结果转换为会话中的工具消息。"""
        if not isinstance(result, ToolResult):
            raise ValueError("result 必须是 ToolResult")

        return cls(
            role=MessageRole.TOOL,
            content=result.content,
            tool_call_id=result.call_id,
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """模型提出的一次结构化工具调用请求。"""

    call_id: str
    tool_name: str
    arguments: dict[str, object]

    def __post_init__(self) -> None:
        """验证工具标识、名称格式和参数容器。"""
        object.__setattr__(self, "call_id", _require_text(self.call_id, "call_id"))
        object.__setattr__(self, "tool_name", require_tool_name(self.tool_name))

        if not isinstance(self.arguments, dict):
            raise ValueError("arguments 必须是字典")


@dataclass(frozen=True, slots=True)
class ToolResult:
    """工具层返回给 Agent Loop 的观察结果。"""

    call_id: str
    tool_name: str
    content: str
    is_error: bool = False

    def __post_init__(self) -> None:
        """验证结果能够对应一次合法工具调用。"""
        object.__setattr__(self, "call_id", _require_text(self.call_id, "call_id"))
        object.__setattr__(self, "tool_name", require_tool_name(self.tool_name))
        object.__setattr__(self, "content", _require_text(self.content, "content"))

        if not isinstance(self.is_error, bool):
            raise ValueError("is_error 必须是布尔值")


@dataclass(slots=True)
class TaskState:
    """单个任务的运行状态，不负责数据库持久化。"""

    task_id: str
    session_id: str
    user_request: str
    status: TaskStatus = TaskStatus.PENDING
    tool_rounds: int = 0

    def __post_init__(self) -> None:
        """验证任务的基本身份和初始轮次。"""
        self.task_id = _require_text(self.task_id, "task_id")
        self.session_id = _require_text(self.session_id, "session_id")
        self.user_request = _require_text(self.user_request, "user_request")

        if not isinstance(self.status, TaskStatus):
            raise ValueError("status 必须是 TaskStatus")

        if (
            isinstance(self.tool_rounds, bool)
            or not isinstance(self.tool_rounds, int)
            or self.tool_rounds < 0
        ):
            raise ValueError("tool_rounds 必须是非负整数")


@dataclass(slots=True)
class Session:
    """一个会话的内存态容器；阶段 3 再将它保存到 SQLite。"""

    session_id: str
    messages: list[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        """验证会话标识与已有消息。"""
        self.session_id = _require_text(self.session_id, "session_id")

        if not all(isinstance(message, Message) for message in self.messages):
            raise ValueError("messages 中的元素必须是 Message")