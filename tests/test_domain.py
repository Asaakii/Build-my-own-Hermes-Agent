"""验证核心领域数据模型的正常路径和安全边界。"""

import pytest

from hermes_lite.domain import (
    Message,
    MessageRole,
    Session,
    TaskState,
    TaskStatus,
    ToolCall,
    ToolResult,
)


def test_message_strips_surrounding_whitespace() -> None:
    """消息内容应去除首尾空白后再保存。"""
    message = Message(role=MessageRole.USER, content="  修复测试  ")

    assert message.content == "修复测试"


@pytest.mark.parametrize("content", ["", "   ", 123])
def test_message_rejects_empty_or_non_text_content(content: object) -> None:
    """空消息和非文本消息不能进入会话。"""
    with pytest.raises(ValueError):
        Message(role=MessageRole.USER, content=content)  # type: ignore[arg-type]


def test_tool_call_accepts_valid_name_and_arguments() -> None:
    """合法工具名和字典参数可以构成工具调用。"""
    tool_call = ToolCall(
        call_id="call-1",
        tool_name="read_file",
        arguments={"path": "example.py"},
    )

    assert tool_call.tool_name == "read_file"


@pytest.mark.parametrize("tool_name", ["../read_file", "read-file", "ReadFile"])
def test_tool_call_rejects_unsafe_or_invalid_tool_name(tool_name: str) -> None:
    """工具名不能借由路径或特殊字符绕过后续注册表。"""
    with pytest.raises(ValueError):
        ToolCall(call_id="call-1", tool_name=tool_name, arguments={})


def test_tool_result_and_task_state_keep_structured_status() -> None:
    """工具观察结果和任务状态应保留明确类型。"""
    result = ToolResult(
        call_id="call-1",
        tool_name="read_file",
        content="文件内容",
    )
    task = TaskState(
        task_id="task-1",
        session_id="session-1",
        user_request="读取文件",
        status=TaskStatus.RUNNING,
    )

    assert result.is_error is False
    assert task.status is TaskStatus.RUNNING


def test_session_rejects_non_message_items() -> None:
    """会话历史不能混入未验证的任意对象。"""
    with pytest.raises(ValueError):
        Session(session_id="session-1", messages=["not-a-message"])  # type: ignore[list-item]


def test_tool_result_rejects_invalid_tool_name() -> None:
    """工具结果也不能携带路径形式的工具名。"""
    with pytest.raises(ValueError):
        ToolResult(
            call_id="call-1",
            tool_name="../read_file",
            content="不应接受",
        )


@pytest.mark.parametrize("tool_rounds", [-1, True, "1"])
def test_task_state_rejects_invalid_tool_rounds(tool_rounds: object) -> None:
    """工具轮次只能是非布尔值的非负整数。"""
    with pytest.raises(ValueError):
        TaskState(
            task_id="task-1",
            session_id="session-1",
            user_request="读取文件",
            tool_rounds=tool_rounds,  # type: ignore[arg-type]
        )


def test_tool_call_rejects_empty_call_id() -> None:
    """工具调用必须拥有可关联结果的标识。"""
    with pytest.raises(ValueError):
        ToolCall(
            call_id="   ",
            tool_name="read_file",
            arguments={},
        )
