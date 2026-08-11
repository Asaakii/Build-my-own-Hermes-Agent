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


def test_assistant_tool_request_keeps_structured_calls() -> None:
    """助手请求工具时不使用伪造的文本内容。"""
    call = ToolCall(
        call_id="call-1",
        tool_name="summarize_text",
        arguments={"text": "测试"},
    )

    message = Message(
        role=MessageRole.ASSISTANT,
        content=None,
        tool_calls=(call,),
    )

    assert message.content is None
    assert message.tool_calls == (call,)


def test_assistant_tool_request_rejects_text_content() -> None:
    """工具请求和普通文本回答不能混为同一种消息。"""
    call = ToolCall(
        call_id="call-1",
        tool_name="summarize_text",
        arguments={"text": "测试"},
    )

    with pytest.raises(ValueError, match="助手工具请求不能包含文本内容"):
        Message(
            role=MessageRole.ASSISTANT,
            content="同时回答并调用工具",
            tool_calls=(call,),
        )


def test_assistant_rejects_non_tuple_tool_calls() -> None:
    """工具调用集合使用元组，避免会话消息创建后被修改。"""
    call = ToolCall(
        call_id="call-1",
        tool_name="summarize_text",
        arguments={"text": "测试"},
    )

    with pytest.raises(ValueError, match="tool_calls 必须是元组"):
        Message(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[call],  # type: ignore[arg-type]
        )


def test_assistant_text_message_still_requires_content() -> None:
    """未请求工具的助手消息仍必须提供非空文本。"""
    with pytest.raises(ValueError, match="content 必须是文本"):
        Message(role=MessageRole.ASSISTANT, content=None)


def test_tool_message_requires_call_id() -> None:
    """工具结果必须能对应到此前的某一次工具调用。"""
    with pytest.raises(ValueError, match="tool_call_id 必须是文本"):
        Message(role=MessageRole.TOOL, content="工具结果")


def test_user_message_rejects_tool_call_id() -> None:
    """用户消息不能伪装为工具结果。"""
    with pytest.raises(ValueError, match="不能包含 tool_call_id"):
        Message(
            role=MessageRole.USER,
            content="用户问题",
            tool_call_id="call-1",
        )


def test_tool_message_rejects_tool_calls() -> None:
    """工具结果消息不能继续携带新的工具请求。"""
    call = ToolCall(
        call_id="call-2",
        tool_name="summarize_text",
        arguments={"text": "测试"},
    )

    with pytest.raises(ValueError, match="不能包含工具调用"):
        Message(
            role=MessageRole.TOOL,
            content="工具结果",
            tool_call_id="call-1",
            tool_calls=(call,),
        )


def test_message_from_tool_result_creates_structured_tool_message() -> None:
    """受控工具结果应以调用标识写回会话。"""
    result = ToolResult(
        call_id="call-1",
        tool_name="summarize_text",
        content="摘要完成",
    )

    message = Message.from_tool_result(result)

    assert message.role is MessageRole.TOOL
    assert message.content == "摘要完成"
    assert message.tool_call_id == "call-1"
