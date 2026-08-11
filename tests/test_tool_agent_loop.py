"""验证离线工具调用 Agent Loop 的状态与停止边界。"""

from collections.abc import Sequence

import pytest

from hermes_lite.domain import (
    Message,
    MessageRole,
    Session,
    ToolCall,
)
from hermes_lite.model_client import ModelClientError
from hermes_lite.tool_agent_loop import ToolAgent
from hermes_lite.tool_registry import (
    ToolDefinition,
    ToolRegistry,
    ToolRiskLevel,
)


class ScriptedToolModel:
    """按预设消息顺序模拟工具感知模型。"""

    def __init__(
        self,
        responses: list[Message] | None = None,
        error: ModelClientError | None = None,
    ) -> None:
        self.responses = responses or []
        self.error = error
        self.calls: list[tuple[list[Message], list[dict[str, object]]]] = []

    def respond(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]],
    ) -> Message:
        """记录模型输入，并返回预设消息或抛出预设错误。"""
        self.calls.append((list(messages), list(tools)))

        if self.error is not None:
            raise self.error

        if not self.responses:
            raise AssertionError("测试模型缺少预设响应")

        return self.responses.pop(0)


def echo_handler(arguments: dict[str, object]) -> str:
    """返回确定性文本，不执行任何外部操作。"""
    return f"回显：{arguments['text']}"


def make_registry() -> ToolRegistry:
    """创建只含一个安全测试工具的注册表。"""
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo_text",
            description="回显给定文本。",
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "需要回显的文本。",
                    },
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            risk_level=ToolRiskLevel.READ_ONLY,
            handler=echo_handler,
        )
    )
    return registry


def make_tool_request(
    call_id: str = "call-1",
    tool_name: str = "echo_text",
) -> Message:
    """创建一条助手工具请求消息。"""
    return Message(
        role=MessageRole.ASSISTANT,
        content=None,
        tool_calls=(
            ToolCall(
                call_id=call_id,
                tool_name=tool_name,
                arguments={"text": "循环验证"},
            ),
        ),
    )


def test_tool_agent_executes_tool_then_returns_final_answer() -> None:
    """工具结果写回会话后，模型可以给出最终文本回答。"""
    model = ScriptedToolModel(
        responses=[
            make_tool_request(),
            Message(role=MessageRole.ASSISTANT, content="工具循环完成。"),
        ]
    )
    agent = ToolAgent(model, make_registry())
    session = Session(session_id="session-1")

    turn = agent.run_turn(session, "请使用工具回显循环验证。", task_id="task-1")

    assert turn.answer == "工具循环完成。"
    assert turn.error_message is None
    assert turn.task.tool_rounds == 1
    assert turn.tool_results[0].content == "回显：循环验证"
    assert turn.tool_results[0].is_error is False
    assert [message.role for message in session.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]

    second_request_messages, _ = model.calls[1]
    assert second_request_messages[-1].role is MessageRole.TOOL
    assert second_request_messages[-1].tool_call_id == "call-1"


def test_tool_agent_returns_error_observation_to_model() -> None:
    """未知工具的失败结果也应写回会话，供模型继续处理。"""
    model = ScriptedToolModel(
        responses=[
            make_tool_request(tool_name="unknown_tool"),
            Message(role=MessageRole.ASSISTANT, content="工具不可用，已停止。"),
        ]
    )
    agent = ToolAgent(model, make_registry())
    session = Session(session_id="session-1")

    turn = agent.run_turn(session, "测试未知工具。", task_id="task-1")

    assert turn.answer == "工具不可用，已停止。"
    assert turn.tool_results[0].is_error is True
    assert "未知工具" in turn.tool_results[0].content
    assert session.messages[2].role is MessageRole.TOOL


def test_tool_agent_stops_after_maximum_tool_rounds() -> None:
    """模型重复请求工具时，循环必须在上限处停止。"""
    model = ScriptedToolModel(
        responses=[
            make_tool_request(call_id="call-1"),
            make_tool_request(call_id="call-2"),
        ]
    )
    agent = ToolAgent(model, make_registry(), max_tool_rounds=1)
    session = Session(session_id="session-1")

    turn = agent.run_turn(session, "持续请求工具。", task_id="task-1")

    assert turn.answer is None
    assert turn.error_message == "工具调用次数超过上限"
    assert turn.task.tool_rounds == 1
    assert len(turn.tool_results) == 1
    assert [message.role for message in session.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]


def test_tool_agent_rejects_non_assistant_model_message() -> None:
    """模型不能把用户或工具消息伪装成自身决策。"""
    model = ScriptedToolModel(
        responses=[Message(role=MessageRole.USER, content="错误角色")]
    )
    agent = ToolAgent(model, make_registry())
    session = Session(session_id="session-1")

    turn = agent.run_turn(session, "测试角色边界。", task_id="task-1")

    assert turn.answer is None
    assert turn.error_message == "模型返回的 Agent 消息必须是 assistant 角色"
    assert session.messages == [
        Message(role=MessageRole.USER, content="测试角色边界。"),
    ]


def test_tool_agent_keeps_user_message_when_model_fails() -> None:
    """模型服务失败时保留用户输入，不伪造助手或工具消息。"""
    model = ScriptedToolModel(
        error=ModelClientError("模型请求失败，请稍后再试")
    )
    agent = ToolAgent(model, make_registry())
    session = Session(session_id="session-1")

    turn = agent.run_turn(session, "测试模型失败。", task_id="task-1")

    assert turn.answer is None
    assert turn.error_message == "模型请求失败，请稍后再试"
    assert turn.tool_results == ()
    assert session.messages == [
        Message(role=MessageRole.USER, content="测试模型失败。"),
    ]


@pytest.mark.parametrize("max_tool_rounds", [0, True])
def test_tool_agent_rejects_invalid_max_tool_rounds(
    max_tool_rounds: object,
) -> None:
    """轮次上限必须是非布尔值的正整数。"""
    with pytest.raises(ValueError, match="max_tool_rounds 必须是正整数"):
        ToolAgent(
            ScriptedToolModel(),
            make_registry(),
            max_tool_rounds=max_tool_rounds,  # type: ignore[arg-type]
        )

def test_tool_agent_groups_multiple_calls_in_one_round() -> None:
    """同一条助手工具请求中的多个调用应归入同一轮摘要。"""
    model = ScriptedToolModel(
        responses=[
            Message(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=(
                    ToolCall(
                        call_id="call-1",
                        tool_name="echo_text",
                        arguments={"text": "第一项"},
                    ),
                    ToolCall(
                        call_id="call-2",
                        tool_name="echo_text",
                        arguments={"text": "第二项"},
                    ),
                ),
            ),
            Message(role=MessageRole.ASSISTANT, content="同轮调用完成。"),
        ]
    )
    agent = ToolAgent(model, make_registry())

    turn = agent.run_turn(Session(session_id="session-1"), "执行两项调用。")

    assert turn.task.tool_rounds == 1
    assert len(turn.round_summaries) == 1
    assert [result.call_id for result in turn.round_summaries[0].results] == [
        "call-1",
        "call-2",
    ]
    assert turn.tool_results == turn.round_summaries[0].results
