"""验证最终回归基线中的跨层安全拒绝场景。"""

from __future__ import annotations

from collections.abc import Sequence

from hermes_lite.domain import Message, MessageRole, Session, ToolCall
from hermes_lite.tool_agent_loop import ToolAgent
from hermes_lite.tool_registry import (
    ToolDefinition,
    ToolRegistry,
    ToolRiskLevel,
)


class ScriptedModel:
    """返回预设模型决策，并记录 Agent 实际提供的观察结果。"""

    def __init__(self, responses: list[Message]) -> None:
        self._responses = responses
        self.calls: list[list[Message]] = []

    def respond(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]],
    ) -> Message:
        """记录上下文后返回下一条离线响应。"""
        del tools
        self.calls.append(list(messages))
        return self._responses.pop(0)


def test_unexpected_model_tool_argument_cannot_reach_handler() -> None:
    """模型越过参数模式时，Agent 必须拒绝而不能执行工具。"""
    handler_calls: list[dict[str, object]] = []

    def safe_handler(arguments: dict[str, object]) -> str:
        """记录真正获得执行许可的参数。"""
        handler_calls.append(arguments)
        return "不应执行。"

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo_text",
            description="回显文本。",
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "待回显的文本。",
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            risk_level=ToolRiskLevel.READ_ONLY,
            handler=safe_handler,
        )
    )
    model = ScriptedModel(
        [
            Message(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=(
                    ToolCall(
                        call_id="call-invalid-argument",
                        tool_name="echo_text",
                        arguments={
                            "text": "正常字段",
                            "unexpected": "模型伪造字段",
                        },
                    ),
                ),
            ),
            Message(role=MessageRole.ASSISTANT, content="工具参数无效，已停止。"),
        ]
    )

    turn = ToolAgent(model, registry).run_turn(
        Session(session_id="regression-session"),
        "请回显文本。",
        task_id="regression-task",
    )

    assert turn.answer == "工具参数无效，已停止。"
    assert handler_calls == []
    assert turn.tool_results[0].is_error is True
    assert "不允许额外参数" in turn.tool_results[0].content
    assert model.calls[1][-1].role is MessageRole.TOOL
    assert "模型伪造字段" not in model.calls[1][-1].content
