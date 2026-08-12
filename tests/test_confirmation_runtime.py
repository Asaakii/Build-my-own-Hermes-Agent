"""验证 ToolAgent 对高风险调用的本地确认流程。"""

from collections.abc import Sequence

from hermes_lite.confirmation_policy import ConfirmationManager
from hermes_lite.domain import Message, MessageRole, Session, TaskStatus, ToolCall
from hermes_lite.tool_agent_loop import ToolAgent
from hermes_lite.tool_registry import ToolDefinition, ToolRegistry, ToolRiskLevel


class RecordingModel:
    """记录模型调用次数，并返回预设工具请求。"""

    def __init__(self, responses: list[Message]) -> None:
        self._responses = responses
        self.call_count = 0

    def respond(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]],
    ) -> Message:
        """返回预设消息；本地确认命令不应调用这里。"""
        del messages, tools
        self.call_count += 1
        return self._responses.pop(0)


def make_tool_request(tool_name: str = "write_text") -> Message:
    """构造模型提出的一条高风险或只读工具调用。"""
    return Message(
        role=MessageRole.ASSISTANT,
        content=None,
        tool_calls=(
            ToolCall(
                call_id="call-1",
                tool_name=tool_name,
                arguments={"text": "待执行内容"},
            ),
        ),
    )


def make_registry(handler_calls: list[dict[str, object]]) -> ToolRegistry:
    """构造一项只读和一项写入测试工具。"""
    registry = ToolRegistry()

    def handler(arguments: dict[str, object]) -> str:
        handler_calls.append(arguments)
        return f"已处理：{arguments["text"]}"

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


def test_high_risk_call_blocks_before_execution_then_confirm_executes() -> None:
    """高风险工具先受阻；正确确认后只执行保存的原始调用。"""
    handler_calls: list[dict[str, object]] = []
    model = RecordingModel([make_tool_request()])
    manager = ConfirmationManager(token_factory=lambda: "confirm-write")
    agent = ToolAgent(
        model,
        make_registry(handler_calls),
        confirmation_manager=manager,
    )
    session = Session(session_id="session-owner")

    waiting_turn = agent.run_turn(session, "请写入内容。")

    assert waiting_turn.task.status is TaskStatus.BLOCKED
    assert waiting_turn.pending_confirmation is not None
    assert waiting_turn.pending_confirmation.token == "confirm-write"
    assert waiting_turn.tool_results[0].is_error is True
    assert "等待确认" in waiting_turn.tool_results[0].content
    assert handler_calls == []
    assert model.call_count == 1

    confirmed_turn = agent.run_turn(session, "/confirm confirm-write")

    assert confirmed_turn.task.status is TaskStatus.COMPLETED
    assert confirmed_turn.answer == "已确认并执行工具: write_text\n已处理：待执行内容"
    assert confirmed_turn.tool_results[0].is_error is False
    assert handler_calls == [{"text": "待执行内容"}]
    assert model.call_count == 1
    assert [message.role for message in session.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]
    assert all(
        "confirm-write" not in (message.content or "")
        for message in session.messages
    )


def test_confirmation_cannot_be_consumed_by_another_session() -> None:
    """其他会话的确认命令不会执行原始写入，令牌仍留给原会话。"""
    handler_calls: list[dict[str, object]] = []
    model = RecordingModel([make_tool_request()])
    manager = ConfirmationManager(token_factory=lambda: "confirm-owner")
    agent = ToolAgent(
        model,
        make_registry(handler_calls),
        confirmation_manager=manager,
    )
    owner_session = Session(session_id="session-owner")
    agent.run_turn(owner_session, "请写入内容。")

    rejected_turn = agent.run_turn(
        Session(session_id="session-other"),
        "/confirm confirm-owner",
    )

    assert rejected_turn.task.status is TaskStatus.FAILED
    assert rejected_turn.error_message == "确认令牌与会话不匹配"
    assert handler_calls == []

    owner_turn = agent.run_turn(owner_session, "/confirm confirm-owner")
    assert owner_turn.task.status is TaskStatus.COMPLETED
    assert handler_calls == [{"text": "待执行内容"}]


def test_read_only_tool_executes_without_confirmation() -> None:
    """只读工具保留原有直接执行与模型继续回答的行为。"""
    handler_calls: list[dict[str, object]] = []
    model = RecordingModel(
        [
            make_tool_request("read_text"),
            Message(role=MessageRole.ASSISTANT, content="只读完成。"),
        ]
    )
    agent = ToolAgent(model, make_registry(handler_calls))

    turn = agent.run_turn(Session(session_id="read-session"), "读取内容。")

    assert turn.task.status is TaskStatus.COMPLETED
    assert turn.answer == "只读完成。"
    assert turn.pending_confirmation is None
    assert handler_calls == [{"text": "待执行内容"}]
    assert model.call_count == 2


def test_invalid_confirmation_command_stops_without_model_request() -> None:
    """缺少令牌的确认命令被本地拒绝，不应交给模型解释。"""
    handler_calls: list[dict[str, object]] = []
    model = RecordingModel([Message(role=MessageRole.ASSISTANT, content="不应调用")])
    agent = ToolAgent(model, make_registry(handler_calls))

    turn = agent.run_turn(Session(session_id="session-1"), "/confirm")

    assert turn.task.status is TaskStatus.FAILED
    assert turn.error_message == "确认命令格式应为：/confirm <token>"
    assert handler_calls == []
    assert model.call_count == 0
