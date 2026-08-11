"""验证 Prompt Builder 在真实 Agent Loop 调用路径中的效果。"""

from __future__ import annotations

from collections.abc import Sequence

from hermes_lite.agent_loop import TextAgent
from hermes_lite.domain import Message, MessageRole, Session, TaskStatus
from hermes_lite.prompt_builder import ContextBudget, PromptBuilder
from hermes_lite.tool_agent_loop import ToolAgent
from hermes_lite.tool_registry import (
    ToolDefinition,
    ToolRegistry,
    ToolRiskLevel,
)


class FixedSummarizer:
    """为运行时测试提供已知的摘要内容。"""

    def __init__(self, summary: str) -> None:
        self.summary = summary
        self.calls: list[tuple[Message, ...]] = []

    def summarize(self, messages: Sequence[Message]) -> str:
        """记录旧历史，并返回预设摘要。"""
        self.calls.append(tuple(messages))
        return self.summary


class SummaryAwareTextModel:
    """只有收到目标摘要时才返回关键事实，验证上下文确实被送达。"""

    def __init__(self) -> None:
        self.calls: list[tuple[Message, ...]] = []

    def ask_messages(self, messages: Sequence[Message]) -> str:
        """检查摘要来源与关键事实，再返回对应回答。"""
        copied_messages = tuple(messages)
        self.calls.append(copied_messages)

        if any(
            message.content is not None
            and "关键事实：项目代号是北斗-17。" in message.content
            for message in copied_messages
        ):
            return "北斗-17"

        return "未找到关键事实"


class RecordingToolModel:
    """记录工具 Agent 的运行时上下文并返回最终文本。"""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Message, ...], tuple[dict[str, object], ...]]] = []

    def respond(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]],
    ) -> Message:
        """保存本轮上下文，确认其中保留摘要和模型工具定义。"""
        self.calls.append((tuple(messages), tuple(tools)))
        return Message(role=MessageRole.ASSISTANT, content="工具上下文已构建。")


def make_compacting_builder() -> PromptBuilder:
    """创建足以保留摘要和最近消息、但会压缩旧消息的预算。"""
    return PromptBuilder(
        ContextBudget(
            max_characters=500,
            recent_message_units=1,
        ),
    )


def make_tool_registry() -> ToolRegistry:
    """创建最小只读工具登记，供工具 Agent 生成真实工具定义。"""
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
                        "description": "要回显的文本。",
                    },
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            risk_level=ToolRiskLevel.READ_ONLY,
            handler=lambda arguments: str(arguments["text"]),
        )
    )
    return registry


def test_text_agent_answers_key_fact_from_runtime_history_summary() -> None:
    """文本 Agent 的实际模型请求应包含可追溯的旧历史摘要。"""
    model = SummaryAwareTextModel()
    summarizer = FixedSummarizer("关键事实：项目代号是北斗-17。")
    agent = TextAgent(
        model,
        prompt_builder=make_compacting_builder(),
        history_summarizer=summarizer,
    )
    session = Session(
        session_id="session-1",
        messages=[
            Message(role=MessageRole.USER, content="项目代号是什么？" + "旧" * 180),
            Message(role=MessageRole.ASSISTANT, content="项目代号已记录。" + "答" * 180),
        ],
    )

    turn = agent.run_turn(
        session,
        "请只回答项目代号。",
        task_id="task-summary",
    )

    assert turn.task.status is TaskStatus.COMPLETED
    assert turn.answer == "北斗-17"
    assert len(summarizer.calls) == 1
    assert len(summarizer.calls[0]) == 2
    assert any(
        message.content is not None
        and "来源：最早的 2 条历史消息" in message.content
        for message in model.calls[0]
    )
    assert not any(
        message.content is not None and "旧" * 100 in message.content
        for message in model.calls[0]
    )


def test_tool_agent_uses_prompt_builder_without_losing_tool_definitions() -> None:
    """工具 Agent 接入 Builder 后仍向模型提供结构化工具定义。"""
    model = RecordingToolModel()
    summarizer = FixedSummarizer("旧工具任务摘要")
    agent = ToolAgent(
        model,
        make_tool_registry(),
        prompt_builder=make_compacting_builder(),
        history_summarizer=summarizer,
    )
    session = Session(
        session_id="session-2",
        messages=[
            Message(role=MessageRole.USER, content="旧工具请求" + "旧" * 180),
            Message(role=MessageRole.ASSISTANT, content="旧工具回答" + "答" * 180),
        ],
    )

    turn = agent.run_turn(session, "请给出最终结论。", task_id="task-tool-summary")

    request_messages, tools = model.calls[0]
    assert turn.task.status is TaskStatus.COMPLETED
    assert turn.answer == "工具上下文已构建。"
    assert len(summarizer.calls) == 1
    assert any(
        message.content is not None and "旧工具任务摘要" in message.content
        for message in request_messages
    )
    assert tools[0]["function"]["name"] == "echo_text"


def test_prompt_builder_error_becomes_safe_agent_failure() -> None:
    """安全规则自身超预算时 Agent 不应调用模型或伪造回答。"""
    model = SummaryAwareTextModel()
    agent = TextAgent(
        model,
        prompt_builder=PromptBuilder(ContextBudget(max_characters=10)),
    )
    session = Session(session_id="session-3")

    turn = agent.run_turn(session, "测试预算失败。", task_id="task-budget")

    assert turn.task.status is TaskStatus.FAILED
    assert turn.answer is None
    assert turn.error_message == "安全规则与任务摘要已超过上下文预算"
    assert model.calls == []
    assert session.messages == [Message(role=MessageRole.USER, content="测试预算失败。")]
