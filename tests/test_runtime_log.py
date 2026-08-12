"""验证任务关联、安全结构化运行日志。"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from io import StringIO
import json
import logging
from types import SimpleNamespace

import pytest

from hermes_lite.agent_loop import TextAgent
from hermes_lite.config import ModelConfig
from hermes_lite.domain import Message, MessageRole, Session, ToolCall
from hermes_lite.model_client import ModelClient
from hermes_lite.runtime_log import (
    RuntimeEvent,
    RuntimeLogError,
    SafeJsonFormatter,
    emit_runtime_event,
    task_log_context,
)
from hermes_lite.tool_agent_loop import ToolAgent
from hermes_lite.tool_registry import (
    ToolDefinition,
    ToolRegistry,
    ToolRiskLevel,
)


@contextmanager
def capture_json_logs(logger_name: str) -> Iterator[StringIO]:
    """为单个项目 Logger 临时配置 JSON 捕获器，并在结束后恢复原状态。"""
    logger = logging.getLogger(logger_name)
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(SafeJsonFormatter())
    old_level = logger.level
    old_propagate = logger.propagate

    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        yield stream
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate


def read_payloads(stream: StringIO) -> list[dict[str, object]]:
    """将捕获到的每一行 JSON 日志转换为便于断言的字典。"""
    return [json.loads(line) for line in stream.getvalue().splitlines()]


class TextModel:
    """只返回固定文本的最小文本模型替身。"""

    def ask_messages(self, messages: Sequence[Message]) -> str:
        """模拟模型成功回答。"""
        return "文本任务完成。"


class ToolModel:
    """按顺序返回工具调用和最终回答的最小模型替身。"""

    def __init__(self) -> None:
        self._responses = [
            Message(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=(
                    ToolCall(
                        call_id="call-log-1",
                        tool_name="echo_text",
                        arguments={"text": "PRIVATE_TOOL_ARGUMENT"},
                    ),
                ),
            ),
            Message(role=MessageRole.ASSISTANT, content="工具任务完成。"),
        ]

    def respond(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]],
    ) -> Message:
        """忽略输入并返回下一条预设响应。"""
        return self._responses.pop(0)


class FakeCompletions:
    """提供模型客户端所需的最小 completions 接口。"""

    def create(self, **kwargs: object) -> object:
        """返回固定 OpenAI 风格文本响应。"""
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="模型任务完成。"),
                )
            ]
        )


class FakeOpenAIClient:
    """提供模型客户端所需的最小 chat 接口。"""

    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())


def make_registry() -> ToolRegistry:
    """构建只有一个安全回显工具的注册表。"""
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
                        "description": "需要回显的文本。",
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            risk_level=ToolRiskLevel.READ_ONLY,
            handler=lambda arguments: f"回显：{arguments[text]}",
        )
    )
    return registry


def make_model_config() -> ModelConfig:
    """构建不包含真实密钥的模型配置。"""
    return ModelConfig(
        provider="demo",
        model="demo-model",
        api_key="test-secret-key",
        base_url="https://api.example.com",
        timeout_seconds=30.0,
    )


def test_safe_json_formatter_keeps_only_whitelisted_fields() -> None:
    """格式化器不应把原始日志消息或任意附加属性写进 JSON。"""
    with capture_json_logs("tests.runtime_log.formatter") as stream:
        logger = logging.getLogger("tests.runtime_log.formatter")
        logger.warning(
            "PRIVATE_USER_REQUEST and PRIVATE_API_KEY",
            extra={
                "event": "manual_event",
                "task_id": "task-log-1",
                "content": "PRIVATE_TOOL_OUTPUT",
            },
        )

    payload = read_payloads(stream)[0]

    assert payload["event"] == "manual_event"
    assert payload["task_id"] == "task-log-1"
    assert "content" not in payload
    assert "PRIVATE_USER_REQUEST" not in stream.getvalue()
    assert "PRIVATE_API_KEY" not in stream.getvalue()
    assert "PRIVATE_TOOL_OUTPUT" not in stream.getvalue()


def test_runtime_event_uses_context_task_id_and_rejects_unsafe_identifier() -> None:
    """运行事件应继承任务上下文，并拒绝不安全的标识符。"""
    with capture_json_logs("tests.runtime_log.event") as stream:
        logger = logging.getLogger("tests.runtime_log.event")
        with task_log_context("task-log-2"):
            emit_runtime_event(
                logger,
                RuntimeEvent.TOOL_REQUESTED,
                tool_name="echo_text",
                round_number=1,
            )

    payload = read_payloads(stream)[0]
    assert payload["event"] == RuntimeEvent.TOOL_REQUESTED.value
    assert payload["task_id"] == "task-log-2"
    assert payload["tool_name"] == "echo_text"
    assert payload["round_number"] == 1

    with pytest.raises(RuntimeLogError, match="task_id 格式无效"):
        with task_log_context("task with spaces"):
            pass


def test_text_agent_logs_task_lifecycle_without_user_content() -> None:
    """文本 Agent 的任务事件必须可关联且不暴露用户原文。"""
    agent = TextAgent(TextModel())
    session = Session(session_id="session-log-1")

    with capture_json_logs("hermes_lite.agent_loop") as stream:
        turn = agent.run_turn(
            session,
            "PRIVATE_TEXT_AGENT_REQUEST",
            task_id="task-text-log-1",
        )

    payloads = read_payloads(stream)
    assert turn.answer == "文本任务完成。"
    assert [payload["event"] for payload in payloads] == [
        RuntimeEvent.TASK_STARTED.value,
        RuntimeEvent.TASK_COMPLETED.value,
    ]
    assert {payload["task_id"] for payload in payloads} == {"task-text-log-1"}
    assert "PRIVATE_TEXT_AGENT_REQUEST" not in stream.getvalue()


def test_tool_agent_logs_tool_lifecycle_without_arguments_or_output() -> None:
    """工具 Agent 应记录生命周期，而不记录工具参数或回显结果。"""
    agent = ToolAgent(ToolModel(), make_registry())
    session = Session(session_id="session-log-2")

    with capture_json_logs("hermes_lite.tool_agent_loop") as stream:
        turn = agent.run_turn(
            session,
            "PRIVATE_TOOL_AGENT_REQUEST",
            task_id="task-tool-log-1",
        )

    payloads = read_payloads(stream)
    assert turn.answer == "工具任务完成。"
    assert [payload["event"] for payload in payloads] == [
        RuntimeEvent.TASK_STARTED.value,
        RuntimeEvent.TOOL_REQUESTED.value,
        RuntimeEvent.TOOL_FINISHED.value,
        RuntimeEvent.TASK_COMPLETED.value,
    ]
    assert {payload["task_id"] for payload in payloads} == {"task-tool-log-1"}
    assert "PRIVATE_TOOL_AGENT_REQUEST" not in stream.getvalue()
    assert "PRIVATE_TOOL_ARGUMENT" not in stream.getvalue()
    assert "回显" not in stream.getvalue()


def test_model_client_inherits_task_id_without_logging_prompts() -> None:
    """模型事件应继承任务 ID，且不记录系统或用户提示词。"""
    client = ModelClient(make_model_config(), client=FakeOpenAIClient())

    with capture_json_logs("hermes_lite.model_client") as stream:
        with task_log_context("task-model-log-1"):
            answer = client.ask_text(
                "PRIVATE_SYSTEM_PROMPT",
                "PRIVATE_USER_PROMPT",
            )

    payloads = read_payloads(stream)
    assert answer == "模型任务完成。"
    assert [payload["event"] for payload in payloads] == [
        RuntimeEvent.MODEL_REQUEST_STARTED.value,
        RuntimeEvent.MODEL_REQUEST_COMPLETED.value,
    ]
    assert {payload["task_id"] for payload in payloads} == {"task-model-log-1"}
    assert "PRIVATE_SYSTEM_PROMPT" not in stream.getvalue()
    assert "PRIVATE_USER_PROMPT" not in stream.getvalue()
