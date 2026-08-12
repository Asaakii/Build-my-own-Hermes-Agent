"""验证持久化聊天运行层不改变现有 Agent 安全边界。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

import hermes_lite.sqlite_state_store as state_store_module
from hermes_lite.chat_runtime import ChatRuntime
from hermes_lite.confirmation_policy import ConfirmationManager
from hermes_lite.domain import Message, MessageRole, TaskStatus, ToolCall
from hermes_lite.sqlite_state_store import SQLiteStateStore, load_sqlite_state_config
from hermes_lite.tool_agent_loop import ToolAgent
from hermes_lite.tool_registry import (
    ToolDefinition,
    ToolRegistry,
    ToolRiskLevel,
)


class ScriptedToolModel:
    """按预设顺序返回文本或工具调用的离线模型替身。"""

    def __init__(self, responses: list[Message]) -> None:
        self._responses = responses
        self.calls: list[list[Message]] = []

    def respond(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]],
    ) -> Message:
        """记录输入并返回下一条预设响应。"""
        del tools
        self.calls.append(list(messages))
        return self._responses.pop(0)


@pytest.fixture
def state_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> SQLiteStateStore:
    """创建独立项目中的 SQLite 状态库。"""
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(state_store_module, "PROJECT_ROOT", project_root)
    store = SQLiteStateStore(load_sqlite_state_config({}))
    store.initialize()
    return store


def test_chat_runtime_restores_history_and_persists_generic_task(
    state_store: SQLiteStateStore,
) -> None:
    """第二个运行实例应恢复既有会话，并持久化普通聊天任务状态。"""
    first_model = ScriptedToolModel(
        [Message(role=MessageRole.ASSISTANT, content="第一轮回答。")]
    )
    first_runtime = ChatRuntime(
        ToolAgent(first_model, ToolRegistry()),
        state_store,
    )

    first_result = first_runtime.run_turn("chat-session", "第一轮问题")

    assert first_result.restored_existing_session is False
    assert first_result.turn.task.status is TaskStatus.COMPLETED
    stored_task = state_store.load_coding_task(first_result.turn.task.task_id)
    assert stored_task is not None
    assert stored_task.report is None

    second_model = ScriptedToolModel(
        [Message(role=MessageRole.ASSISTANT, content="第二轮回答。")]
    )
    second_runtime = ChatRuntime(
        ToolAgent(second_model, ToolRegistry()),
        state_store,
    )

    second_result = second_runtime.run_turn("chat-session", "第二轮问题")

    assert second_result.restored_existing_session is True
    assert [message.role for message in second_model.calls[0]] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
    ]
    assert [message.content for message in second_model.calls[0][-3:]] == [
        "第一轮问题",
        "第一轮回答。",
        "第二轮问题",
    ]


def test_chat_runtime_never_persists_confirmation_token(
    state_store: SQLiteStateStore,
) -> None:
    """高风险调用的令牌只能留在当次内存结果，不能写入会话数据库。"""
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="write_text",
            description="写入测试文本。",
            parameters_schema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "待写入内容。",
                    }
                },
                "required": ["content"],
                "additionalProperties": False,
            },
            risk_level=ToolRiskLevel.WRITE,
            handler=lambda arguments: "写入完成",
        )
    )
    model = ScriptedToolModel(
        [
            Message(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=(
                    ToolCall(
                        call_id="call-confirm-1",
                        tool_name="write_text",
                        arguments={"content": "PRIVATE_TOOL_ARGUMENT"},
                    ),
                ),
            )
        ]
    )
    runtime = ChatRuntime(
        ToolAgent(
            model,
            registry,
            confirmation_manager=ConfirmationManager(
                token_factory=lambda: "PRIVATE_CONFIRM_TOKEN",
            ),
        ),
        state_store,
    )

    result = runtime.run_turn("chat-session", "执行写入")
    confirmed_result = runtime.run_turn(
        "chat-session",
        "/confirm PRIVATE_CONFIRM_TOKEN",
    )
    restored = state_store.load_session("chat-session")
    stored_task = state_store.load_coding_task(confirmed_result.turn.task.task_id)

    assert result.turn.task.status is TaskStatus.BLOCKED
    assert result.turn.pending_confirmation is not None
    assert result.turn.pending_confirmation.token == "PRIVATE_CONFIRM_TOKEN"
    assert confirmed_result.turn.task.status is TaskStatus.COMPLETED
    assert restored is not None
    assert stored_task is not None
    persisted_content = "\n".join(
        message.content or "" for message in restored.messages
    )
    assert "PRIVATE_CONFIRM_TOKEN" not in persisted_content
    assert "PRIVATE_CONFIRM_TOKEN" not in stored_task.task.user_request
    assert stored_task.task.user_request == "确认命令（令牌已隐藏）"
