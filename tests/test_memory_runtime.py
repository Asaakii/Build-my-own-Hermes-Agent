"""验证 /remember 命令与长期记忆的 Agent 运行时边界。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from hermes_lite.agent_loop import TextAgent
from hermes_lite.domain import Message, MessageRole, Session, TaskStatus
from hermes_lite.memory_store import SQLiteMemoryStore, parse_remember_command
from hermes_lite.sqlite_state_store import SQLiteStateStore, load_sqlite_state_config
from hermes_lite.tool_agent_loop import ToolAgent
from hermes_lite.tool_registry import (
    ToolDefinition,
    ToolRegistry,
    ToolRiskLevel,
)
import hermes_lite.sqlite_state_store as state_store_module


class RecordingTextModel:
    """记录文本模型输入，按已授权记忆返回确定性回答。"""

    def __init__(self) -> None:
        self.calls: list[tuple[Message, ...]] = []

    def ask_messages(self, messages: Sequence[Message]) -> str:
        """只有收到已授权项目代号时才返回该代号。"""
        copied_messages = tuple(messages)
        self.calls.append(copied_messages)

        if any(
            message.content is not None and "项目代号是北斗-17" in message.content
            for message in copied_messages
        ):
            return "北斗-17"

        return "未找到长期记忆"


class RecordingToolModel:
    """记录工具模型输入，用于验证长期记忆和工具定义并存。"""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Message, ...], tuple[dict[str, object], ...]]] = []

    def respond(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]],
    ) -> Message:
        """保存上下文并返回确定性最终文本。"""
        self.calls.append((tuple(messages), tuple(tools)))
        return Message(role=MessageRole.ASSISTANT, content="工具任务完成。")


@pytest.fixture
def memory_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> SQLiteMemoryStore:
    """创建绑定独立项目 SQLite 状态的长期记忆存储。"""
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(state_store_module, "PROJECT_ROOT", project_root)
    state_store = SQLiteStateStore(load_sqlite_state_config({}))
    return SQLiteMemoryStore(state_store)


def make_registry() -> ToolRegistry:
    """创建一个最小只读工具，以检查工具定义不会被记忆注入覆盖。"""
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


def test_remember_command_saves_without_model_call(
    memory_store: SQLiteMemoryStore,
) -> None:
    """/remember 由程序处理，不能交给模型假装已保存。"""
    model = RecordingTextModel()
    agent = TextAgent(model, memory_store=memory_store)
    session = Session(session_id="session-save")

    turn = agent.run_turn(
        session,
        "/remember 项目代号是北斗-17",
        task_id="task-save",
    )

    assert turn.task.status is TaskStatus.COMPLETED
    assert turn.answer == "长期记忆已保存。"
    assert model.calls == []
    assert [message.content for message in session.messages] == [
        "/remember 项目代号是北斗-17",
        "长期记忆已保存。",
    ]
    assert [memory.content for memory in memory_store.list_memories()] == [
        "项目代号是北斗-17",
    ]


def test_authorized_memory_is_injected_cross_session_but_chat_text_is_not(
    memory_store: SQLiteMemoryStore,
) -> None:
    """授权内容可跨会话注入，未授权聊天内容不会进入模型上下文。"""
    saver = TextAgent(RecordingTextModel(), memory_store=memory_store)
    saver.run_turn(
        Session(session_id="session-save"),
        "/remember 项目代号是北斗-17",
    )

    temporary_session = Session(session_id="session-temporary")
    temporary_model = RecordingTextModel()
    temporary_agent = TextAgent(temporary_model, memory_store=memory_store)
    temporary_agent.run_turn(temporary_session, "临时昵称是海盐蓝")

    reader_model = RecordingTextModel()
    reader = TextAgent(reader_model, memory_store=memory_store)
    turn = reader.run_turn(
        Session(session_id="session-reader"),
        "请只回答项目代号。",
        task_id="task-reader",
    )

    sent_context = reader_model.calls[0]
    assert turn.answer == "北斗-17"
    assert any(
        message.content is not None and "项目代号是北斗-17" in message.content
        for message in sent_context
    )
    assert not any(
        message.content is not None and "临时昵称是海盐蓝" in message.content
        for message in sent_context
    )


def test_remember_command_fails_safely_without_configured_store() -> None:
    """未配置持久化存储时，/remember 不应被模型误认为普通聊天。"""
    model = RecordingTextModel()
    agent = TextAgent(model)
    session = Session(session_id="session-no-store")

    turn = agent.run_turn(session, "/remember 偏好简洁回答")

    assert turn.task.status is TaskStatus.FAILED
    assert turn.error_message == "长期记忆存储尚未配置"
    assert model.calls == []
    assert session.messages == [
        Message(role=MessageRole.USER, content="/remember 偏好简洁回答"),
    ]


def test_sensitive_remember_command_is_not_sent_to_model(
    memory_store: SQLiteMemoryStore,
) -> None:
    """命中凭据规则的命令必须停止在本地，不泄露给模型上下文。"""
    model = RecordingTextModel()
    agent = TextAgent(model, memory_store=memory_store)

    turn = agent.run_turn(
        Session(session_id="session-sensitive"),
        "/remember api_key=abcdefghi",
    )

    assert turn.task.status is TaskStatus.FAILED
    assert "敏感凭据" in (turn.error_message or "")
    assert model.calls == []
    assert memory_store.list_memories() == ()


def test_tool_agent_injects_authorized_memory_and_handles_remember_command(
    memory_store: SQLiteMemoryStore,
) -> None:
    """工具 Agent 也应复用相同授权边界与长期记忆上下文。"""
    request = parse_remember_command(
        "session-save",
        "/remember 用户偏好使用中文",
    )
    assert request is not None
    memory_store.save_authorized(request)

    model = RecordingToolModel()
    agent = ToolAgent(model, make_registry(), memory_store=memory_store)
    normal_turn = agent.run_turn(
        Session(session_id="session-tool"),
        "请给出结论。",
    )
    messages, tools = model.calls[0]

    command_turn = agent.run_turn(
        Session(session_id="session-tool-command"),
        "/remember 用户偏好使用中文",
    )

    assert normal_turn.task.status is TaskStatus.COMPLETED
    assert any(
        message.content is not None and "用户偏好使用中文" in message.content
        for message in messages
    )
    assert tools[0]["function"]["name"] == "echo_text"
    assert command_turn.answer == "长期记忆已存在。"
    assert len(model.calls) == 1
