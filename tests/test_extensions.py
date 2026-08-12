"""验证 Gateway、Telegram 与可审查计划任务扩展。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import sqlite3
from pathlib import Path

import pytest

import hermes_lite.sqlite_state_store as state_module
from hermes_lite.chat_runtime import ChatTurnResult
from hermes_lite.domain import Session, TaskState, TaskStatus
from hermes_lite.gateway import (
    GatewayError,
    GatewayService,
    TelegramChannel,
    TelegramConfig,
    load_gateway_config,
)
from hermes_lite.memory_store import SQLiteMemoryStore
from hermes_lite.scheduler import SchedulerStore
from hermes_lite.sqlite_state_store import SQLiteStateStore, load_sqlite_state_config
from hermes_lite.tool_agent_loop import ToolAgentTurn


class RuntimeStub:
    """返回固定 Agent 结果，避免扩展测试产生真实模型请求。"""

    def __init__(self, status: TaskStatus = TaskStatus.COMPLETED) -> None:
        self.status = status
        self.calls: list[tuple[str, str]] = []

    def run_turn(self, session_id: str, user_request: str, skill_name: object | None = None) -> ChatTurnResult:
        del skill_name
        self.calls.append((session_id, user_request))
        task = TaskState("task-extension", session_id, user_request, self.status)
        turn = ToolAgentTurn(
            task=task,
            answer="渠道转发成功。" if self.status is TaskStatus.COMPLETED else None,
            error_message=None if self.status is TaskStatus.COMPLETED else ("需要确认" if self.status is TaskStatus.BLOCKED else "失败"),
            tool_results=(),
            round_summaries=(),
        )
        return ChatTurnResult(Session(session_id), turn, False, 0)


class TelegramApiStub:
    """存放 Telegram 轮询输入和发送输出。"""

    def __init__(self, updates: list[dict[str, object]]) -> None:
        self.updates = updates
        self.sent: list[tuple[int, str]] = []

    def get_updates(self, offset: int | None, timeout: int) -> list[dict[str, object]]:
        del offset, timeout
        return self.updates

    def send_message(self, chat_id: int, text: str) -> None:
        self.sent.append((chat_id, text))


def test_gateway_config_is_loopback_only() -> None:
    """Gateway 不能因环境变量意外监听局域网地址。"""
    config = load_gateway_config({"HERMES_GATEWAY_TOKEN": "a" * 16})
    assert config.host == "127.0.0.1"
    with pytest.raises(GatewayError, match="127.0.0.1"):
        load_gateway_config({"HERMES_GATEWAY_TOKEN": "a" * 16, "HERMES_GATEWAY_HOST": "0.0.0.0"})


def test_gateway_service_reuses_runtime_and_hides_confirmation() -> None:
    """渠道消息应进入同一运行时，确认令牌不能从外部渠道流入。"""
    runtime = RuntimeStub()
    service = GatewayService(runtime)
    assert service.handle_message("telegram:7", "你好") == "渠道转发成功。"
    assert runtime.calls == [("telegram:7", "你好")]
    assert "确认令牌" in service.handle_message("telegram:7", "/confirm secret")
    assert len(runtime.calls) == 1


def test_telegram_channel_accepts_only_whitelisted_private_text() -> None:
    """群聊、其他用户和非文本更新都不能进入 Agent。"""
    api = TelegramApiStub([
        {"update_id": 1, "message": {"from": {"id": 9}, "chat": {"id": 9, "type": "private"}, "text": "你好"}},
        {"update_id": 2, "message": {"from": {"id": 8}, "chat": {"id": 8, "type": "private"}, "text": "拒绝"}},
        {"update_id": 3, "message": {"from": {"id": 9}, "chat": {"id": -1, "type": "group"}, "text": "拒绝"}},
    ])
    runtime = RuntimeStub()
    channel = TelegramChannel(TelegramConfig("bot-token", 9), api, GatewayService(runtime))
    assert channel.poll_once() == 1
    assert runtime.calls == [("telegram:9", "你好")]
    assert api.sent == [(9, "渠道转发成功。")]


def test_scheduler_creates_candidate_then_requires_explicit_memory_approval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """到期任务只产生候选复盘，审批前不能进入长期记忆。"""
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(state_module, "PROJECT_ROOT", project_root)
    store = SQLiteStateStore(load_sqlite_state_config({}))
    scheduler = SchedulerStore(store)
    task = scheduler.create("local-default", "复习工具安全边界", 1)

    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE scheduled_tasks SET due_at = ? WHERE task_id = ?",
            ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(), task.task_id),
        )

    delivered: list[str] = []
    delivered_tasks = scheduler.run_due(lambda due: delivered.append(due.message))
    assert [item.task_id for item in delivered_tasks] == [task.task_id]
    assert delivered == ["复习工具安全边界"]
    candidates = scheduler.list_candidates()
    assert len(candidates) == 1
    assert candidates[0].status == "pending"
    assert SQLiteMemoryStore(store).list_memories() == ()

    assert scheduler.approve_memory_candidate(candidates[0].candidate_id) is True
    assert scheduler.list_candidates()[0].status == "approved"
    assert len(SQLiteMemoryStore(store).list_memories()) == 1
