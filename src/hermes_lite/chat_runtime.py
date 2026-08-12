"""HermesLite 持久化聊天运行层。

本模块复用既有 ToolAgent、SQLite 状态库、记忆和受限工作区；不实现新的
模型决策或工具执行逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass

from hermes_lite.config import load_model_config
from hermes_lite.domain import Session
from hermes_lite.memory_store import SQLiteMemoryStore
from hermes_lite.model_client import ModelClient
from hermes_lite.sqlite_state_store import (
    SQLiteStateStore,
    SQLiteStateStoreError,
    load_sqlite_state_config,
)
from hermes_lite.tool_agent_loop import ToolAgent, ToolAgentTurn
from hermes_lite.workspace import Workspace, load_workspace_config
from hermes_lite.workspace_tools import build_workspace_tool_registry


class ChatRuntimeError(RuntimeError):
    """表示聊天会话恢复或持久化未能完成。"""


@dataclass(frozen=True, slots=True)
class ChatTurnResult:
    """一次聊天运行的结果，以及本轮开始时的恢复信息。"""

    session: Session
    turn: ToolAgentTurn
    restored_existing_session: bool
    skipped_message_records: int

    def __post_init__(self) -> None:
        """确保结果由已验证会话、任务结果和恢复状态构成。"""
        if not isinstance(self.session, Session):
            raise ValueError("session 必须是 Session")
        if not isinstance(self.turn, ToolAgentTurn):
            raise ValueError("turn 必须是 ToolAgentTurn")
        if not isinstance(self.restored_existing_session, bool):
            raise ValueError("restored_existing_session 必须是布尔值")
        if (
            isinstance(self.skipped_message_records, bool)
            or not isinstance(self.skipped_message_records, int)
            or self.skipped_message_records < 0
        ):
            raise ValueError("skipped_message_records 必须是非负整数")


class ChatRuntime:
    """恢复会话、运行已有 ToolAgent，并持久化真实运行状态。"""

    def __init__(
        self,
        tool_agent: ToolAgent,
        state_store: SQLiteStateStore,
    ) -> None:
        """保存受控 Agent 和项目内唯一状态库。"""
        if not isinstance(tool_agent, ToolAgent):
            raise ValueError("tool_agent 必须是 ToolAgent")
        if not isinstance(state_store, SQLiteStateStore):
            raise ValueError("state_store 必须是 SQLiteStateStore")

        self._tool_agent = tool_agent
        self._state_store = state_store

    def run_turn(
        self,
        session_id: str,
        user_request: str,
        skill_name: object | None = None,
    ) -> ChatTurnResult:
        """运行并保存一轮聊天；会话保存失败时不伪造持久化成功。"""
        try:
            restored = self._state_store.restore_session(session_id)
        except SQLiteStateStoreError as error:
            raise ChatRuntimeError("无法恢复本地会话") from error

        session = restored.session or Session(session_id=session_id)
        turn = self._tool_agent.run_turn(
            session,
            user_request,
            skill_name=skill_name,
        )

        try:
            # 任务记录依赖会话外键，因此先保存会话，再保存本轮任务状态。
            self._state_store.save_session(session)
            self._state_store.save_task_state(
                turn.task,
                turn.round_summaries,
            )
        except SQLiteStateStoreError as error:
            raise ChatRuntimeError("无法保存本轮聊天状态") from error

        return ChatTurnResult(
            session=session,
            turn=turn,
            restored_existing_session=restored.session is not None,
            skipped_message_records=restored.skipped_message_records,
        )


def build_local_chat_runtime() -> ChatRuntime:
    """按本地已验证配置组合模型、状态、记忆、审计和受限工具。"""
    model_client = ModelClient(load_model_config())
    state_store = SQLiteStateStore(load_sqlite_state_config())
    state_store.initialize()
    workspace = Workspace(load_workspace_config())
    tool_agent = ToolAgent(
        model=model_client,
        registry=build_workspace_tool_registry(workspace),
        memory_store=SQLiteMemoryStore(state_store),
        audit_recorder=state_store,
    )
    return ChatRuntime(tool_agent, state_store)
