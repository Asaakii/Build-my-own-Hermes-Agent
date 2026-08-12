"""验证 HermesLite 在隔离练习项目中的受控毕业任务。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

import hermes_lite.sqlite_state_store as sqlite_state_store_module
import hermes_lite.workspace as workspace_module
from hermes_lite.audit_log import AuditEventType, InMemoryAuditLog
from hermes_lite.coding_task import CodingTaskReport, VerificationStatus
from hermes_lite.confirmation_policy import ConfirmationManager
from hermes_lite.domain import (
    Message,
    MessageRole,
    Session,
    TaskState,
    TaskStatus,
    ToolCall,
)
from hermes_lite.memory_store import SQLiteMemoryStore
from hermes_lite.sqlite_state_store import (
    SQLiteStateStore,
    load_sqlite_state_config,
)
from hermes_lite.tool_agent_loop import ToolAgent
from hermes_lite.workspace import Workspace, load_workspace_config
from hermes_lite.workspace_tools import build_workspace_tool_registry


class ScriptedToolModel:
    """按预设顺序模拟毕业任务中的模型决策，不访问真实模型。"""

    def __init__(self, responses: list[Message]) -> None:
        """保存尚未返回的结构化工具请求或最终摘要。"""
        self._responses = responses

    def respond(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]],
    ) -> Message:
        """忽略外部服务，仅返回下一条可重复的决策。"""
        del messages, tools

        if not self._responses:
            raise AssertionError("毕业任务缺少预设模型响应")

        return self._responses.pop(0)


def make_tool_request(
    call_id: str,
    tool_name: str,
    arguments: dict[str, object],
) -> Message:
    """构造一条不携带自然语言的结构化工具调用请求。"""
    return Message(
        role=MessageRole.ASSISTANT,
        content=None,
        tool_calls=(
            ToolCall(
                call_id=call_id,
                tool_name=tool_name,
                arguments=arguments,
            ),
        ),
    )


def test_graduation_demo_repairs_isolated_project_with_confirmations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """毕业任务必须完成受控修复、复测、恢复与长期记忆边界验证。"""
    project_root = tmp_path / "graduation-project"
    workspace_root = project_root / "sandbox_workspace"
    workspace_root.mkdir(parents=True)
    outside_file = project_root / "outside.py"
    outside_file.write_text("不得修改\n", encoding="utf-8")

    # 工作区与 SQLite 均指向临时项目，确保演示不会触碰真实项目文件。
    monkeypatch.setattr(workspace_module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sqlite_state_store_module, "PROJECT_ROOT", project_root)
    workspace = Workspace(load_workspace_config({}))

    (workspace.root / "TASK.md").write_text(
        "# 修复任务\n\n"
        "修复 add 的简单错误，并运行既有测试。\n"
        "未授权长期记忆标记：毕业任务不应自动写入记忆。\n",
        encoding="utf-8",
    )
    (workspace.root / "calculator.py").write_text(
        "def add(left: int, right: int) -> int:\n"
        "    return left - right\n",
        encoding="utf-8",
    )
    tests_directory = workspace.root / "tests"
    tests_directory.mkdir()
    (tests_directory / "test_calculator.py").write_text(
        "from pathlib import Path\n"
        "import sys\n\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
        "from calculator import add\n\n"
        "def test_add() -> None:\n"
        "    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )

    state_config = load_sqlite_state_config({})
    state_store = SQLiteStateStore(state_config)
    memory_store = SQLiteMemoryStore(state_store)
    audit_log = InMemoryAuditLog()
    confirmation_manager = ConfirmationManager(
        token_factory=lambda: "graduation-confirm-token",
    )
    model = ScriptedToolModel(
        [
            make_tool_request("call-read-task", "read_file", {"path": "TASK.md"}),
            make_tool_request(
                "call-read-source",
                "read_file",
                {"path": "calculator.py"},
            ),
            make_tool_request(
                "call-first-test",
                "run_pytest",
                {"target": "tests/test_calculator.py"},
            ),
            make_tool_request(
                "call-minimal-patch",
                "replace_text_once",
                {
                    "path": "calculator.py",
                    "expected_text": "return left - right",
                    "replacement": "return left + right",
                },
            ),
            make_tool_request(
                "call-retest",
                "run_pytest",
                {"target": "tests/test_calculator.py"},
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content=(
                    "已将 add 的减法改为加法；复测通过。"
                    "未解决风险：本演示只覆盖 add 的单一行为。"
                    "审计摘要：读取、测试、等待确认、确认执行与复测均已记录。"
                ),
            ),
        ]
    )
    agent = ToolAgent(
        model,
        build_workspace_tool_registry(workspace),
        max_tool_rounds=3,
        memory_store=memory_store,
        confirmation_manager=confirmation_manager,
        audit_recorder=audit_log,
    )
    session = Session(session_id="graduation-session")
    task_id = "graduation-task"

    # 第一次测试、最小补丁和复测均属于高风险操作，必须逐次确认。
    inspected = agent.run_turn(
        session,
        "阅读 TASK.md 和相关文件，运行测试定位问题。",
        task_id=task_id,
    )
    assert inspected.task.status is TaskStatus.BLOCKED
    assert inspected.pending_confirmation is not None
    assert inspected.pending_confirmation.tool_call.tool_name == "run_pytest"
    first_test = agent.run_turn(
        session,
        f"/confirm {inspected.pending_confirmation.token}",
        task_id=task_id,
    )
    assert "测试结束，退出码: 1" in first_test.tool_results[0].content

    proposed_patch = agent.run_turn(
        session,
        "测试失败，请提出并执行最小修复。",
        task_id=task_id,
    )
    assert proposed_patch.task.status is TaskStatus.BLOCKED
    assert proposed_patch.pending_confirmation is not None
    assert proposed_patch.pending_confirmation.tool_call.tool_name == "replace_text_once"
    patched = agent.run_turn(
        session,
        f"/confirm {proposed_patch.pending_confirmation.token}",
        task_id=task_id,
    )
    assert patched.task.status is TaskStatus.COMPLETED

    proposed_retest = agent.run_turn(
        session,
        "请重新运行测试验证修复。",
        task_id=task_id,
    )
    assert proposed_retest.task.status is TaskStatus.BLOCKED
    assert proposed_retest.pending_confirmation is not None
    retested = agent.run_turn(
        session,
        f"/confirm {proposed_retest.pending_confirmation.token}",
        task_id=task_id,
    )
    assert "测试结束，退出码: 0" in retested.tool_results[0].content

    summary_turn = agent.run_turn(
        session,
        "请依据已有工具观察输出修改、验证、风险和审计摘要。",
        task_id=task_id,
    )
    assert summary_turn.task.status is TaskStatus.COMPLETED
    assert summary_turn.answer is not None

    # 最终报告以真实复测观察为依据，而不是以模型声称“完成”为依据。
    report = CodingTaskReport(
        task_id=task_id,
        status=TaskStatus.COMPLETED,
        verification=VerificationStatus.PASSED,
        summary=summary_turn.answer,
        rounds=retested.round_summaries,
    )
    persisted_task = TaskState(
        task_id=task_id,
        session_id=session.session_id,
        user_request="修复 add 的简单错误，并运行既有测试。",
        status=TaskStatus.COMPLETED,
        tool_rounds=5,
    )
    state_store.save_session(session)
    state_store.save_coding_task(persisted_task, report)

    # 以新的存储实例模拟重启；任务摘要可恢复，未授权内容不能变成长久记忆。
    restarted_store = SQLiteStateStore(state_config)
    restored_session = restarted_store.restore_session(session.session_id).session
    restored_task = restarted_store.load_coding_task(task_id)
    assert restored_session is not None
    assert restored_task is not None
    assert restored_task.report is not None
    assert restored_task.report.summary == summary_turn.answer
    assert "未解决风险" in restored_task.report.summary
    assert SQLiteMemoryStore(restarted_store).list_memories() == ()

    assert (workspace.root / "calculator.py").read_text(encoding="utf-8") == (
        "def add(left: int, right: int) -> int:\n"
        "    return left + right\n"
    )
    assert outside_file.read_text(encoding="utf-8") == "不得修改\n"
    assert all(
        "graduation-confirm-token" not in (message.content or "")
        for message in restored_session.messages
    )

    event_types = [event.event_type for event in audit_log.events]
    assert event_types.count(AuditEventType.CONFIRMATION_REQUIRED) == 3
    assert event_types.count(AuditEventType.CONFIRMATION_ACCEPTED) == 3
    assert event_types.count(AuditEventType.TOOL_EXECUTED) == 5
    assert all(
        "毕业任务不应自动写入记忆" not in str(event)
        and "graduation-confirm-token" not in str(event)
        for event in audit_log.events
    )
