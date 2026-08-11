"""验证 SQLite 会话与编码任务的读写恢复。"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_lite.sqlite_state_store as state_store_module
from hermes_lite.coding_task import (
    CodingTaskReport,
    ToolRoundSummary,
    VerificationStatus,
)
from hermes_lite.domain import (
    Message,
    MessageRole,
    Session,
    TaskState,
    TaskStatus,
    ToolCall,
    ToolResult,
)
from hermes_lite.sqlite_state_store import (
    SQLiteStateStore,
    SQLiteStateStoreError,
    load_sqlite_state_config,
)


@pytest.fixture
def store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> SQLiteStateStore:
    """创建使用临时项目目录的已初始化状态数据库。"""
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(state_store_module, "PROJECT_ROOT", project_root)
    state_store = SQLiteStateStore(load_sqlite_state_config({}))
    state_store.initialize()

    return state_store


def make_session() -> Session:
    """创建包含结构化工具调用和工具结果的会话。"""
    tool_call = ToolCall(
        call_id="call-1",
        tool_name="read_file",
        arguments={"path": "example.py"},
    )

    return Session(
        session_id="session-1",
        messages=[
            Message(role=MessageRole.SYSTEM, content="系统规则"),
            Message(role=MessageRole.USER, content="读取文件"),
            Message(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=(tool_call,),
            ),
            Message(
                role=MessageRole.TOOL,
                content="文件内容",
                tool_call_id="call-1",
            ),
        ],
    )


def make_task_and_report() -> tuple[TaskState, CodingTaskReport]:
    """创建一条可保存的任务状态及最终报告。"""
    task = TaskState(
        task_id="task-1",
        session_id="session-1",
        user_request="修复失败测试",
        status=TaskStatus.COMPLETED,
        tool_rounds=2,
    )
    report = CodingTaskReport(
        task_id="task-1",
        status=TaskStatus.COMPLETED,
        verification=VerificationStatus.PASSED,
        summary="修改完成，测试通过。",
        rounds=(
            ToolRoundSummary(
                round_number=1,
                results=(
                    ToolResult(
                        call_id="call-read",
                        tool_name="read_file",
                        content="测试文件内容",
                    ),
                ),
            ),
            ToolRoundSummary(
                round_number=2,
                results=(
                    ToolResult(
                        call_id="call-test",
                        tool_name="run_pytest",
                        content="测试结束，退出码: 0",
                    ),
                ),
            ),
        ),
    )

    return task, report


def test_save_and_load_session_preserves_message_protocol(
    store: SQLiteStateStore,
) -> None:
    """会话往返后必须保留消息角色、调用标识和工具参数。"""
    session = make_session()

    store.save_session(session)
    restored = store.load_session(session.session_id)

    assert restored is not None
    assert [message.role for message in restored.messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]
    assert restored.messages[2].tool_calls[0].arguments == {
        "path": "example.py",
    }
    assert restored.messages[3].tool_call_id == "call-1"


def test_save_session_replaces_old_history_without_duplicates(
    store: SQLiteStateStore,
) -> None:
    """重复保存同一会话时，数据库应以新历史替换旧历史。"""
    session = make_session()
    store.save_session(session)
    session.messages = [Message(role=MessageRole.USER, content="新的历史")]

    store.save_session(session)
    restored = store.load_session(session.session_id)

    assert restored is not None
    assert restored.messages == [
        Message(role=MessageRole.USER, content="新的历史"),
    ]


def test_save_and_load_coding_task_preserves_report_rounds(
    store: SQLiteStateStore,
) -> None:
    """任务、每轮工具结果和最终报告应可完整恢复。"""
    store.save_session(Session(session_id="session-1"))
    task, report = make_task_and_report()

    store.save_coding_task(task, report)
    restored = store.load_coding_task(task.task_id)

    assert restored is not None
    assert restored.task.session_id == "session-1"
    assert restored.task.tool_rounds == 2
    assert restored.report is not None
    assert restored.report.verification is VerificationStatus.PASSED
    assert [summary.round_number for summary in restored.report.rounds] == [1, 2]
    assert restored.report.rounds[1].results[0].tool_name == "run_pytest"


def test_save_task_requires_existing_session(
    store: SQLiteStateStore,
) -> None:
    """任务不能绕过会话边界被单独写入。"""
    task, report = make_task_and_report()

    with pytest.raises(SQLiteStateStoreError, match="会话尚未保存"):
        store.save_coding_task(task, report)


def test_load_missing_session_and_task_returns_none(
    store: SQLiteStateStore,
) -> None:
    """不存在的状态标识应明确返回 None。"""
    assert store.load_session("missing-session") is None
    assert store.load_coding_task("missing-task") is None


def test_save_task_rejects_mismatched_report_identifier(
    store: SQLiteStateStore,
) -> None:
    """任务与报告不能通过不同标识串联。"""
    task, report = make_task_and_report()
    mismatched_report = CodingTaskReport(
        task_id="other-task",
        status=report.status,
        verification=report.verification,
        summary=report.summary,
        rounds=report.rounds,
    )

    with pytest.raises(SQLiteStateStoreError, match="task_id 不一致"):
        store.save_coding_task(task, mismatched_report)


def test_save_session_rejects_non_json_tool_arguments(
    store: SQLiteStateStore,
) -> None:
    """无法稳定 JSON 序列化的工具参数不能进入数据库。"""
    session = Session(
        session_id="session-1",
        messages=[
            Message(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=(
                    ToolCall(
                        call_id="call-1",
                        tool_name="read_file",
                        arguments={"path": object()},
                    ),
                ),
            ),
        ],
    )

    with pytest.raises(SQLiteStateStoreError, match="无法序列化"):
        store.save_session(session)
