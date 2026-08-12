"""验证 HermesLite 的最小只读 CLI。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

import hermes_lite.cli as cli_module
from hermes_lite.chat_runtime import ChatTurnResult
from hermes_lite.config import ModelConfig
from hermes_lite.doctor import DoctorCheck, DoctorCheckStatus, DoctorReport
from hermes_lite.confirmation_policy import PendingConfirmation
from hermes_lite.coding_task import ToolRoundSummary
from hermes_lite.domain import (
    Message,
    MessageRole,
    Session,
    TaskState,
    TaskStatus,
    ToolCall,
    ToolResult,
)
from hermes_lite.memory_store import LongTermMemory
from hermes_lite.skill_loader import Skill
from hermes_lite.sqlite_state_store import SessionRestoreResult
from hermes_lite.tool_agent_loop import ToolAgentTurn

@dataclass
class StateStoreStub:
    """只提供会话恢复结果的最小状态库替身。"""

    result: SessionRestoreResult

    def restore_session(self, session_id: object) -> SessionRestoreResult:
        """返回预设会话恢复结果。"""
        del session_id
        return self.result

@dataclass
class MemoryStoreStub:
    """只提供列表和检索结果的最小记忆服务替身。"""

    memories: tuple[LongTermMemory, ...]

    def list_memories(self, max_results: int) -> tuple[LongTermMemory, ...]:
        """返回预设列表结果。"""
        del max_results
        return self.memories

    def search(
        self,
        query: object,
        max_results: int,
    ) -> tuple[LongTermMemory, ...]:
        """返回预设检索结果。"""
        del query, max_results
        return self.memories


@dataclass
class ChatRuntimeStub:
    """提供固定聊天结果，并记录 CLI 转交的请求。"""

    result: ChatTurnResult
    calls: list[tuple[str, str, object | None]]

    def run_turn(
        self,
        session_id: str,
        user_request: str,
        skill_name: object | None = None,
    ) -> ChatTurnResult:
        """记录参数后返回预设聊天结果。"""
        self.calls.append((session_id, user_request, skill_name))
        return self.result


def make_completed_chat_result() -> ChatTurnResult:
    """构造无需工具的普通聊天成功结果。"""
    task = TaskState(
        task_id="chat-task",
        session_id="chat-session",
        user_request="测试请求",
        status=TaskStatus.COMPLETED,
    )
    return ChatTurnResult(
        session=Session(session_id="chat-session"),
        turn=ToolAgentTurn(
            task=task,
            answer="聊天已写入。",
            error_message=None,
            tool_results=(),
            round_summaries=(),
        ),
        restored_existing_session=False,
        skipped_message_records=0,
    )


def make_blocked_chat_result() -> ChatTurnResult:
    """构造带内存确认令牌的高风险操作结果。"""
    tool_call = ToolCall(
        call_id="call-confirm",
        tool_name="write_text",
        arguments={"content": "PRIVATE_ARGUMENT"},
    )
    result = ToolResult(
        call_id="call-confirm",
        tool_name="write_text",
        content="工具调用等待确认: write_text。",
        is_error=True,
    )
    task = TaskState(
        task_id="blocked-task",
        session_id="chat-session",
        user_request="执行写入",
        status=TaskStatus.BLOCKED,
        tool_rounds=1,
    )
    return ChatTurnResult(
        session=Session(session_id="chat-session"),
        turn=ToolAgentTurn(
            task=task,
            answer=None,
            error_message="高风险工具等待确认。",
            tool_results=(result,),
            round_summaries=(ToolRoundSummary(1, (result,)),),
            pending_confirmation=PendingConfirmation(
                token="PRIVATE_CONFIRM_TOKEN",
                session_id="chat-session",
                tool_call=tool_call,
                expires_at=datetime.now(UTC),
            ),
        ),
        restored_existing_session=False,
        skipped_message_records=0,
    )


def make_memory() -> LongTermMemory:
    """构造一条已授权的测试记忆。"""
    return LongTermMemory(
        memory_id=1,
        source_session_id="session-source",
        content="偏好使用简洁中文。",
        created_at="2026-08-12T00:00:00+00:00",
    )


def test_help_lists_available_read_only_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """帮助文本应列出当前已实现的命令组。"""
    exit_code = cli_module.main(["--help"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "config" in captured.out
    assert "doctor" in captured.out
    assert "chat" in captured.out
    assert "sessions" in captured.out
    assert "memory" in captured.out
    assert "skills" in captured.out


def test_unknown_command_returns_predictable_usage_code(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """未知命令必须以稳定退出码失败，且不回显原始输入。"""
    exit_code = cli_module.main(["PRIVATE_UNKNOWN_COMMAND"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "命令参数错误" in captured.err
    assert "PRIVATE_UNKNOWN_COMMAND" not in captured.err


def test_config_command_uses_redacted_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """配置命令必须展示可用摘要而不是 API Key。"""
    monkeypatch.setattr(
        cli_module,
        "load_model_config",
        lambda: ModelConfig(
            provider="demo",
            model="demo-model",
            api_key="PRIVATE_API_KEY",
            base_url="https://api.example.com",
            timeout_seconds=30.0,
        ),
    )

    exit_code = cli_module.main(["config"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "模型供应商: demo" in captured.out
    assert "API Key: 已配置（已隐藏）" in captured.out
    assert "PRIVATE_API_KEY" not in captured.out


def test_sessions_show_reports_metadata_without_message_content(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """会话查看只输出元数据，不复制历史消息正文。"""
    session = Session(
        session_id="session-1",
        messages=[
            Message(role=MessageRole.USER, content="PRIVATE_SESSION_CONTENT"),
        ],
    )
    monkeypatch.setattr(
        cli_module,
        "_load_state_store",
        lambda: StateStoreStub(SessionRestoreResult(session=session)),
    )

    exit_code = cli_module.main(["sessions", "show", "session-1"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "会话: session-1" in captured.out
    assert "消息数: 1" in captured.out
    assert "PRIVATE_SESSION_CONTENT" not in captured.out


def test_sessions_show_returns_one_when_session_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """不存在会话不能被误报为已恢复。"""
    monkeypatch.setattr(
        cli_module,
        "_load_state_store",
        lambda: StateStoreStub(SessionRestoreResult(session=None)),
    )

    exit_code = cli_module.main(["sessions", "show", "missing"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "会话不存在" in captured.err


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (["memory", "list"], "记忆 #1: 偏好使用简洁中文。"),
        (["memory", "search", "简洁"], "记忆 #1: 偏好使用简洁中文。"),
    ],
)
def test_memory_commands_call_authorized_memory_service(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: list[str],
    expected: str,
) -> None:
    """记忆命令只读取既有的授权记忆服务。"""
    monkeypatch.setattr(
        cli_module,
        "_load_memory_store",
        lambda: MemoryStoreStub((make_memory(),)),
    )

    exit_code = cli_module.main(command)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert expected in captured.out
    assert "session-source" not in captured.out


def test_memory_limit_rejects_zero_before_accessing_storage(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """非法列表上限应作为参数错误停止，不访问记忆数据库。"""
    def fail_if_called() -> MemoryStoreStub:
        raise AssertionError("不应访问记忆存储")

    monkeypatch.setattr(cli_module, "_load_memory_store", fail_if_called)

    exit_code = cli_module.main(["memory", "list", "--limit", "0"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "命令参数错误" in captured.err


def test_skills_list_displays_validated_metadata_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """技能命令应展示元数据，不打印技能完整指令。"""
    monkeypatch.setattr(
        cli_module,
        "list_available_skills",
        lambda: (
            Skill(
                name="fix_test",
                description="修复已有测试。",
                allowed_tools=("read_file", "run_pytest"),
                instructions="PRIVATE_SKILL_INSTRUCTIONS",
            ),
        ),
    )

    exit_code = cli_module.main(["skills", "list"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "技能: fix_test" in captured.out
    assert "允许工具: read_file, run_pytest" in captured.out
    assert "PRIVATE_SKILL_INSTRUCTIONS" not in captured.out


def test_chat_command_runs_a_single_persisted_turn(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """单次聊天应把消息和可选技能交给持久化运行层。"""
    runtime = ChatRuntimeStub(make_completed_chat_result(), [])
    monkeypatch.setattr(cli_module, "build_local_chat_runtime", lambda: runtime)

    exit_code = cli_module.main(
        ["chat", "--session-id", "chat-1", "--skill", "fix_test", "测试消息"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Agent: 聊天已写入。" in captured.out
    assert runtime.calls == [("chat-1", "测试消息", "fix_test")]


def test_chat_rejects_invalid_session_identifier_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """路径式会话标识不能进入运行层或数据库。"""
    def fail_if_called() -> ChatRuntimeStub:
        raise AssertionError("不应创建聊天运行层")

    monkeypatch.setattr(cli_module, "build_local_chat_runtime", fail_if_called)

    exit_code = cli_module.main(["chat", "--session-id", "../unsafe", "测试消息"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "命令参数错误" in captured.err


def test_one_shot_chat_hides_confirmation_token(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """一次性聊天不能泄露只能用于当前进程的确认令牌。"""
    runtime = ChatRuntimeStub(make_blocked_chat_result(), [])
    monkeypatch.setattr(cli_module, "build_local_chat_runtime", lambda: runtime)

    exit_code = cli_module.main(["chat", "执行写入"])

    captured = capsys.readouterr()
    assert exit_code == 3
    assert "PRIVATE_CONFIRM_TOKEN" not in captured.out
    assert "同一交互会话内确认" in captured.out


def test_interactive_chat_forwards_remember_command_to_agent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """交互层只处理本地退出帮助，授权命令继续交给 ToolAgent。"""
    runtime = ChatRuntimeStub(make_completed_chat_result(), [])
    requests = iter(["/help", "/remember 偏好简洁回答", "/quit"])

    exit_code = cli_module.run_interactive_chat(
        runtime,
        "chat-1",
        input_fn=lambda prompt: next(requests),
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "本地命令" in captured.out
    assert "Agent: 聊天已写入。" in captured.out
    assert runtime.calls == [("chat-1", "/remember 偏好简洁回答", None)]


def make_doctor_report(*, healthy: bool) -> DoctorReport:
    """构造供 CLI 路由测试使用的固定诊断报告。"""
    status = DoctorCheckStatus.PASSED if healthy else DoctorCheckStatus.FAILED
    return DoctorReport(
        (
            DoctorCheck(
                name="模型配置",
                status=status,
                message="测试诊断结果。",
            ),
        )
    )


@pytest.mark.parametrize("check_model", [False, True])
def test_doctor_command_forwards_explicit_model_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    check_model: bool,
) -> None:
    """doctor 子命令只将显式模型检查意图转交给诊断层。"""
    requested: list[bool] = []

    def fake_run_doctor(*, check_model: bool) -> DoctorReport:
        requested.append(check_model)
        return make_doctor_report(healthy=True)

    monkeypatch.setattr(cli_module, "run_doctor", fake_run_doctor)
    command = ["doctor"] + (["--check-model"] if check_model else [])

    exit_code = cli_module.main(command)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert requested == [check_model]
    assert "HermesLite 本地诊断" in captured.out


def test_doctor_command_returns_one_for_failed_check(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """诊断失败必须有非零退出码，便于脚本识别。"""
    monkeypatch.setattr(
        cli_module,
        "run_doctor",
        lambda check_model: make_doctor_report(healthy=False),
    )

    exit_code = cli_module.main(["doctor"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "[失败]" in captured.out
