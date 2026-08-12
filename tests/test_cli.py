"""验证 HermesLite 的最小只读 CLI。"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import hermes_lite.cli as cli_module
from hermes_lite.config import ModelConfig
from hermes_lite.domain import Message, MessageRole, Session
from hermes_lite.memory_store import LongTermMemory
from hermes_lite.skill_loader import Skill
from hermes_lite.sqlite_state_store import SessionRestoreResult


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
