"""验证离线编码任务闭环。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

import hermes_lite.workspace as workspace_module
from hermes_lite.coding_agent import CodingAgent
from hermes_lite.coding_task import VerificationStatus
from hermes_lite.domain import Message, MessageRole, Session, TaskStatus, ToolCall
from hermes_lite.model_client import ModelClientError
from hermes_lite.tool_agent_loop import ToolAgent
from hermes_lite.workspace import Workspace, load_workspace_config
from hermes_lite.workspace_tools import build_workspace_tool_registry


class ScriptedToolModel:
    """按固定顺序返回离线工具决策。"""

    def __init__(
        self,
        responses: list[Message] | None = None,
        error: ModelClientError | None = None,
    ) -> None:
        self._responses = responses or []
        self._error = error

    def respond(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]],
    ) -> Message:
        """返回预设响应，不访问真实模型服务。"""
        del messages, tools

        if self._error is not None:
            raise self._error

        if not self._responses:
            raise AssertionError("测试模型缺少预设响应")

        return self._responses.pop(0)


@pytest.fixture
def workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Workspace:
    """创建带有独立工作区的临时项目。"""
    project_root = tmp_path / "project"
    workspace_root = project_root / "sandbox_workspace"
    workspace_root.mkdir(parents=True)

    monkeypatch.setattr(workspace_module, "PROJECT_ROOT", project_root)

    return Workspace(load_workspace_config({}))


def make_tool_request(
    call_id: str,
    tool_name: str,
    arguments: dict[str, object],
) -> Message:
    """构造一条不含自然语言的结构化工具请求。"""
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


def make_agent(
    workspace: Workspace,
    responses: list[Message] | None = None,
    error: ModelClientError | None = None,
) -> CodingAgent:
    """将离线模型、受限工具注册表与编码协调层组合起来。"""
    tool_agent = ToolAgent(
        ScriptedToolModel(responses=responses, error=error),
        build_workspace_tool_registry(workspace),
        max_tool_rounds=3,
    )
    return CodingAgent(tool_agent)


def test_coding_agent_repairs_sample_then_passes_tests(
    workspace: Workspace,
) -> None:
    """离线闭环应读取、精确修改、运行测试并生成完成报告。"""
    (workspace.root / "calculator.py").write_text(
        "def add(left: int, right: int) -> int:\n"
        "    return left - right\n",
        encoding="utf-8",
    )
    tests_directory = workspace.root / "tests"
    tests_directory.mkdir()
    (tests_directory / "test_calculator.py").write_text(
        "from calculator import add\n\n"
        "def test_add() -> None:\n"
        "    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    outside_file = workspace.root.parent / "outside.py"
    outside_file.write_text("不能被修改", encoding="utf-8")

    agent = make_agent(
        workspace,
        responses=[
            make_tool_request(
                "call-read",
                "read_file",
                {"path": "tests/test_calculator.py"},
            ),
            make_tool_request(
                "call-replace",
                "replace_text_once",
                {
                    "path": "calculator.py",
                    "expected_text": "return left - right",
                    "replacement": "return left + right",
                },
            ),
            make_tool_request(
                "call-test",
                "run_pytest",
                {"target": "tests"},
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content="已修复加法实现，测试通过。",
            ),
        ],
    )

    report = agent.run_task(
        Session(session_id="coding-session"),
        "修复加法函数并运行测试。",
        task_id="coding-task-1",
    )

    assert report.status is TaskStatus.COMPLETED
    assert report.verification is VerificationStatus.PASSED
    assert report.summary == "已修复加法实现，测试通过。"
    assert [summary.round_number for summary in report.rounds] == [1, 2, 3]
    assert [summary.results[0].tool_name for summary in report.rounds] == [
        "read_file",
        "replace_text_once",
        "run_pytest",
    ]
    assert "return left + right" in (workspace.root / "calculator.py").read_text(
        encoding="utf-8",
    )
    assert outside_file.read_text(encoding="utf-8") == "不能被修改"


def test_coding_agent_blocks_completion_without_test_verification(
    workspace: Workspace,
) -> None:
    """模型直接声称完成时，报告必须标记为未验证受阻。"""
    agent = make_agent(
        workspace,
        responses=[
            Message(
                role=MessageRole.ASSISTANT,
                content="已经修复完成。",
            ),
        ],
    )

    report = agent.run_task(
        Session(session_id="coding-session"),
        "修复一个问题。",
    )

    assert report.status is TaskStatus.BLOCKED
    assert report.verification is VerificationStatus.NOT_RUN


def test_coding_agent_marks_failed_pytest_as_failed_task(
    workspace: Workspace,
) -> None:
    """实际测试失败时，最终报告不能被模型文本掩盖。"""
    tests_directory = workspace.root / "tests"
    tests_directory.mkdir()
    (tests_directory / "test_failure.py").write_text(
        "def test_failure() -> None:\n"
        "    assert False\n",
        encoding="utf-8",
    )
    agent = make_agent(
        workspace,
        responses=[
            make_tool_request(
                "call-test",
                "run_pytest",
                {"target": "tests"},
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content="已经完成修复。",
            ),
        ],
    )

    report = agent.run_task(
        Session(session_id="coding-session"),
        "运行测试。",
    )

    assert report.status is TaskStatus.FAILED
    assert report.verification is VerificationStatus.FAILED


def test_coding_agent_blocks_when_model_fails_before_testing(
    workspace: Workspace,
) -> None:
    """未运行测试就发生模型故障时，报告应保留受阻原因。"""
    agent = make_agent(
        workspace,
        error=ModelClientError("模型服务暂不可用"),
    )

    report = agent.run_task(
        Session(session_id="coding-session"),
        "修复一个问题。",
    )

    assert report.status is TaskStatus.BLOCKED
    assert report.verification is VerificationStatus.NOT_RUN
    assert "模型服务暂不可用" in report.summary


def test_coding_agent_rejects_non_tool_agent() -> None:
    """协调层只能包装具备受控工具边界的 ToolAgent。"""
    with pytest.raises(ValueError, match="ToolAgent"):
        CodingAgent(object())  # type: ignore[arg-type]
