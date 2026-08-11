"""验证受控 Pytest 命令策略。"""

from pathlib import Path
import sys

import pytest

import hermes_lite.workspace as workspace_module
from hermes_lite.test_runner import (
    PytestCommand,
    PytestPolicyError,
    build_pytest_command,
)
from hermes_lite.workspace import Workspace, load_workspace_config


@pytest.fixture
def workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Workspace:
    """创建隔离的测试工作区。"""
    project_root = tmp_path / "project"
    workspace_root = project_root / "sandbox_workspace"
    workspace_root.mkdir(parents=True)

    monkeypatch.setattr(workspace_module, "PROJECT_ROOT", project_root)

    return Workspace(load_workspace_config({}))


def test_build_pytest_command_uses_fixed_prefix(
    workspace: Workspace,
) -> None:
    """根目录测试只能转换为固定参数数组。"""
    command = build_pytest_command(workspace, {"target": "."})

    assert command.argv == (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        ".",
    )
    assert command.cwd == workspace.root


def test_build_pytest_command_accepts_python_test_file(
    workspace: Workspace,
) -> None:
    """允许选择工作区内已有的 Python 测试文件。"""
    tests_directory = workspace.root / "tests"
    tests_directory.mkdir()
    (tests_directory / "test_sample.py").write_text(
        "def test_sample() -> None:\n    assert True\n",
        encoding="utf-8",
    )

    command = build_pytest_command(
        workspace,
        {"target": "tests/test_sample.py"},
    )

    assert command.argv[-1] == "tests/test_sample.py"


@pytest.mark.parametrize(
    "target",
    ["/tmp/outside.py", "../outside.py"],
)
def test_build_pytest_command_rejects_unsafe_path(
    workspace: Workspace,
    target: str,
) -> None:
    """绝对路径和父目录穿越都不能成为测试目标。"""
    with pytest.raises(PytestPolicyError):
        build_pytest_command(workspace, {"target": target})


def test_build_pytest_command_rejects_missing_target(
    workspace: Workspace,
) -> None:
    """不存在的路径不能进入后续执行器。"""
    with pytest.raises(PytestPolicyError, match="不存在"):
        build_pytest_command(
            workspace,
            {"target": "tests/test_missing.py"},
        )


def test_build_pytest_command_rejects_non_python_file(
    workspace: Workspace,
) -> None:
    """单文件目标必须是 Python 文件。"""
    (workspace.root / "notes.txt").write_text(
        "不是测试文件",
        encoding="utf-8",
    )

    with pytest.raises(PytestPolicyError, match=".py"):
        build_pytest_command(
            workspace,
            {"target": "notes.txt"},
        )


def test_pytest_command_rejects_non_fixed_argv(
    workspace: Workspace,
) -> None:
    """命令对象本身也不能被构造为任意命令。"""
    with pytest.raises(PytestPolicyError, match="固定"):
        PytestCommand(
            argv=("rm", "-rf", ".", "unused", "unused"),
            cwd=workspace.root,
        )