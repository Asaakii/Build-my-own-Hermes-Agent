"""验证受控 Pytest 命令策略。"""

from pathlib import Path
import subprocess
import sys

import pytest

import hermes_lite.workspace as workspace_module
from hermes_lite.test_runner import (
    PytestCommand,
    PytestExecutionError,
    PytestPolicyError,
    build_pytest_command,
    execute_pytest,
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

def test_execute_pytest_uses_fixed_safe_subprocess_options(
    workspace: Workspace,
) -> None:
    """执行器必须以参数数组、固定 cwd 和 shell=False 运行。"""
    command = build_pytest_command(workspace, {"target": "."})
    received: dict[str, object] = {}

    def fake_runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        received["args"] = args
        received["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="1 passed", stderr="")

    result = execute_pytest(command, runner=fake_runner)

    assert result.returncode == 0
    assert result.timed_out is False
    assert result.output == "1 passed"
    assert received["args"] == (command.argv,)
    options = received["kwargs"]
    assert isinstance(options, dict)
    assert options["cwd"] == workspace.root
    assert options["shell"] is False
    assert options["check"] is False
    assert options["timeout"] == 10.0


def test_execute_pytest_keeps_non_zero_exit_as_observation(
    workspace: Workspace,
) -> None:
    """失败测试应返回退出码，而不是被误判为执行器崩溃。"""
    command = build_pytest_command(workspace, {"target": "."})

    def fake_runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="1 failed", stderr="")

    result = execute_pytest(command, runner=fake_runner)

    assert result.returncode == 1
    assert result.timed_out is False
    assert "退出码: 1" in result.to_text()


def test_execute_pytest_truncates_returned_output(
    workspace: Workspace,
) -> None:
    """过长输出返回给 Agent 前必须截断。"""
    command = build_pytest_command(workspace, {"target": "."})

    def fake_runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="abcdefgh", stderr="")

    result = execute_pytest(command, max_output_characters=4, runner=fake_runner)

    assert result.output == "abcd\n[输出已截断]"


def test_execute_pytest_returns_timeout_result(
    workspace: Workspace,
) -> None:
    """超时应返回受控结果，而不是抛出未处理异常。"""
    command = build_pytest_command(workspace, {"target": "."})

    def timeout_runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"], output=b"partial output", stderr=b"timeout details")

    result = execute_pytest(command, runner=timeout_runner)

    assert result.returncode is None
    assert result.timed_out is True
    assert "partial output" in result.output
    assert "[stderr]" in result.output


def test_execute_pytest_converts_process_start_error(
    workspace: Workspace,
) -> None:
    """无法启动进程时不能泄露底层异常细节。"""
    command = build_pytest_command(workspace, {"target": "."})

    def broken_runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("内部路径信息")

    with pytest.raises(PytestExecutionError, match="无法启动"):
        execute_pytest(command, runner=broken_runner)


def test_execute_pytest_runs_self_authored_workspace_test(
    workspace: Workspace,
) -> None:
    """执行器可运行测试夹具自行创建的最小测试文件。"""
    tests_directory = workspace.root / "tests"
    tests_directory.mkdir()
    (tests_directory / "test_sample.py").write_text(
        "def test_sample() -> None:\n    assert True\n",
        encoding="utf-8",
    )
    command = build_pytest_command(workspace, {"target": "tests"})

    result = execute_pytest(command)

    assert result.returncode == 0
    assert result.timed_out is False
    assert "1 passed" in result.output
