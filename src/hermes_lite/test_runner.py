"""HermesLite 的受控 Pytest 命令策略。

本模块只构造并校验命令，不启动子进程。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

from hermes_lite.workspace import Workspace, WorkspaceError


class PytestPolicyError(ValueError):
    """测试目标或受控命令不符合安全策略时抛出。"""


class PytestExecutionError(RuntimeError):
    """受控 Pytest 进程无法正常启动时抛出。"""


@dataclass(frozen=True, slots=True)
class TestRunResult:
    """一次受控 Pytest 执行的脱敏结果。"""

    returncode: int | None
    output: str
    timed_out: bool = False

    def __post_init__(self) -> None:
        """验证执行结果字段。"""
        if self.returncode is not None and (
            isinstance(self.returncode, bool)
            or not isinstance(self.returncode, int)
        ):
            raise ValueError("returncode 必须是整数或 None")

        if not isinstance(self.output, str):
            raise ValueError("output 必须是文本")

        if not isinstance(self.timed_out, bool):
            raise ValueError("timed_out 必须是布尔值")

    def to_text(self) -> str:
        """转换为可安全写回 Agent 会话的文本观察结果。"""
        if self.timed_out:
            status = "测试超时：进程已停止。"
        else:
            status = f"测试结束，退出码: {self.returncode}"

        output = self.output or "（测试进程没有输出）"

        return f"{status}\n测试输出:\n{output}"


@dataclass(frozen=True, slots=True)
class PytestCommand:
    """一条已经通过策略验证、可在后续阶段执行的 Pytest 命令。"""

    argv: tuple[str, ...]
    cwd: Path

    def __post_init__(self) -> None:
        """确保命令结构固定，不能携带任意 shell 参数。"""
        if not isinstance(self.argv, tuple):
            raise PytestPolicyError("argv 必须是元组")

        if not all(isinstance(item, str) and item for item in self.argv):
            raise PytestPolicyError("argv 中的命令项必须是非空文本")

        expected_prefix = (sys.executable, "-m", "pytest", "-q")

        if self.argv[:4] != expected_prefix or len(self.argv) != 5:
            raise PytestPolicyError("只允许固定的 python -m pytest -q 命令")

        target = Path(self.argv[4])

        if target.is_absolute() or ".." in target.parts:
            raise PytestPolicyError("测试目标必须是安全相对路径")

        if not isinstance(self.cwd, Path) or not self.cwd.is_dir():
            raise PytestPolicyError("cwd 必须是存在的目录")


def _require_target(arguments: object) -> str:
    """读取模型提交的测试目标文本。"""
    if not isinstance(arguments, dict):
        raise PytestPolicyError("工具参数必须是字典")

    target = arguments.get("target")

    if not isinstance(target, str):
        raise PytestPolicyError("target 必须是文本")

    cleaned_target = target.strip()

    if not cleaned_target:
        raise PytestPolicyError("target 不能为空")

    if len(cleaned_target) > 512:
        raise PytestPolicyError("target 长度不能超过 512 个字符")

    return cleaned_target


def build_pytest_command(
    workspace: Workspace,
    arguments: object,
) -> PytestCommand:
    """将受限工作区中的测试目标转换为固定 Pytest 参数数组。"""
    if not isinstance(workspace, Workspace):
        raise PytestPolicyError("workspace 必须是 Workspace")

    raw_target = _require_target(arguments)

    try:
        target_path = workspace.resolve_path(raw_target)
    except WorkspaceError as error:
        raise PytestPolicyError(str(error)) from error

    if not target_path.exists():
        raise PytestPolicyError("测试目标不存在")

    if not target_path.is_file() and not target_path.is_dir():
        raise PytestPolicyError("测试目标必须是文件或目录")

    if target_path.is_file() and target_path.suffix != ".py":
        raise PytestPolicyError("测试目标文件必须是 .py 文件")

    relative_target = target_path.relative_to(workspace.root)

    if relative_target == Path("."):
        target_argument = "."
    else:
        target_argument = relative_target.as_posix()

    return PytestCommand(
        argv=(
            sys.executable,
            "-m",
            "pytest",
            "-q",
            target_argument,
        ),
        cwd=workspace.root,
    )


DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_RETURNED_OUTPUT_CHARACTERS = 12_000

CompletedProcessRunner = Callable[..., subprocess.CompletedProcess[str]]


def _require_positive_number(value: object, field_name: str) -> float:
    """验证执行器内部使用的正数限制。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} 必须是正数")

    number = float(value)

    if number <= 0:
        raise ValueError(f"{field_name} 必须大于 0")

    return number


def _require_positive_integer(value: object, field_name: str) -> int:
    """验证执行器内部使用的正整数限制。"""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} 必须是正整数")

    return value


def _coerce_process_output(value: object) -> str:
    """将 subprocess 可能返回的文本或字节转换为 UTF-8 文本。"""
    if value is None:
        return ""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, str):
        return value

    return str(value)


def _merge_process_output(stdout: object, stderr: object) -> str:
    """合并标准输出与标准错误，保留来源提示。"""
    stdout_text = _coerce_process_output(stdout)
    stderr_text = _coerce_process_output(stderr)

    if not stderr_text:
        return stdout_text

    if not stdout_text:
        return f"[stderr]\n{stderr_text}"

    return f"{stdout_text}\n[stderr]\n{stderr_text}"


def _truncate_returned_output(output: str, max_characters: int) -> str:
    """限制返回给 Agent 的输出，避免挤占会话上下文。"""
    if len(output) <= max_characters:
        return output

    return output[:max_characters] + "\n[输出已截断]"


def execute_pytest(
    command: PytestCommand,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_output_characters: int = MAX_RETURNED_OUTPUT_CHARACTERS,
    runner: CompletedProcessRunner = subprocess.run,
) -> TestRunResult:
    """执行已验证的 Pytest 命令，并返回受限观察结果。"""
    if not isinstance(command, PytestCommand):
        raise PytestExecutionError("command 必须是 PytestCommand")

    timeout = _require_positive_number(
        timeout_seconds,
        "timeout_seconds",
    )
    output_limit = _require_positive_integer(
        max_output_characters,
        "max_output_characters",
    )

    try:
        completed = runner(
            command.argv,
            cwd=command.cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as error:
        output = _merge_process_output(error.output, error.stderr)

        return TestRunResult(
            returncode=None,
            output=_truncate_returned_output(output, output_limit),
            timed_out=True,
        )
    except OSError as error:
        raise PytestExecutionError("无法启动受控 Pytest 进程") from error

    output = _merge_process_output(completed.stdout, completed.stderr)

    return TestRunResult(
        returncode=completed.returncode,
        output=_truncate_returned_output(output, output_limit),
    )