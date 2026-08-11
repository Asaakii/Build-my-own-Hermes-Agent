"""HermesLite 的受控 Pytest 命令策略。

本模块只构造并校验命令，不启动子进程。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from hermes_lite.workspace import Workspace, WorkspaceError


class PytestPolicyError(ValueError):
    """测试目标或受控命令不符合安全策略时抛出。"""


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