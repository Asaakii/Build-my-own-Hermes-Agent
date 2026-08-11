"""HermesLite 受限工作区的配置与路径边界。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

from hermes_lite.config import DOTENV_PATH, PROJECT_ROOT


class WorkspaceError(ValueError):
    """工作区配置或用户路径违反安全边界时抛出。"""


@dataclass(frozen=True, slots=True)
class WorkspaceConfig:
    """已验证的项目内工作区配置。"""

    root: Path

    def __post_init__(self) -> None:
        """确保工作区是项目内一个真实存在的子目录。"""
        if not isinstance(self.root, Path):
            raise WorkspaceError("root 必须是 Path")

        project_root = PROJECT_ROOT.resolve()
        root = self.root.resolve()

        if not root.is_dir():
            raise WorkspaceError("工作区目录不存在或不是目录")

        try:
            root.relative_to(project_root)
        except ValueError as error:
            raise WorkspaceError("工作区必须位于项目根目录内") from error

        if root == project_root:
            raise WorkspaceError("工作区不能直接使用项目根目录")

        object.__setattr__(self, "root", root)


def _get_workspace_relative_path(
    environment: Mapping[str, str],
) -> Path:
    """读取项目内相对工作区路径。"""
    raw_path = environment.get("AGENT_WORKSPACE_PATH")

    if raw_path is None:
        return Path("sandbox_workspace")

    if not isinstance(raw_path, str) or not raw_path.strip():
        raise WorkspaceError("AGENT_WORKSPACE_PATH 必须是非空文本")

    relative_path = Path(raw_path.strip())

    if relative_path.is_absolute():
        raise WorkspaceError("AGENT_WORKSPACE_PATH 只能是项目内相对路径")

    if ".." in relative_path.parts:
        raise WorkspaceError("AGENT_WORKSPACE_PATH 不能包含父目录穿越")

    return relative_path


def load_workspace_config(
    environment: Mapping[str, str] | None = None,
) -> WorkspaceConfig:
    """加载并验证工作区配置。"""
    if environment is None:
        load_dotenv(DOTENV_PATH, override=False)
        environment = os.environ

    relative_path = _get_workspace_relative_path(environment)
    return WorkspaceConfig(root=PROJECT_ROOT / relative_path)


class Workspace:
    """只解析位于已验证工作区内的相对路径。"""

    def __init__(self, config: WorkspaceConfig) -> None:
        """保存已验证的工作区配置。"""
        if not isinstance(config, WorkspaceConfig):
            raise WorkspaceError("config 必须是 WorkspaceConfig")

        self._root = config.root

    @property
    def root(self) -> Path:
        """返回已验证工作区根目录。"""
        return self._root

    def resolve_path(self, user_path: object) -> Path:
        """将用户相对路径解析为工作区内的绝对路径。"""
        if not isinstance(user_path, str) or not user_path.strip():
            raise WorkspaceError("路径必须是非空文本")

        relative_path = Path(user_path.strip())

        if relative_path.is_absolute():
            raise WorkspaceError("不允许使用绝对路径")

        if ".." in relative_path.parts:
            raise WorkspaceError("不允许使用父目录穿越")

        resolved_path = (self._root / relative_path).resolve()

        try:
            resolved_path.relative_to(self._root)
        except ValueError as error:
            raise WorkspaceError("路径不能逃离受限工作区") from error

        return resolved_path