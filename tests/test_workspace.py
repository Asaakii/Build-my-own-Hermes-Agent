"""验证受限工作区配置与路径逃逸防护。"""

from pathlib import Path

import pytest

import hermes_lite.workspace as workspace_module
from hermes_lite.workspace import (
    Workspace,
    WorkspaceConfig,
    WorkspaceError,
    load_workspace_config,
)


@pytest.fixture
def project_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    """创建隔离项目根目录，并替换模块中的项目根路径。"""
    root = tmp_path / "project"
    (root / "sandbox_workspace").mkdir(parents=True)
    monkeypatch.setattr(workspace_module, "PROJECT_ROOT", root)
    return root


def test_load_workspace_config_uses_default_workspace(
    project_root: Path,
) -> None:
    """未配置时应使用项目内默认工作区。"""
    config = load_workspace_config({})

    assert config.root == (project_root / "sandbox_workspace").resolve()


def test_load_workspace_config_accepts_project_relative_workspace(
    project_root: Path,
) -> None:
    """允许选择项目内另一个已存在的子目录。"""
    (project_root / "custom_workspace").mkdir()

    config = load_workspace_config(
        {"AGENT_WORKSPACE_PATH": "custom_workspace"}
    )

    assert config.root == (project_root / "custom_workspace").resolve()


def test_load_workspace_config_rejects_absolute_path(
    project_root: Path,
    tmp_path: Path,
) -> None:
    """配置不能把工作区指向任意绝对路径。"""
    with pytest.raises(WorkspaceError, match="相对路径"):
        load_workspace_config(
            {"AGENT_WORKSPACE_PATH": str(tmp_path / "outside")}
        )


def test_load_workspace_config_rejects_parent_traversal(
    project_root: Path,
) -> None:
    """配置路径不能通过父目录离开项目。"""
    with pytest.raises(WorkspaceError, match="父目录穿越"):
        load_workspace_config({"AGENT_WORKSPACE_PATH": "../outside"})


def test_load_workspace_config_rejects_project_root(
    project_root: Path,
) -> None:
    """项目根目录过宽，不能直接作为 Agent 工作区。"""
    with pytest.raises(WorkspaceError, match="不能直接使用项目根目录"):
        load_workspace_config({"AGENT_WORKSPACE_PATH": "."})


def test_load_workspace_config_rejects_missing_directory(
    project_root: Path,
) -> None:
    """工作区必须在启动前实际存在。"""
    with pytest.raises(WorkspaceError, match="不存在"):
        load_workspace_config({"AGENT_WORKSPACE_PATH": "missing_workspace"})


def test_workspace_resolves_nested_relative_path(
    project_root: Path,
) -> None:
    """工作区内嵌套相对路径应正确解析。"""
    config = load_workspace_config({})
    workspace = Workspace(config)

    resolved_path = workspace.resolve_path("notes/today.md")

    assert resolved_path == (config.root / "notes" / "today.md").resolve()


def test_workspace_rejects_absolute_user_path(
    project_root: Path,
    tmp_path: Path,
) -> None:
    """工具调用不能传入绝对路径。"""
    workspace = Workspace(load_workspace_config({}))

    with pytest.raises(WorkspaceError, match="绝对路径"):
        workspace.resolve_path(str(tmp_path / "outside.txt"))


def test_workspace_rejects_parent_traversal_user_path(
    project_root: Path,
) -> None:
    """工具调用不能用 ../ 读取工作区外文件。"""
    workspace = Workspace(load_workspace_config({}))

    with pytest.raises(WorkspaceError, match="父目录穿越"):
        workspace.resolve_path("../outside.txt")


def test_workspace_rejects_symlink_escape(
    project_root: Path,
    tmp_path: Path,
) -> None:
    """工作区内符号链接指向外部时也必须拒绝。"""
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    (outside_directory / "secret.txt").write_text("secret", encoding="utf-8")

    workspace = Workspace(load_workspace_config({}))
    (workspace.root / "escape").symlink_to(
        outside_directory,
        target_is_directory=True,
    )

    with pytest.raises(WorkspaceError, match="逃离"):
        workspace.resolve_path("escape/secret.txt")