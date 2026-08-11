"""验证受限工作区的只读观察工具。"""

from pathlib import Path

import pytest

import hermes_lite.workspace as workspace_module
import hermes_lite.workspace_tools as workspace_tools_module
from hermes_lite.domain import ToolCall
from hermes_lite.tool_registry import ToolExecutionError
from hermes_lite.workspace import Workspace, load_workspace_config
from hermes_lite.workspace_tools import (
    build_workspace_tool_registry,
    list_files,
    read_file,
    search_text,
    create_text_file,
    replace_text_once,
)


@pytest.fixture
def workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Workspace:
    """创建一个隔离且真实存在的测试工作区。"""
    project_root = tmp_path / "project"
    workspace_root = project_root / "sandbox_workspace"
    workspace_root.mkdir(parents=True)

    monkeypatch.setattr(workspace_module, "PROJECT_ROOT", project_root)

    return Workspace(load_workspace_config({}))


def test_list_files_returns_sorted_immediate_entries(
    workspace: Workspace,
) -> None:
    """目录工具应按名称列出第一层文件与目录。"""
    (workspace.root / "notes").mkdir()
    (workspace.root / "zeta.txt").write_text("z", encoding="utf-8")
    (workspace.root / "alpha.txt").write_text("a", encoding="utf-8")

    result = list_files(workspace, {"path": "."})

    assert result == (
        "目录: .\n"
        "- alpha.txt\n"
        "- notes/\n"
        "- zeta.txt"
    )


def test_list_files_skips_symlink_escaping_workspace(
    workspace: Workspace,
    tmp_path: Path,
) -> None:
    """目录工具不能列出指向工作区外的符号链接目标。"""
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    (outside_directory / "secret.txt").write_text(
        "secret",
        encoding="utf-8",
    )
    (workspace.root / "escape").symlink_to(
        outside_directory,
        target_is_directory=True,
    )

    result = list_files(workspace, {"path": "."})

    assert "escape" not in result
    assert "已跳过不安全的符号链接" in result


def test_read_file_returns_utf8_text(
    workspace: Workspace,
) -> None:
    """文本读取工具应返回工作区内文件内容。"""
    notes_directory = workspace.root / "notes"
    notes_directory.mkdir()
    (notes_directory / "today.txt").write_text(
        "HermesLite 只读工具验证。",
        encoding="utf-8",
    )

    result = read_file(workspace, {"path": "notes/today.txt"})

    assert result == (
        "文件: notes/today.txt\n"
        "HermesLite 只读工具验证。"
    )


def test_read_file_rejects_parent_traversal(
    workspace: Workspace,
) -> None:
    """读取工具不能通过 ../ 访问工作区外内容。"""
    with pytest.raises(ToolExecutionError, match="父目录穿越"):
        read_file(workspace, {"path": "../secret.txt"})


def test_read_file_rejects_binary_content(
    workspace: Workspace,
) -> None:
    """包含空字节的内容不能作为文本交给模型。"""
    (workspace.root / "binary.bin").write_bytes(b"hello\x00world")

    with pytest.raises(ToolExecutionError, match="二进制"):
        read_file(workspace, {"path": "binary.bin"})


def test_read_file_rejects_oversized_file(
    workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """超过读取上限的文件必须拒绝。"""
    monkeypatch.setattr(workspace_tools_module, "MAX_FILE_BYTES", 5)
    (workspace.root / "large.txt").write_text(
        "超过五个字节",
        encoding="utf-8",
    )

    with pytest.raises(ToolExecutionError, match="超过读取上限"):
        read_file(workspace, {"path": "large.txt"})


def test_read_file_rejects_missing_file(
    workspace: Workspace,
) -> None:
    """不存在的文件不能伪装成空内容。"""
    with pytest.raises(ToolExecutionError, match="不存在"):
        read_file(workspace, {"path": "missing.txt"})


def test_search_text_returns_matching_file_and_line(
    workspace: Workspace,
) -> None:
    """检索工具应返回相对路径、行号和匹配行。"""
    notes_directory = workspace.root / "notes"
    notes_directory.mkdir()
    (notes_directory / "today.txt").write_text(
        "第一行\n目标文本在第二行\n第三行",
        encoding="utf-8",
    )
    (workspace.root / "other.txt").write_text(
        "没有匹配内容",
        encoding="utf-8",
    )

    result = search_text(
        workspace,
        {"path": ".", "query": "目标文本"},
    )

    assert result == "检索结果:\nnotes/today.txt:2: 目标文本在第二行"


def test_search_text_rejects_multiline_query(
    workspace: Workspace,
) -> None:
    """检索词不能携带换行，避免污染工具输出结构。"""
    with pytest.raises(ToolExecutionError, match="不能包含换行"):
        search_text(workspace, {"path": ".", "query": "第一行\n第二行"})


def test_search_text_marks_truncated_results(
    workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """检索结果达到上限时必须明确标记截断。"""
    monkeypatch.setattr(workspace_tools_module, "MAX_SEARCH_RESULTS", 1)
    (workspace.root / "first.txt").write_text(
        "关键字",
        encoding="utf-8",
    )
    (workspace.root / "second.txt").write_text(
        "关键字",
        encoding="utf-8",
    )

    result = search_text(workspace, {"path": ".", "query": "关键字"})

    assert "[检索结果已截断，最多显示 1 条]" in result


def test_workspace_tools_run_through_registry(
    workspace: Workspace,
) -> None:
    """三个只读工具必须通过注册表的受控执行入口运行。"""
    (workspace.root / "hello.txt").write_text(
        "hello",
        encoding="utf-8",
    )
    registry = build_workspace_tool_registry(workspace)

    result = registry.execute(
        ToolCall(
            call_id="call-list",
            tool_name="list_files",
            arguments={"path": "."},
        )
    )

    assert result.is_error is False
    assert "hello.txt" in result.content
    assert len(registry.list_model_definitions()) == 6


def test_search_text_marks_truncated_paths(
    workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """路径数量达到上限时必须停止继续遍历。"""
    monkeypatch.setattr(workspace_tools_module, "MAX_SEARCH_PATHS", 1)
    (workspace.root / "first.txt").write_text(
        "关键字",
        encoding="utf-8",
    )
    (workspace.root / "second.txt").write_text(
        "关键字",
        encoding="utf-8",
    )

    result = search_text(workspace, {"path": ".", "query": "关键字"})

    assert "[检索路径数已截断，最多检查 1 个路径]" in result


def test_create_text_file_writes_utf8_content(
    workspace: Workspace,
) -> None:
    """创建工具应在已有目录内写入指定 UTF-8 文本。"""
    (workspace.root / "notes").mkdir()

    result = create_text_file(
        workspace,
        {
            "path": "notes/today.txt",
            "content": "创建型写入验证。",
        },
    )

    assert "已创建文本文件: notes/today.txt" in result
    assert (workspace.root / "notes" / "today.txt").read_text(
        encoding="utf-8"
    ) == "创建型写入验证。"


def test_create_text_file_rejects_existing_file(
    workspace: Workspace,
) -> None:
    """创建工具不能覆盖已有文件，原内容必须保留。"""
    target = workspace.root / "existing.txt"
    target.write_text("原始内容", encoding="utf-8")

    with pytest.raises(ToolExecutionError, match="拒绝覆盖"):
        create_text_file(
            workspace,
            {"path": "existing.txt", "content": "新内容"},
        )

    assert target.read_text(encoding="utf-8") == "原始内容"


def test_create_text_file_rejects_missing_parent(
    workspace: Workspace,
) -> None:
    """创建工具不能隐式创建未知目录。"""
    with pytest.raises(ToolExecutionError, match="父目录不存在"):
        create_text_file(
            workspace,
            {"path": "missing/new.txt", "content": "内容"},
        )


def test_create_text_file_rejects_parent_traversal(
    workspace: Workspace,
) -> None:
    """创建工具不能通过 ../ 写到工作区外。"""
    with pytest.raises(ToolExecutionError, match="父目录穿越"):
        create_text_file(
            workspace,
            {"path": "../outside.txt", "content": "内容"},
        )


def test_create_text_file_rejects_null_byte(
    workspace: Workspace,
) -> None:
    """写入内容不能含空字节，避免生成二进制文件。"""
    with pytest.raises(ToolExecutionError, match="空字节"):
        create_text_file(
            workspace,
            {"path": "binary.txt", "content": "hello\x00world"},
        )


def test_create_text_file_rejects_oversized_content(
    workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """超过字节上限时不能创建文件。"""
    monkeypatch.setattr(workspace_tools_module, "MAX_WRITE_BYTES", 5)

    with pytest.raises(ToolExecutionError, match="超过写入上限"):
        create_text_file(
            workspace,
            {"path": "large.txt", "content": "超过五个字节"},
        )

    assert not (workspace.root / "large.txt").exists()


def test_create_text_file_runs_through_registry(
    workspace: Workspace,
) -> None:
    """创建工具也必须经过注册表的受控执行入口。"""
    registry = build_workspace_tool_registry(workspace)

    result = registry.execute(
        ToolCall(
            call_id="call-create",
            tool_name="create_text_file",
            arguments={
                "path": "created.txt",
                "content": "注册表创建验证。",
            },
        )
    )

    assert result.is_error is False
    assert (workspace.root / "created.txt").read_text(
        encoding="utf-8"
    ) == "注册表创建验证。"


def test_replace_text_once_updates_unique_match(
    workspace: Workspace,
) -> None:
    """唯一旧文本应被替换，并返回差异预览。"""
    target = workspace.root / "note.txt"
    target.write_text("标题\n旧内容\n结尾", encoding="utf-8")

    result = replace_text_once(
        workspace,
        {
            "path": "note.txt",
            "expected_text": "旧内容",
            "replacement": "新内容",
        },
    )

    assert target.read_text(encoding="utf-8") == "标题\n新内容\n结尾"
    assert "已原子替换文本: note.txt" in result
    assert "-旧内容" in result
    assert "+新内容" in result


def test_replace_text_once_rejects_missing_match(
    workspace: Workspace,
) -> None:
    """旧文本不存在时必须拒绝，原文件保持不变。"""
    target = workspace.root / "note.txt"
    target.write_text("原始内容", encoding="utf-8")

    with pytest.raises(ToolExecutionError, match="未找到"):
        replace_text_once(
            workspace,
            {
                "path": "note.txt",
                "expected_text": "不存在",
                "replacement": "新内容",
            },
        )

    assert target.read_text(encoding="utf-8") == "原始内容"


def test_replace_text_once_rejects_ambiguous_match(
    workspace: Workspace,
) -> None:
    """旧文本出现多次时不能猜测替换位置。"""
    target = workspace.root / "note.txt"
    target.write_text("重复 重复", encoding="utf-8")

    with pytest.raises(ToolExecutionError, match="出现多次"):
        replace_text_once(
            workspace,
            {
                "path": "note.txt",
                "expected_text": "重复",
                "replacement": "新内容",
            },
        )

    assert target.read_text(encoding="utf-8") == "重复 重复"


def test_replace_text_once_rejects_binary_file(
    workspace: Workspace,
) -> None:
    """二进制文件不能进入文本替换流程。"""
    target = workspace.root / "binary.bin"
    target.write_bytes(b"hello\x00world")

    with pytest.raises(ToolExecutionError, match="二进制"):
        replace_text_once(
            workspace,
            {
                "path": "binary.bin",
                "expected_text": "hello",
                "replacement": "world",
            },
        )


def test_replace_text_once_keeps_original_when_replace_fails(
    workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """原子替换失败时，原文件和临时文件状态应可控。"""
    target = workspace.root / "note.txt"
    target.write_text("旧内容", encoding="utf-8")

    def raise_replace_error(source: object, destination: object) -> None:
        raise OSError("模拟替换失败")

    monkeypatch.setattr(
        workspace_tools_module.os,
        "replace",
        raise_replace_error,
    )

    with pytest.raises(ToolExecutionError, match="无法原子更新"):
        replace_text_once(
            workspace,
            {
                "path": "note.txt",
                "expected_text": "旧内容",
                "replacement": "新内容",
            },
        )

    assert target.read_text(encoding="utf-8") == "旧内容"
    assert not list(workspace.root.glob(".hermes-lite-*.tmp"))


def test_replace_text_once_runs_through_registry(
    workspace: Workspace,
) -> None:
    """精确替换必须通过注册表执行。"""
    target = workspace.root / "note.txt"
    target.write_text("旧内容", encoding="utf-8")
    registry = build_workspace_tool_registry(workspace)

    result = registry.execute(
        ToolCall(
            call_id="call-replace",
            tool_name="replace_text_once",
            arguments={
                "path": "note.txt",
                "expected_text": "旧内容",
                "replacement": "新内容",
            },
        )
    )

    assert result.is_error is False
    assert target.read_text(encoding="utf-8") == "新内容"

def test_run_pytest_runs_through_registry(
    workspace: Workspace,
) -> None:
    """测试执行工具也必须通过注册表进入受控执行器。"""
    tests_directory = workspace.root / "tests"
    tests_directory.mkdir()
    (tests_directory / "test_sample.py").write_text(
        "def test_sample() -> None:\n    assert True\n",
        encoding="utf-8",
    )
    registry = build_workspace_tool_registry(workspace)

    result = registry.execute(
        ToolCall(
            call_id="call-pytest",
            tool_name="run_pytest",
            arguments={"target": "tests"},
        )
    )

    assert result.is_error is False
    assert "退出码: 0" in result.content
