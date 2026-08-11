"""HermesLite 工作区内的受控只读观察工具。"""

from __future__ import annotations

from difflib import unified_diff
import os
from os import walk
from pathlib import Path
from tempfile import NamedTemporaryFile

from hermes_lite.domain import ToolCall
from hermes_lite.tool_registry import (
    ToolDefinition,
    ToolExecutionError,
    ToolRegistry,
    ToolRiskLevel,
)
from hermes_lite.workspace import Workspace, WorkspaceError


# 这些限制用于防止单次工具调用占用过多内存或模型上下文。
MAX_FILE_BYTES = 128 * 1024
MAX_READ_CHARACTERS = 12_000
MAX_LIST_ENTRIES = 100
MAX_SEARCH_FILES = 200
# 限制目录遍历量，避免超大目录被完整收集到内存。
MAX_SEARCH_PATHS = 1_000
MAX_SEARCH_RESULTS = 100
MAX_SEARCH_LINE_CHARACTERS = 300
MAX_PATH_CHARACTERS = 512
MAX_QUERY_CHARACTERS = 200
# 写入限制独立于读取限制，避免一次创建过大内容。
MAX_WRITE_BYTES = 64 * 1024
MAX_WRITE_CHARACTERS = 16_000
# 差异预览也属于工具输出，需要独立限制。
MAX_DIFF_LINES = 80
MAX_DIFF_CHARACTERS = 6_000


def _require_text_argument(
    arguments: object,
    name: str,
    max_characters: int,
) -> str:
    """读取并校验一个非空且长度受限的文本参数。"""
    if not isinstance(arguments, dict):
        raise ToolExecutionError("工具参数必须是字典")

    value = arguments.get(name)

    if not isinstance(value, str):
        raise ToolExecutionError(f"{name} 必须是文本")

    cleaned_value = value.strip()

    if not cleaned_value:
        raise ToolExecutionError(f"{name} 不能为空")

    if len(cleaned_value) > max_characters:
        raise ToolExecutionError(f"{name} 长度不能超过 {max_characters} 个字符")

    return cleaned_value


def _resolve_existing_path(workspace: Workspace, raw_path: str) -> Path:
    """解析路径，并确保目标在工作区内且实际存在。"""
    try:
        path = workspace.resolve_path(raw_path)
    except WorkspaceError as error:
        raise ToolExecutionError(str(error)) from error

    if not path.exists():
        raise ToolExecutionError("目标路径不存在")

    return path


def _resolve_new_file_path(workspace: Workspace, raw_path: str) -> Path:
    """解析待创建路径，并确认它不会覆盖已有目标。"""
    try:
        path = workspace.resolve_path(raw_path)
    except WorkspaceError as error:
        raise ToolExecutionError(str(error)) from error

    if path.exists():
        raise ToolExecutionError("目标文件已经存在，拒绝覆盖")

    if not path.parent.is_dir():
        raise ToolExecutionError("目标父目录不存在或不是目录")

    return path


def _require_write_text_argument(
    arguments: object,
    name: str,
    *,
    allow_empty: bool,
) -> str:
    """读取写入类文本参数，保留空白并限制 UTF-8 大小。"""
    if not isinstance(arguments, dict):
        raise ToolExecutionError("工具参数必须是字典")

    value = arguments.get(name)

    if not isinstance(value, str):
        raise ToolExecutionError(f"{name} 必须是文本")

    if not allow_empty and not value:
        raise ToolExecutionError(f"{name} 不能为空")

    if "\x00" in value:
        raise ToolExecutionError(f"{name} 不能包含空字节")

    if len(value) > MAX_WRITE_CHARACTERS:
        raise ToolExecutionError(
            f"{name} 长度不能超过 {MAX_WRITE_CHARACTERS} 个字符"
        )

    if len(value.encode("utf-8")) > MAX_WRITE_BYTES:
        raise ToolExecutionError(
            f"{name} 超过写入上限：{MAX_WRITE_BYTES} 字节"
        )

    return value


def _require_write_content(arguments: object) -> str:
    """读取创建文件时允许为空的文本内容。"""
    return _require_write_text_argument(
        arguments,
        "content",
        allow_empty=True,
    )


def _display_relative_path(workspace: Workspace, path: Path) -> str:
    """将工作区内绝对路径转换为适合工具结果展示的相对路径。"""
    relative_path = path.relative_to(workspace.root)

    if relative_path == Path("."):
        return "."

    return relative_path.as_posix()


def _load_utf8_text(path: Path) -> str:
    """读取大小受限的 UTF-8 文本文件，并拒绝二进制内容。"""
    if not path.is_file():
        raise ToolExecutionError("目标不是普通文件")

    try:
        size = path.stat().st_size
    except OSError as error:
        raise ToolExecutionError("无法读取目标文件") from error

    if size > MAX_FILE_BYTES:
        raise ToolExecutionError(
            f"文件超过读取上限：{MAX_FILE_BYTES} 字节"
        )

    try:
        content = path.read_bytes()
    except OSError as error:
        raise ToolExecutionError("无法读取目标文件") from error

    # 在 stat 与 read_bytes 之间文件可能发生变化，因此再次检查。
    if len(content) > MAX_FILE_BYTES:
        raise ToolExecutionError(
            f"文件超过读取上限：{MAX_FILE_BYTES} 字节"
        )

    if b"\x00" in content:
        raise ToolExecutionError("目标文件可能是二进制文件")

    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ToolExecutionError("目标文件不是 UTF-8 文本") from error


def list_files(workspace: Workspace, arguments: object) -> str:
    """列出工作区内指定目录的第一层安全条目。"""
    raw_path = _require_text_argument(
        arguments,
        "path",
        MAX_PATH_CHARACTERS,
    )
    target = _resolve_existing_path(workspace, raw_path)

    if not target.is_dir():
        raise ToolExecutionError("目标不是目录")

    try:
        children = sorted(target.iterdir(), key=lambda child: child.name)
    except OSError as error:
        raise ToolExecutionError("无法列出目标目录") from error

    entries: list[str] = []
    skipped_unsafe_entries = 0

    for child in children:
        relative_path = child.relative_to(workspace.root).as_posix()

        try:
            safe_path = workspace.resolve_path(relative_path)
        except WorkspaceError:
            # 工作区内若存在指向外部的符号链接，不能向模型暴露其内容。
            skipped_unsafe_entries += 1
            continue

        suffix = "/" if safe_path.is_dir() else ""
        entries.append(f"- {relative_path}{suffix}")

        if len(entries) >= MAX_LIST_ENTRIES:
            break

    displayed_path = _display_relative_path(workspace, target)

    if not entries:
        result = f"目录为空: {displayed_path}"
    else:
        result = f"目录: {displayed_path}\n" + "\n".join(entries)

    if len(children) > len(entries) + skipped_unsafe_entries:
        result += f"\n[目录结果已截断，最多显示 {MAX_LIST_ENTRIES} 项]"

    if skipped_unsafe_entries:
        result += "\n[已跳过不安全的符号链接]"

    return result


def create_text_file(workspace: Workspace, arguments: object) -> str:
    """在工作区已有目录中创建一个新的 UTF-8 文本文件。"""
    raw_path = _require_text_argument(
        arguments,
        "path",
        MAX_PATH_CHARACTERS,
    )
    content = _require_write_content(arguments)
    target = _resolve_new_file_path(workspace, raw_path)

    try:
        # x 模式只允许新建；目标若在检查后被其他进程创建，也不会覆盖。
        with target.open("x", encoding="utf-8") as file:
            file.write(content)
    except FileExistsError as error:
        raise ToolExecutionError("目标文件已经存在，拒绝覆盖") from error
    except OSError as error:
        raise ToolExecutionError("无法创建目标文件") from error

    displayed_path = _display_relative_path(workspace, target)
    byte_count = len(content.encode("utf-8"))

    return (
        f"已创建文本文件: {displayed_path}"
        f"（{len(content)} 个字符，{byte_count} 字节）"
    )


def read_file(workspace: Workspace, arguments: object) -> str:
    """读取工作区内一个大小受限的 UTF-8 文本文件。"""
    raw_path = _require_text_argument(
        arguments,
        "path",
        MAX_PATH_CHARACTERS,
    )
    target = _resolve_existing_path(workspace, raw_path)
    content = _load_utf8_text(target)

    if len(content) > MAX_READ_CHARACTERS:
        content = (
            content[:MAX_READ_CHARACTERS]
            + "\n\n[文件内容已截断]"
        )

    displayed_path = _display_relative_path(workspace, target)

    if not content:
        return f"文件为空: {displayed_path}"

    return f"文件: {displayed_path}\n{content}"


def _count_text_occurrences(content: str, expected_text: str) -> int:
    """计算文本中所有（包含重叠情况的）匹配次数。"""
    count = 0
    start_index = 0

    while True:
        found_index = content.find(expected_text, start_index)

        if found_index == -1:
            return count

        count += 1
        start_index = found_index + 1


def _build_diff_preview(
    displayed_path: str,
    original_content: str,
    updated_content: str,
) -> str:
    """生成受行数和字符数限制的统一差异预览。"""
    preview_lines: list[str] = []
    character_count = 0
    truncated = False

    for line in unified_diff(
        original_content.splitlines(),
        updated_content.splitlines(),
        fromfile=displayed_path,
        tofile=displayed_path,
        lineterm="",
    ):
        if (
            len(preview_lines) >= MAX_DIFF_LINES
            or character_count + len(line) > MAX_DIFF_CHARACTERS
        ):
            truncated = True
            break

        preview_lines.append(line)
        character_count += len(line)

    if not preview_lines:
        return "（无文本差异）"

    preview = "\n".join(preview_lines)

    if truncated:
        preview += "\n[差异预览已截断]"

    return preview


def _replace_text_atomically(
    target: Path,
    original_content: str,
    updated_content: str,
) -> None:
    """先写入同目录临时文件，再原子替换目标文件。"""
    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=".hermes-lite-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(updated_content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        try:
            current_content = _load_utf8_text(target)
        except ToolExecutionError as error:
            raise ToolExecutionError(
                "目标文件在写入期间发生变化"
            ) from error

        if current_content != original_content:
            raise ToolExecutionError("目标文件在写入期间发生变化")

        # 临时文件与目标在同一目录，replace 可避免半写入状态。
        os.replace(temporary_path, target)
        temporary_path = None
    except ToolExecutionError:
        raise
    except OSError as error:
        raise ToolExecutionError("无法原子更新目标文件") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def replace_text_once(workspace: Workspace, arguments: object) -> str:
    """在 UTF-8 文本文件中精确替换唯一出现的一段文本。"""
    raw_path = _require_text_argument(
        arguments,
        "path",
        MAX_PATH_CHARACTERS,
    )
    expected_text = _require_write_text_argument(
        arguments,
        "expected_text",
        allow_empty=False,
    )
    replacement = _require_write_text_argument(
        arguments,
        "replacement",
        allow_empty=True,
    )

    if expected_text == replacement:
        raise ToolExecutionError("replacement 不能与 expected_text 相同")

    target = _resolve_existing_path(workspace, raw_path)
    original_content = _load_utf8_text(target)
    occurrence_count = _count_text_occurrences(
        original_content,
        expected_text,
    )

    if occurrence_count == 0:
        raise ToolExecutionError("未找到需要替换的目标文本")

    if occurrence_count > 1:
        raise ToolExecutionError("目标文本出现多次，拒绝不明确替换")

    updated_content = original_content.replace(
        expected_text,
        replacement,
        1,
    )
    displayed_path = _display_relative_path(workspace, target)
    diff_preview = _build_diff_preview(
        displayed_path,
        original_content,
        updated_content,
    )

    _replace_text_atomically(
        target,
        original_content,
        updated_content,
    )

    return (
        f"已原子替换文本: {displayed_path}\n"
        "匹配次数: 1\n"
        f"差异预览:\n{diff_preview}"
    )


def search_text(workspace: Workspace, arguments: object) -> str:
    """在工作区内指定目录的 UTF-8 文本文件中进行字面量检索。"""
    raw_path = _require_text_argument(
        arguments,
        "path",
        MAX_PATH_CHARACTERS,
    )
    query = _require_text_argument(
        arguments,
        "query",
        MAX_QUERY_CHARACTERS,
    )

    if "\n" in query or "\r" in query:
        raise ToolExecutionError("query 不能包含换行")

    target = _resolve_existing_path(workspace, raw_path)

    if not target.is_dir():
        raise ToolExecutionError("检索目标必须是目录")

    matches: list[str] = []
    scanned_files = 0
    inspected_paths = 0
    files_truncated = False
    paths_truncated = False
    results_truncated = False

    # walk 是逐层生成器；不使用 sorted(rglob(...))，
    # 避免先把整个目录树一次性收集到内存。
    for raw_directory, directory_names, file_names in walk(
        target,
        followlinks=False,
    ):
        directory_names.sort()
        file_names.sort()
        current_directory = Path(raw_directory)
        safe_directory_names: list[str] = []

        # 检查子目录并过滤不安全的符号链接。
        for directory_name in directory_names:
            inspected_paths += 1

            if inspected_paths > MAX_SEARCH_PATHS:
                paths_truncated = True
                break

            candidate = current_directory / directory_name
            relative_path = candidate.relative_to(
                workspace.root
            ).as_posix()

            try:
                safe_directory = workspace.resolve_path(relative_path)
            except WorkspaceError:
                continue

            if safe_directory.is_dir():
                safe_directory_names.append(directory_name)

        if paths_truncated:
            break

        # 只允许 os.walk 继续进入已经确认安全的目录。
        directory_names[:] = safe_directory_names

        for file_name in file_names:
            inspected_paths += 1

            if inspected_paths > MAX_SEARCH_PATHS:
                paths_truncated = True
                break

            candidate = current_directory / file_name
            relative_path = candidate.relative_to(
                workspace.root
            ).as_posix()

            try:
                safe_path = workspace.resolve_path(relative_path)
            except WorkspaceError:
                continue

            if not safe_path.is_file():
                continue

            scanned_files += 1

            if scanned_files > MAX_SEARCH_FILES:
                files_truncated = True
                break

            try:
                content = _load_utf8_text(safe_path)
            except ToolExecutionError:
                # 二进制、过大或不可读取文件不会中断整次检索。
                continue

            for line_number, line in enumerate(
                content.splitlines(),
                start=1,
            ):
                if query not in line:
                    continue

                displayed_line = line.strip()

                if len(displayed_line) > MAX_SEARCH_LINE_CHARACTERS:
                    displayed_line = (
                        displayed_line[:MAX_SEARCH_LINE_CHARACTERS]
                        + "…"
                    )

                matches.append(
                    f"{relative_path}:{line_number}: {displayed_line}"
                )

                if len(matches) >= MAX_SEARCH_RESULTS:
                    results_truncated = True
                    break

            if files_truncated or results_truncated:
                break

        if paths_truncated or files_truncated or results_truncated:
            break

    if matches:
        result = "检索结果:\n" + "\n".join(matches)
    else:
        result = f"未找到包含“{query}”的文本。"

    if paths_truncated:
        result += (
            f"\n[检索路径数已截断，最多检查 {MAX_SEARCH_PATHS} 个路径]"
        )

    if files_truncated:
        result += (
            f"\n[检索文件数已截断，最多扫描 {MAX_SEARCH_FILES} 个文件]"
        )

    if results_truncated:
        result += (
            f"\n[检索结果已截断，最多显示 {MAX_SEARCH_RESULTS} 条]"
        )

    return result


def build_workspace_tool_registry(workspace: Workspace) -> ToolRegistry:
    """创建绑定到指定受限工作区的只读工具注册表。"""
    if not isinstance(workspace, Workspace):
        raise ValueError("workspace 必须是 Workspace")

    registry = ToolRegistry()

    def list_files_handler(arguments: dict[str, object]) -> str:
        """调用目录观察工具。"""
        return list_files(workspace, arguments)

    def read_file_handler(arguments: dict[str, object]) -> str:
        """调用文本读取工具。"""
        return read_file(workspace, arguments)

    def search_text_handler(arguments: dict[str, object]) -> str:
        """调用文本检索工具。"""
        return search_text(workspace, arguments)

    def create_text_file_handler(arguments: dict[str, object]) -> str:
        """调用创建文本文件工具。"""
        return create_text_file(workspace, arguments)

    def replace_text_once_handler(arguments: dict[str, object]) -> str:
        """调用精确文本替换工具。"""
        return replace_text_once(workspace, arguments)

    registry.register(
        ToolDefinition(
            name="list_files",
            description="列出受限工作区内指定目录的第一层内容。",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "工作区内相对目录路径；使用 . 表示根目录。",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            risk_level=ToolRiskLevel.READ_ONLY,
            handler=list_files_handler,
        )
    )
    registry.register(
        ToolDefinition(
            name="read_file",
            description="读取受限工作区内一个 UTF-8 文本文件。",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "工作区内相对文件路径。",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            risk_level=ToolRiskLevel.READ_ONLY,
            handler=read_file_handler,
        )
    )
    registry.register(
        ToolDefinition(
            name="search_text",
            description="在受限工作区内的文本文件中检索字面量文本。",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "工作区内相对目录路径。",
                    },
                    "query": {
                        "type": "string",
                        "description": "需要检索的文本。",
                    },
                },
                "required": ["path", "query"],
                "additionalProperties": False,
            },
            risk_level=ToolRiskLevel.READ_ONLY,
            handler=search_text_handler,
        )
    )
    registry.register(
        ToolDefinition(
            name="create_text_file",
            description="在受限工作区已有目录中创建新的 UTF-8 文本文件，不允许覆盖已有文件。",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "工作区内相对文件路径；父目录必须已存在。",
                    },
                    "content": {
                        "type": "string",
                        "description": "需要写入的新文件文本内容。",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            risk_level=ToolRiskLevel.WRITE,
            handler=create_text_file_handler,
        )
    )
    registry.register(
        ToolDefinition(
            name="replace_text_once",
            description=(
                "在受限工作区的 UTF-8 文本文件中，"
                "精确替换唯一出现的一段文本。"
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "工作区内相对文件路径。",
                    },
                    "expected_text": {
                        "type": "string",
                        "description": "原文件中必须恰好出现一次的旧文本。",
                    },
                    "replacement": {
                        "type": "string",
                        "description": "替换后的新文本；允许为空以删除旧文本。",
                    },
                },
                "required": [
                    "path",
                    "expected_text",
                    "replacement",
                ],
                "additionalProperties": False,
            },
            risk_level=ToolRiskLevel.WRITE,
            handler=replace_text_once_handler,
        )
    )
    return registry