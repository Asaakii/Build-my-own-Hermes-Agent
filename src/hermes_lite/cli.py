"""HermesLite 的最小本地命令行入口。

本模块只解析命令、调用既有核心服务并展示结果；不会自行实现模型、
会话、记忆、技能或工具的业务逻辑。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import re
import sys

from hermes_lite.chat_runtime import (
    ChatRuntime,
    ChatRuntimeError,
    ChatTurnResult,
    build_local_chat_runtime,
)
from hermes_lite.config import (
    ConfigurationError,
    format_config_summary,
    load_model_config,
)
from hermes_lite.domain import TaskStatus
from hermes_lite.memory_store import (
    LongTermMemory,
    MemoryStoreError,
    SQLiteMemoryStore,
)
from hermes_lite.skill_loader import (
    Skill,
    SkillLoadError,
    list_available_skills,
)
from hermes_lite.sqlite_state_store import (
    SQLiteStateStore,
    SQLiteStateStoreError,
    load_sqlite_state_config,
)
from hermes_lite.workspace import WorkspaceError


_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9:_-]{1,80}$")


class CliUsageError(ValueError):
    """表示命令参数不符合 CLI 的固定接口。"""


class CliArgumentParser(argparse.ArgumentParser):
    """将 argparse 的参数错误转换为可预测的项目错误。"""

    def error(self, message: str) -> None:
        """不回显原始参数，避免把意外敏感输入打印到终端。"""
        del message
        raise CliUsageError("命令参数无效")


def _positive_integer(value: str) -> int:
    """解析供列表命令使用的正整数上限。"""
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须是正整数") from error

    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")

    return number


def _session_id(value: str) -> str:
    """限制 CLI 会话标识为可展示、可恢复的固定字符集合。"""
    if not _SESSION_ID_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError("会话标识格式无效")
    return value


def build_parser() -> CliArgumentParser:
    """构建本地 CLI 的命令与参数结构。"""
    parser = CliArgumentParser(
        prog="hermeslite",
        description="HermesLite 受限本地 Agent 的命令行工具。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("config", help="检查并显示脱敏模型配置。")

    chat_parser = subparsers.add_parser(
        "chat",
        help="运行可恢复的受限工具聊天；省略消息进入交互模式。",
    )
    chat_parser.add_argument(
        "--session-id",
        type=_session_id,
        default="local-default",
        help="会话标识，默认 local-default。",
    )
    chat_parser.add_argument(
        "--skill",
        help="本轮显式选择的受限技能名称。",
    )
    chat_parser.add_argument(
        "message",
        nargs="?",
        help="单次发送的用户消息；省略后进入交互模式。",
    )

    sessions_parser = subparsers.add_parser(
        "sessions",
        help="查看已保存会话的元数据。",
    )
    sessions_subparsers = sessions_parser.add_subparsers(
        dest="sessions_command",
        required=True,
    )
    sessions_show_parser = sessions_subparsers.add_parser(
        "show",
        help="显示单个会话的消息数量，不显示正文。",
    )
    sessions_show_parser.add_argument(
        "session_id",
        type=_session_id,
        help="要查看的会话标识。",
    )

    memory_parser = subparsers.add_parser(
        "memory",
        help="查看或检索已明确授权的长期记忆。",
    )
    memory_subparsers = memory_parser.add_subparsers(
        dest="memory_command",
        required=True,
    )
    memory_list_parser = memory_subparsers.add_parser(
        "list",
        help="列出已授权长期记忆。",
    )
    memory_list_parser.add_argument(
        "--limit",
        type=_positive_integer,
        default=5,
        help="最多显示的记忆条数，默认 5。",
    )
    memory_search_parser = memory_subparsers.add_parser(
        "search",
        help="按关键词检索已授权长期记忆。",
    )
    memory_search_parser.add_argument("query", help="检索关键词。")
    memory_search_parser.add_argument(
        "--limit",
        type=_positive_integer,
        default=5,
        help="最多显示的记忆条数，默认 5。",
    )

    skills_parser = subparsers.add_parser(
        "skills",
        help="列出项目内已经通过校验的技能。",
    )
    skills_subparsers = skills_parser.add_subparsers(
        dest="skills_command",
        required=True,
    )
    skills_subparsers.add_parser("list", help="列出可用技能。")

    return parser


def _load_state_store() -> SQLiteStateStore:
    """加载并初始化唯一的本地 SQLite 状态库。"""
    state_store = SQLiteStateStore(load_sqlite_state_config())
    state_store.initialize()
    return state_store


def _load_memory_store() -> SQLiteMemoryStore:
    """基于同一状态库构建长期记忆服务。"""
    return SQLiteMemoryStore(_load_state_store())


def _print_memories(memories: Sequence[LongTermMemory]) -> None:
    """显示用户已明确授权保存的记忆，不显示来源会话标识。"""
    if not memories:
        print("没有已授权的长期记忆。")
        return

    for memory in memories:
        print(f"记忆 #{memory.memory_id}: {memory.content}")


def _print_skills(skills: Sequence[Skill]) -> None:
    """显示已验证的技能元数据，不打印完整指令正文。"""
    if not skills:
        print("没有可用技能。")
        return

    for skill in skills:
        allowed_tools = ", ".join(skill.allowed_tools) or "无"
        print(f"技能: {skill.name}")
        print(f"说明: {skill.description}")
        print(f"允许工具: {allowed_tools}")


def _print_chat_turn(
    result: ChatTurnResult,
    *,
    interactive: bool,
) -> int:
    """按任务最终状态显示安全聊天结果，并返回对应退出码。"""
    turn = result.turn

    if result.skipped_message_records:
        print(f"提示: 已跳过 {result.skipped_message_records} 条损坏消息记录。")

    if turn.task.status is TaskStatus.COMPLETED:
        assert turn.answer is not None
        print(f"Agent: {turn.answer}")
        return 0

    if turn.task.status is TaskStatus.BLOCKED:
        print(f"Agent: {turn.error_message}")
        pending = turn.pending_confirmation
        if interactive and pending is not None:
            print(f"请在本次会话中发送: /confirm {pending.token}")
        else:
            print("提示: 高风险操作只能在同一交互会话内确认，请重新进入交互模式后发起。")
        return 3

    print(f"Agent 任务失败: {turn.error_message or "未获得可用回答。"}")
    return 1


def run_interactive_chat(
    runtime: ChatRuntime,
    session_id: str,
    skill_name: object | None = None,
    input_fn: Callable[[str], str] = input,
) -> int:
    """运行本地交互聊天；/remember 和 /confirm 继续交给既有 ToolAgent。"""
    print("HermesLite 交互聊天已启动。输入 /help 查看本地命令，/exit 或 /quit 退出。")

    while True:
        try:
            user_request = input_fn("你: ")
        except (EOFError, KeyboardInterrupt):
            print("聊天已结束。")
            return 0

        if not isinstance(user_request, str) or not user_request.strip():
            print("消息不能为空，请重新输入。")
            continue

        normalized_request = user_request.strip()
        if normalized_request in {"/exit", "/quit"}:
            print("聊天已结束。")
            return 0

        if normalized_request == "/help":
            print("本地命令: /help、/exit、/quit。/remember 与 /confirm 由 Agent 安全策略处理。")
            continue

        try:
            result = runtime.run_turn(
                session_id,
                user_request,
                skill_name=skill_name,
            )
        except ChatRuntimeError:
            print("聊天状态保存失败，请检查本地数据库和工作区配置。")
            continue

        _print_chat_turn(result, interactive=True)


def _run_config() -> int:
    """检查模型配置并显示脱敏摘要。"""
    print(format_config_summary(load_model_config()))
    return 0


def _run_chat(
    session_id: str,
    skill_name: object | None,
    message: str | None,
) -> int:
    """执行单次聊天或启动交互聊天。"""
    runtime = build_local_chat_runtime()

    if message is None:
        return run_interactive_chat(runtime, session_id, skill_name)

    if not message.strip():
        print("命令参数错误。请提供非空消息。", file=sys.stderr)
        return 2

    try:
        result = runtime.run_turn(session_id, message, skill_name=skill_name)
    except ChatRuntimeError:
        print("聊天状态保存失败，请检查本地数据库和工作区配置。", file=sys.stderr)
        return 1

    return _print_chat_turn(result, interactive=False)


def _run_sessions_show(session_id: str) -> int:
    """显示指定已保存会话的元数据，避免输出完整聊天正文。"""
    result = _load_state_store().restore_session(session_id)
    if result.session is None:
        print("会话不存在。", file=sys.stderr)
        return 1

    print(f"会话: {result.session.session_id}")
    print(f"消息数: {len(result.session.messages)}")
    if result.skipped_message_records:
        print(f"提示: 已跳过 {result.skipped_message_records} 条损坏消息记录。")
    return 0


def _run_memory_list(limit: int) -> int:
    """列出有限条已授权长期记忆。"""
    _print_memories(_load_memory_store().list_memories(limit))
    return 0


def _run_memory_search(query: str, limit: int) -> int:
    """检索有限条已授权长期记忆。"""
    _print_memories(_load_memory_store().search(query, limit))
    return 0


def _run_skills_list() -> int:
    """列出项目目录中全部通过校验的技能。"""
    _print_skills(list_available_skills())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令并返回稳定退出码，便于脚本和自动化测试调用。"""
    parser = build_parser()
    try:
        arguments = parser.parse_args(argv)
    except CliUsageError:
        print("命令参数错误。请使用 --help 查看可用命令。", file=sys.stderr)
        return 2
    except SystemExit as error:
        return int(error.code)

    try:
        if arguments.command == "config":
            return _run_config()
        if arguments.command == "chat":
            return _run_chat(
                arguments.session_id,
                arguments.skill,
                arguments.message,
            )
        if arguments.command == "sessions":
            return _run_sessions_show(arguments.session_id)
        if arguments.command == "memory":
            if arguments.memory_command == "list":
                return _run_memory_list(arguments.limit)
            return _run_memory_search(arguments.query, arguments.limit)
        if arguments.command == "skills":
            return _run_skills_list()
    except (
        ConfigurationError,
        MemoryStoreError,
        SkillLoadError,
        SQLiteStateStoreError,
        WorkspaceError,
    ):
        print("命令执行失败，请检查本地配置或数据。", file=sys.stderr)
        return 1

    print("命令参数错误。请使用 --help 查看可用命令。", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
