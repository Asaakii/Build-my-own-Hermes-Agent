"""HermesLite 的 SQLite 状态数据库初始化与领域对象读写。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3

from dotenv import load_dotenv

from hermes_lite.coding_task import (
    CodingTaskReport,
    ToolRoundSummary,
    VerificationStatus,
)
from hermes_lite.config import DOTENV_PATH, PROJECT_ROOT
from hermes_lite.domain import (
    Message,
    MessageRole,
    Session,
    TaskState,
    TaskStatus,
    ToolCall,
    ToolResult,
)


DEFAULT_STATE_DB_RELATIVE_PATH = Path("data") / "hermes_lite.sqlite3"
SCHEMA_VERSION = 1


class SQLiteStateStoreError(ValueError):
    """SQLite 状态数据库配置、初始化或记录恢复失败时抛出。"""


@dataclass(frozen=True, slots=True)
class SQLiteStateConfig:
    """已验证的项目内 SQLite 数据库位置。"""

    database_path: Path

    def __post_init__(self) -> None:
        """限制数据库仅能位于项目内 data 目录。"""
        if not isinstance(self.database_path, Path):
            raise SQLiteStateStoreError("database_path 必须是 Path")

        project_root = PROJECT_ROOT.resolve()
        database_path = self.database_path.resolve()

        try:
            relative_path = database_path.relative_to(project_root)
        except ValueError as error:
            raise SQLiteStateStoreError("状态数据库必须位于项目根目录内") from error

        if not relative_path.parts or relative_path.parts[0] != "data":
            raise SQLiteStateStoreError("状态数据库必须位于 data 目录内")

        if database_path.suffix != ".sqlite3":
            raise SQLiteStateStoreError("状态数据库必须使用 .sqlite3 后缀")

        object.__setattr__(self, "database_path", database_path)


@dataclass(frozen=True, slots=True)
class SessionRestoreResult:
    """一次会话恢复的结果，包括被跳过的损坏消息数量。"""

    session: Session | None
    skipped_message_records: int = 0

    def __post_init__(self) -> None:
        """验证恢复结果不会伪造会话或跳过数量。"""
        if self.session is not None and not isinstance(self.session, Session):
            raise SQLiteStateStoreError("session 必须是 Session 或 None")

        if (
            isinstance(self.skipped_message_records, bool)
            or not isinstance(self.skipped_message_records, int)
            or self.skipped_message_records < 0
        ):
            raise SQLiteStateStoreError("skipped_message_records 必须是非负整数")


@dataclass(frozen=True, slots=True)
class StoredCodingTask:
    """从 SQLite 恢复的一条任务状态及其可选最终报告。"""

    task: TaskState
    report: CodingTaskReport | None

    def __post_init__(self) -> None:
        """确保恢复结果仍使用已验证的领域对象。"""
        if not isinstance(self.task, TaskState):
            raise SQLiteStateStoreError("task 必须是 TaskState")

        if self.report is not None and not isinstance(
            self.report,
            CodingTaskReport,
        ):
            raise SQLiteStateStoreError("report 必须是 CodingTaskReport 或 None")


def _get_database_relative_path(environment: Mapping[str, str]) -> Path:
    """读取并验证项目内相对数据库路径。"""
    raw_path = environment.get("AGENT_STATE_DB_PATH")

    if raw_path is None:
        return DEFAULT_STATE_DB_RELATIVE_PATH

    if not isinstance(raw_path, str) or not raw_path.strip():
        raise SQLiteStateStoreError("AGENT_STATE_DB_PATH 必须是非空文本")

    relative_path = Path(raw_path.strip())

    if relative_path.is_absolute():
        raise SQLiteStateStoreError("AGENT_STATE_DB_PATH 只能是项目内相对路径")

    if ".." in relative_path.parts:
        raise SQLiteStateStoreError("AGENT_STATE_DB_PATH 不能包含父目录穿越")

    return relative_path


def load_sqlite_state_config(
    environment: Mapping[str, str] | None = None,
) -> SQLiteStateConfig:
    """加载并验证 SQLite 状态数据库配置。"""
    if environment is None:
        load_dotenv(DOTENV_PATH, override=False)
        environment = os.environ

    relative_path = _get_database_relative_path(environment)

    return SQLiteStateConfig(database_path=PROJECT_ROOT / relative_path)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    session_id TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls_json TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (session_id, sequence_number),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_request TEXT NOT NULL,
    status TEXT NOT NULL,
    tool_rounds INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS tool_results (
    task_id TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    result_sequence INTEGER NOT NULL,
    call_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    content TEXT NOT NULL,
    is_error INTEGER NOT NULL,
    PRIMARY KEY (task_id, round_number, result_sequence),
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

CREATE TABLE IF NOT EXISTS task_reports (
    task_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    verification TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);
"""


def _utc_now_text() -> str:
    """生成可保存到 SQLite 的 UTC 时间文本。"""
    return datetime.now(timezone.utc).isoformat()


def _serialize_tool_calls(calls: tuple[ToolCall, ...]) -> str | None:
    """把助手工具请求转换为稳定的 JSON 文本。"""
    if not calls:
        return None

    payload = [
        {
            "call_id": call.call_id,
            "tool_name": call.tool_name,
            "arguments": call.arguments,
        }
        for call in calls
    ]

    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SQLiteStateStoreError("工具调用参数无法序列化为 JSON") from error


def _deserialize_tool_calls(raw_value: object) -> tuple[ToolCall, ...]:
    """把 SQLite 中的工具请求 JSON 恢复为领域对象。"""
    if raw_value is None:
        return ()

    if not isinstance(raw_value, str):
        raise SQLiteStateStoreError("工具调用记录格式无效")

    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise SQLiteStateStoreError("工具调用记录不是有效 JSON") from error

    if not isinstance(payload, list):
        raise SQLiteStateStoreError("工具调用记录必须是列表")

    if not all(isinstance(item, dict) for item in payload):
        raise SQLiteStateStoreError("工具调用记录内容无效")

    try:
        return tuple(
            ToolCall(
                call_id=item["call_id"],
                tool_name=item["tool_name"],
                arguments=item["arguments"],
            )
            for item in payload
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SQLiteStateStoreError("工具调用记录内容无效") from error


class SQLiteStateStore:
    """负责 SQLite 模式管理及正常领域对象的读写恢复。"""

    def __init__(self, config: SQLiteStateConfig) -> None:
        """保存已经验证的数据库配置。"""
        if not isinstance(config, SQLiteStateConfig):
            raise SQLiteStateStoreError("config 必须是 SQLiteStateConfig")

        self._config = config

    @property
    def database_path(self) -> Path:
        """返回已验证的数据库文件路径。"""
        return self._config.database_path

    def _connect(self) -> sqlite3.Connection:
        """建立启用外键约束的 SQLite 连接。"""
        try:
            connection = sqlite3.connect(self._config.database_path)
            connection.execute("PRAGMA foreign_keys = ON")
            return connection
        except sqlite3.Error as error:
            raise SQLiteStateStoreError("无法连接状态数据库") from error

    def initialize(self) -> None:
        """创建父目录、数据表和当前模式版本，可安全重复执行。"""
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SQLiteStateStoreError("无法创建状态数据库目录") from error

        connection: sqlite3.Connection | None = None

        try:
            connection = self._connect()

            with connection:
                connection.executescript(_SCHEMA_SQL)
                connection.execute(
                    "INSERT OR IGNORE INTO schema_metadata (key, value) "
                    "VALUES (?, ?)",
                    ("schema_version", str(SCHEMA_VERSION)),
                )
        except sqlite3.Error as error:
            raise SQLiteStateStoreError("无法初始化状态数据库") from error
        finally:
            if connection is not None:
                connection.close()

    def schema_version(self) -> int:
        """读取已初始化数据库的模式版本。"""
        self.initialize()
        connection: sqlite3.Connection | None = None

        try:
            connection = self._connect()
            row = connection.execute(
                "SELECT value FROM schema_metadata WHERE key = ?",
                ("schema_version",),
            ).fetchone()
        except sqlite3.Error as error:
            raise SQLiteStateStoreError("状态数据库尚未初始化") from error
        finally:
            if connection is not None:
                connection.close()

        if row is None:
            raise SQLiteStateStoreError("状态数据库缺少模式版本")

        try:
            return int(row[0])
        except (TypeError, ValueError) as error:
            raise SQLiteStateStoreError("状态数据库模式版本无效") from error

    def save_session(self, session: Session) -> None:
        """覆盖保存一条会话的完整、有序消息历史。"""
        if not isinstance(session, Session):
            raise SQLiteStateStoreError("session 必须是 Session")

        self.initialize()
        connection: sqlite3.Connection | None = None

        try:
            connection = self._connect()
            now = _utc_now_text()

            with connection:
                connection.execute(
                    "INSERT OR IGNORE INTO sessions (session_id, created_at) "
                    "VALUES (?, ?)",
                    (session.session_id, now),
                )
                connection.execute(
                    "DELETE FROM messages WHERE session_id = ?",
                    (session.session_id,),
                )
                rows = [
                    (
                        session.session_id,
                        index,
                        message.role.value,
                        message.content,
                        message.tool_call_id,
                        _serialize_tool_calls(message.tool_calls),
                        now,
                    )
                    for index, message in enumerate(session.messages)
                ]

                if rows:
                    connection.executemany(
                        "INSERT INTO messages "
                        "(session_id, sequence_number, role, content, "
                        "tool_call_id, tool_calls_json, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        rows,
                    )
        except sqlite3.Error as error:
            raise SQLiteStateStoreError("无法保存会话记录") from error
        finally:
            if connection is not None:
                connection.close()

    def restore_session(self, session_id: object) -> SessionRestoreResult:
        """恢复会话，并跳过单条无法构造成领域消息的损坏记录。"""
        if not isinstance(session_id, str) or not session_id.strip():
            raise SQLiteStateStoreError("session_id 必须是非空文本")

        self.initialize()
        normalized_session_id = session_id.strip()
        connection: sqlite3.Connection | None = None

        try:
            connection = self._connect()
            exists = connection.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (normalized_session_id,),
            ).fetchone()

            if exists is None:
                return SessionRestoreResult(session=None)

            rows = connection.execute(
                "SELECT role, content, tool_call_id, tool_calls_json "
                "FROM messages WHERE session_id = ? "
                "ORDER BY sequence_number",
                (normalized_session_id,),
            ).fetchall()
            messages: list[Message] = []
            skipped_message_records = 0

            for row in rows:
                try:
                    messages.append(
                        Message(
                            role=MessageRole(row[0]),
                            content=row[1],
                            tool_call_id=row[2],
                            tool_calls=_deserialize_tool_calls(row[3]),
                        )
                    )
                except (SQLiteStateStoreError, TypeError, ValueError):
                    skipped_message_records += 1

            return SessionRestoreResult(
                session=Session(
                    session_id=normalized_session_id,
                    messages=messages,
                ),
                skipped_message_records=skipped_message_records,
            )
        except sqlite3.Error as error:
            raise SQLiteStateStoreError("会话记录无法恢复") from error
        finally:
            if connection is not None:
                connection.close()

    def load_session(self, session_id: object) -> Session | None:
        """按会话标识恢复消息历史，不存在时返回 None。"""
        return self.restore_session(session_id).session

    def save_coding_task(
        self,
        task: TaskState,
        report: CodingTaskReport,
    ) -> None:
        """保存任务状态、每轮工具结果和最终编码报告。"""
        if not isinstance(task, TaskState):
            raise SQLiteStateStoreError("task 必须是 TaskState")

        if not isinstance(report, CodingTaskReport):
            raise SQLiteStateStoreError("report 必须是 CodingTaskReport")

        if task.task_id != report.task_id:
            raise SQLiteStateStoreError("任务状态与报告 task_id 不一致")

        self.initialize()
        connection: sqlite3.Connection | None = None

        try:
            connection = self._connect()
            now = _utc_now_text()

            with connection:
                session_row = connection.execute(
                    "SELECT session_id FROM sessions WHERE session_id = ?",
                    (task.session_id,),
                ).fetchone()

                if session_row is None:
                    raise SQLiteStateStoreError("任务所属会话尚未保存")

                connection.execute(
                    "INSERT INTO tasks "
                    "(task_id, session_id, user_request, status, tool_rounds, "
                    "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(task_id) DO UPDATE SET "
                    "session_id = excluded.session_id, "
                    "user_request = excluded.user_request, "
                    "status = excluded.status, "
                    "tool_rounds = excluded.tool_rounds, "
                    "updated_at = excluded.updated_at",
                    (
                        task.task_id,
                        task.session_id,
                        task.user_request,
                        task.status.value,
                        task.tool_rounds,
                        now,
                        now,
                    ),
                )
                connection.execute(
                    "DELETE FROM tool_results WHERE task_id = ?",
                    (task.task_id,),
                )

                result_rows = [
                    (
                        task.task_id,
                        summary.round_number,
                        result_index,
                        result.call_id,
                        result.tool_name,
                        result.content,
                        int(result.is_error),
                    )
                    for summary in report.rounds
                    for result_index, result in enumerate(summary.results)
                ]

                if result_rows:
                    connection.executemany(
                        "INSERT INTO tool_results "
                        "(task_id, round_number, result_sequence, call_id, "
                        "tool_name, content, is_error) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        result_rows,
                    )

                connection.execute(
                    "INSERT INTO task_reports "
                    "(task_id, status, verification, summary, created_at) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(task_id) DO UPDATE SET "
                    "status = excluded.status, "
                    "verification = excluded.verification, "
                    "summary = excluded.summary, "
                    "created_at = excluded.created_at",
                    (
                        report.task_id,
                        report.status.value,
                        report.verification.value,
                        report.summary,
                        now,
                    ),
                )
        except sqlite3.Error as error:
            raise SQLiteStateStoreError("无法保存任务记录") from error
        finally:
            if connection is not None:
                connection.close()

    def load_coding_task(self, task_id: object) -> StoredCodingTask | None:
        """恢复任务状态、工具结果和最终报告，不存在时返回 None。"""
        if not isinstance(task_id, str) or not task_id.strip():
            raise SQLiteStateStoreError("task_id 必须是非空文本")

        self.initialize()
        normalized_task_id = task_id.strip()
        connection: sqlite3.Connection | None = None

        try:
            connection = self._connect()
            task_row = connection.execute(
                "SELECT task_id, session_id, user_request, status, tool_rounds "
                "FROM tasks WHERE task_id = ?",
                (normalized_task_id,),
            ).fetchone()

            if task_row is None:
                return None

            task = TaskState(
                task_id=task_row[0],
                session_id=task_row[1],
                user_request=task_row[2],
                status=TaskStatus(task_row[3]),
                tool_rounds=task_row[4],
            )
            report_row = connection.execute(
                "SELECT status, verification, summary "
                "FROM task_reports WHERE task_id = ?",
                (normalized_task_id,),
            ).fetchone()

            if report_row is None:
                return StoredCodingTask(task=task, report=None)

            result_rows = connection.execute(
                "SELECT round_number, result_sequence, call_id, tool_name, "
                "content, is_error FROM tool_results WHERE task_id = ? "
                "ORDER BY round_number, result_sequence",
                (normalized_task_id,),
            ).fetchall()
            grouped_results: dict[int, list[ToolResult]] = {}

            for row in result_rows:
                grouped_results.setdefault(row[0], []).append(
                    ToolResult(
                        call_id=row[2],
                        tool_name=row[3],
                        content=row[4],
                        is_error=bool(row[5]),
                    )
                )

            report = CodingTaskReport(
                task_id=task.task_id,
                status=TaskStatus(report_row[0]),
                verification=VerificationStatus(report_row[1]),
                summary=report_row[2],
                rounds=tuple(
                    ToolRoundSummary(
                        round_number=round_number,
                        results=tuple(results),
                    )
                    for round_number, results in grouped_results.items()
                ),
            )
            return StoredCodingTask(task=task, report=report)
        except (sqlite3.Error, ValueError, TypeError) as error:
            raise SQLiteStateStoreError("任务记录损坏或无法恢复") from error
        finally:
            if connection is not None:
                connection.close()
