"""HermesLite 的 SQLite 状态数据库初始化与模式管理。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import sqlite3

from dotenv import load_dotenv

from hermes_lite.config import DOTENV_PATH, PROJECT_ROOT


DEFAULT_STATE_DB_RELATIVE_PATH = Path("data") / "hermes_lite.sqlite3"
SCHEMA_VERSION = 1


class SQLiteStateStoreError(ValueError):
    """SQLite 状态数据库配置或初始化失败时抛出。"""


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


class SQLiteStateStore:
    """负责建立 SQLite 状态数据库，不在本步读写领域对象。"""

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
