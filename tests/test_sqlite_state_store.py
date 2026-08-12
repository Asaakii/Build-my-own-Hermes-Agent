"""验证 SQLite 状态数据库配置与初始化。"""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

import hermes_lite.sqlite_state_store as state_store_module
from hermes_lite.sqlite_state_store import (
    SCHEMA_VERSION,
    SQLiteStateStore,
    SQLiteStateStoreError,
    load_sqlite_state_config,
)


@pytest.fixture
def project_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    """将状态数据库配置限制到独立临时项目。"""
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(state_store_module, "PROJECT_ROOT", root)

    return root


def test_load_config_uses_project_data_directory(project_root: Path) -> None:
    """默认数据库位置必须在项目内的 data 目录。"""
    config = load_sqlite_state_config({})

    assert config.database_path == project_root / "data" / "hermes_lite.sqlite3"


@pytest.mark.parametrize(
    "raw_path",
    [
        "/tmp/state.sqlite3",
        "../state.sqlite3",
        "outside/state.sqlite3",
        "data/state.db",
    ],
)
def test_load_config_rejects_unsafe_or_invalid_path(
    project_root: Path,
    raw_path: str,
) -> None:
    """数据库路径不能越界、离开 data 目录或使用错误后缀。"""
    del project_root

    with pytest.raises(SQLiteStateStoreError):
        load_sqlite_state_config({"AGENT_STATE_DB_PATH": raw_path})


def test_initialize_creates_all_state_tables(project_root: Path) -> None:
    """初始化应创建全部状态表并写入当前模式版本。"""
    del project_root
    config = load_sqlite_state_config({})
    store = SQLiteStateStore(config)

    store.initialize()

    assert store.database_path.is_file()
    assert store.schema_version() == SCHEMA_VERSION

    connection = sqlite3.connect(store.database_path)
    try:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = ?",
                ("table",),
            )
        }
    finally:
        connection.close()

    assert {
        "schema_metadata",
        "sessions",
        "messages",
        "tasks",
        "tool_results",
        "task_reports",
        "long_term_memories",
        "audit_events",
    }.issubset(table_names)


def test_initialize_is_idempotent(project_root: Path) -> None:
    """重复初始化不能破坏既有模式版本。"""
    del project_root
    store = SQLiteStateStore(load_sqlite_state_config({}))

    store.initialize()
    store.initialize()

    assert store.schema_version() == SCHEMA_VERSION


def test_initialize_reports_unusable_parent_directory(
    project_root: Path,
) -> None:
    """数据库父路径无法创建时，应提供受控错误。"""
    blocked_path = project_root / "data"
    blocked_path.write_text("不是目录", encoding="utf-8")
    config = load_sqlite_state_config(
        {"AGENT_STATE_DB_PATH": "data/state.sqlite3"},
    )

    with pytest.raises(SQLiteStateStoreError, match="无法创建"):
        SQLiteStateStore(config).initialize()



def test_initialize_upgrades_previous_schema_version(project_root: Path) -> None:
    """已有旧版本数据库初始化后应升级到当前模式版本。"""
    config = load_sqlite_state_config({})
    config.database_path.parent.mkdir(parents=True)
    connection = sqlite3.connect(config.database_path)
    try:
        with connection:
            connection.execute(
                "CREATE TABLE schema_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            )
            connection.execute(
                "INSERT INTO schema_metadata (key, value) VALUES (?, ?)",
                ("schema_version", "1"),
            )
    finally:
        connection.close()

    store = SQLiteStateStore(config)
    store.initialize()

    assert store.schema_version() == SCHEMA_VERSION
    connection = sqlite3.connect(config.database_path)
    try:
        table_row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = ? AND name = ?",
            ("table", "long_term_memories"),
        ).fetchone()
        audit_table_row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = ? AND name = ?",
            ("table", "audit_events"),
        ).fetchone()
    finally:
        connection.close()

    assert table_row == ("long_term_memories",)
    assert audit_table_row == ("audit_events",)


def test_read_schema_version_does_not_initialize_missing_database(
    project_root: Path,
) -> None:
    """只读诊断不能因为检查版本而创建状态数据库。"""
    del project_root
    store = SQLiteStateStore(load_sqlite_state_config({}))

    with pytest.raises(SQLiteStateStoreError, match="尚未初始化"):
        store.read_schema_version()

    assert not store.database_path.exists()


def test_read_schema_version_reads_existing_database_without_migration(
    project_root: Path,
) -> None:
    """已初始化数据库可被只读检查，不改变其模式版本。"""
    del project_root
    store = SQLiteStateStore(load_sqlite_state_config({}))
    store.initialize()

    assert store.read_schema_version() == SCHEMA_VERSION
