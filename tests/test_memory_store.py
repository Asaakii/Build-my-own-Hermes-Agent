"""验证显式授权的长期记忆保存、去重、检索与隐私边界。"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_lite.sqlite_state_store as state_store_module
from hermes_lite.memory_store import (
    MAX_MEMORY_CHARACTERS,
    MemoryPrivacyError,
    MemoryStoreError,
    RememberRequest,
    SQLiteMemoryStore,
    parse_remember_command,
)
from hermes_lite.sqlite_state_store import (
    SQLiteStateStore,
    load_sqlite_state_config,
)


@pytest.fixture
def memory_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> SQLiteMemoryStore:
    """创建使用独立临时 SQLite 数据库的长期记忆存储。"""
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(state_store_module, "PROJECT_ROOT", project_root)
    state_store = SQLiteStateStore(load_sqlite_state_config({}))
    return SQLiteMemoryStore(state_store)


def test_normal_chat_text_never_becomes_memory_request(
    memory_store: SQLiteMemoryStore,
) -> None:
    """普通聊天与相似命令不会自动获得长期保存授权。"""
    assert parse_remember_command("session-1", "记住：偏好简洁回答") is None
    assert parse_remember_command("session-1", "/remembering 偏好简洁回答") is None

    with pytest.raises(MemoryStoreError, match="只能保存 RememberRequest"):
        memory_store.save_authorized("偏好简洁回答")  # type: ignore[arg-type]


def test_remember_command_persists_and_cross_instance_searches(
    memory_store: SQLiteMemoryStore,
) -> None:
    """明确授权后可从同一 SQLite 数据库的另一存储实例检索。"""
    request = parse_remember_command(
        "session-1",
        "/remember 偏好使用简洁中文回答",
    )

    assert request is not None
    saved = memory_store.save_authorized(request)
    reloaded_store = SQLiteMemoryStore(memory_store._state_store)
    found = reloaded_store.search("简洁中文")

    assert saved.created
    assert saved.memory.source_session_id == "session-1"
    assert [memory.content for memory in found] == ["偏好使用简洁中文回答"]


def test_remember_deduplicates_normalized_content(
    memory_store: SQLiteMemoryStore,
) -> None:
    """仅空白与大小写不同的重复记忆不能创建第二条记录。"""
    first = memory_store.save_authorized(
        RememberRequest("session-1", "偏好使用 Python"),
    )
    second = memory_store.save_authorized(
        RememberRequest("session-2", "偏好使用   python"),
    )

    assert first.created
    assert not second.created
    assert second.memory.memory_id == first.memory.memory_id
    assert second.memory.source_session_id == "session-1"
    assert len(memory_store.search("偏好")) == 1


def test_search_returns_empty_when_memory_is_not_found(
    memory_store: SQLiteMemoryStore,
) -> None:
    """未命中长期记忆应返回空元组，而不是伪造答案。"""
    memory_store.save_authorized(
        RememberRequest("session-1", "偏好使用中文"),
    )

    assert memory_store.search("不存在的偏好") == ()


@pytest.mark.parametrize(
    "content",
    [
        "api_key=abc123456789",
        "password: abc123456789",
        "Bearer abcdefghijklmnop",
        "ghp_abcdefghijklmnop",
    ],
)
def test_remember_rejects_basic_sensitive_credential_patterns(content: str) -> None:
    """明显凭据格式不能因 /remember 获得持久化资格。"""
    with pytest.raises(MemoryPrivacyError, match="敏感凭据"):
        parse_remember_command("session-1", f"/remember {content}")


def test_remember_requires_explicit_nonempty_content() -> None:
    """授权命令必须带非空内容，不能把命令本身写入记忆。"""
    with pytest.raises(MemoryStoreError, match="/remember 后必须提供"):
        parse_remember_command("session-1", "/remember")


def test_memory_rejects_oversized_content() -> None:
    """单条记忆大小受限，避免持久化任意大段对话。"""
    with pytest.raises(MemoryStoreError, match="不能超过"):
        RememberRequest("session-1", "长" * (MAX_MEMORY_CHARACTERS + 1))


def test_search_rejects_invalid_limit(memory_store: SQLiteMemoryStore) -> None:
    """检索数量边界不能接受零、布尔值或过大值。"""
    with pytest.raises(MemoryStoreError, match="max_results"):
        memory_store.search("偏好", max_results=True)

    with pytest.raises(MemoryStoreError, match="max_results"):
        memory_store.search("偏好", max_results=21)
