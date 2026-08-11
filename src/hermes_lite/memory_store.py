"""HermesLite 的显式授权长期记忆存储。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import re
import sqlite3

from hermes_lite.sqlite_state_store import SQLiteStateStore


MAX_MEMORY_CHARACTERS = 500
_SENSITIVE_PATTERNS = (
    re.compile(
        r"(?i)\b(?:api[ _-]?key|password|secret|access[ _-]?token)\b\s*[:=]",
    ),
    re.compile(r"(?i)\b(?:sk-|ghp_|bearer\s+)[a-z0-9._-]{8,}"),
)


class MemoryStoreError(ValueError):
    """表示授权命令、隐私过滤或长期记忆存储失败。"""


class MemoryPrivacyError(MemoryStoreError):
    """表示待保存内容命中基础敏感信息规则。"""


def _require_text(value: object, field_name: str) -> str:
    """验证非空文本并移除首尾空白。"""
    if not isinstance(value, str):
        raise MemoryStoreError(f"{field_name} 必须是文本")

    normalized_value = value.strip()
    if not normalized_value:
        raise MemoryStoreError(f"{field_name} 不能为空")

    return normalized_value


def _normalize_content(content: str) -> str:
    """生成用于精确去重和基础检索的规范化内容。"""
    return " ".join(content.split()).casefold()


def _validate_memory_content(value: object) -> str:
    """限制记忆大小并拒绝明显凭据样式的内容。"""
    content = _require_text(value, "content")

    if len(content) > MAX_MEMORY_CHARACTERS:
        raise MemoryStoreError(
            f"content 不能超过 {MAX_MEMORY_CHARACTERS} 个字符",
        )

    if any(pattern.search(content) for pattern in _SENSITIVE_PATTERNS):
        raise MemoryPrivacyError("记忆内容疑似包含敏感凭据，已拒绝保存")

    return content


@dataclass(frozen=True, slots=True)
class RememberRequest:
    """由明确 /remember 命令生成的长期记忆保存请求。"""

    source_session_id: str
    content: str

    def __post_init__(self) -> None:
        """验证请求来源与待保存内容。"""
        object.__setattr__(
            self,
            "source_session_id",
            _require_text(self.source_session_id, "source_session_id"),
        )
        object.__setattr__(self, "content", _validate_memory_content(self.content))


@dataclass(frozen=True, slots=True)
class LongTermMemory:
    """一条可跨会话读取的、已授权本地记忆。"""

    memory_id: int
    source_session_id: str
    content: str
    created_at: str

    def __post_init__(self) -> None:
        """确保从 SQLite 恢复的数据仍满足领域边界。"""
        if (
            isinstance(self.memory_id, bool)
            or not isinstance(self.memory_id, int)
            or self.memory_id <= 0
        ):
            raise MemoryStoreError("memory_id 必须是正整数")

        object.__setattr__(
            self,
            "source_session_id",
            _require_text(self.source_session_id, "source_session_id"),
        )
        object.__setattr__(self, "content", _validate_memory_content(self.content))
        object.__setattr__(self, "created_at", _require_text(self.created_at, "created_at"))


@dataclass(frozen=True, slots=True)
class MemorySaveResult:
    """一次已授权保存的结果，说明是否实际新建记录。"""

    memory: LongTermMemory
    created: bool

    def __post_init__(self) -> None:
        """保证结果不会把重复保存伪装为新写入。"""
        if not isinstance(self.memory, LongTermMemory):
            raise MemoryStoreError("memory 必须是 LongTermMemory")

        if not isinstance(self.created, bool):
            raise MemoryStoreError("created 必须是布尔值")


def parse_remember_command(
    source_session_id: object,
    user_input: object,
) -> RememberRequest | None:
    """仅将严格的 /remember 命令解析成保存请求。"""
    text = _require_text(user_input, "user_input")
    command, separator, content = text.partition(" ")

    if command != "/remember":
        return None

    if not separator or not content.strip():
        raise MemoryStoreError("/remember 后必须提供要保存的内容")

    return RememberRequest(
        source_session_id=_require_text(source_session_id, "source_session_id"),
        content=content,
    )


class SQLiteMemoryStore:
    """将显式授权的长期记忆保存到现有项目 SQLite 数据库。"""

    def __init__(self, state_store: SQLiteStateStore) -> None:
        """复用已验证的项目内 SQLite 数据库位置。"""
        if not isinstance(state_store, SQLiteStateStore):
            raise MemoryStoreError("state_store 必须是 SQLiteStateStore")

        self._state_store = state_store

    def _connect(self) -> sqlite3.Connection:
        """建立独立连接；模式和位置仍由状态存储统一初始化。"""
        try:
            return sqlite3.connect(self._state_store.database_path)
        except sqlite3.Error as error:
            raise MemoryStoreError("无法连接长期记忆数据库") from error

    @staticmethod
    def _from_row(row: Sequence[object]) -> LongTermMemory:
        """把查询行恢复为经过验证的长期记忆对象。"""
        try:
            return LongTermMemory(
                memory_id=row[0],
                source_session_id=row[1],
                content=row[2],
                created_at=row[3],
            )
        except (IndexError, MemoryStoreError) as error:
            raise MemoryStoreError("长期记忆记录损坏") from error

    def save_authorized(self, request: RememberRequest) -> MemorySaveResult:
        """保存一条明确授权的记忆，重复内容返回既有记录。"""
        if not isinstance(request, RememberRequest):
            raise MemoryStoreError("只能保存 RememberRequest 授权请求")

        self._state_store.initialize()
        normalized_content = _normalize_content(request.content)
        now = datetime.now(timezone.utc).isoformat()
        connection: sqlite3.Connection | None = None

        try:
            connection = self._connect()
            with connection:
                cursor = connection.execute(
                    "INSERT INTO long_term_memories "
                    "(source_session_id, content, normalized_content, created_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(normalized_content) DO NOTHING",
                    (
                        request.source_session_id,
                        request.content,
                        normalized_content,
                        now,
                    ),
                )
                row = connection.execute(
                    "SELECT memory_id, source_session_id, content, created_at "
                    "FROM long_term_memories WHERE normalized_content = ?",
                    (normalized_content,),
                ).fetchone()
        except sqlite3.Error as error:
            raise MemoryStoreError("无法保存长期记忆") from error
        finally:
            if connection is not None:
                connection.close()

        if row is None:
            raise MemoryStoreError("长期记忆保存后无法读取")

        return MemorySaveResult(
            memory=self._from_row(row),
            created=cursor.rowcount == 1,
        )

    def list_memories(self, max_results: int = 5) -> tuple[LongTermMemory, ...]:
        """按保存顺序读取有限条已授权记忆，供受限提示词注入。"""
        if (
            isinstance(max_results, bool)
            or not isinstance(max_results, int)
            or max_results <= 0
            or max_results > 20
        ):
            raise MemoryStoreError("max_results 必须是 1 到 20 的整数")

        self._state_store.initialize()
        connection: sqlite3.Connection | None = None

        try:
            connection = self._connect()
            rows = connection.execute(
                "SELECT memory_id, source_session_id, content, created_at "
                "FROM long_term_memories ORDER BY memory_id LIMIT ?",
                (max_results,),
            ).fetchall()
        except sqlite3.Error as error:
            raise MemoryStoreError("无法读取长期记忆") from error
        finally:
            if connection is not None:
                connection.close()

        return tuple(self._from_row(row) for row in rows)

    def search(self, query: object, max_results: int = 5) -> tuple[LongTermMemory, ...]:
        """按规范化子串检索已授权记忆，未命中时返回空元组。"""
        query_text = _require_text(query, "query")
        if (
            isinstance(max_results, bool)
            or not isinstance(max_results, int)
            or max_results <= 0
            or max_results > 20
        ):
            raise MemoryStoreError("max_results 必须是 1 到 20 的整数")

        self._state_store.initialize()
        normalized_query = _normalize_content(query_text)
        connection: sqlite3.Connection | None = None

        try:
            connection = self._connect()
            rows = connection.execute(
                "SELECT memory_id, source_session_id, content, created_at "
                "FROM long_term_memories WHERE normalized_content LIKE ? "
                "ORDER BY memory_id LIMIT ?",
                (f"%{normalized_query}%", max_results),
            ).fetchall()
        except sqlite3.Error as error:
            raise MemoryStoreError("无法检索长期记忆") from error
        finally:
            if connection is not None:
                connection.close()

        return tuple(self._from_row(row) for row in rows)
