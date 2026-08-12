"""HermesLite 的可审查计划任务与复盘候选服务。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import sqlite3
from uuid import uuid4

from hermes_lite.memory_store import RememberRequest, SQLiteMemoryStore
from hermes_lite.sqlite_state_store import SQLiteStateStore, SQLiteStateStoreError


class SchedulerError(ValueError):
    """计划任务或复盘候选不符合受控状态规则。"""


@dataclass(frozen=True, slots=True)
class ScheduledTask:
    """一条只会投递提醒文本的本地计划任务。"""

    task_id: str
    session_id: str
    message: str
    due_at: str
    status: str


@dataclass(frozen=True, slots=True)
class ReviewCandidate:
    """尚未写入长期记忆或技能的候选复盘项。"""

    candidate_id: str
    source_task_id: str
    kind: str
    content: str
    status: str


def _text(value: object, field_name: str, max_length: int = 1000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchedulerError(f"{field_name} 必须是非空文本")
    result = value.strip()
    if len(result) > max_length:
        raise SchedulerError(f"{field_name} 过长")
    return result


class SchedulerStore:
    """复用项目 SQLite 数据库保存任务与待人工审批的候选项。"""

    def __init__(self, state_store: SQLiteStateStore) -> None:
        if not isinstance(state_store, SQLiteStateStore):
            raise SchedulerError("state_store 必须是 SQLiteStateStore")
        self._state_store = state_store
        self._state_store.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._state_store.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def create(self, session_id: object, message: object, delay_seconds: object) -> ScheduledTask:
        session = _text(session_id, "session_id", 80)
        content = _text(message, "message", 500)
        if isinstance(delay_seconds, bool) or not isinstance(delay_seconds, int):
            raise SchedulerError("delay_seconds 必须是正整数")
        if delay_seconds <= 0 or delay_seconds > 7 * 24 * 60 * 60:
            raise SchedulerError("delay_seconds 必须在 1 到 604800 之间")
        task = ScheduledTask(
            task_id=f"schedule-{uuid4().hex}",
            session_id=session,
            message=content,
            due_at=(datetime.now(UTC) + timedelta(seconds=delay_seconds)).isoformat(),
            status="pending",
        )
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO scheduled_tasks (task_id, session_id, message, due_at, status) VALUES (?, ?, ?, ?, ?)",
                    (task.task_id, task.session_id, task.message, task.due_at, task.status),
                )
        except sqlite3.Error as error:
            raise SchedulerError("无法保存计划任务") from error
        return task

    def list(self) -> tuple[ScheduledTask, ...]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT task_id, session_id, message, due_at, status FROM scheduled_tasks ORDER BY due_at, task_id"
                ).fetchall()
        except sqlite3.Error as error:
            raise SchedulerError("无法读取计划任务") from error
        return tuple(ScheduledTask(*tuple(row)) for row in rows)

    def cancel(self, task_id: object) -> bool:
        normalized_id = _text(task_id, "task_id", 100)
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "UPDATE scheduled_tasks SET status = 'cancelled' WHERE task_id = ? AND status = 'pending'",
                    (normalized_id,),
                )
        except sqlite3.Error as error:
            raise SchedulerError("无法取消计划任务") from error
        return cursor.rowcount == 1

    def run_due(self, deliver: Callable[[ScheduledTask], None]) -> tuple[ScheduledTask, ...]:
        """投递到期任务，并只创建候选复盘项，绝不自动写长期记忆。"""
        if not callable(deliver):
            raise SchedulerError("deliver 必须是可调用对象")
        now = datetime.now(UTC).isoformat()
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT task_id, session_id, message, due_at, status FROM scheduled_tasks WHERE status = 'pending' AND due_at <= ? ORDER BY due_at, task_id",
                    (now,),
                ).fetchall()
        except sqlite3.Error as error:
            raise SchedulerError("无法读取到期任务") from error

        delivered: list[ScheduledTask] = []
        for row in rows:
            task = ScheduledTask(*tuple(row))
            deliver(task)
            candidate = ReviewCandidate(
                candidate_id=f"candidate-{uuid4().hex}",
                source_task_id=task.task_id,
                kind="memory",
                content=f"计划任务已投递：{task.message}",
                status="pending",
            )
            try:
                with self._connect() as connection:
                    connection.execute("UPDATE scheduled_tasks SET status = 'delivered' WHERE task_id = ?", (task.task_id,))
                    connection.execute(
                        "INSERT INTO review_candidates (candidate_id, source_task_id, kind, content, status) VALUES (?, ?, ?, ?, ?)",
                        (candidate.candidate_id, candidate.source_task_id, candidate.kind, candidate.content, candidate.status),
                    )
            except sqlite3.Error as error:
                raise SchedulerError("无法保存投递结果") from error
            delivered.append(task)
        return tuple(delivered)

    def list_candidates(self) -> tuple[ReviewCandidate, ...]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT candidate_id, source_task_id, kind, content, status FROM review_candidates ORDER BY candidate_id"
                ).fetchall()
        except sqlite3.Error as error:
            raise SchedulerError("无法读取复盘候选") from error
        return tuple(ReviewCandidate(*tuple(row)) for row in rows)

    def approve_memory_candidate(self, candidate_id: object) -> bool:
        """用户显式审批后才将候选保存为长期记忆。"""
        normalized_id = _text(candidate_id, "candidate_id", 100)
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT candidate.source_task_id, candidate.kind, candidate.content, candidate.status, task.session_id FROM review_candidates AS candidate JOIN scheduled_tasks AS task ON task.task_id = candidate.source_task_id WHERE candidate.candidate_id = ?",
                    (normalized_id,),
                ).fetchone()
        except sqlite3.Error as error:
            raise SchedulerError("无法读取复盘候选") from error
        if row is None or row[3] != "pending" or row[1] != "memory":
            return False
        try:
            SQLiteMemoryStore(self._state_store).save_authorized(
                RememberRequest(source_session_id=row[4], content=row[2]),
            )
            with self._connect() as connection:
                connection.execute("UPDATE review_candidates SET status = 'approved' WHERE candidate_id = ?", (normalized_id,))
        except (SQLiteStateStoreError, sqlite3.Error) as error:
            raise SchedulerError("无法审批复盘候选") from error
        return True
