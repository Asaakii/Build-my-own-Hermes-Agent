"""HermesLite 编码任务的结构化摘要与最终报告。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from hermes_lite.domain import TaskStatus, ToolResult


class VerificationStatus(str, Enum):
    """编码任务的测试验证状态。"""

    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"


def _require_text(value: object, field_name: str) -> str:
    """验证必填文本并去除首尾空白。"""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} 必须是文本")

    cleaned_value = value.strip()

    if not cleaned_value:
        raise ValueError(f"{field_name} 不能为空")

    return cleaned_value


@dataclass(frozen=True, slots=True)
class ToolRoundSummary:
    """一轮工具调用产生的受控观察结果。"""

    round_number: int
    results: tuple[ToolResult, ...]

    def __post_init__(self) -> None:
        """验证轮次编号和工具结果集合。"""
        if (
            isinstance(self.round_number, bool)
            or not isinstance(self.round_number, int)
            or self.round_number <= 0
        ):
            raise ValueError("round_number 必须是正整数")

        if not isinstance(self.results, tuple):
            raise ValueError("results 必须是元组")

        if not self.results:
            raise ValueError("results 不能为空")

        if not all(isinstance(result, ToolResult) for result in self.results):
            raise ValueError("results 中的元素必须是 ToolResult")


@dataclass(frozen=True, slots=True)
class CodingTaskReport:
    """一次编码任务结束后可展示、可测试的最终报告。"""

    task_id: str
    status: TaskStatus
    verification: VerificationStatus
    summary: str
    rounds: tuple[ToolRoundSummary, ...]

    def __post_init__(self) -> None:
        """保证最终任务状态、验证状态和摘要相互一致。"""
        object.__setattr__(self, "task_id", _require_text(self.task_id, "task_id"))
        object.__setattr__(self, "summary", _require_text(self.summary, "summary"))

        if not isinstance(self.status, TaskStatus):
            raise ValueError("status 必须是 TaskStatus")

        if not isinstance(self.verification, VerificationStatus):
            raise ValueError("verification 必须是 VerificationStatus")

        if not isinstance(self.rounds, tuple):
            raise ValueError("rounds 必须是元组")

        if not all(isinstance(round_, ToolRoundSummary) for round_ in self.rounds):
            raise ValueError("rounds 中的元素必须是 ToolRoundSummary")

        if self.status not in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
        }:
            raise ValueError("编码任务报告只能记录最终状态")

        if (
            self.status is TaskStatus.COMPLETED
            and self.verification is not VerificationStatus.PASSED
        ):
            raise ValueError("已完成任务必须通过测试验证")
