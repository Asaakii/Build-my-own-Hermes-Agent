"""验证编码任务摘要与最终报告的边界。"""

import pytest

from hermes_lite.coding_task import (
    CodingTaskReport,
    ToolRoundSummary,
    VerificationStatus,
)
from hermes_lite.domain import TaskStatus, ToolResult


def make_result() -> ToolResult:
    """创建一条最小工具观察结果。"""
    return ToolResult(
        call_id="call-1",
        tool_name="run_pytest",
        content="测试结束，退出码: 0",
    )


def test_completed_report_requires_passed_verification() -> None:
    """完成的编码任务必须有通过的测试验证。"""
    report = CodingTaskReport(
        task_id="coding-task-1",
        status=TaskStatus.COMPLETED,
        verification=VerificationStatus.PASSED,
        summary="已修复练习项目中的失败测试。",
        rounds=(ToolRoundSummary(round_number=1, results=(make_result(),)),),
    )

    assert report.verification is VerificationStatus.PASSED
    assert report.rounds[0].results[0].tool_name == "run_pytest"


@pytest.mark.parametrize(
    ("status", "verification"),
    [
        (TaskStatus.COMPLETED, VerificationStatus.FAILED),
        (TaskStatus.FAILED, VerificationStatus.PASSED),
        (TaskStatus.BLOCKED, VerificationStatus.PASSED),
    ],
)
def test_report_rejects_inconsistent_final_status(
    status: TaskStatus,
    verification: VerificationStatus,
) -> None:
    """最终任务状态不能与测试验证状态矛盾。"""
    with pytest.raises(ValueError):
        CodingTaskReport(
            task_id="coding-task-1",
            status=status,
            verification=verification,
            summary="状态不一致。",
            rounds=(),
        )


@pytest.mark.parametrize("status", [TaskStatus.PENDING, TaskStatus.RUNNING])
def test_report_rejects_unfinished_task_status(status: TaskStatus) -> None:
    """最终报告不能记录尚未结束的任务。"""
    with pytest.raises(ValueError, match="最终状态"):
        CodingTaskReport(
            task_id="coding-task-1",
            status=status,
            verification=VerificationStatus.NOT_RUN,
            summary="任务仍在执行。",
            rounds=(),
        )


def test_round_rejects_empty_results() -> None:
    """工具轮次必须确实包含至少一条工具观察结果。"""
    with pytest.raises(ValueError, match="results 不能为空"):
        ToolRoundSummary(round_number=1, results=())


def test_round_rejects_invalid_number() -> None:
    """工具轮次编号必须从一开始的正整数。"""
    with pytest.raises(ValueError, match="round_number"):
        ToolRoundSummary(round_number=0, results=(make_result(),))
