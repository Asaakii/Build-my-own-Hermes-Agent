"""HermesLite 的离线编码任务协调层。"""

from __future__ import annotations

from hermes_lite.coding_task import CodingTaskReport, VerificationStatus
from hermes_lite.domain import Session, TaskStatus, ToolResult
from hermes_lite.tool_agent_loop import ToolAgent, ToolAgentTurn


class CodingAgent:
    """根据工具循环的真实观察结果生成编码任务报告。"""

    def __init__(self, tool_agent: ToolAgent) -> None:
        """保存已经配置好受控工具的通用工具 Agent。"""
        if not isinstance(tool_agent, ToolAgent):
            raise ValueError("tool_agent 必须是 ToolAgent")

        self._tool_agent = tool_agent

    @staticmethod
    def _verification_from_results(
        tool_results: tuple[ToolResult, ...],
    ) -> VerificationStatus:
        """根据最后一次实际 Pytest 观察结果判断验证状态。"""
        for result in reversed(tool_results):
            if result.tool_name != "run_pytest":
                continue

            if result.is_error:
                return VerificationStatus.NOT_RUN

            if "测试结束，退出码: 0" in result.content:
                return VerificationStatus.PASSED

            if (
                "测试结束，退出码:" in result.content
                or "测试超时：" in result.content
            ):
                return VerificationStatus.FAILED

            return VerificationStatus.NOT_RUN

        return VerificationStatus.NOT_RUN

    @staticmethod
    def _report_status(
        turn: ToolAgentTurn,
        verification: VerificationStatus,
    ) -> TaskStatus:
        """将工具循环结果映射为编码任务的最终状态。"""
        if (
            turn.task.status is TaskStatus.COMPLETED
            and verification is VerificationStatus.PASSED
        ):
            return TaskStatus.COMPLETED

        if verification is VerificationStatus.NOT_RUN:
            return TaskStatus.BLOCKED

        return TaskStatus.FAILED

    @staticmethod
    def _summary_from_turn(turn: ToolAgentTurn) -> str:
        """优先使用模型最终文本；没有时给出可复现的失败摘要。"""
        if turn.answer is not None:
            return turn.answer

        if turn.error_message is not None:
            return f"编码任务未完成：{turn.error_message}"

        return "编码任务未完成，未获得最终回答。"

    def run_task(
        self,
        session: Session,
        user_request: str,
        task_id: str | None = None,
    ) -> CodingTaskReport:
        """运行一次受控工具循环，并以真实测试结果生成最终报告。"""
        turn = self._tool_agent.run_turn(
            session,
            user_request,
            task_id=task_id,
        )
        verification = self._verification_from_results(turn.tool_results)

        return CodingTaskReport(
            task_id=turn.task.task_id,
            status=self._report_status(turn, verification),
            verification=verification,
            summary=self._summary_from_turn(turn),
            rounds=turn.round_summaries,
        )
