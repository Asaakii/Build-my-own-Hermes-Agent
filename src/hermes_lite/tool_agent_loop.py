"""HermesLite 的离线工具调用 Agent Loop。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from hermes_lite.coding_task import ToolRoundSummary
from hermes_lite.domain import (
    Message,
    MessageRole,
    Session,
    TaskState,
    TaskStatus,
    ToolResult,
)
from hermes_lite.model_client import ModelClientError
from hermes_lite.tool_registry import ToolRegistry


DEFAULT_TOOL_SYSTEM_PROMPT = (
    "你是 HermesLite。需要外部观察时，只能请求已提供的工具；"
    "收到工具结果后继续判断；完成后返回简洁文本回答。"
)


class ToolConversationModel(Protocol):
    """ToolAgent 依赖的模型接口。"""

    def respond(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]],
    ) -> Message:
        """根据会话和允许工具，返回一条助手消息。"""


@dataclass(frozen=True, slots=True)
class ToolAgentTurn:
    """一次可包含多轮工具调用的 Agent 运行结果。"""

    task: TaskState
    answer: str | None
    error_message: str | None
    tool_results: tuple[ToolResult, ...]
    round_summaries: tuple[ToolRoundSummary, ...]

    def __post_init__(self) -> None:
        """验证成功或失败状态与工具结果摘要一致。"""
        if not all(isinstance(result, ToolResult) for result in self.tool_results):
            raise ValueError("tool_results 中的元素必须是 ToolResult")

        if not isinstance(self.round_summaries, tuple):
            raise ValueError("round_summaries 必须是元组")

        if not all(
            isinstance(summary, ToolRoundSummary)
            for summary in self.round_summaries
        ):
            raise ValueError("round_summaries 中的元素必须是 ToolRoundSummary")

        summarized_results = tuple(
            result
            for summary in self.round_summaries
            for result in summary.results
        )

        if summarized_results != self.tool_results:
            raise ValueError("round_summaries 必须完整保留工具结果顺序")

        if self.task.status is TaskStatus.COMPLETED:
            if not isinstance(self.answer, str) or not self.answer:
                raise ValueError("已完成任务必须包含回答")

            if self.error_message is not None:
                raise ValueError("已完成任务不能包含错误信息")

        if self.task.status is TaskStatus.FAILED:
            if self.answer is not None:
                raise ValueError("失败任务不能包含回答")

            if not isinstance(self.error_message, str) or not self.error_message:
                raise ValueError("失败任务必须包含错误信息")


class ToolAgent:
    """协调模型决策、受控工具执行和会话观察结果。"""

    def __init__(
        self,
        model: ToolConversationModel,
        registry: ToolRegistry,
        max_tool_rounds: int = 3,
        system_prompt: str = DEFAULT_TOOL_SYSTEM_PROMPT,
    ) -> None:
        """保存模型、工具注册表和每次任务允许的最大工具轮数。"""
        if not isinstance(registry, ToolRegistry):
            raise ValueError("registry 必须是 ToolRegistry")

        if (
            isinstance(max_tool_rounds, bool)
            or not isinstance(max_tool_rounds, int)
            or max_tool_rounds <= 0
        ):
            raise ValueError("max_tool_rounds 必须是正整数")

        self._model = model
        self._registry = registry
        self._max_tool_rounds = max_tool_rounds
        self._system_message = Message(
            role=MessageRole.SYSTEM,
            content=system_prompt,
        )

    def _failed_turn(
        self,
        task: TaskState,
        tool_results: list[ToolResult],
        round_summaries: list[ToolRoundSummary],
        error_message: str,
    ) -> ToolAgentTurn:
        """把失败状态统一转换为结构化运行结果。"""
        task.status = TaskStatus.FAILED
        return ToolAgentTurn(
            task=task,
            answer=None,
            error_message=error_message,
            tool_results=tuple(tool_results),
            round_summaries=tuple(round_summaries),
        )

    def run_turn(
        self,
        session: Session,
        user_request: str,
        task_id: str | None = None,
    ) -> ToolAgentTurn:
        """执行一轮用户请求，以及其内部有限次工具循环。"""
        user_message = Message(
            role=MessageRole.USER,
            content=user_request,
        )
        task = TaskState(
            task_id=task_id or f"task-{uuid4().hex}",
            session_id=session.session_id,
            user_request=user_message.content,
            status=TaskStatus.RUNNING,
        )
        tool_results: list[ToolResult] = []
        round_summaries: list[ToolRoundSummary] = []

        # 用户真实输入先写入会话；之后即使模型失败也不丢失。
        session.messages.append(user_message)

        while True:
            try:
                response = self._model.respond(
                    [self._system_message, *session.messages],
                    self._registry.list_model_definitions(),
                )
            except ModelClientError as error:
                return self._failed_turn(
                    task,
                    tool_results,
                    round_summaries,
                    str(error),
                )

            if not isinstance(response, Message):
                return self._failed_turn(
                    task,
                    tool_results,
                    round_summaries,
                    "模型返回了无效的 Agent 消息",
                )

            if response.role is not MessageRole.ASSISTANT:
                return self._failed_turn(
                    task,
                    tool_results,
                    round_summaries,
                    "模型返回的 Agent 消息必须是 assistant 角色",
                )

            session.messages.append(response)

            if not response.tool_calls:
                assert isinstance(response.content, str)
                task.status = TaskStatus.COMPLETED
                return ToolAgentTurn(
                    task=task,
                    answer=response.content,
                    error_message=None,
                    tool_results=tuple(tool_results),
                    round_summaries=tuple(round_summaries),
                )

            if task.tool_rounds >= self._max_tool_rounds:
                return self._failed_turn(
                    task,
                    tool_results,
                    round_summaries,
                    "工具调用次数超过上限",
                )

            task.tool_rounds += 1
            current_round_results: list[ToolResult] = []

            for call in response.tool_calls:
                result = self._registry.execute(call)
                tool_results.append(result)
                current_round_results.append(result)
                session.messages.append(Message.from_tool_result(result))

            round_summaries.append(
                ToolRoundSummary(
                    round_number=task.tool_rounds,
                    results=tuple(current_round_results),
                )
            )
