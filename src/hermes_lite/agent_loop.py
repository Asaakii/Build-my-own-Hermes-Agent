"""HermesLite 的最小文本 Agent Loop。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from hermes_lite.domain import (
    Message,
    MessageRole,
    Session,
    TaskState,
    TaskStatus,
)
from hermes_lite.model_client import ModelClientError


DEFAULT_SYSTEM_PROMPT = (
    "你是 HermesLite，一个用于学习 Agent 开发的助手。"
    "当前只能进行文本对话，不能声称自己执行了工具或修改了文件。"
)


class ConversationModel(Protocol):
    """TextAgent 依赖的最小模型接口。"""

    def ask_messages(self, messages: Sequence[Message]) -> str:
        """根据完整会话消息返回一段文本回答。"""


@dataclass(frozen=True, slots=True)
class AgentTurn:
    """一次 Agent 运行的结构化结果。"""

    task: TaskState
    answer: str | None
    error_message: str | None

    def __post_init__(self) -> None:
        """确保任务状态与回答/错误信息彼此一致。"""
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


class TextAgent:
    """管理内存态会话和文本模型调用的最小 Agent。"""

    def __init__(
        self,
        model: ConversationModel,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        """保存模型接口和固定系统提示词。"""
        self._model = model
        self._system_message = Message(
            role=MessageRole.SYSTEM,
            content=system_prompt,
        )

    def run_turn(
        self,
        session: Session,
        user_request: str,
        task_id: str | None = None,
    ) -> AgentTurn:
        """执行一轮文本对话，并将可确认的消息写入会话。"""
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

        # 用户消息是真实输入，即使模型临时失败也应留在会话中。
        session.messages.append(user_message)

        try:
            answer = self._model.ask_messages(
                [self._system_message, *session.messages],
            )
        except ModelClientError as error:
            task.status = TaskStatus.FAILED
            return AgentTurn(
                task=task,
                answer=None,
                error_message=str(error),
            )

        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=answer,
        )
        session.messages.append(assistant_message)
        task.status = TaskStatus.COMPLETED

        return AgentTurn(
            task=task,
            answer=assistant_message.content,
            error_message=None,
        )