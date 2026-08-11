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
from hermes_lite.memory_store import (
    MemoryStoreError,
    SQLiteMemoryStore,
    parse_remember_command,
)
from hermes_lite.model_client import ModelClientError
from hermes_lite.prompt_builder import (
    HistorySummarizer,
    PromptBuilder,
    PromptBuilderError,
)


DEFAULT_SYSTEM_PROMPT = (
    "你是 HermesLite，一个用于学习 Agent 开发的助手。"
    "当前只能进行文本对话，不能声称自己执行了工具或修改了文件。"
)

DEFAULT_TEXT_WORKSPACE_RULES = (
    "当前文本模式不提供工作区访问权限，也不能声称执行了文件操作。"
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
        prompt_builder: PromptBuilder | None = None,
        history_summarizer: HistorySummarizer | None = None,
        memory_store: SQLiteMemoryStore | None = None,
    ) -> None:
        """保存模型与可替换的上下文构建依赖。"""
        if prompt_builder is not None and not isinstance(
            prompt_builder,
            PromptBuilder,
        ):
            raise ValueError("prompt_builder 必须是 PromptBuilder 或 None")

        if memory_store is not None and not isinstance(
            memory_store,
            SQLiteMemoryStore,
        ):
            raise ValueError("memory_store 必须是 SQLiteMemoryStore 或 None")

        self._model = model
        self._system_message = Message(
            role=MessageRole.SYSTEM,
            content=system_prompt,
        )
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._history_summarizer = history_summarizer
        self._memory_store = memory_store

    def _long_term_memory_contents(self) -> tuple[str, ...]:
        """读取少量已授权记忆，供本轮可信上下文注入。"""
        if self._memory_store is None:
            return ()

        return tuple(
            memory.content
            for memory in self._memory_store.list_memories()
        )

    def _build_model_messages(self, session: Session) -> tuple[Message, ...]:
        """通过统一 Builder 构建本轮可发送给模型的上下文。"""
        return self._prompt_builder.build(
            safety_rules=self._system_message.content,
            workspace_rules=DEFAULT_TEXT_WORKSPACE_RULES,
            long_term_memories=self._long_term_memory_contents(),
            history=session.messages,
            summarizer=self._history_summarizer,
        ).messages

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
            remember_request = parse_remember_command(
                session.session_id,
                user_message.content,
            )
        except MemoryStoreError as error:
            task.status = TaskStatus.FAILED
            return AgentTurn(task=task, answer=None, error_message=str(error))

        if remember_request is not None:
            if self._memory_store is None:
                task.status = TaskStatus.FAILED
                return AgentTurn(
                    task=task,
                    answer=None,
                    error_message="长期记忆存储尚未配置",
                )

            try:
                save_result = self._memory_store.save_authorized(remember_request)
            except MemoryStoreError as error:
                task.status = TaskStatus.FAILED
                return AgentTurn(task=task, answer=None, error_message=str(error))

            answer = (
                "长期记忆已保存。"
                if save_result.created
                else "长期记忆已存在。"
            )
            session.messages.append(
                Message(role=MessageRole.ASSISTANT, content=answer),
            )
            task.status = TaskStatus.COMPLETED
            return AgentTurn(task=task, answer=answer, error_message=None)

        try:
            answer = self._model.ask_messages(
                self._build_model_messages(session),
            )
        except (ModelClientError, PromptBuilderError, MemoryStoreError) as error:
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