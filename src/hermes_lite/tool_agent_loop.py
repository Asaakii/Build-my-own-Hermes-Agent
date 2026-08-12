"""HermesLite 的离线工具调用 Agent Loop。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging
from typing import Protocol
from uuid import uuid4

from hermes_lite.audit_log import (
    AuditEvent,
    AuditEventRecorder,
    AuditEventType,
    InMemoryAuditLog,
)
from hermes_lite.coding_task import ToolRoundSummary
from hermes_lite.confirmation_policy import (
    ConfirmationManager,
    ConfirmationPolicyError,
    PendingConfirmation,
    parse_confirmation_command,
    requires_confirmation,
)
from hermes_lite.domain import (
    Message,
    MessageRole,
    Session,
    TaskState,
    ToolCall,
    TaskStatus,
    ToolResult,
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
from hermes_lite.runtime_log import (
    RuntimeEvent,
    emit_runtime_event,
    task_log_context,
)
from hermes_lite.skill_loader import Skill, SkillLoadError, load_skill
from hermes_lite.tool_registry import ToolRegistry, ToolRegistryError


DEFAULT_TOOL_SYSTEM_PROMPT = (
    "你是 HermesLite。需要外部观察时，只能请求已提供的工具；"
    "收到工具结果后继续判断；完成后返回简洁文本回答。高风险工具必须等待程序收到用户确认。"
)

DEFAULT_TOOL_WORKSPACE_RULES = (
    "工具只能访问已配置的受限工作区，不能绕过注册表或路径边界。"
)

logger = logging.getLogger(__name__)


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
    pending_confirmation: PendingConfirmation | None = None

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

        if (
            self.pending_confirmation is not None
            and not isinstance(self.pending_confirmation, PendingConfirmation)
        ):
            raise ValueError("pending_confirmation 必须是 PendingConfirmation 或 None")

        if self.task.status is TaskStatus.COMPLETED:
            if not isinstance(self.answer, str) or not self.answer:
                raise ValueError("已完成任务必须包含回答")

            if self.error_message is not None:
                raise ValueError("已完成任务不能包含错误信息")

            if self.pending_confirmation is not None:
                raise ValueError("已完成任务不能保留待确认操作")

        if self.task.status is TaskStatus.FAILED:
            if self.answer is not None:
                raise ValueError("失败任务不能包含回答")

            if not isinstance(self.error_message, str) or not self.error_message:
                raise ValueError("失败任务必须包含错误信息")

            if self.pending_confirmation is not None:
                raise ValueError("失败任务不能保留待确认操作")

        if self.task.status is TaskStatus.BLOCKED:
            if self.answer is not None:
                raise ValueError("受阻任务不能包含回答")

            if not isinstance(self.error_message, str) or not self.error_message:
                raise ValueError("受阻任务必须包含说明")

            if self.pending_confirmation is None:
                raise ValueError("受阻任务必须包含待确认操作")


class ToolAgent:
    """协调模型决策、受控工具执行和会话观察结果。"""

    def __init__(
        self,
        model: ToolConversationModel,
        registry: ToolRegistry,
        max_tool_rounds: int = 3,
        system_prompt: str = DEFAULT_TOOL_SYSTEM_PROMPT,
        prompt_builder: PromptBuilder | None = None,
        history_summarizer: HistorySummarizer | None = None,
        memory_store: SQLiteMemoryStore | None = None,
        confirmation_manager: ConfirmationManager | None = None,
        audit_recorder: AuditEventRecorder | None = None,
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

        if confirmation_manager is not None and not isinstance(
            confirmation_manager,
            ConfirmationManager,
        ):
            raise ValueError(
                "confirmation_manager 必须是 ConfirmationManager 或 None"
            )

        if audit_recorder is not None and not isinstance(
            audit_recorder,
            AuditEventRecorder,
        ):
            raise ValueError("audit_recorder 必须实现 AuditEventRecorder")

        self._model = model
        self._registry = registry
        self._max_tool_rounds = max_tool_rounds
        self._system_message = Message(
            role=MessageRole.SYSTEM,
            content=system_prompt,
        )
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._history_summarizer = history_summarizer
        self._memory_store = memory_store
        self._confirmation_manager = confirmation_manager or ConfirmationManager()
        self._audit_recorder = audit_recorder or InMemoryAuditLog()

    def _record_tool_event(
        self,
        task: TaskState,
        event_type: AuditEventType,
        tool_call: ToolCall,
        reason_code: str | None = None,
    ) -> str | None:
        """写入一条脱敏工具事件；失败时阻止后续敏感操作。"""
        try:
            self._audit_recorder.record_audit_event(
                AuditEvent.for_tool_call(
                    session_id=task.session_id,
                    task_id=task.task_id,
                    event_type=event_type,
                    tool_call=tool_call,
                    reason_code=reason_code,
                )
            )
        except Exception:
            return "审计事件记录失败，已停止工具操作"

        return None

    def _record_execution_outcome(
        self,
        task: TaskState,
        tool_call: ToolCall,
        result: ToolResult,
    ) -> str | None:
        """按标准结果文本区分注册表拒绝、工具失败和成功执行。"""
        if result.content.startswith("工具调用被拒绝:"):
            event_type = AuditEventType.TOOL_REJECTED
            reason_code = "rejected_by_registry"
        elif result.is_error:
            event_type = AuditEventType.TOOL_EXECUTED
            reason_code = "tool_error"
        else:
            event_type = AuditEventType.TOOL_EXECUTED
            reason_code = "success"

        return self._record_tool_event(
            task,
            event_type,
            tool_call,
            reason_code,
        )

    def _record_confirmation_rejection(
        self,
        task: TaskState,
        reason_code: str,
    ) -> str | None:
        """记录确认被拒绝的事实，不保存用户提交的令牌。"""
        try:
            self._audit_recorder.record_audit_event(
                AuditEvent(
                    session_id=task.session_id,
                    task_id=task.task_id,
                    event_type=AuditEventType.CONFIRMATION_REJECTED,
                    reason_code=reason_code,
                )
            )
        except Exception:
            return "审计事件记录失败，已停止工具操作"

        return None

    def _long_term_memory_contents(self) -> tuple[str, ...]:
        """读取少量已授权记忆，供本轮可信上下文注入。"""
        if self._memory_store is None:
            return ()

        return tuple(
            memory.content
            for memory in self._memory_store.list_memories()
        )

    def _build_model_messages(
        self,
        session: Session,
        tool_definitions: Sequence[dict[str, object]],
        selected_skill: Skill | None,
    ) -> tuple[Message, ...]:
        """构建含受限工具、显式技能和历史的模型上下文。"""
        skills = ()
        if selected_skill is not None:
            skills = (
                f"技能名称：{selected_skill.name}\n"
                f"技能说明：{selected_skill.description}\n"
                f"技能步骤：\n{selected_skill.instructions}",
            )

        return self._prompt_builder.build(
            safety_rules=self._system_message.content,
            workspace_rules=DEFAULT_TOOL_WORKSPACE_RULES,
            tool_definitions=tool_definitions,
            skills=skills,
            long_term_memories=self._long_term_memory_contents(),
            history=session.messages,
            summarizer=self._history_summarizer,
        ).messages

    def _failed_turn(
        self,
        task: TaskState,
        tool_results: list[ToolResult],
        round_summaries: list[ToolRoundSummary],
        error_message: str,
    ) -> ToolAgentTurn:
        """把失败状态统一转换为结构化运行结果。"""
        task.status = TaskStatus.FAILED
        emit_runtime_event(
            logger,
            RuntimeEvent.TASK_FAILED,
            task_id=task.task_id,
            task_status=task.status.value,
            level=logging.WARNING,
        )
        return ToolAgentTurn(
            task=task,
            answer=None,
            error_message=error_message,
            tool_results=tuple(tool_results),
            round_summaries=tuple(round_summaries),
        )

    def _blocked_turn(
        self,
        task: TaskState,
        tool_results: list[ToolResult],
        round_summaries: list[ToolRoundSummary],
        pending_confirmation: PendingConfirmation,
    ) -> ToolAgentTurn:
        """将等待用户确认的调用保存为可恢复的受阻任务。"""
        task.status = TaskStatus.BLOCKED
        emit_runtime_event(
            logger,
            RuntimeEvent.TASK_BLOCKED,
            task_id=task.task_id,
            task_status=task.status.value,
            level=logging.WARNING,
        )
        return ToolAgentTurn(
            task=task,
            answer=None,
            error_message=(
                f"工具调用等待确认: {pending_confirmation.tool_call.tool_name}"
            ),
            tool_results=tuple(tool_results),
            round_summaries=tuple(round_summaries),
            pending_confirmation=pending_confirmation,
        )

    def _requires_confirmation_for_call(self, call: ToolCall) -> bool:
        """只为已登记的高风险工具签发确认，未知工具仍交给注册表拒绝。"""
        try:
            definition = self._registry.get(call.tool_name)
        except ToolRegistryError:
            return False

        return requires_confirmation(definition.risk_level)

    def _confirm_pending_call(
        self,
        session: Session,
        task: TaskState,
        confirmation_token: str,
    ) -> ToolAgentTurn:
        """消费确认令牌并执行其保存的原始调用，不请求模型。"""
        tool_results: list[ToolResult] = []
        round_summaries: list[ToolRoundSummary] = []

        try:
            pending = self._confirmation_manager.consume_for_session(
                confirmation_token,
                session.session_id,
            )
        except ConfirmationPolicyError as error:
            audit_error = self._record_confirmation_rejection(
                task,
                "invalid_or_unusable_token",
            )
            return self._failed_turn(
                task,
                tool_results,
                round_summaries,
                audit_error or str(error),
            )

        audit_error = self._record_tool_event(
            task,
            AuditEventType.CONFIRMATION_ACCEPTED,
            pending.tool_call,
            "user_confirmed",
        )
        if audit_error is not None:
            return self._failed_turn(
                task,
                tool_results,
                round_summaries,
                audit_error,
            )

        task.tool_rounds = 1
        result = self._registry.execute(pending.tool_call)
        emit_runtime_event(
            logger,
            RuntimeEvent.TOOL_FINISHED,
            task_id=task.task_id,
            tool_name=result.tool_name,
            round_number=task.tool_rounds,
        )
        tool_results.append(result)
        round_summaries.append(
            ToolRoundSummary(round_number=task.tool_rounds, results=(result,))
        )
        session.messages.append(Message.from_tool_result(result))

        audit_error = self._record_execution_outcome(
            task,
            pending.tool_call,
            result,
        )
        if audit_error is not None:
            return self._failed_turn(
                task,
                tool_results,
                round_summaries,
                audit_error,
            )

        if result.is_error:
            return self._failed_turn(
                task,
                tool_results,
                round_summaries,
                f"确认后工具执行失败: {result.content}",
            )

        answer = f"已确认并执行工具: {result.tool_name}\n{result.content}"
        session.messages.append(Message(role=MessageRole.ASSISTANT, content=answer))
        task.status = TaskStatus.COMPLETED
        emit_runtime_event(
            logger,
            RuntimeEvent.TASK_COMPLETED,
            task_id=task.task_id,
            task_status=task.status.value,
        )
        return ToolAgentTurn(
            task=task,
            answer=answer,
            error_message=None,
            tool_results=tuple(tool_results),
            round_summaries=tuple(round_summaries),
        )

    def run_turn(
        self,
        session: Session,
        user_request: str,
        task_id: str | None = None,
        skill_name: object | None = None,
    ) -> ToolAgentTurn:
        """执行一轮用户请求，以及其内部有限次工具循环。

        只有调用方显式提供 skill_name 时才加载技能；默认仍可使用完整注册表。
        """
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

        emit_runtime_event(
            logger,
            RuntimeEvent.TASK_STARTED,
            task_id=task.task_id,
            task_status=task.status.value,
        )

        # 用户真实输入先写入会话；之后即使模型失败也不丢失。
        session.messages.append(user_message)

        try:
            confirmation_token = parse_confirmation_command(
                user_message.content,
            )
        except ConfirmationPolicyError as error:
            audit_error = self._record_confirmation_rejection(
                task,
                "malformed_command",
            )
            return self._failed_turn(
                task,
                tool_results,
                round_summaries,
                audit_error or str(error),
            )

        if confirmation_token is not None:
            return self._confirm_pending_call(
                session,
                task,
                confirmation_token,
            )

        try:
            selected_skill = (
                load_skill(skill_name)
                if skill_name is not None
                else None
            )
            tool_definitions = (
                selected_skill.allowed_tool_definitions(self._registry)
                if selected_skill is not None
                else self._registry.list_model_definitions()
            )
        except SkillLoadError as error:
            return self._failed_turn(
                task,
                tool_results,
                round_summaries,
                str(error),
            )

        try:
            remember_request = parse_remember_command(
                session.session_id,
                user_message.content,
            )
        except MemoryStoreError as error:
            return self._failed_turn(
                task,
                tool_results,
                round_summaries,
                str(error),
            )

        if remember_request is not None:
            if self._memory_store is None:
                return self._failed_turn(
                    task,
                    tool_results,
                    round_summaries,
                    "长期记忆存储尚未配置",
                )

            try:
                save_result = self._memory_store.save_authorized(remember_request)
            except MemoryStoreError as error:
                return self._failed_turn(
                    task,
                    tool_results,
                    round_summaries,
                    str(error),
                )

            answer = (
                "长期记忆已保存。"
                if save_result.created
                else "长期记忆已存在。"
            )
            session.messages.append(
                Message(role=MessageRole.ASSISTANT, content=answer),
            )
            task.status = TaskStatus.COMPLETED
            emit_runtime_event(
                logger,
                RuntimeEvent.TASK_COMPLETED,
                task_id=task.task_id,
                task_status=task.status.value,
            )
            return ToolAgentTurn(
                task=task,
                answer=answer,
                error_message=None,
                tool_results=(),
                round_summaries=(),
            )

        while True:
            try:
                with task_log_context(task.task_id):
                    response = self._model.respond(
                        self._build_model_messages(
                            session,
                            tool_definitions,
                            selected_skill,
                        ),
                        tool_definitions,
                    )
            except (ModelClientError, PromptBuilderError, MemoryStoreError) as error:
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
                emit_runtime_event(
                    logger,
                    RuntimeEvent.TASK_COMPLETED,
                    task_id=task.task_id,
                    task_status=task.status.value,
                )
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
                audit_error = self._record_tool_event(
                    task,
                    AuditEventType.TOOL_REQUESTED,
                    call,
                )
                if audit_error is not None:
                    return self._failed_turn(
                        task,
                        tool_results,
                        round_summaries,
                        audit_error,
                    )

                emit_runtime_event(
                    logger,
                    RuntimeEvent.TOOL_REQUESTED,
                    task_id=task.task_id,
                    tool_name=call.tool_name,
                    round_number=task.tool_rounds,
                )

                if (
                    selected_skill is not None
                    and call.tool_name not in selected_skill.allowed_tools
                ):
                    # 即使模型伪造了未暴露工具，也不把它交给注册表执行。
                    result = ToolResult(
                        call_id=call.call_id,
                        tool_name=call.tool_name,
                        content=(
                            "工具调用被拒绝：当前技能不允许工具 "
                            f"{call.tool_name}"
                        ),
                        is_error=True,
                    )
                    audit_error = self._record_tool_event(
                        task,
                        AuditEventType.TOOL_REJECTED,
                        call,
                        "not_allowed_by_skill",
                    )
                    if audit_error is not None:
                        return self._failed_turn(
                            task,
                            tool_results,
                            round_summaries,
                            audit_error,
                        )
                elif self._requires_confirmation_for_call(call):
                    try:
                        pending = self._confirmation_manager.issue(
                            session.session_id,
                            call,
                        )
                    except ConfirmationPolicyError as error:
                        return self._failed_turn(
                            task,
                            tool_results,
                            round_summaries,
                            str(error),
                        )

                    audit_error = self._record_tool_event(
                        task,
                        AuditEventType.CONFIRMATION_REQUIRED,
                        call,
                        "high_risk",
                    )
                    if audit_error is not None:
                        return self._failed_turn(
                            task,
                            tool_results,
                            round_summaries,
                            audit_error,
                        )

                    result = ToolResult(
                        call_id=call.call_id,
                        tool_name=call.tool_name,
                        content=(
                            f"工具调用等待确认: {call.tool_name}。"
                            f"请发送 /confirm {pending.token} 后执行。"
                        ),
                        is_error=True,
                    )
                    tool_results.append(result)
                    current_round_results.append(result)
                    session.messages.append(Message.from_tool_result(result))
                    round_summaries.append(
                        ToolRoundSummary(
                            round_number=task.tool_rounds,
                            results=tuple(current_round_results),
                        )
                    )
                    return self._blocked_turn(
                        task,
                        tool_results,
                        round_summaries,
                        pending,
                    )
                else:
                    result = self._registry.execute(call)
                    audit_error = self._record_execution_outcome(
                        task,
                        call,
                        result,
                    )
                    if audit_error is not None:
                        emit_runtime_event(
                            logger,
                            RuntimeEvent.TOOL_FINISHED,
                            task_id=task.task_id,
                            tool_name=result.tool_name,
                            round_number=task.tool_rounds,
                        )
                        tool_results.append(result)
                        current_round_results.append(result)
                        session.messages.append(Message.from_tool_result(result))
                        round_summaries.append(
                            ToolRoundSummary(
                                round_number=task.tool_rounds,
                                results=tuple(current_round_results),
                            )
                        )
                        return self._failed_turn(
                            task,
                            tool_results,
                            round_summaries,
                            audit_error,
                        )

                emit_runtime_event(
                    logger,
                    RuntimeEvent.TOOL_FINISHED,
                    task_id=task.task_id,
                    tool_name=result.tool_name,
                    round_number=task.tool_rounds,
                )
                tool_results.append(result)
                current_round_results.append(result)
                session.messages.append(Message.from_tool_result(result))

            round_summaries.append(
                ToolRoundSummary(
                    round_number=task.tool_rounds,
                    results=tuple(current_round_results),
                )
            )
