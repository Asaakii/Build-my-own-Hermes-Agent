"""HermesLite 的受控提示词构建与上下文预算。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Protocol

from hermes_lite.domain import Message, MessageRole


class PromptBuilderError(ValueError):
    """表示提示词输入、预算或摘要结果不符合约束。"""


def _require_text(value: object, field_name: str) -> str:
    """验证必填文本，并返回去除首尾空白后的值。"""
    if not isinstance(value, str):
        raise PromptBuilderError(f"{field_name} 必须是文本")

    cleaned_value = value.strip()
    if not cleaned_value:
        raise PromptBuilderError(f"{field_name} 不能为空")

    return cleaned_value


def _require_text_items(
    values: Sequence[object],
    field_name: str,
) -> tuple[str, ...]:
    """验证文本片段序列，避免空片段悄悄进入系统提示词。"""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PromptBuilderError(f"{field_name} 必须是文本序列")

    return tuple(
        _require_text(value, f"{field_name}[{index}]")
        for index, value in enumerate(values)
    )


class HistorySummarizer(Protocol):
    """为旧消息生成摘要的最小依赖接口。"""

    def summarize(self, messages: Sequence[Message]) -> str:
        """根据旧消息返回非空摘要文本。"""


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """控制单次模型请求的字符近似预算与最近历史保留数量。"""

    max_characters: int = 12_000
    recent_message_units: int = 2

    def __post_init__(self) -> None:
        """验证预算不会退化成没有可用上下文的配置。"""
        for field_name, value in (
            ("max_characters", self.max_characters),
            ("recent_message_units", self.recent_message_units),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise PromptBuilderError(f"{field_name} 必须是正整数")


@dataclass(frozen=True, slots=True)
class PromptBuildResult:
    """一次提示词构建的消息结果及其可观察预算状态。"""

    messages: tuple[Message, ...]
    system_prompt: str
    context_characters: int
    summarized_message_count: int = 0
    compression_attempted: bool = False
    compression_failed: bool = False
    exceeds_budget: bool = False

    def __post_init__(self) -> None:
        """保证调用方可依赖构建结果的基础结构。"""
        if not self.messages or self.messages[0].role is not MessageRole.SYSTEM:
            raise PromptBuilderError("messages 必须以系统消息开始")

        if self.messages[0].content != self.system_prompt:
            raise PromptBuilderError("system_prompt 必须对应第一条系统消息")

        for field_name, value in (
            ("context_characters", self.context_characters),
            ("summarized_message_count", self.summarized_message_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PromptBuilderError(f"{field_name} 必须是非负整数")

        if not isinstance(self.compression_attempted, bool):
            raise PromptBuilderError("compression_attempted 必须是布尔值")

        if not isinstance(self.compression_failed, bool):
            raise PromptBuilderError("compression_failed 必须是布尔值")

        if not isinstance(self.exceeds_budget, bool):
            raise PromptBuilderError("exceeds_budget 必须是布尔值")

        if self.compression_failed and not self.compression_attempted:
            raise PromptBuilderError("压缩失败必须对应一次压缩尝试")


def _message_character_count(message: Message) -> int:
    """近似计算一条消息发送给模型时占用的字符数。"""
    count = len(message.content or "")
    count += len(message.tool_call_id or "")

    for call in message.tool_calls:
        try:
            arguments_text = json.dumps(
                call.arguments,
                ensure_ascii=False,
                sort_keys=True,
            )
        except (TypeError, ValueError):
            arguments_text = repr(call.arguments)

        count += len(call.call_id) + len(call.tool_name) + len(arguments_text)

    return count


def _messages_character_count(messages: Sequence[Message]) -> int:
    """汇总消息列表的字符近似值。"""
    return sum(_message_character_count(message) for message in messages)


def _group_history_messages(
    history: Sequence[Message],
) -> tuple[tuple[Message, ...], ...]:
    """将一次工具请求及其紧随的工具结果视为不可拆分单元。"""
    groups: list[tuple[Message, ...]] = []
    index = 0

    while index < len(history):
        current_message = history[index]
        group = [current_message]
        index += 1

        if (
            current_message.role is MessageRole.ASSISTANT
            and current_message.tool_calls
        ):
            while (
                index < len(history)
                and history[index].role is MessageRole.TOOL
            ):
                group.append(history[index])
                index += 1

        groups.append(tuple(group))

    return tuple(groups)


def _flatten(groups: Sequence[Sequence[Message]]) -> tuple[Message, ...]:
    """保留组内和组间顺序地展开消息。"""
    return tuple(message for group in groups for message in group)


class PromptBuilder:
    """按固定顺序组装受控系统提示词和有限历史。"""

    def __init__(self, budget: ContextBudget | None = None) -> None:
        """保存上下文预算，默认使用适合学习项目的保守字符上限。"""
        if budget is not None and not isinstance(budget, ContextBudget):
            raise PromptBuilderError("budget 必须是 ContextBudget 或 None")

        self._budget = budget or ContextBudget()

    @staticmethod
    def _format_list(items: Sequence[str]) -> str:
        """把可信文本片段变成稳定、可阅读的 Markdown 列表。"""
        if not items:
            return "- 无"

        return "\n".join(f"- {item}" for item in items)

    @staticmethod
    def _format_tool_definitions(
        tool_definitions: Sequence[Mapping[str, object]],
    ) -> str:
        """稳定序列化模型可见工具定义，供系统提示词说明可用能力。"""
        if isinstance(tool_definitions, (str, bytes)) or not isinstance(
            tool_definitions,
            Sequence,
        ):
            raise PromptBuilderError("tool_definitions 必须是字典序列")

        serialized_tools: list[str] = []
        for index, definition in enumerate(tool_definitions):
            if not isinstance(definition, Mapping):
                raise PromptBuilderError(
                    f"tool_definitions[{index}] 必须是字典",
                )

            try:
                serialized_tools.append(
                    json.dumps(
                        dict(definition),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            except (TypeError, ValueError) as error:
                raise PromptBuilderError(
                    f"tool_definitions[{index}] 无法 JSON 序列化",
                ) from error

        return PromptBuilder._format_list(serialized_tools)

    def _build_system_prompt(
        self,
        safety_rules: object,
        workspace_rules: object,
        tool_definitions: Sequence[Mapping[str, object]],
        skills: Sequence[object],
        long_term_memories: Sequence[object],
    ) -> str:
        """按不可变顺序构造系统提示词中的可信片段。"""
        sections = (
            ("安全规则", _require_text(safety_rules, "safety_rules")),
            ("工作区规则", _require_text(workspace_rules, "workspace_rules")),
            ("允许工具", self._format_tool_definitions(tool_definitions)),
            ("已加载技能", self._format_list(_require_text_items(skills, "skills"))),
            (
                "经授权的长期记忆",
                self._format_list(
                    _require_text_items(long_term_memories, "long_term_memories"),
                ),
            ),
        )
        return "\n\n".join(
            f"## {title}\n{content}"
            for title, content in sections
        )

    @staticmethod
    def _validate_history(history: Sequence[Message]) -> tuple[Message, ...]:
        """验证会话历史只承载历史消息，不允许注入额外系统规则。"""
        if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
            raise PromptBuilderError("history 必须是 Message 序列")

        normalized_history = tuple(history)
        if not all(isinstance(message, Message) for message in normalized_history):
            raise PromptBuilderError("history 中的元素必须是 Message")

        if any(message.role is MessageRole.SYSTEM for message in normalized_history):
            raise PromptBuilderError("history 不能包含 system 消息")

        return normalized_history

    @staticmethod
    def _summary_message(summary: object, source_count: int) -> Message:
        """将摘要器输出包裹为可追溯的历史替代消息。"""
        summary_text = _require_text(summary, "history_summary")
        return Message(
            role=MessageRole.SYSTEM,
            content=(
                f"历史摘要（来源：最早的 {source_count} 条历史消息）\n"
                f"{summary_text}"
            ),
        )

    def build(
        self,
        *,
        safety_rules: object,
        workspace_rules: object,
        tool_definitions: Sequence[Mapping[str, object]] = (),
        skills: Sequence[object] = (),
        long_term_memories: Sequence[object] = (),
        history: Sequence[Message] = (),
        task_summary: object | None = None,
        summarizer: HistorySummarizer | None = None,
    ) -> PromptBuildResult:
        """构造固定顺序上下文，并在可能时用摘要替换过旧历史。"""
        system_prompt = self._build_system_prompt(
            safety_rules,
            workspace_rules,
            tool_definitions,
            skills,
            long_term_memories,
        )
        normalized_history = self._validate_history(history)
        system_message = Message(role=MessageRole.SYSTEM, content=system_prompt)
        task_message: Message | None = None

        if task_summary is not None:
            task_message = Message(
                role=MessageRole.SYSTEM,
                content=f"当前任务摘要\n{_require_text(task_summary, "task_summary")}",
            )

        protected_messages = (system_message,) + (
            (task_message,) if task_message is not None else ()
        )
        protected_characters = _messages_character_count(protected_messages)

        if protected_characters > self._budget.max_characters:
            raise PromptBuilderError("安全规则与任务摘要已超过上下文预算")

        raw_messages = (system_message, *normalized_history)
        if task_message is not None:
            raw_messages = (*raw_messages, task_message)

        raw_characters = _messages_character_count(raw_messages)
        if raw_characters <= self._budget.max_characters:
            return PromptBuildResult(
                messages=raw_messages,
                system_prompt=system_prompt,
                context_characters=raw_characters,
            )

        if summarizer is None:
            return PromptBuildResult(
                messages=raw_messages,
                system_prompt=system_prompt,
                context_characters=raw_characters,
                exceeds_budget=True,
            )

        groups = _group_history_messages(normalized_history)
        recent_groups = groups[-self._budget.recent_message_units :]
        old_groups = groups[: -self._budget.recent_message_units]
        old_messages = _flatten(old_groups)

        if not old_messages:
            return PromptBuildResult(
                messages=raw_messages,
                system_prompt=system_prompt,
                context_characters=raw_characters,
                compression_attempted=True,
                compression_failed=True,
                exceeds_budget=True,
            )

        try:
            summary_message = self._summary_message(
                summarizer.summarize(old_messages),
                len(old_messages),
            )
        except Exception:
            return PromptBuildResult(
                messages=raw_messages,
                system_prompt=system_prompt,
                context_characters=raw_characters,
                compression_attempted=True,
                compression_failed=True,
                exceeds_budget=True,
            )

        compressed_messages = (
            system_message,
            summary_message,
            *_flatten(recent_groups),
        )
        if task_message is not None:
            compressed_messages = (*compressed_messages, task_message)

        compressed_characters = _messages_character_count(compressed_messages)
        if compressed_characters > self._budget.max_characters:
            return PromptBuildResult(
                messages=raw_messages,
                system_prompt=system_prompt,
                context_characters=raw_characters,
                compression_attempted=True,
                compression_failed=True,
                exceeds_budget=True,
            )

        return PromptBuildResult(
            messages=compressed_messages,
            system_prompt=system_prompt,
            context_characters=compressed_characters,
            summarized_message_count=len(old_messages),
            compression_attempted=True,
        )
