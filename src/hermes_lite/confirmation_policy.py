"""HermesLite 的工具风险确认策略和一次性确认令牌。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import secrets

from hermes_lite.domain import ToolCall
from hermes_lite.tool_registry import ToolRiskLevel


DEFAULT_CONFIRMATION_TTL_SECONDS = 300


class ConfirmationPolicyError(ValueError):
    """表示确认策略、令牌或待确认调用不符合要求。"""


def _require_text(value: object, field_name: str) -> str:
    """验证非空文本并去除首尾空白。"""
    if not isinstance(value, str):
        raise ConfirmationPolicyError(f"{field_name} 必须是文本")

    cleaned_value = value.strip()
    if not cleaned_value:
        raise ConfirmationPolicyError(f"{field_name} 不能为空")

    return cleaned_value


def _require_utc_time(value: object, field_name: str) -> datetime:
    """验证时钟值是带 UTC 时区的时间，避免比较本地时间。"""
    if not isinstance(value, datetime):
        raise ConfirmationPolicyError(f"{field_name} 必须是 datetime")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ConfirmationPolicyError(f"{field_name} 必须包含时区")

    return value.astimezone(UTC)


def requires_confirmation(risk_level: object) -> bool:
    """只读工具无需确认，其余现有风险等级必须有一次确认。"""
    if not isinstance(risk_level, ToolRiskLevel):
        raise ConfirmationPolicyError("risk_level 必须是 ToolRiskLevel")

    return risk_level is not ToolRiskLevel.READ_ONLY


@dataclass(frozen=True, slots=True)
class PendingConfirmation:
    """一项绑定会话与完整工具调用的一次性待确认记录。"""

    token: str
    session_id: str
    tool_call: ToolCall
    expires_at: datetime

    def __post_init__(self) -> None:
        """确保待确认记录不会缺少关联对象或失去时区信息。"""
        object.__setattr__(self, "token", _require_text(self.token, "token"))
        object.__setattr__(
            self,
            "session_id",
            _require_text(self.session_id, "session_id"),
        )
        if not isinstance(self.tool_call, ToolCall):
            raise ConfirmationPolicyError("tool_call 必须是 ToolCall")
        object.__setattr__(
            self,
            "expires_at",
            _require_utc_time(self.expires_at, "expires_at"),
        )


class ConfirmationManager:
    """在当前进程中签发并一次性消费工具操作确认令牌。"""

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_CONFIRMATION_TTL_SECONDS,
        clock: Callable[[], datetime] | None = None,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        """保存可注入时钟和令牌工厂，使有效期规则可重复测试。"""
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, int)
            or ttl_seconds <= 0
        ):
            raise ConfirmationPolicyError("ttl_seconds 必须是正整数")

        if clock is not None and not callable(clock):
            raise ConfirmationPolicyError("clock 必须是可调用对象或 None")

        if token_factory is not None and not callable(token_factory):
            raise ConfirmationPolicyError("token_factory 必须是可调用对象或 None")

        self._ttl = timedelta(seconds=ttl_seconds)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._token_factory = token_factory or secrets.token_urlsafe
        self._pending_by_token: dict[str, PendingConfirmation] = {}

    def _now(self) -> datetime:
        """读取并规范化当前时间，拒绝不带时区的测试时钟输出。"""
        return _require_utc_time(self._clock(), "clock 返回值")

    def issue(
        self,
        session_id: object,
        tool_call: ToolCall,
    ) -> PendingConfirmation:
        """为一条已被策略拦截的工具调用签发一次性令牌。"""
        normalized_session_id = _require_text(session_id, "session_id")
        if not isinstance(tool_call, ToolCall):
            raise ConfirmationPolicyError("tool_call 必须是 ToolCall")

        token = _require_text(self._token_factory(), "token_factory 返回值")
        if token in self._pending_by_token:
            raise ConfirmationPolicyError("确认令牌冲突，请重新发起操作")

        pending = PendingConfirmation(
            token=token,
            session_id=normalized_session_id,
            tool_call=tool_call,
            expires_at=self._now() + self._ttl,
        )
        self._pending_by_token[token] = pending
        return pending

    def consume(
        self,
        token: object,
        session_id: object,
        tool_call: ToolCall,
    ) -> PendingConfirmation:
        """验证并消费令牌；令牌只可用于原会话中的原始调用一次。"""
        normalized_token = _require_text(token, "token")
        normalized_session_id = _require_text(session_id, "session_id")
        if not isinstance(tool_call, ToolCall):
            raise ConfirmationPolicyError("tool_call 必须是 ToolCall")

        try:
            pending = self._pending_by_token[normalized_token]
        except KeyError as error:
            raise ConfirmationPolicyError("确认令牌不存在或已使用") from error

        if self._now() >= pending.expires_at:
            del self._pending_by_token[normalized_token]
            raise ConfirmationPolicyError("确认令牌已过期")

        if pending.session_id != normalized_session_id:
            raise ConfirmationPolicyError("确认令牌与会话不匹配")

        if pending.tool_call != tool_call:
            raise ConfirmationPolicyError("确认令牌与待确认工具调用不匹配")

        del self._pending_by_token[normalized_token]
        return pending
