"""验证工具风险确认策略和一次性令牌边界。"""

from datetime import UTC, datetime, timedelta

import pytest

from hermes_lite.confirmation_policy import (
    ConfirmationManager,
    ConfirmationPolicyError,
    requires_confirmation,
)
from hermes_lite.domain import ToolCall
from hermes_lite.tool_registry import ToolRiskLevel


class AdjustableClock:
    """提供可手动推进的确定性 UTC 时钟。"""

    def __init__(self) -> None:
        self.now = datetime(2026, 5, 10, tzinfo=UTC)

    def __call__(self) -> datetime:
        """返回当前受控时间。"""
        return self.now

    def advance(self, seconds: int) -> None:
        """推进指定秒数，模拟确认令牌过期。"""
        self.now += timedelta(seconds=seconds)


def make_tool_call(text: str = "修改内容") -> ToolCall:
    """构造可绑定到确认令牌的稳定测试调用。"""
    return ToolCall(
        call_id="call-write-1",
        tool_name="write_text",
        arguments={"text": text},
    )


@pytest.mark.parametrize(
    ("risk_level", "expected"),
    [
        (ToolRiskLevel.READ_ONLY, False),
        (ToolRiskLevel.WRITE, True),
        (ToolRiskLevel.EXECUTE, True),
        (ToolRiskLevel.DESTRUCTIVE, True),
    ],
)
def test_risk_policy_requires_confirmation_for_non_read_only_tools(
    risk_level: ToolRiskLevel,
    expected: bool,
) -> None:
    """风险等级决定是否必须取得用户确认。"""
    assert requires_confirmation(risk_level) is expected


def test_risk_policy_rejects_unknown_risk_value() -> None:
    """策略不接受字符串伪装的风险等级。"""
    with pytest.raises(ConfirmationPolicyError, match="ToolRiskLevel"):
        requires_confirmation("write")


def test_confirmation_token_binds_one_tool_call_and_is_single_use() -> None:
    """令牌只能由原会话消费一次，且返回原始待确认记录。"""
    clock = AdjustableClock()
    manager = ConfirmationManager(
        clock=clock,
        token_factory=lambda: "confirm-fixed-token",
    )
    tool_call = make_tool_call()

    pending = manager.issue("session-1", tool_call)
    consumed = manager.consume("confirm-fixed-token", "session-1", tool_call)

    assert pending == consumed
    assert pending.expires_at == datetime(2026, 5, 10, 0, 5, tzinfo=UTC)
    with pytest.raises(ConfirmationPolicyError, match="不存在或已使用"):
        manager.consume("confirm-fixed-token", "session-1", tool_call)


def test_confirmation_token_rejects_other_session_without_consuming() -> None:
    """其他会话不能消费令牌，原会话仍可在有效期内确认。"""
    manager = ConfirmationManager(token_factory=lambda: "confirm-session")
    tool_call = make_tool_call()
    manager.issue("session-owner", tool_call)

    with pytest.raises(ConfirmationPolicyError, match="与会话不匹配"):
        manager.consume("confirm-session", "session-other", tool_call)

    assert manager.consume("confirm-session", "session-owner", tool_call).token == (
        "confirm-session"
    )


def test_confirmation_token_rejects_changed_tool_arguments_without_consuming() -> None:
    """令牌不能被替换为同名但不同参数的工具调用。"""
    manager = ConfirmationManager(token_factory=lambda: "confirm-call")
    original_call = make_tool_call("原始内容")
    manager.issue("session-1", original_call)

    with pytest.raises(ConfirmationPolicyError, match="工具调用不匹配"):
        manager.consume("confirm-call", "session-1", make_tool_call("篡改内容"))

    assert manager.consume("confirm-call", "session-1", original_call).tool_call == (
        original_call
    )


def test_confirmation_token_expires_and_cannot_be_reused() -> None:
    """过期令牌会被删除，之后不能恢复或重放。"""
    clock = AdjustableClock()
    manager = ConfirmationManager(
        ttl_seconds=10,
        clock=clock,
        token_factory=lambda: "confirm-expired",
    )
    tool_call = make_tool_call()
    manager.issue("session-1", tool_call)
    clock.advance(10)

    with pytest.raises(ConfirmationPolicyError, match="已过期"):
        manager.consume("confirm-expired", "session-1", tool_call)

    with pytest.raises(ConfirmationPolicyError, match="不存在或已使用"):
        manager.consume("confirm-expired", "session-1", tool_call)


@pytest.mark.parametrize("ttl_seconds", [0, -1, True])
def test_confirmation_manager_rejects_invalid_ttl(ttl_seconds: object) -> None:
    """有效期必须是非布尔值正整数。"""
    with pytest.raises(ConfirmationPolicyError, match="正整数"):
        ConfirmationManager(ttl_seconds=ttl_seconds)  # type: ignore[arg-type]
