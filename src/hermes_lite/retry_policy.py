"""HermesLite 的有限模型重试策略。"""

from __future__ import annotations

from dataclasses import dataclass
import math


class RetryPolicyError(ValueError):
    """表示重试次数或退避参数不符合安全边界。"""


def _require_non_negative_number(value: object, field_name: str) -> float:
    """验证可用于等待时间的有限非负数。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RetryPolicyError(f"{field_name} 必须是非负数字")

    normalized_value = float(value)
    if not math.isfinite(normalized_value) or normalized_value < 0:
        raise RetryPolicyError(f"{field_name} 必须是有限非负数字")

    return normalized_value


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """限制模型请求次数，并计算确定性的指数退避等待时间。"""

    max_attempts: int = 2
    initial_delay_seconds: float = 0.25
    max_delay_seconds: float = 2.0

    def __post_init__(self) -> None:
        """拒绝无限重试、负等待和不一致的退避上限。"""
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts <= 0
        ):
            raise RetryPolicyError("max_attempts 必须是正整数")

        initial_delay_seconds = _require_non_negative_number(
            self.initial_delay_seconds,
            "initial_delay_seconds",
        )
        max_delay_seconds = _require_non_negative_number(
            self.max_delay_seconds,
            "max_delay_seconds",
        )
        if max_delay_seconds < initial_delay_seconds:
            raise RetryPolicyError(
                "max_delay_seconds 不能小于 initial_delay_seconds",
            )

        object.__setattr__(self, "initial_delay_seconds", initial_delay_seconds)
        object.__setattr__(self, "max_delay_seconds", max_delay_seconds)

    def should_retry(self, *, retryable: bool, attempts_made: int) -> bool:
        """仅在错误可重试且尚未达到总尝试次数时返回真。"""
        if not isinstance(retryable, bool):
            raise RetryPolicyError("retryable 必须是布尔值")
        if (
            isinstance(attempts_made, bool)
            or not isinstance(attempts_made, int)
            or attempts_made <= 0
        ):
            raise RetryPolicyError("attempts_made 必须是正整数")

        return retryable and attempts_made < self.max_attempts

    def delay_after_failure(self, attempts_made: int) -> float:
        """计算本次失败后的等待时间：初始值乘以二并受上限限制。"""
        if (
            isinstance(attempts_made, bool)
            or not isinstance(attempts_made, int)
            or attempts_made <= 0
        ):
            raise RetryPolicyError("attempts_made 必须是正整数")

        delay = self.initial_delay_seconds * (2 ** (attempts_made - 1))
        return min(delay, self.max_delay_seconds)
