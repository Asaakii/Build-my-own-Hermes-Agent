"""验证有限指数退避策略的参数与边界。"""

from __future__ import annotations

import pytest

from hermes_lite.retry_policy import RetryPolicy, RetryPolicyError


def test_retry_policy_limits_total_attempts_and_doubles_delay() -> None:
    """可重试错误只能在上限内继续，并使用受限指数退避。"""
    policy = RetryPolicy(
        max_attempts=4,
        initial_delay_seconds=0.1,
        max_delay_seconds=0.25,
    )

    assert policy.should_retry(retryable=True, attempts_made=1) is True
    assert policy.should_retry(retryable=True, attempts_made=3) is True
    assert policy.should_retry(retryable=True, attempts_made=4) is False
    assert policy.should_retry(retryable=False, attempts_made=1) is False
    assert policy.delay_after_failure(1) == 0.1
    assert policy.delay_after_failure(2) == 0.2
    assert policy.delay_after_failure(3) == 0.25


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"max_attempts": True},
        {"initial_delay_seconds": -0.1},
        {"initial_delay_seconds": float("inf")},
        {"initial_delay_seconds": 0.5, "max_delay_seconds": 0.1},
    ],
)
def test_retry_policy_rejects_unsafe_configuration(
    kwargs: dict[str, object],
) -> None:
    """无限、负数或倒置的等待策略不能进入运行时。"""
    with pytest.raises(RetryPolicyError):
        RetryPolicy(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("attempts_made", [0, True])
def test_retry_policy_rejects_invalid_attempt_count(attempts_made: object) -> None:
    """调用方不能以零或布尔值绕过重试次数边界。"""
    policy = RetryPolicy()

    with pytest.raises(RetryPolicyError):
        policy.should_retry(retryable=True, attempts_made=attempts_made)  # type: ignore[arg-type]
