"""验证 HermesLite doctor 默认不发起模型请求。"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import hermes_lite.doctor as doctor_module
from hermes_lite.config import ConfigurationError, ModelConfig
from hermes_lite.doctor import (
    DoctorCheckStatus,
    format_doctor_report,
    run_doctor,
)
from hermes_lite.model_client import ModelClientError


@dataclass
class StateStoreStub:
    """提供固定模式版本，避免诊断测试写入真实数据库。"""

    schema_value: int = 3

    def read_schema_version(self) -> int:
        """返回已验证的模式版本。"""
        return self.schema_value


@dataclass
class ModelClientStub:
    """记录显式连通性请求，不进行网络访问。"""

    response: str = "PRIVATE_MODEL_RESPONSE"
    calls: list[tuple[str, str]] | None = None
    error: ModelClientError | None = None

    def __post_init__(self) -> None:
        """为调用记录建立独立容器。"""
        if self.calls is None:
            self.calls = []

    def ask_text(self, system_prompt: str, user_prompt: str) -> str:
        """记录固定诊断提示或抛出预设错误。"""
        assert self.calls is not None
        self.calls.append((system_prompt, user_prompt))
        if self.error is not None:
            raise self.error
        return self.response


def make_config() -> ModelConfig:
    """构造不会泄露到诊断输出的测试配置。"""
    return ModelConfig(
        provider="demo",
        model="demo-model",
        api_key="PRIVATE_API_KEY",
        base_url="https://api.example.com",
        timeout_seconds=30.0,
    )


@pytest.fixture
def local_checks_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """用纯内存替身固定三项本地诊断结果。"""
    monkeypatch.setattr(doctor_module, "load_model_config", make_config)
    monkeypatch.setattr(doctor_module, "load_sqlite_state_config", lambda: object())
    monkeypatch.setattr(doctor_module, "SQLiteStateStore", lambda config: StateStoreStub())
    monkeypatch.setattr(doctor_module, "load_workspace_config", lambda: object())
    monkeypatch.setattr(doctor_module, "Workspace", lambda config: object())


def test_default_doctor_skips_model_client(
    local_checks_pass: None,
) -> None:
    """默认诊断只能检查本地依赖，不能构造模型客户端。"""
    def fail_if_called(config: ModelConfig) -> ModelClientStub:
        del config
        raise AssertionError("默认 doctor 不应请求模型")

    report = run_doctor(model_client_factory=fail_if_called)

    assert report.is_healthy is True
    assert [check.status for check in report.checks] == [
        DoctorCheckStatus.PASSED,
        DoctorCheckStatus.PASSED,
        DoctorCheckStatus.PASSED,
        DoctorCheckStatus.SKIPPED,
    ]


def test_explicit_model_check_uses_fixed_minimal_request(
    local_checks_pass: None,
) -> None:
    """只有显式参数才请求模型，且报告不回显模型回答。"""
    client = ModelClientStub()

    report = run_doctor(
        check_model=True,
        model_client_factory=lambda config: client,
    )

    assert report.is_healthy is True
    assert client.calls == [
        ("你是 HermesLite 连通性诊断助手。", "请只回复：连通性检查成功。")
    ]
    rendered = format_doctor_report(report)
    assert "模型连通性" in rendered
    assert "PRIVATE_MODEL_RESPONSE" not in rendered


def test_invalid_model_config_skips_explicit_connection_request(
    local_checks_pass: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """配置无效时不能构造客户端或发送请求。"""
    def raise_configuration_error() -> ModelConfig:
        raise ConfigurationError("缺少必备配置：LLM_API_KEY")

    def fail_if_called(config: ModelConfig) -> ModelClientStub:
        del config
        raise AssertionError("配置无效时不应请求模型")

    monkeypatch.setattr(doctor_module, "load_model_config", raise_configuration_error)

    report = run_doctor(check_model=True, model_client_factory=fail_if_called)

    assert report.is_healthy is False
    assert report.checks[0].status is DoctorCheckStatus.FAILED
    assert report.checks[-1].status is DoctorCheckStatus.SKIPPED


def test_model_connection_error_marks_report_unhealthy(
    local_checks_pass: None,
) -> None:
    """显式模型请求失败应在报告中如实标记失败。"""
    client = ModelClientStub(error=ModelClientError("模型请求失败，请稍后再试"))

    report = run_doctor(
        check_model=True,
        model_client_factory=lambda config: client,
    )

    assert report.is_healthy is False
    assert report.checks[-1].status is DoctorCheckStatus.FAILED
