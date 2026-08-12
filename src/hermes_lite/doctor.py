"""HermesLite 的本地运行环境诊断。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from hermes_lite.config import ConfigurationError, ModelConfig, load_model_config
from hermes_lite.model_client import ModelClient, ModelClientError
from hermes_lite.sqlite_state_store import (
    SCHEMA_VERSION,
    SQLiteStateStore,
    SQLiteStateStoreError,
    load_sqlite_state_config,
)
from hermes_lite.workspace import Workspace, WorkspaceError, load_workspace_config


class DoctorCheckStatus(str, Enum):
    """单项诊断的固定状态，避免用自由文本判断结果。"""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    """一项不包含敏感配置值的本地诊断结果。"""

    name: str
    status: DoctorCheckStatus
    message: str

    def __post_init__(self) -> None:
        """确保诊断显示字段都有明确类型和文本。"""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name 必须是非空文本")
        if not isinstance(self.status, DoctorCheckStatus):
            raise ValueError("status 必须是 DoctorCheckStatus")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message 必须是非空文本")


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """一次诊断的完整结果；跳过显式模型检查不是失败。"""

    checks: tuple[DoctorCheck, ...]

    def __post_init__(self) -> None:
        """保证报告至少有一项并全部来自固定诊断模型。"""
        if not isinstance(self.checks, tuple) or not self.checks:
            raise ValueError("checks 必须是非空元组")
        if not all(isinstance(check, DoctorCheck) for check in self.checks):
            raise ValueError("checks 中的元素必须是 DoctorCheck")

    @property
    def is_healthy(self) -> bool:
        """只要有本地检查失败，命令就应返回失败状态。"""
        return all(
            check.status is not DoctorCheckStatus.FAILED
            for check in self.checks
        )


def run_doctor(
    *,
    check_model: bool = False,
    model_client_factory: Callable[[ModelConfig], ModelClient] = ModelClient,
) -> DoctorReport:
    """执行本地诊断；只有显式选择时才检查模型连通性。"""
    if not isinstance(check_model, bool):
        raise ValueError("check_model 必须是布尔值")
    if not callable(model_client_factory):
        raise ValueError("model_client_factory 必须可调用")

    checks: list[DoctorCheck] = []
    model_config: ModelConfig | None = None

    try:
        model_config = load_model_config()
    except ConfigurationError as error:
        checks.append(
            DoctorCheck(
                name="模型配置",
                status=DoctorCheckStatus.FAILED,
                message=f"配置无效：{error}",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                name="模型配置",
                status=DoctorCheckStatus.PASSED,
                message="配置格式有效，API Key 已隐藏。",
            )
        )

    try:
        state_store = SQLiteStateStore(load_sqlite_state_config())
        schema_version = state_store.read_schema_version()
    except SQLiteStateStoreError as error:
        checks.append(
            DoctorCheck(
                name="状态数据库",
                status=DoctorCheckStatus.FAILED,
                message=f"本地状态不可用：{error}",
            )
        )
    else:
        if schema_version != SCHEMA_VERSION:
            checks.append(
                DoctorCheck(
                    name="状态数据库",
                    status=DoctorCheckStatus.FAILED,
                    message=(
                        f"模式版本 {schema_version} 与程序要求的 "
                        f"{SCHEMA_VERSION} 不一致；诊断不会自动迁移。"
                    ),
                )
            )
        else:
            checks.append(
                DoctorCheck(
                    name="状态数据库",
                    status=DoctorCheckStatus.PASSED,
                    message=f"模式版本 {schema_version} 可用。",
                )
            )

    try:
        workspace = Workspace(load_workspace_config())
    except WorkspaceError as error:
        checks.append(
            DoctorCheck(
                name="受限工作区",
                status=DoctorCheckStatus.FAILED,
                message=f"工作区不可用：{error}",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                name="受限工作区",
                status=DoctorCheckStatus.PASSED,
                message="工作区边界有效。",
            )
        )
        del workspace

    if not check_model:
        checks.append(
            DoctorCheck(
                name="模型连通性",
                status=DoctorCheckStatus.SKIPPED,
                message="未检查；使用 --check-model 才会发送模型请求。",
            )
        )
    elif model_config is None:
        checks.append(
            DoctorCheck(
                name="模型连通性",
                status=DoctorCheckStatus.SKIPPED,
                message="模型配置未通过，未发送模型请求。",
            )
        )
    else:
        try:
            model_client = model_client_factory(model_config)
            model_client.ask_text(
                system_prompt="你是 HermesLite 连通性诊断助手。",
                user_prompt="请只回复：连通性检查成功。",
            )
        except ModelClientError as error:
            checks.append(
                DoctorCheck(
                    name="模型连通性",
                    status=DoctorCheckStatus.FAILED,
                    message=f"请求失败：{error}",
                )
            )
        else:
            checks.append(
                DoctorCheck(
                    name="模型连通性",
                    status=DoctorCheckStatus.PASSED,
                    message="显式连通性请求成功。",
                )
            )

    return DoctorReport(tuple(checks))


def format_doctor_report(report: DoctorReport) -> str:
    """用稳定中文输出报告，不显示 API Key 或模型原始回答。"""
    if not isinstance(report, DoctorReport):
        raise ValueError("report 必须是 DoctorReport")

    status_text = {
        DoctorCheckStatus.PASSED: "通过",
        DoctorCheckStatus.FAILED: "失败",
        DoctorCheckStatus.SKIPPED: "跳过",
    }
    lines = ["HermesLite 本地诊断："]
    for check in report.checks:
        lines.append(
            f"- [{status_text[check.status]}] {check.name}：{check.message}"
        )

    return "\n".join(lines)
