"""HermesLite 的模型配置加载与校验。"""

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"
PLACEHOLDER_API_KEY = "replace-with-your-api-key"


class ConfigurationError(ValueError):
    """表示本地配置缺失或格式不正确。"""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """模型客户端运行所需的已验证配置。"""

    provider: str
    model: str
    api_key: str = field(repr=False)
    base_url: str
    timeout_seconds: float


def _get_required_setting(
    environment: Mapping[str, str],
    setting_name: str,
) -> str:
    """读取必填文本配置，并拒绝空值。"""
    raw_value = environment.get(setting_name)

    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ConfigurationError(f"缺少必备配置：{setting_name}")

    return raw_value.strip()


def _get_base_url(environment: Mapping[str, str]) -> str:
    """读取并验证 HTTP 或 HTTPS 模型服务地址。"""
    base_url = _get_required_setting(environment, "LLM_BASE_URL")
    parsed_url = urlparse(base_url)

    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ConfigurationError("LLM_BASE_URL 必须是有效的 HTTP 或 HTTPS 地址")

    return base_url.rstrip("/")


def _get_timeout_seconds(environment: Mapping[str, str]) -> float:
    """读取并验证模型请求超时时间。"""
    raw_timeout = _get_required_setting(environment, "LLM_TIMEOUT_SECONDS")

    try:
        timeout_seconds = float(raw_timeout)
    except ValueError as error:
        raise ConfigurationError("LLM_TIMEOUT_SECONDS 必须是数字") from error

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ConfigurationError("LLM_TIMEOUT_SECONDS 必须大于 0")

    return timeout_seconds


def load_model_config(
    environment: Mapping[str, str] | None = None,
) -> ModelConfig:
    """加载环境变量，并返回已完成校验的模型配置。"""
    if environment is None:
        # override=False：终端显式设置的环境变量优先于 .env。
        load_dotenv(DOTENV_PATH, override=False)
        environment = os.environ

    api_key = _get_required_setting(environment, "LLM_API_KEY")
    if api_key == PLACEHOLDER_API_KEY:
        raise ConfigurationError("LLM_API_KEY 仍是示例占位值，请在 .env 中替换")

    return ModelConfig(
        provider=_get_required_setting(environment, "LLM_PROVIDER"),
        model=_get_required_setting(environment, "LLM_MODEL"),
        api_key=api_key,
        base_url=_get_base_url(environment),
        timeout_seconds=_get_timeout_seconds(environment),
    )


def format_config_summary(config: ModelConfig) -> str:
    """生成可显示的配置摘要，绝不输出真实 API Key。"""
    return "\n".join(
        [
            f"模型供应商: {config.provider}",
            f"模型名称: {config.model}",
            f"模型地址: {config.base_url}",
            f"请求超时秒数: {config.timeout_seconds}",
            "API Key: 已配置（已隐藏）",
        ]
    )


def main() -> int:
    """提供手动配置检查入口。"""
    try:
        config = load_model_config()
    except ConfigurationError as error:
        print(f"配置检查失败: {error}")
        return 1

    print(format_config_summary(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())