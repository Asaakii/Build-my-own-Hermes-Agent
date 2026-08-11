"""验证模型配置的读取、校验与脱敏展示。"""

import pytest
import hermes_lite.config as config_module

from hermes_lite.config import (
    ConfigurationError,
    PLACEHOLDER_API_KEY,
    format_config_summary,
    load_model_config,
)


def make_environment(**overrides: str) -> dict[str, str]:
    """构造不依赖真实 .env 文件的测试配置。"""
    environment = {
        "LLM_PROVIDER": "deepseek",
        "LLM_MODEL": "demo-model",
        "LLM_API_KEY": "test-secret-key",
        "LLM_BASE_URL": "https://api.example.com/",
        "LLM_TIMEOUT_SECONDS": "30",
    }
    environment.update(overrides)
    return environment


def test_load_model_config_returns_validated_values() -> None:
    """合法配置应转换为带类型的 ModelConfig。"""
    config = load_model_config(make_environment())

    assert config.provider == "deepseek"
    assert config.model == "demo-model"
    assert config.base_url == "https://api.example.com"
    assert config.timeout_seconds == 30.0


@pytest.mark.parametrize(
    "setting_name",
    [
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_TIMEOUT_SECONDS",
    ],
)
def test_load_model_config_rejects_missing_required_setting(
    setting_name: str,
) -> None:
    """任一必填配置缺失时都应立即失败。"""
    environment = make_environment()
    del environment[setting_name]

    with pytest.raises(ConfigurationError, match=setting_name):
        load_model_config(environment)


@pytest.mark.parametrize("timeout", ["0", "-1", "not-a-number", "inf"])
def test_load_model_config_rejects_invalid_timeout(timeout: str) -> None:
    """超时必须是有限的正数。"""
    with pytest.raises(ConfigurationError):
        load_model_config(make_environment(LLM_TIMEOUT_SECONDS=timeout))


@pytest.mark.parametrize("base_url", ["api.example.com", "ftp://example.com"])
def test_load_model_config_rejects_invalid_base_url(base_url: str) -> None:
    """模型地址只能是有效的 HTTP 或 HTTPS 地址。"""
    with pytest.raises(ConfigurationError):
        load_model_config(make_environment(LLM_BASE_URL=base_url))


def test_load_model_config_rejects_example_api_key() -> None:
    """示例占位 Key 不能被误当作真实配置。"""
    with pytest.raises(ConfigurationError, match="占位值"):
        load_model_config(make_environment(LLM_API_KEY=PLACEHOLDER_API_KEY))


def test_config_summary_hides_api_key() -> None:
    """配置展示和对象 repr 都不能泄露 API Key。"""
    config = load_model_config(make_environment())
    summary = format_config_summary(config)

    assert "test-secret-key" not in summary
    assert "test-secret-key" not in repr(config)
    assert "API Key: 已配置（已隐藏）" in summary


def test_shell_environment_takes_priority_over_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """终端显式配置应覆盖 .env 中同名配置。"""
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=from-dotenv",
                "LLM_MODEL=from-dotenv-model",
                "LLM_API_KEY=dotenv-test-key",
                "LLM_BASE_URL=https://api.example.com",
                "LLM_TIMEOUT_SECONDS=15",
            ],
        ),
        encoding="utf-8",
    )

    setting_names = [
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_TIMEOUT_SECONDS",
    ]
    for setting_name in setting_names:
        monkeypatch.delenv(setting_name, raising=False)

    monkeypatch.setattr(config_module, "DOTENV_PATH", dotenv_path)
    monkeypatch.setenv("LLM_MODEL", "from-shell-model")

    config = config_module.load_model_config()

    assert config.provider == "from-dotenv"
    assert config.model == "from-shell-model"
