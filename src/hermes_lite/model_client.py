"""HermesLite 的 OpenAI 兼容模型客户端。"""

from __future__ import annotations

import logging
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from hermes_lite.config import ConfigurationError, ModelConfig, load_model_config


logger = logging.getLogger(__name__)


class ModelClientError(RuntimeError):
    """表示模型请求无法获得可用回答。"""


class ModelTimeoutError(ModelClientError):
    """表示模型服务在指定时间内没有响应。"""


class ModelServiceError(ModelClientError):
    """表示模型服务或网络连接出现问题。"""


class ModelResponseError(ModelClientError):
    """表示模型返回结构不完整或没有有效文本。"""


def _require_prompt(value: object, field_name: str) -> str:
    """验证模型请求中的提示词文本。"""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} 必须是文本")

    prompt = value.strip()
    if not prompt:
        raise ValueError(f"{field_name} 不能为空")

    return prompt


class ModelClient:
    """通过 OpenAI 兼容接口请求纯文本模型回答。"""

    def __init__(
        self,
        config: ModelConfig,
        client: Any | None = None,
    ) -> None:
        """保存已验证配置，并允许测试注入假的 SDK 客户端。"""
        self._config = config
        self._client = client if client is not None else OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    def ask_text(self, system_prompt: str, user_prompt: str) -> str:
        """发送一轮纯文本请求，并返回去除首尾空白后的回答。"""
        system_prompt = _require_prompt(system_prompt, "system_prompt")
        user_prompt = _require_prompt(user_prompt, "user_prompt")

        logger.info(
            "开始模型请求：provider=%s model=%s",
            self._config.provider,
            self._config.model,
        )

        try:
            completion = self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
        except (APITimeoutError, TimeoutError) as error:
            raise ModelTimeoutError("模型请求超时，请稍后再试") from error
        except (APIConnectionError, ConnectionError) as error:
            raise ModelServiceError("无法连接模型服务，请检查网络") from error
        except APIStatusError as error:
            raise ModelServiceError(
                f"模型服务返回错误状态码: {error.status_code}",
            ) from error
        except Exception as error:
            raise ModelClientError("模型请求失败，请稍后再试") from error

        try:
            content = completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as error:
            raise ModelResponseError("模型未返回有效回答，请稍后再试") from error

        if not isinstance(content, str) or not content.strip():
            raise ModelResponseError("模型未返回有效回答，请稍后再试")

        logger.info("模型请求成功")
        return content.strip()


def main() -> int:
    """提供一次真实模型连接验证入口。"""
    try:
        config = load_model_config()
        client = ModelClient(config)
        answer = client.ask_text(
            system_prompt="你是模型连接验证助手。",
            user_prompt="请只回复：模型连接验证成功。",
        )
    except (ConfigurationError, ModelClientError) as error:
        print(f"模型请求失败: {error}")
        return 1

    print(f"模型回答: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())