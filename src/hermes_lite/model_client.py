"""HermesLite 的 OpenAI 兼容模型客户端。"""

from __future__ import annotations

import json
from collections.abc import Sequence
import logging
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from hermes_lite.domain import Message, MessageRole, ToolCall
from hermes_lite.config import (
    ConfigurationError,
    ModelConfig,
    load_model_config,
)

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
    """通过 OpenAI 兼容接口请求文本回答或结构化工具调用。"""

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

    def _coerce_messages(
        self,
        messages: Sequence[Message],
    ) -> list[Message]:
        """验证并复制调用方传入的会话消息。"""
        message_list = list(messages)

        if not message_list:
            raise ValueError("messages 不能为空")

        if not all(isinstance(message, Message) for message in message_list):
            raise ValueError("messages 中的元素必须是 Message")

        return message_list


    def _to_tool_request_message(
        self,
        message: Message,
    ) -> dict[str, object]:
        """将领域消息转换为 OpenAI 兼容的请求消息。"""
        request_message: dict[str, object] = {
            "role": message.role.value,
            "content": message.content,
        }

        if message.role is MessageRole.ASSISTANT and message.tool_calls:
            request_message["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.tool_name,
                        "arguments": json.dumps(
                            call.arguments,
                            ensure_ascii=False,
                        ),
                    },
                }
                for call in message.tool_calls
            ]

        if message.role is MessageRole.TOOL:
            request_message["tool_call_id"] = message.tool_call_id

        return request_message


    def _request_completion(
        self,
        request_messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        """统一发送模型请求，并转换 SDK 层异常。"""
        request_options: dict[str, object] = {
            "model": self._config.model,
            "messages": request_messages,
            "temperature": 0.2,
        }

        if tools:
            request_options["tools"] = tools
            request_options["tool_choice"] = "auto"

        logger.info(
            "开始模型请求：provider=%s model=%s messages=%d tools=%d",
            self._config.provider,
            self._config.model,
            len(request_messages),
            len(tools or []),
        )

        try:
            return self._client.chat.completions.create(**request_options)
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


    def _parse_assistant_message(self, completion: object) -> Message:
        """将 OpenAI 兼容响应转换为领域层助手消息。"""
        try:
            response_message = completion.choices[0].message
        except (AttributeError, IndexError, TypeError) as error:
            raise ModelResponseError("模型未返回有效回答，请稍后再试") from error

        raw_tool_calls = getattr(response_message, "tool_calls", None)

        if raw_tool_calls:
            if not isinstance(raw_tool_calls, (list, tuple)):
                raise ModelResponseError("模型返回的工具调用格式无效")

            tool_calls: list[ToolCall] = []

            for raw_call in raw_tool_calls:
                try:
                    raw_arguments = raw_call.function.arguments
                    arguments = json.loads(raw_arguments)
                except (
                    AttributeError,
                    TypeError,
                    json.JSONDecodeError,
                ) as error:
                    raise ModelResponseError(
                        "模型返回的工具参数不是有效 JSON",
                    ) from error

                if not isinstance(arguments, dict):
                    raise ModelResponseError(
                        "模型返回的工具参数必须是 JSON 对象",
                    )

                try:
                    tool_calls.append(
                        ToolCall(
                            call_id=raw_call.id,
                            tool_name=raw_call.function.name,
                            arguments=arguments,
                        )
                    )
                except (AttributeError, ValueError) as error:
                    raise ModelResponseError(
                        "模型返回的工具调用格式无效",
                    ) from error

            return Message(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=tuple(tool_calls),
            )

        try:
            return Message(
                role=MessageRole.ASSISTANT,
                content=response_message.content,
            )
        except (AttributeError, ValueError) as error:
            raise ModelResponseError("模型未返回有效回答，请稍后再试") from error

    def ask_messages(self, messages: Sequence[Message]) -> str:
        """发送纯文本会话，并返回模型文本回答。"""
        message_list = self._coerce_messages(messages)

        if any(
            message.role is MessageRole.TOOL
            or message.tool_calls
            or message.tool_call_id is not None
            for message in message_list
        ):
            raise ValueError("当前文本客户端暂不支持工具消息")

        completion = self._request_completion(
            [
                self._to_tool_request_message(message)
                for message in message_list
            ]
        )
        assistant_message = self._parse_assistant_message(completion)

        if assistant_message.tool_calls:
            raise ModelResponseError("纯文本请求不应返回工具调用")

        assert isinstance(assistant_message.content, str)
        logger.info("模型请求成功")
        return assistant_message.content

    def respond(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]],
    ) -> Message:
        """发送工具感知请求，并返回文本回答或结构化工具请求。"""
        message_list = self._coerce_messages(messages)
        tool_list = list(tools)

        if not all(isinstance(tool, dict) for tool in tool_list):
            raise ValueError("tools 中的元素必须是字典")

        completion = self._request_completion(
            [
                self._to_tool_request_message(message)
                for message in message_list
            ],
            tools=tool_list,
        )
        assistant_message = self._parse_assistant_message(completion)
        logger.info("工具感知模型请求成功")
        return assistant_message

    def ask_text(self, system_prompt: str, user_prompt: str) -> str:
        """提供单轮系统提示词与用户提示词的便捷接口。"""
        system_prompt = _require_prompt(system_prompt, "system_prompt")
        user_prompt = _require_prompt(user_prompt, "user_prompt")

        return self.ask_messages(
            [
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                Message(role=MessageRole.USER, content=user_prompt),
            ],
        )


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