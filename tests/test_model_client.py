"""验证模型客户端的请求、回答处理与错误转换。"""

from types import SimpleNamespace

import pytest
import httpx
from openai import APIStatusError

from hermes_lite.config import ModelConfig
from hermes_lite.model_client import (
    ModelClient,
    ModelClientError,
    ModelResponseError,
    ModelServiceError,
    ModelTimeoutError,
)


def make_config() -> ModelConfig:
    """构造不含真实密钥的测试配置。"""
    return ModelConfig(
        provider="demo",
        model="demo-model",
        api_key="test-secret-key",
        base_url="https://api.example.com",
        timeout_seconds=30.0,
    )


def make_completion(content: object) -> SimpleNamespace:
    """构造最小的 OpenAI 风格文本回答。"""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            ),
        ],
    )


class FakeCompletions:
    """记录请求参数，并返回预设回答或抛出预设异常。"""

    def __init__(
        self,
        response: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        """模拟 SDK 的 chat.completions.create 方法。"""
        self.requests.append(kwargs)

        if self.error is not None:
            raise self.error

        return self.response


class FakeOpenAIClient:
    """提供 ModelClient 当前需要的最小 SDK 接口。"""

    def __init__(
        self,
        response: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.completions = FakeCompletions(response=response, error=error)
        self.chat = SimpleNamespace(completions=self.completions)


def test_ask_text_returns_trimmed_answer() -> None:
    """模型客户端应返回去除首尾空白后的文本。"""
    fake_client = FakeOpenAIClient(response=make_completion("  验证成功  "))
    client = ModelClient(make_config(), client=fake_client)

    answer = client.ask_text("系统提示", "用户问题")

    assert answer == "验证成功"
    assert fake_client.completions.requests == [
        {
            "model": "demo-model",
            "messages": [
                {"role": "system", "content": "系统提示"},
                {"role": "user", "content": "用户问题"},
            ],
            "temperature": 0.2,
        },
    ]


@pytest.mark.parametrize("content", ["", "   ", None])
def test_ask_text_rejects_empty_model_response(content: object) -> None:
    """空回答不能被误当作有效模型结果。"""
    fake_client = FakeOpenAIClient(response=make_completion(content))
    client = ModelClient(make_config(), client=fake_client)

    with pytest.raises(ModelResponseError):
        client.ask_text("系统提示", "用户问题")


def test_ask_text_rejects_response_without_choices() -> None:
    """缺少 choices 的异常响应应被转换为统一错误。"""
    fake_client = FakeOpenAIClient(response=SimpleNamespace(choices=[]))
    client = ModelClient(make_config(), client=fake_client)

    with pytest.raises(ModelResponseError):
        client.ask_text("系统提示", "用户问题")


def test_ask_text_converts_timeout_error() -> None:
    """超时应转换为调用方可识别的错误类型。"""
    fake_client = FakeOpenAIClient(error=TimeoutError("timeout"))
    client = ModelClient(make_config(), client=fake_client)

    with pytest.raises(ModelTimeoutError):
        client.ask_text("系统提示", "用户问题")


def test_ask_text_converts_connection_error() -> None:
    """网络错误应转换为模型服务错误。"""
    fake_client = FakeOpenAIClient(error=ConnectionError("offline"))
    client = ModelClient(make_config(), client=fake_client)

    with pytest.raises(ModelServiceError):
        client.ask_text("系统提示", "用户问题")


def test_ask_text_converts_unexpected_error() -> None:
    """未知 SDK 错误不能直接泄露给用户。"""
    fake_client = FakeOpenAIClient(error=RuntimeError("internal detail"))
    client = ModelClient(make_config(), client=fake_client)

    with pytest.raises(ModelClientError, match="模型请求失败"):
        client.ask_text("系统提示", "用户问题")


@pytest.mark.parametrize("system_prompt, user_prompt", [("", "问题"), ("系统", "")])
def test_ask_text_rejects_empty_prompts(
    system_prompt: str,
    user_prompt: str,
) -> None:
    """空提示词不应发送到模型服务。"""
    client = ModelClient(make_config(), client=FakeOpenAIClient())

    with pytest.raises(ValueError):
        client.ask_text(system_prompt, user_prompt)


def test_ask_text_converts_api_status_error() -> None:
    """HTTP 状态错误应转换为不暴露服务细节的统一错误。"""
    response = httpx.Response(
        status_code=401,
        request=httpx.Request(
            "POST",
            "https://api.example.com/chat/completions",
        ),
    )
    status_error = APIStatusError(
        "unauthorized",
        response=response,
        body={"error": "invalid key"},
    )
    fake_client = FakeOpenAIClient(error=status_error)
    client = ModelClient(make_config(), client=fake_client)

    with pytest.raises(ModelServiceError, match="401"):
        client.ask_text("系统提示", "用户问题")