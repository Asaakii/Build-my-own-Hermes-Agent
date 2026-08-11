"""验证模型客户端的请求、回答处理与错误转换。"""

from types import SimpleNamespace

import pytest
import httpx
from openai import APIStatusError
from hermes_lite.domain import Message, MessageRole

from hermes_lite.config import ConfigurationError, ModelConfig
from hermes_lite.model_client import (
    ModelClient,
    ModelClientError,
    ModelResponseError,
    ModelServiceError,
    ModelTimeoutError,
)
import hermes_lite.model_client as model_client_module
from hermes_lite.domain import Message, MessageRole, ToolCall

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


def test_ask_messages_sends_complete_conversation_history() -> None:
    """多消息接口应按原顺序发送完整会话历史。"""
    fake_client = FakeOpenAIClient(response=make_completion("第二轮回答"))
    client = ModelClient(make_config(), client=fake_client)
    messages = [
        Message(role=MessageRole.SYSTEM, content="系统提示"),
        Message(role=MessageRole.USER, content="第一轮问题"),
        Message(role=MessageRole.ASSISTANT, content="第一轮回答"),
        Message(role=MessageRole.USER, content="第二轮问题"),
    ]

    answer = client.ask_messages(messages)

    assert answer == "第二轮回答"
    assert fake_client.completions.requests[0]["messages"] == [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "第一轮问题"},
        {"role": "assistant", "content": "第一轮回答"},
        {"role": "user", "content": "第二轮问题"},
    ]


def test_ask_messages_rejects_empty_history() -> None:
    """模型客户端不能请求空会话。"""
    client = ModelClient(make_config(), client=FakeOpenAIClient())

    with pytest.raises(ValueError, match="messages 不能为空"):
        client.ask_messages([])


def test_ask_messages_rejects_non_message_item() -> None:
    """调用方不能把任意对象伪装成会话消息。"""
    client = ModelClient(make_config(), client=FakeOpenAIClient())

    with pytest.raises(ValueError, match="Message"):
        client.ask_messages(["not-a-message"])  # type: ignore[list-item]


def test_ask_messages_rejects_tool_message_before_tool_support() -> None:
    """工具消息在未实现 tool_call_id 前必须明确拒绝。"""
    client = ModelClient(make_config(), client=FakeOpenAIClient())
    messages = [
        Message(
            role=MessageRole.TOOL,
            content="工具结果",
            tool_call_id="call-1",
        ),
    ]

    with pytest.raises(ValueError, match="暂不支持工具消息"):
        client.ask_messages(messages)


def test_ask_messages_rejects_assistant_tool_request_before_tool_support() -> None:
    """纯文本客户端不能提前发送助手工具请求。"""
    fake_client = FakeOpenAIClient()
    client = ModelClient(make_config(), client=fake_client)
    call = ToolCall(
        call_id="call-1",
        tool_name="summarize_text",
        arguments={"text": "测试"},
    )
    messages = [
        Message(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=(call,),
        ),
    ]

    with pytest.raises(ValueError, match="暂不支持工具消息"):
        client.ask_messages(messages)

    assert fake_client.completions.requests == []


class FakeRuntimeClient:
    """用于验证命令行入口的最小模型客户端替身。"""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def ask_text(self, system_prompt: str, user_prompt: str) -> str:
        """返回固定回答，避免入口测试访问真实模型。"""
        return "模型连接验证成功。"


def test_main_prints_answer_after_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """命令行入口在成功时应输出模型回答并返回零。"""
    monkeypatch.setattr(model_client_module, "load_model_config", make_config)
    monkeypatch.setattr(model_client_module, "ModelClient", FakeRuntimeClient)

    exit_code = model_client_module.main()

    assert exit_code == 0
    assert capsys.readouterr().out == "模型回答: 模型连接验证成功。\n"


def test_main_prints_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """命令行入口应把配置错误转换为可理解的提示。"""
    def raise_configuration_error() -> ModelConfig:
        raise ConfigurationError("缺少必备配置：LLM_API_KEY")

    monkeypatch.setattr(
        model_client_module,
        "load_model_config",
        raise_configuration_error,
    )

    exit_code = model_client_module.main()

    assert exit_code == 1
    assert (
        capsys.readouterr().out
        == "模型请求失败: 缺少必备配置：LLM_API_KEY\n"
    )
