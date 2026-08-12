"""验证模型客户端的请求、回答处理与错误转换。"""

from types import SimpleNamespace

import json
import pytest
import httpx
from openai import APIStatusError
from hermes_lite.domain import Message, MessageRole

from hermes_lite.config import ConfigurationError, ModelConfig
from hermes_lite.retry_policy import RetryPolicy
from hermes_lite.model_client import (
    ModelClient,
    ModelClientError,
    ModelErrorKind,
    ModelResponseError,
    ModelServiceError,
    ModelTimeoutError,
)
import hermes_lite.model_client as model_client_module
from hermes_lite.domain import Message, MessageRole, ToolCall, ToolResult

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


def make_tool_completion(raw_arguments: object) -> SimpleNamespace:
    """构造最小的 OpenAI 风格工具调用响应。"""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(
                                name="echo_text",
                                arguments=raw_arguments,
                            ),
                        )
                    ],
                )
            )
        ]
    )


def make_tool_definition() -> dict[str, object]:
    """构造供模型客户端测试使用的工具定义。"""
    return {
        "type": "function",
        "function": {
            "name": "echo_text",
            "description": "回显文本。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "需要回显的文本。",
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    }


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


def test_respond_returns_text_assistant_message() -> None:
    """工具感知接口也可以返回普通文本回答。"""
    fake_client = FakeOpenAIClient(response=make_completion("最终回答"))
    client = ModelClient(make_config(), client=fake_client)

    response = client.respond(
        [Message(role=MessageRole.USER, content="请回答。")],
        [make_tool_definition()],
    )

    assert response.role is MessageRole.ASSISTANT
    assert response.content == "最终回答"
    assert response.tool_calls == ()
    assert fake_client.completions.requests[0]["tool_choice"] == "auto"


def test_respond_parses_structured_tool_call() -> None:
    """模型返回的函数调用应转换为领域 ToolCall。"""
    fake_client = FakeOpenAIClient(
        response=make_tool_completion('{"text": "工具验证"}')
    )
    client = ModelClient(make_config(), client=fake_client)

    response = client.respond(
        [Message(role=MessageRole.USER, content="请使用 echo_text。")],
        [make_tool_definition()],
    )

    assert response.content is None
    assert response.tool_calls[0] == ToolCall(
        call_id="call-1",
        tool_name="echo_text",
        arguments={"text": "工具验证"},
    )


def test_respond_serializes_structured_tool_history() -> None:
    """工具请求和工具结果必须按 OpenAI 兼容字段发送。"""
    fake_client = FakeOpenAIClient(response=make_completion("处理完成"))
    client = ModelClient(make_config(), client=fake_client)
    call = ToolCall(
        call_id="call-1",
        tool_name="echo_text",
        arguments={"text": "test"},
    )
    tool_message = Message.from_tool_result(
        ToolResult(
            call_id="call-1",
            tool_name="echo_text",
            content="回显：test",
        )
    )

    client.respond(
        [
            Message(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=(call,),
            ),
            tool_message,
        ],
        [make_tool_definition()],
    )

    request_messages = fake_client.completions.requests[0]["messages"]
    assert request_messages[0]["tool_calls"][0]["id"] == "call-1"
    assert request_messages[0]["tool_calls"][0]["function"]["arguments"] == (
        json.dumps({"text": "test"}, ensure_ascii=False)
    )
    assert request_messages[1] == {
        "role": "tool",
        "content": "回显：test",
        "tool_call_id": "call-1",
    }


@pytest.mark.parametrize("raw_arguments", ["not-json", "[]"])
def test_respond_rejects_invalid_tool_arguments(
    raw_arguments: str,
) -> None:
    """工具参数必须是 JSON 对象，不能是任意文本或数组。"""
    fake_client = FakeOpenAIClient(
        response=make_tool_completion(raw_arguments)
    )
    client = ModelClient(make_config(), client=fake_client)

    with pytest.raises(ModelResponseError):
        client.respond(
            [Message(role=MessageRole.USER, content="测试")],
            [make_tool_definition()],
        )


def test_respond_rejects_response_without_text_or_tool_call() -> None:
    """模型不能返回既无文本也无工具调用的空助手消息。"""
    fake_client = FakeOpenAIClient(response=make_completion(None))
    client = ModelClient(make_config(), client=fake_client)

    with pytest.raises(ModelResponseError):
        client.respond(
            [Message(role=MessageRole.USER, content="测试")],
            [make_tool_definition()],
        )


def test_ask_messages_rejects_unexpected_tool_call_response() -> None:
    """纯文本接口收到工具调用响应时必须明确失败。"""
    fake_client = FakeOpenAIClient(
        response=make_tool_completion('{"text": "测试"}')
    )
    client = ModelClient(make_config(), client=fake_client)

    with pytest.raises(ModelResponseError, match="纯文本请求不应返回工具调用"):
        client.ask_messages(
            [Message(role=MessageRole.USER, content="普通文本问题")]
        )


class SequencedCompletions:
    """按顺序返回结果或异常，用于验证有限重试次数。"""

    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = outcomes
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        """记录请求，并消费下一项预设结果。"""
        self.requests.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class SequencedOpenAIClient:
    """提供顺序化的 chat.completions 测试替身。"""

    def __init__(self, outcomes: list[object]) -> None:
        self.completions = SequencedCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


def make_status_error(status_code: int) -> APIStatusError:
    """构造不含真实服务数据的 HTTP 状态异常。"""
    return APIStatusError(
        "request failed",
        response=httpx.Response(
            status_code=status_code,
            request=httpx.Request(
                "POST",
                "https://api.example.com/chat/completions",
            ),
        ),
        body={"error": "sanitized"},
    )


def test_model_client_retries_timeout_then_returns_answer() -> None:
    """超时属于暂时错误，客户端按退避策略重试后可成功返回。"""
    fake_client = SequencedOpenAIClient(
        [
            TimeoutError("temporary timeout"),
            TimeoutError("temporary timeout"),
            make_completion("重试成功"),
        ]
    )
    wait_calls: list[float] = []
    client = ModelClient(
        make_config(),
        client=fake_client,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_delay_seconds=0.1,
            max_delay_seconds=0.5,
        ),
        sleep=wait_calls.append,
    )

    answer = client.ask_text("系统提示", "用户问题")

    assert answer == "重试成功"
    assert len(fake_client.completions.requests) == 3
    assert wait_calls == [0.1, 0.2]


def test_model_client_stops_after_retry_limit() -> None:
    """持续超时达到总次数上限后，应返回最后一次标准错误。"""
    fake_client = SequencedOpenAIClient(
        [
            TimeoutError("temporary timeout"),
            TimeoutError("temporary timeout"),
            TimeoutError("temporary timeout"),
        ]
    )
    wait_calls: list[float] = []
    client = ModelClient(
        make_config(),
        client=fake_client,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_delay_seconds=0.1,
            max_delay_seconds=0.5,
        ),
        sleep=wait_calls.append,
    )

    with pytest.raises(ModelTimeoutError) as error_info:
        client.ask_text("系统提示", "用户问题")

    assert error_info.value.kind is ModelErrorKind.TIMEOUT
    assert error_info.value.retryable is True
    assert len(fake_client.completions.requests) == 3
    assert wait_calls == [0.1, 0.2]


def test_model_client_does_not_retry_authentication_error() -> None:
    """认证失败不是暂时错误，必须立即停止以避免重复无效请求。"""
    fake_client = SequencedOpenAIClient([make_status_error(401)])
    wait_calls: list[float] = []
    client = ModelClient(
        make_config(),
        client=fake_client,
        retry_policy=RetryPolicy(max_attempts=3),
        sleep=wait_calls.append,
    )

    with pytest.raises(ModelServiceError) as error_info:
        client.ask_text("系统提示", "用户问题")

    assert error_info.value.kind is ModelErrorKind.AUTHENTICATION
    assert error_info.value.retryable is False
    assert len(fake_client.completions.requests) == 1
    assert wait_calls == []


@pytest.mark.parametrize(
    ("status_code", "expected_kind", "expected_retryable"),
    [
        (429, ModelErrorKind.RATE_LIMIT, True),
        (503, ModelErrorKind.SERVICE, True),
        (400, ModelErrorKind.REQUEST, False),
    ],
)
def test_model_client_classifies_status_errors(
    status_code: int,
    expected_kind: ModelErrorKind,
    expected_retryable: bool,
) -> None:
    """状态码应映射为稳定类别，并明确是否可重试。"""
    fake_client = SequencedOpenAIClient([make_status_error(status_code)])
    client = ModelClient(
        make_config(),
        client=fake_client,
        retry_policy=RetryPolicy(max_attempts=1),
        sleep=lambda delay: None,
    )

    with pytest.raises(ModelServiceError) as error_info:
        client.ask_text("系统提示", "用户问题")

    assert error_info.value.kind is expected_kind
    assert error_info.value.retryable is expected_retryable
    assert len(fake_client.completions.requests) == 1
