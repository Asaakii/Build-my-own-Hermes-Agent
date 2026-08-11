"""验证最小文本 Agent Loop 的会话和任务状态。"""

from collections.abc import Sequence

import pytest

from hermes_lite.agent_loop import TextAgent
from hermes_lite.domain import Message, MessageRole, Session, TaskStatus
from hermes_lite.model_client import ModelClientError


class ScriptedModel:
    """按预设顺序返回回答的模型替身。"""

    def __init__(
        self,
        answers: list[str] | None = None,
        error: ModelClientError | None = None,
    ) -> None:
        self.answers = answers or []
        self.error = error
        self.calls: list[list[Message]] = []

    def ask_messages(self, messages: Sequence[Message]) -> str:
        """记录会话，并返回预设回答或抛出预设错误。"""
        self.calls.append(list(messages))

        if self.error is not None:
            raise self.error

        if not self.answers:
            raise AssertionError("测试模型缺少预设回答")

        return self.answers.pop(0)


def test_run_turn_records_user_and_assistant_messages() -> None:
    """成功调用后，会话应按顺序保存用户和助手消息。"""
    model = ScriptedModel(answers=["你好，文本循环验证成功。"])
    agent = TextAgent(model)
    session = Session(session_id="session-1")

    turn = agent.run_turn(
        session,
        "请只回复：文本循环验证成功。",
        task_id="task-1",
    )

    assert turn.task.status is TaskStatus.COMPLETED
    assert turn.answer == "你好，文本循环验证成功。"
    assert turn.error_message is None
    assert [(message.role, message.content) for message in session.messages] == [
        (MessageRole.USER, "请只回复：文本循环验证成功。"),
        (MessageRole.ASSISTANT, "你好，文本循环验证成功。"),
    ]
    assert [message.role for message in model.calls[0]] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
    ]


def test_run_turn_sends_previous_history_to_next_turn() -> None:
    """第二轮请求必须携带第一轮已确认的会话历史。"""
    model = ScriptedModel(answers=["第一轮回答", "第二轮回答"])
    agent = TextAgent(model)
    session = Session(session_id="session-1")

    agent.run_turn(session, "第一轮问题", task_id="task-1")
    turn = agent.run_turn(session, "第二轮问题", task_id="task-2")

    assert turn.answer == "第二轮回答"
    assert [(message.role, message.content) for message in model.calls[1]] == [
        (MessageRole.SYSTEM, model.calls[1][0].content),
        (MessageRole.USER, "第一轮问题"),
        (MessageRole.ASSISTANT, "第一轮回答"),
        (MessageRole.USER, "第二轮问题"),
    ]


def test_run_turn_keeps_user_message_when_model_fails() -> None:
    """模型失败时不应伪造助手回答，但应保留真实用户输入。"""
    model = ScriptedModel(error=ModelClientError("模型请求失败，请稍后再试"))
    agent = TextAgent(model)
    session = Session(session_id="session-1")

    turn = agent.run_turn(session, "测试失败路径", task_id="task-1")

    assert turn.task.status is TaskStatus.FAILED
    assert turn.answer is None
    assert turn.error_message == "模型请求失败，请稍后再试"
    assert [(message.role, message.content) for message in session.messages] == [
        (MessageRole.USER, "测试失败路径"),
    ]


@pytest.mark.parametrize("user_request", ["", "   "])
def test_run_turn_rejects_empty_user_request(user_request: str) -> None:
    """空用户输入不能改变会话，也不能访问模型。"""
    model = ScriptedModel(answers=["不应被调用"])
    agent = TextAgent(model)
    session = Session(session_id="session-1")

    with pytest.raises(ValueError):
        agent.run_turn(session, user_request)

    assert session.messages == []
    assert model.calls == []