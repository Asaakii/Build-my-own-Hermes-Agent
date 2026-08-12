"""验证显式技能选择如何收窄 ToolAgent 的工具边界。"""

from collections.abc import Sequence
from pathlib import Path

import pytest

import hermes_lite.skill_loader as skill_loader_module
from hermes_lite.domain import Message, MessageRole, Session, ToolCall
from hermes_lite.tool_agent_loop import ToolAgent
from hermes_lite.tool_registry import ToolDefinition, ToolRegistry, ToolRiskLevel


class RecordingToolModel:
    """记录模型可见上下文，并按顺序返回预设消息。"""

    def __init__(self, responses: list[Message]) -> None:
        self._responses = responses
        self.calls: list[tuple[list[Message], list[dict[str, object]]]] = []

    def respond(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]],
    ) -> Message:
        """记录本次可见工具，再返回一条预设模型消息。"""
        self.calls.append((list(messages), list(tools)))
        return self._responses.pop(0)


def echo_handler(arguments: dict[str, object]) -> str:
    """提供确定性的安全测试工具结果。"""
    return f"回显：{arguments["text"]}"


def make_registry(other_handler: object | None = None) -> ToolRegistry:
    """创建包含两个工具的注册表，便于验证子集收窄。"""
    registry = ToolRegistry()
    for name, handler in (
        ("echo_text", echo_handler),
        ("other_text", other_handler or echo_handler),
    ):
        registry.register(
            ToolDefinition(
                name=name,
                description=f"{name} 的测试说明。",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "测试文本。",
                        },
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
                risk_level=ToolRiskLevel.READ_ONLY,
                handler=handler,  # type: ignore[arg-type]
            )
        )
    return registry


def write_skill(project_root: Path) -> None:
    """写入允许 echo_text 的受限技能样例。"""
    skill_path = project_root / "skills" / "focus_read" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        "---\n"
        "{\"name\":\"focus_read\",\"description\":\"只允许回显工具。\","
        "\"allowed_tools\":[\"echo_text\"]}\n"
        "---\n"
        "# 受限检查\n"
        "1. 只使用已允许的工具。\n",
        encoding="utf-8",
    )


@pytest.fixture
def isolated_skills_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """把技能根目录隔离到临时项目，避免依赖真实工作区。"""
    write_skill(tmp_path)
    monkeypatch.setattr(skill_loader_module, "PROJECT_ROOT", tmp_path)
    return tmp_path


def tool_names(definitions: list[dict[str, object]]) -> list[str]:
    """从模型工具定义中提取函数名称，简化断言。"""
    return [str(definition["function"]["name"]) for definition in definitions]


def test_selected_skill_is_injected_and_limits_visible_tools(
    isolated_skills_root: Path,
) -> None:
    """显式选择技能后，模型只接收该技能的说明和工具子集。"""
    model = RecordingToolModel(
        [
            Message(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=(
                    ToolCall(
                        call_id="call-allowed",
                        tool_name="echo_text",
                        arguments={"text": "技能验证"},
                    ),
                ),
            ),
            Message(role=MessageRole.ASSISTANT, content="已按技能执行。"),
        ]
    )
    agent = ToolAgent(model, make_registry())

    turn = agent.run_turn(
        Session(session_id="skill-session"),
        "执行受限任务。",
        skill_name="focus_read",
    )

    assert turn.answer == "已按技能执行。"
    assert turn.tool_results[0].content == "回显：技能验证"
    messages, definitions = model.calls[0]
    assert tool_names(definitions) == ["echo_text"]
    assert "技能名称：focus_read" in (messages[0].content or "")
    assert "只使用已允许的工具。" in (messages[0].content or "")


def test_selected_skill_rejects_a_hallucinated_disallowed_tool(
    isolated_skills_root: Path,
) -> None:
    """模型伪造未授权工具时，执行层必须拒绝且不调用处理函数。"""
    other_tool_calls: list[dict[str, object]] = []

    def other_handler(arguments: dict[str, object]) -> str:
        other_tool_calls.append(arguments)
        return "不应执行"

    model = RecordingToolModel(
        [
            Message(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=(
                    ToolCall(
                        call_id="call-disallowed",
                        tool_name="other_text",
                        arguments={"text": "伪造调用"},
                    ),
                ),
            ),
            Message(role=MessageRole.ASSISTANT, content="已收到拒绝结果。"),
        ]
    )
    agent = ToolAgent(model, make_registry(other_handler))

    turn = agent.run_turn(
        Session(session_id="skill-session"),
        "执行受限任务。",
        skill_name="focus_read",
    )

    assert turn.answer == "已收到拒绝结果。"
    assert turn.tool_results[0].is_error is True
    assert "当前技能不允许工具 other_text" in turn.tool_results[0].content
    assert other_tool_calls == []
    assert model.calls[1][0][-1].role is MessageRole.TOOL


def test_tool_agent_does_not_auto_load_skills(
    isolated_skills_root: Path,
) -> None:
    """未提供 skill_name 时保留完整注册表，不从目录自动挑选技能。"""
    model = RecordingToolModel(
        [Message(role=MessageRole.ASSISTANT, content="普通工具会话。")]
    )
    agent = ToolAgent(model, make_registry())

    turn = agent.run_turn(Session(session_id="normal-session"), "普通任务。")

    assert turn.answer == "普通工具会话。"
    messages, definitions = model.calls[0]
    assert tool_names(definitions) == ["echo_text", "other_text"]
    assert "## 已加载技能\n- 无" in (messages[0].content or "")


def test_missing_skill_fails_before_model_request(
    isolated_skills_root: Path,
) -> None:
    """不存在的显式技能要停止本轮，不能静默退化成普通工具会话。"""
    model = RecordingToolModel(
        [Message(role=MessageRole.ASSISTANT, content="不应请求模型。")]
    )
    session = Session(session_id="missing-skill-session")
    agent = ToolAgent(model, make_registry())

    turn = agent.run_turn(session, "执行任务。", skill_name="missing_skill")

    assert turn.answer is None
    assert turn.error_message == "技能不存在: missing_skill"
    assert model.calls == []
    assert session.messages == [
        Message(role=MessageRole.USER, content="执行任务。"),
    ]
