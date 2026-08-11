"""验证工具注册表的声明导出与参数安全边界。"""

import pytest

from hermes_lite.tool_registry import (
    ToolDefinition,
    ToolRegistry,
    ToolRegistryError,
    ToolRiskLevel,
)


def make_summary_definition() -> ToolDefinition:
    """创建一项供测试使用的只读工具定义。"""
    return ToolDefinition(
        name="summarize_text",
        description="总结给定文本。",
        parameters_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "需要总结的文本。",
                },
                "max_length": {
                    "type": "integer",
                    "description": "最大摘要长度。",
                },
                "temperature": {
                    "type": "number",
                    "description": "数值参数示例。",
                },
                "strict": {
                    "type": "boolean",
                    "description": "是否启用严格模式。",
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        risk_level=ToolRiskLevel.READ_ONLY,
    )


def make_registry() -> ToolRegistry:
    """创建已经登记测试工具的注册表。"""
    registry = ToolRegistry()
    registry.register(make_summary_definition())
    return registry


def test_registry_exports_declared_tool_for_model() -> None:
    """模型只能看见已登记的工具和受限参数模式。"""
    registry = make_registry()

    definitions = registry.list_model_definitions()

    assert definitions[0]["type"] == "function"
    assert definitions[0]["function"]["name"] == "summarize_text"
    assert definitions[0]["function"]["parameters"]["additionalProperties"] is False


def test_registry_rejects_duplicate_tool_name() -> None:
    """同名工具不能覆盖已有定义。"""
    registry = make_registry()

    with pytest.raises(ToolRegistryError, match="工具已登记"):
        registry.register(make_summary_definition())


def test_registry_rejects_unknown_tool() -> None:
    """未登记工具不能被查询或进入后续执行层。"""
    registry = ToolRegistry()

    with pytest.raises(ToolRegistryError, match="未知工具"):
        registry.get("delete_all_files")


def test_validate_arguments_returns_independent_copy() -> None:
    """合法参数会通过校验，并以独立字典返回。"""
    registry = make_registry()
    arguments = {
        "text": "Agent 开发需要明确工具边界。",
        "max_length": 20,
        "temperature": 0.5,
        "strict": True,
    }

    validated_arguments = registry.validate_arguments(
        "summarize_text",
        arguments,
    )
    validated_arguments["text"] = "已修改"

    assert arguments["text"] == "Agent 开发需要明确工具边界。"


def test_validate_arguments_rejects_missing_required_parameter() -> None:
    """缺少声明为必填的参数时必须拒绝。"""
    registry = make_registry()

    with pytest.raises(ToolRegistryError, match="缺少必填参数"):
        registry.validate_arguments("summarize_text", {})


def test_validate_arguments_rejects_extra_parameter() -> None:
    """模型附加未声明字段时不能通过。"""
    registry = make_registry()

    with pytest.raises(ToolRegistryError, match="不允许额外参数"):
        registry.validate_arguments(
            "summarize_text",
            {
                "text": "测试文本",
                "shell": "rm -rf .",
            },
        )


def test_validate_arguments_rejects_non_dictionary() -> None:
    """工具参数容器不能是任意列表或文本。"""
    registry = make_registry()

    with pytest.raises(ToolRegistryError, match="必须是字典"):
        registry.validate_arguments("summarize_text", ["text"])


@pytest.mark.parametrize(
    ("arguments", "parameter_name"),
    [
        ({"text": 123}, "text"),
        ({"text": "测试", "max_length": True}, "max_length"),
        ({"text": "测试", "temperature": float("inf")}, "temperature"),
        ({"text": "测试", "strict": "true"}, "strict"),
    ],
)
def test_validate_arguments_rejects_wrong_json_type(
    arguments: dict[str, object],
    parameter_name: str,
) -> None:
    """每个参数都必须符合声明的 JSON 类型。"""
    registry = make_registry()

    with pytest.raises(ToolRegistryError, match=parameter_name):
        registry.validate_arguments("summarize_text", arguments)


def test_tool_definition_rejects_unsupported_parameter_type() -> None:
    """当前学习版不应假装支持尚未实现的复杂 JSON 类型。"""
    with pytest.raises(ToolRegistryError, match="暂不支持"):
        ToolDefinition(
            name="invalid_tool",
            description="错误定义。",
            parameters_schema={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "暂不支持。",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            risk_level=ToolRiskLevel.READ_ONLY,
        )