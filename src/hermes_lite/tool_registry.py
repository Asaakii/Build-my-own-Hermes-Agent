"""HermesLite 的工具定义与参数校验注册表。"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import math
import re

from hermes_lite.domain import ToolCall, ToolResult, require_tool_name

_PARAMETER_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_SUPPORTED_JSON_TYPES = frozenset({"string", "integer", "number", "boolean"})


class ToolRegistryError(ValueError):
    """工具声明、查询或参数校验不符合规则时抛出。"""


ToolHandler = Callable[[dict[str, object]], str]


class ToolExecutionError(Exception):
    """工具主动报告可安全展示的执行失败时抛出。"""


class ToolRiskLevel(str, Enum):
    """工具能力的风险等级。"""

    READ_ONLY = "read_only"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


def _require_text(value: object, field_name: str) -> str:
    """验证注册表内部需要的非空文本。"""
    if not isinstance(value, str):
        raise ToolRegistryError(f"{field_name} 必须是文本")

    cleaned_value = value.strip()
    if not cleaned_value:
        raise ToolRegistryError(f"{field_name} 不能为空")

    return cleaned_value


def _require_parameter_name(value: object) -> str:
    """限制参数名，避免模型传入路径或特殊字符形式的字段。"""
    parameter_name = _require_text(value, "parameter_name")

    if not _PARAMETER_NAME_PATTERN.fullmatch(parameter_name):
        raise ToolRegistryError("parameter_name 只能包含小写字母、数字和下划线")

    return parameter_name


def _copy_schema(schema: dict[str, object]) -> dict[str, object]:
    """返回一份可安全交给调用方的独立参数模式副本。"""
    properties = schema["properties"]
    assert isinstance(properties, dict)

    return {
        "type": "object",
        "properties": {
            name: dict(parameter_schema)
            for name, parameter_schema in properties.items()
        },
        "required": list(schema["required"]),
        "additionalProperties": False,
    }


def _validate_schema(schema: object) -> dict[str, object]:
    """验证并规范化当前学习版支持的 JSON 参数模式。"""
    if not isinstance(schema, dict):
        raise ToolRegistryError("parameters_schema 必须是字典")

    if schema.get("type") != "object":
        raise ToolRegistryError("parameters_schema.type 必须是 object")

    raw_properties = schema.get("properties")
    if not isinstance(raw_properties, dict):
        raise ToolRegistryError("parameters_schema.properties 必须是字典")

    normalized_properties: dict[str, dict[str, str]] = {}

    for raw_name, raw_parameter_schema in raw_properties.items():
        parameter_name = _require_parameter_name(raw_name)

        if not isinstance(raw_parameter_schema, dict):
            raise ToolRegistryError(f"参数 {parameter_name} 的模式必须是字典")

        parameter_type = raw_parameter_schema.get("type")
        if parameter_type not in _SUPPORTED_JSON_TYPES:
            raise ToolRegistryError(
                f"参数 {parameter_name} 使用了暂不支持的 JSON 类型"
            )

        description = _require_text(
            raw_parameter_schema.get("description"),
            f"参数 {parameter_name} 的 description",
        )
        normalized_properties[parameter_name] = {
            "type": parameter_type,
            "description": description,
        }

    raw_required = schema.get("required", [])
    if not isinstance(raw_required, list):
        raise ToolRegistryError("parameters_schema.required 必须是列表")

    required: list[str] = []
    for raw_name in raw_required:
        parameter_name = _require_parameter_name(raw_name)

        if parameter_name not in normalized_properties:
            raise ToolRegistryError(
                f"required 参数未在 properties 中声明: {parameter_name}"
            )

        if parameter_name in required:
            raise ToolRegistryError(
                f"required 参数不能重复: {parameter_name}"
            )

        required.append(parameter_name)

    if schema.get("additionalProperties") is not False:
        raise ToolRegistryError("parameters_schema 必须禁止额外参数")

    return {
        "type": "object",
        "properties": normalized_properties,
        "required": required,
        "additionalProperties": False,
    }


def _matches_json_type(value: object, expected_type: str) -> bool:
    """判断 Python 值是否对应当前支持的 JSON 类型。"""
    if expected_type == "string":
        return isinstance(value, str)

    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)

    if expected_type == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

    if expected_type == "boolean":
        return isinstance(value, bool)

    return False


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """一项可被模型看见的受控工具声明。"""

    name: str
    description: str
    parameters_schema: dict[str, object]
    risk_level: ToolRiskLevel
    handler: ToolHandler | None = None

    def __post_init__(self) -> None:
        """验证工具定义，避免不完整声明进入注册表。"""
        object.__setattr__(self, "name", require_tool_name(self.name))
        object.__setattr__(
            self,
            "description",
            _require_text(self.description, "description"),
        )
        object.__setattr__(
            self,
            "parameters_schema",
            _validate_schema(self.parameters_schema),
        )

        if not isinstance(self.risk_level, ToolRiskLevel):
            raise ToolRegistryError("risk_level 必须是 ToolRiskLevel")

        if self.handler is not None and not callable(self.handler):
            raise ToolRegistryError("handler 必须是可调用对象")

    def to_model_definition(self) -> dict[str, object]:
        """转换为模型可见的 OpenAI 兼容工具定义。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _copy_schema(self.parameters_schema),
            },
        }


class ToolRegistry:
    """集中保存工具声明，并校验模型提供的调用参数。"""

    def __init__(self) -> None:
        """创建初始为空的工具注册表。"""
        self._definitions: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        """登记一项工具，不允许名称重复。"""
        if not isinstance(definition, ToolDefinition):
            raise ToolRegistryError("definition 必须是 ToolDefinition")

        if definition.handler is None:
            raise ToolRegistryError(f"工具缺少执行函数: {definition.name}")

        if definition.name in self._definitions:
            raise ToolRegistryError(f"工具已登记: {definition.name}")

        self._definitions[definition.name] = definition

    def get(self, tool_name: object) -> ToolDefinition:
        """按名称取得已登记工具，未知名称会被拒绝。"""
        normalized_name = require_tool_name(tool_name)

        try:
            return self._definitions[normalized_name]
        except KeyError as error:
            raise ToolRegistryError(f"未知工具: {normalized_name}") from error

    def list_model_definitions(self) -> list[dict[str, object]]:
        """导出当前允许模型调用的工具定义。"""
        return [
            definition.to_model_definition()
            for definition in self._definitions.values()
        ]

    def validate_arguments(
        self,
        tool_name: object,
        arguments: object,
    ) -> dict[str, object]:
        """验证工具参数，并返回一份独立副本。"""
        definition = self.get(tool_name)

        if not isinstance(arguments, dict):
            raise ToolRegistryError("工具参数必须是字典")

        schema = definition.parameters_schema
        properties = schema["properties"]
        required = schema["required"]

        assert isinstance(properties, dict)
        assert isinstance(required, list)

        missing_names = [
            parameter_name
            for parameter_name in required
            if parameter_name not in arguments
        ]
        if missing_names:
            raise ToolRegistryError(
                f"缺少必填参数: {', '.join(missing_names)}"
            )

        extra_names = [
            parameter_name
            for parameter_name in arguments
            if parameter_name not in properties
        ]
        if extra_names:
            raise ToolRegistryError(
                f"不允许额外参数: {', '.join(extra_names)}"
            )

        for parameter_name, value in arguments.items():
            parameter_schema = properties[parameter_name]
            assert isinstance(parameter_schema, dict)

            expected_type = parameter_schema["type"]
            assert isinstance(expected_type, str)

            if not _matches_json_type(value, expected_type):
                raise ToolRegistryError(
                    f"参数 {parameter_name} 必须是 {expected_type}"
                )

        return dict(arguments)

    def execute(self, call: ToolCall) -> ToolResult:
        """校验并执行已登记工具，始终返回结构化结果。"""
        if not isinstance(call, ToolCall):
            raise TypeError("call 必须是 ToolCall")

        try:
            definition = self.get(call.tool_name)
            validated_arguments = self.validate_arguments(
                call.tool_name,
                call.arguments,
            )
        except ToolRegistryError as error:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                content=f"工具调用被拒绝: {error}",
                is_error=True,
            )

        handler = definition.handler
        assert handler is not None

        try:
            content = handler(validated_arguments)

            if not isinstance(content, str) or not content.strip():
                raise ToolExecutionError("工具返回了无效内容")
        except ToolExecutionError as error:
            detail = str(error).strip() or "工具主动报告失败"
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                content=f"工具执行失败: {detail}",
                is_error=True,
            )
        except Exception:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                content="工具执行失败：工具内部发生未预期错误",
                is_error=True,
            )

        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            content=content,
            is_error=False,
        )
