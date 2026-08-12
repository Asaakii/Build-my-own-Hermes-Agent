"""HermesLite 的受限 Markdown 技能加载器。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path

from hermes_lite.config import PROJECT_ROOT
from hermes_lite.domain import require_tool_name
from hermes_lite.tool_registry import ToolRegistry, ToolRegistryError


SKILLS_DIRECTORY_NAME = "skills"
SKILL_FILENAME = "SKILL.md"
MAX_SKILL_CHARACTERS = 12_000
_REQUIRED_METADATA_FIELDS = frozenset(
    {"name", "description", "allowed_tools"},
)


class SkillLoadError(ValueError):
    """表示技能名称、路径、元数据或正文不符合受限规则。"""


class SkillPolicyError(SkillLoadError):
    """表示技能声明了当前注册表中不存在的工具。"""


def _require_text(value: object, field_name: str) -> str:
    """验证非空文本并去除首尾空白。"""
    if not isinstance(value, str):
        raise SkillLoadError(f"{field_name} 必须是文本")

    cleaned_value = value.strip()
    if not cleaned_value:
        raise SkillLoadError(f"{field_name} 不能为空")

    return cleaned_value


def _normalize_skill_name(value: object) -> str:
    """复用工具名称格式，拒绝路径片段和特殊字符。"""
    try:
        return require_tool_name(value)
    except ValueError as error:
        raise SkillLoadError(str(error).replace("tool_name", "skill_name")) from error


@dataclass(frozen=True, slots=True)
class Skill:
    """一份只含任务说明和允许工具声明的受限技能。"""

    name: str
    description: str
    allowed_tools: tuple[str, ...]
    instructions: str

    def __post_init__(self) -> None:
        """验证技能只包含可展示的文本和合法工具名称。"""
        object.__setattr__(self, "name", _normalize_skill_name(self.name))
        object.__setattr__(
            self,
            "description",
            _require_text(self.description, "description"),
        )
        object.__setattr__(
            self,
            "instructions",
            _require_text(self.instructions, "instructions"),
        )

        if not isinstance(self.allowed_tools, tuple):
            raise SkillLoadError("allowed_tools 必须是元组")

        normalized_tools = tuple(
            _normalize_skill_name(tool_name)
            for tool_name in self.allowed_tools
        )
        if len(set(normalized_tools)) != len(normalized_tools):
            raise SkillLoadError("allowed_tools 不能重复")

        object.__setattr__(self, "allowed_tools", normalized_tools)

    def allowed_tool_definitions(
        self,
        registry: ToolRegistry,
    ) -> list[dict[str, object]]:
        """从既有注册表取出技能允许的子集，绝不创建新工具。"""
        if not isinstance(registry, ToolRegistry):
            raise SkillPolicyError("registry 必须是 ToolRegistry")

        definitions: list[dict[str, object]] = []
        for tool_name in self.allowed_tools:
            try:
                definitions.append(registry.get(tool_name).to_model_definition())
            except ToolRegistryError as error:
                raise SkillPolicyError(
                    f"技能声明了未登记工具: {tool_name}",
                ) from error

        return definitions


def _split_front_matter(content: str) -> tuple[str, str]:
    """提取以 JSON 表示的固定 Markdown 前置元数据。"""
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillLoadError("技能文件必须以 --- JSON 元数据开始")

    try:
        closing_index = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as error:
        raise SkillLoadError("技能元数据缺少结束分隔符") from error

    metadata_text = "\n".join(lines[1:closing_index]).strip()
    instructions = "\n".join(lines[closing_index + 1 :]).strip()
    return metadata_text, instructions


def _parse_metadata(metadata_text: str) -> tuple[str, str, tuple[str, ...]]:
    """解析严格 JSON 元数据，避免引入通用 YAML 解析和隐式类型。"""
    try:
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError as error:
        raise SkillLoadError("技能元数据必须是有效 JSON") from error

    if not isinstance(metadata, dict):
        raise SkillLoadError("技能元数据必须是 JSON 对象")

    if set(metadata) != _REQUIRED_METADATA_FIELDS:
        raise SkillLoadError("技能元数据字段必须且只能包含 name、description、allowed_tools")

    name = _normalize_skill_name(metadata["name"])
    description = _require_text(metadata["description"], "description")
    raw_allowed_tools = metadata["allowed_tools"]
    if not isinstance(raw_allowed_tools, list):
        raise SkillLoadError("allowed_tools 必须是列表")

    allowed_tools = tuple(
        _normalize_skill_name(tool_name)
        for tool_name in raw_allowed_tools
    )
    if len(set(allowed_tools)) != len(allowed_tools):
        raise SkillLoadError("allowed_tools 不能重复")

    return name, description, allowed_tools


def _get_skill_path(skill_name: str) -> Path:
    """返回已解析并确认仍位于项目技能目录的 Markdown 路径。"""
    skills_root = (PROJECT_ROOT / SKILLS_DIRECTORY_NAME).resolve()
    candidate_path = (skills_root / skill_name / SKILL_FILENAME).resolve()

    try:
        candidate_path.relative_to(skills_root)
    except ValueError as error:
        raise SkillLoadError("技能路径不能逃离 skills 目录") from error

    return candidate_path


def load_skill(value: object) -> Skill:
    """从项目受限技能目录读取并验证一份 Markdown 技能。"""
    skill_name = _normalize_skill_name(value)
    skill_path = _get_skill_path(skill_name)

    if not skill_path.is_file():
        raise SkillLoadError(f"技能不存在: {skill_name}")

    try:
        content = skill_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise SkillLoadError("技能文件必须是 UTF-8 文本") from error
    except OSError as error:
        raise SkillLoadError("无法读取技能文件") from error

    if len(content) > MAX_SKILL_CHARACTERS:
        raise SkillLoadError(f"技能文件不能超过 {MAX_SKILL_CHARACTERS} 个字符")

    metadata_text, instructions = _split_front_matter(content)
    metadata_name, description, allowed_tools = _parse_metadata(metadata_text)

    if metadata_name != skill_name:
        raise SkillLoadError("技能目录名称与元数据 name 不一致")

    return Skill(
        name=metadata_name,
        description=description,
        allowed_tools=allowed_tools,
        instructions=instructions,
    )



def list_available_skills() -> tuple[Skill, ...]:
    """加载项目技能目录中的全部直接子目录技能，供只读 CLI 展示。"""
    skills_root = (PROJECT_ROOT / SKILLS_DIRECTORY_NAME).resolve()

    if not skills_root.exists():
        return ()

    if not skills_root.is_dir():
        raise SkillLoadError("skills 路径不是目录")

    skills: list[Skill] = []
    try:
        directories = sorted(
            path for path in skills_root.iterdir() if path.is_dir()
        )
    except OSError as error:
        raise SkillLoadError("无法列出技能目录") from error

    for directory in directories:
        skills.append(load_skill(directory.name))

    return tuple(skills)
