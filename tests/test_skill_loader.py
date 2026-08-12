"""验证受限 Markdown 技能的加载与工具权限边界。"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_lite.skill_loader as skill_loader_module
from hermes_lite.skill_loader import SkillLoadError, SkillPolicyError, load_skill
from hermes_lite.tool_registry import ToolDefinition, ToolRegistry, ToolRiskLevel


VALID_METADATA = (
    "{\"name\":\"fix_failing_test\","
    "\"description\":\"修复已有失败测试。\","
    "\"allowed_tools\":[\"read_file\",\"run_pytest\"]}"
)


@pytest.fixture
def skills_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    """创建独立项目技能目录，避免测试读取真实项目文件。"""
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(skill_loader_module, "PROJECT_ROOT", project_root)
    root = project_root / "skills"
    root.mkdir()
    return root


def write_skill(
    skills_root: Path,
    name: str,
    metadata: str = VALID_METADATA,
    instructions: str = "# 步骤\n\n读取失败测试后做最小修改。",
) -> Path:
    """写入一份测试技能 Markdown，并返回文件路径。"""
    skill_directory = skills_root / name
    skill_directory.mkdir()
    skill_path = skill_directory / "SKILL.md"
    skill_path.write_text(
        f"---\n{metadata}\n---\n{instructions}\n",
        encoding="utf-8",
    )
    return skill_path


def make_registry(*tool_names: str) -> ToolRegistry:
    """创建只登记指定名称工具的注册表。"""
    registry = ToolRegistry()
    for tool_name in tool_names:
        registry.register(
            ToolDefinition(
                name=tool_name,
                description=f"{tool_name} 的测试工具。",
                parameters_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                risk_level=ToolRiskLevel.READ_ONLY,
                handler=lambda arguments: "完成",
            )
        )
    return registry


def test_load_skill_returns_validated_metadata_and_instructions(
    skills_root: Path,
) -> None:
    """正常技能应按元数据和 Markdown 正文被完整加载。"""
    skill_path = write_skill(skills_root, "fix_failing_test")
    original_content = skill_path.read_text(encoding="utf-8")

    skill = load_skill("fix_failing_test")

    assert skill.name == "fix_failing_test"
    assert skill.description == "修复已有失败测试。"
    assert skill.allowed_tools == ("read_file", "run_pytest")
    assert "最小修改" in skill.instructions
    assert skill_path.read_text(encoding="utf-8") == original_content


@pytest.mark.parametrize("skill_name", ["../outside", "bad-name", "UpperCase"])
def test_load_skill_rejects_invalid_name_before_path_access(
    skills_root: Path,
    skill_name: str,
) -> None:
    """名称非法时不能把用户输入当作路径处理。"""
    del skills_root

    with pytest.raises(SkillLoadError, match="skill_name"):
        load_skill(skill_name)


def test_load_skill_rejects_missing_skill(skills_root: Path) -> None:
    """缺失技能必须明确报错，不能创建目录或回退到其他文件。"""
    del skills_root

    with pytest.raises(SkillLoadError, match="技能不存在"):
        load_skill("missing_skill")


@pytest.mark.parametrize(
    "metadata",
    [
        "not-json",
        "[]",
        "{\"name\":\"other_skill\",\"description\":\"说明\",\"allowed_tools\":[]}",
        "{\"name\":\"fix_failing_test\",\"description\":\"说明\",\"allowed_tools\":[\"read_file\",\"read_file\"]}",
    ],
)
def test_load_skill_rejects_invalid_metadata(
    skills_root: Path,
    metadata: str,
) -> None:
    """元数据结构、名称一致性和工具重复都必须在加载时拒绝；空工具列表可作为零权限技能。"""
    write_skill(skills_root, "fix_failing_test", metadata=metadata)

    with pytest.raises(SkillLoadError):
        load_skill("fix_failing_test")


def test_load_skill_rejects_symlink_that_escapes_skills_directory(
    skills_root: Path,
    tmp_path: Path,
) -> None:
    """即使名称合法，外部符号链接也不能把技能读取带出项目目录。"""
    outside_directory = tmp_path / "outside_skill"
    outside_directory.mkdir()
    (outside_directory / "SKILL.md").write_text(
        f"---\n{VALID_METADATA}\n---\n# 外部技能\n",
        encoding="utf-8",
    )
    (skills_root / "fix_failing_test").symlink_to(
        outside_directory,
        target_is_directory=True,
    )

    with pytest.raises(SkillLoadError, match="不能逃离"):
        load_skill("fix_failing_test")


def test_skill_only_selects_registered_tool_subset(skills_root: Path) -> None:
    """技能允许工具只能从已有注册表选择，不能增加额外能力。"""
    write_skill(skills_root, "fix_failing_test")
    skill = load_skill("fix_failing_test")
    registry = make_registry("read_file", "run_pytest", "create_text_file")

    definitions = skill.allowed_tool_definitions(registry)

    assert [definition["function"]["name"] for definition in definitions] == [
        "read_file",
        "run_pytest",
    ]


def test_skill_rejects_unregistered_tool_declaration(skills_root: Path) -> None:
    """技能声明未知工具时不能获得新的执行能力。"""
    metadata = (
        "{\"name\":\"fix_failing_test\","
        "\"description\":\"修复已有失败测试。\","
        "\"allowed_tools\":[\"read_file\",\"delete_all_files\"]}"
    )
    write_skill(skills_root, "fix_failing_test", metadata=metadata)
    skill = load_skill("fix_failing_test")
    registry = make_registry("read_file")

    with pytest.raises(SkillPolicyError, match="未登记工具"):
        skill.allowed_tool_definitions(registry)



def test_list_available_skills_loads_directories_in_name_order(
    skills_root: Path,
) -> None:
    """技能列表应复用严格加载逻辑，并按目录名称稳定排序。"""
    write_skill(
        skills_root,
        "zeta_skill",
        metadata=(
            "{\"name\":\"zeta_skill\",\"description\":\"Z 技能。\","
            "\"allowed_tools\":[]}"
        ),
    )
    write_skill(
        skills_root,
        "alpha_skill",
        metadata=(
            "{\"name\":\"alpha_skill\",\"description\":\"A 技能。\","
            "\"allowed_tools\":[]}"
        ),
    )

    skills = skill_loader_module.list_available_skills()

    assert [skill.name for skill in skills] == ["alpha_skill", "zeta_skill"]
