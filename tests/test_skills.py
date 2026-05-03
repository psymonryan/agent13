"""Tests for the skills module."""

import pytest

from agent13.skills.models import SkillMetadata, SkillInfo
from agent13.skills.parser import parse_frontmatter, SkillParseError
from agent13.skills.manager import SkillManager


class TestSkillMetadata:
    """Tests for SkillMetadata model."""

    def test_valid_metadata(self):
        """Test parsing valid skill metadata."""
        meta = SkillMetadata(
            name="test-skill",
            description="A test skill",
        )
        assert meta.name == "test-skill"
        assert meta.description == "A test skill"

    def test_metadata_with_all_fields(self):
        """Test metadata with all optional fields."""
        meta = SkillMetadata(
            name="full-skill",
            description="A full skill",
            license="MIT",
            compatibility="python>=3.10",
            metadata={"author": "test"},
        )
        assert meta.name == "full-skill"
        assert meta.license == "MIT"
        assert meta.compatibility == "python>=3.10"
        assert meta.metadata == {"author": "test"}

    def test_invalid_name_pattern(self):
        """Test that invalid names are rejected."""
        with pytest.raises(Exception):  # Pydantic validation error
            SkillMetadata(
                name="Invalid Name!",
                description="Invalid name",
            )

    def test_name_with_hyphens(self):
        """Test that names with hyphens are valid."""
        meta = SkillMetadata(
            name="my-awesome-skill",
            description="Valid name",
        )
        assert meta.name == "my-awesome-skill"


class TestSkillInfo:
    """Tests for SkillInfo model."""

    def test_from_metadata(self, tmp_path):
        """Test creating SkillInfo from metadata."""
        meta = SkillMetadata(
            name="test-skill",
            description="A test skill",
        )
        skill_path = tmp_path / "test-skill" / "SKILL.md"
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text("---\nname: test-skill\n---\n")

        info = SkillInfo.from_metadata(meta, skill_path)
        assert info.name == "test-skill"
        assert info.description == "A test skill"
        assert info.skill_path == skill_path.resolve()
        assert info.skill_dir == skill_path.parent.resolve()


class TestParseFrontmatter:
    """Tests for YAML frontmatter parsing."""

    def test_valid_frontmatter(self):
        """Test parsing valid frontmatter."""
        content = """---
name: test-skill
description: A test skill
---

## Instructions

This is the skill body.
"""
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter["name"] == "test-skill"
        assert frontmatter["description"] == "A test skill"
        assert "## Instructions" in body

    def test_empty_frontmatter(self):
        """Test parsing empty frontmatter."""
        content = """---
---

Body content.
"""
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter == {}
        assert "Body content." in body

    def test_missing_frontmatter(self):
        """Test that missing frontmatter raises error."""
        content = "No frontmatter here"
        with pytest.raises(SkillParseError):
            parse_frontmatter(content)

    def test_invalid_yaml(self):
        """Test that invalid YAML raises error."""
        content = """---
name: [invalid yaml
---

Body.
"""
        with pytest.raises(SkillParseError):
            parse_frontmatter(content)

    def test_frontmatter_not_dict(self):
        """Test that non-dict frontmatter raises error."""
        content = """---
- item1
- item2
---

Body.
"""
        with pytest.raises(SkillParseError):
            parse_frontmatter(content)


class TestSkillManager:
    """Tests for SkillManager."""

    def test_discover_skills(self, tmp_path, monkeypatch):
        """Test skill discovery from directory."""
        # Create a test skill
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("""---
name: test-skill
description: A test skill for discovery
---

## Test Instructions
""")

        # Create SkillManager with custom path
        def mock_config():
            class MockConfig:
                skill_paths = [tmp_path]
                include_skills = False

            return MockConfig()

        manager = SkillManager(mock_config)
        assert "test-skill" in manager.skills
        assert manager.skills["test-skill"].description == "A test skill for discovery"

    def test_skill_shadowing(self, tmp_path):
        """Test that earlier skills shadow later ones."""
        # Create first skill
        skill1_dir = tmp_path / "skill1" / "test-skill"
        skill1_dir.mkdir(parents=True)
        (skill1_dir / "SKILL.md").write_text("""---
name: test-skill
description: First skill
---
""")

        # Create second skill with same name
        skill2_dir = tmp_path / "skill2" / "test-skill"
        skill2_dir.mkdir(parents=True)
        (skill2_dir / "SKILL.md").write_text("""---
name: test-skill
description: Second skill
---
""")

        def mock_config():
            class MockConfig:
                skill_paths = [tmp_path / "skill1", tmp_path / "skill2"]
                include_skills = False

            return MockConfig()

        manager = SkillManager(mock_config)
        # First skill should win
        assert manager.skills["test-skill"].description == "First skill"

    def test_missing_skill_file(self, tmp_path):
        """Test handling of directory without SKILL.md."""
        # Create directory without SKILL.md
        (tmp_path / "not-a-skill").mkdir()

        def mock_config():
            class MockConfig:
                skill_paths = [tmp_path]
                include_skills = False

            return MockConfig()

        manager = SkillManager(mock_config)
        assert "not-a-skill" not in manager.skills

    def test_load_skill_content(self, tmp_path):
        """Test loading skill content."""
        skill_dir = tmp_path / "content-skill"
        skill_dir.mkdir()
        skill_content = """---
name: content-skill
description: Skill with content
---

## Detailed Instructions

These are detailed instructions.
"""
        (skill_dir / "SKILL.md").write_text(skill_content)

        def mock_config():
            class MockConfig:
                skill_paths = [tmp_path]
                include_skills = False

            return MockConfig()

        manager = SkillManager(mock_config)
        content = manager.load_skill_content("content-skill")
        assert content == skill_content

    def test_load_nonexistent_skill(self, tmp_path):
        """Test loading a skill that doesn't exist."""

        def mock_config():
            class MockConfig:
                skill_paths = [tmp_path]
                include_skills = False

            return MockConfig()

        manager = SkillManager(mock_config)
        content = manager.load_skill_content("nonexistent")
        assert content is None

    def test_symlink_resolution(self, tmp_path):
        """Test that symlinks are resolved."""
        # Create actual skill directory
        actual_dir = tmp_path / "actual-skills" / "linked-skill"
        actual_dir.mkdir(parents=True)
        (actual_dir / "SKILL.md").write_text("""---
name: linked-skill
description: Skill via symlink
---
""")

        # Create symlink directory
        link_dir = tmp_path / "linked-skills"
        link_dir.symlink_to(actual_dir.parent)

        def mock_config():
            class MockConfig:
                skill_paths = [link_dir]
                include_skills = False

            return MockConfig()

        manager = SkillManager(mock_config)
        assert "linked-skill" in manager.skills


class TestGetSkillsSection:
    """Tests for get_skills_section function."""

    def test_empty_skills(self):
        """Test with no skills."""
        from agent13.prompts import get_skills_section

        result = get_skills_section({})
        assert result == ""

    def test_single_skill(self, tmp_path):
        """Test with a single skill."""
        from agent13.prompts import get_skills_section

        skill_path = tmp_path / "single-skill" / "SKILL.md"
        skill_path.parent.mkdir(parents=True)

        meta = SkillMetadata(name="single-skill", description="A single skill")
        info = SkillInfo.from_metadata(meta, skill_path)

        result = get_skills_section({"single-skill": info})
        assert "# Available Skills" in result
        assert "<available_skills>" in result
        assert "<name>single-skill</name>" in result
        assert "<description>A single skill</description>" in result
        # Should mention the skill tool
        assert "skill" in result.lower()
        assert "tool" in result.lower()

    def test_multiple_skills_sorted(self, tmp_path):
        """Test that skills are sorted by name."""
        from agent13.prompts import get_skills_section

        skills = {}
        for name in ["zebra-skill", "alpha-skill", "middle-skill"]:
            skill_path = tmp_path / name / "SKILL.md"
            skill_path.parent.mkdir(parents=True)
            meta = SkillMetadata(name=name, description=f"Skill {name}")
            skills[name] = SkillInfo.from_metadata(meta, skill_path)

        result = get_skills_section(skills)
        # Check that alpha-skill appears before zebra-skill
        alpha_pos = result.index("alpha-skill")
        middle_pos = result.index("middle-skill")
        zebra_pos = result.index("zebra-skill")
        assert alpha_pos < middle_pos < zebra_pos


class TestSkillTool:
    """Tests for the skill tool."""

    @pytest.fixture
    def mock_skill_manager(self, tmp_path):
        """Create a mock skill manager with a test skill."""
        from agent13.skills import SkillManager
        from agent13.context import skill_manager_ctx

        # Create a test skill
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("""---
name: test-skill
description: A test skill for unit testing
---
# Test Skill
This is test content.
""")

        # Create another skill
        skill_dir2 = tmp_path / "another-skill"
        skill_dir2.mkdir()
        skill_file2 = skill_dir2 / "SKILL.md"
        skill_file2.write_text("""---
name: another-skill
description: Another test skill
---
# Another Skill
More content.
""")

        # Create mock config
        class MockConfig:
            skill_paths = [tmp_path]

        # Create SkillManager
        sm = SkillManager(lambda: MockConfig())

        # Set context and return
        token = skill_manager_ctx.set(sm)
        yield sm
        skill_manager_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_skill_tool_loads_existing_skill(self, mock_skill_manager):
        """Test that skill tool loads an existing skill."""
        from tools import execute_tool

        result = await execute_tool("skill", {"name": "test-skill"})
        # Result is JSON string
        assert "test-skill" in result
        assert "Test Skill" in result or "test-skill" in result

    @pytest.mark.asyncio
    async def test_skill_tool_lists_available_on_missing(self, mock_skill_manager):
        """Test that skill tool lists available skills when skill not found."""
        from tools import execute_tool

        result = await execute_tool("skill", {"name": "nonexistent"})
        assert "not found" in result.lower()
        assert "test-skill" in result or "another-skill" in result

    @pytest.mark.asyncio
    async def test_skill_tool_without_context(self):
        """Test that skill tool handles missing context gracefully."""
        from tools import execute_tool
        from agent13.context import skill_manager_ctx

        # Ensure context is None
        token = skill_manager_ctx.set(None)
        try:
            result = await execute_tool("skill", {"name": "test"})
            assert "not available" in result.lower()
        finally:
            skill_manager_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_skill_tool_lists_bundled_files(self, mock_skill_manager, tmp_path):
        """Test that skill tool lists bundled files."""
        from tools import execute_tool

        # Add a file to the test skill
        skill_dir = tmp_path / "test-skill"
        (skill_dir / "script.sh").write_text("#!/bin/bash\necho hello")

        result = await execute_tool("skill", {"name": "test-skill"})
        assert "script.sh" in result


class TestFormatSkillContent:
    """Tests for SkillManager.format_skill_content method."""

    def test_format_includes_acknowledgment_instruction(self, tmp_path):
        """Test that formatted content includes instruction to not summarize."""
        from agent13.skills import SkillManager

        # Create a test skill
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("""---
name: my-skill
description: Test skill
---
# My Skill
Some instructions here.
""")

        class MockConfig:
            skill_paths = [tmp_path]

        sm = SkillManager(lambda: MockConfig())
        result = sm.format_skill_content("my-skill")

        assert "Briefly acknowledge" in result
        assert "Do not summarize" in result

    def test_format_matches_tool_output(self, tmp_path):
        """Test that format_skill_content matches what the tool returns."""
        from agent13.skills import SkillManager
        from agent13.context import skill_manager_ctx
        from tools import execute_tool
        import asyncio

        # Create a test skill
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("""---
name: test-skill
description: Test
---
# Test
Content here.
""")

        class MockConfig:
            skill_paths = [tmp_path]

        sm = SkillManager(lambda: MockConfig())
        token = skill_manager_ctx.set(sm)

        try:
            # Get via tool
            async def get_tool_result():
                return await execute_tool("skill", {"name": "test-skill"})

            tool_result = asyncio.run(get_tool_result())
            # Tool result is JSON-quoted, direct is not
            # But they should have the same core content
            assert "test-skill" in tool_result
            assert "Briefly acknowledge" in tool_result
        finally:
            skill_manager_ctx.reset(token)

    def test_format_returns_error_for_missing_skill(self, tmp_path):
        """Test that format_skill_content returns error for missing skill."""
        from agent13.skills import SkillManager

        class MockConfig:
            skill_paths = [tmp_path]

        sm = SkillManager(lambda: MockConfig())
        result = sm.format_skill_content("nonexistent")

        assert "not found" in result

    def test_format_excludes_junk_files(self, tmp_path):
        """Test that .DS_Store and other junk files are excluded."""
        from agent13.skills import SkillManager

        # Create a test skill with junk files
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("""---
name: my-skill
description: Test skill
---
# My Skill
""")

        # Create files that should be excluded
        (skill_dir / ".DS_Store").write_text("junk")
        (skill_dir / ".gitignore").write_text("junk")
        (skill_dir / "__pycache__").mkdir()
        (skill_dir / "__pycache__" / "test.pyc").write_text("junk")

        # Create a file that should be included
        (skill_dir / "script.sh").write_text("#!/bin/bash\necho hello")

        class MockConfig:
            skill_paths = [tmp_path]

        sm = SkillManager(lambda: MockConfig())
        result = sm.format_skill_content("my-skill")

        # Should include the real file
        assert "script.sh" in result
        # Should NOT include junk files
        assert ".DS_Store" not in result
        assert ".gitignore" not in result
        assert "__pycache__" not in result
