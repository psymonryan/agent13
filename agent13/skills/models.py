"""Data models for skills."""

from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, field_validator


class SkillMetadata(BaseModel):
    """Parsed YAML frontmatter from SKILL.md."""

    model_config = {"populate_by_name": True}

    name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$",
        description="Skill identifier. Lowercase letters, numbers, hyphens.",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="What this skill does and when to use it.",
    )
    license: str | None = Field(default=None)
    compatibility: str | None = Field(default=None, max_length=500)
    metadata: dict[str, Any] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(
        default_factory=list,
        validation_alias="allowed-tools",
    )
    user_invocable: bool = Field(
        default=True,
        validation_alias="user-invocable",
    )

    @field_validator("allowed_tools", mode="before")
    @classmethod
    def parse_allowed_tools(cls, v: str | list[str] | None) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return v.split()
        return list(v)


class SkillInfo(BaseModel):
    """Complete skill information including filesystem path."""

    name: str
    description: str
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)
    user_invocable: bool = True
    skill_path: Path  # Path to SKILL.md file

    model_config = {"arbitrary_types_allowed": True}

    @property
    def skill_dir(self) -> Path:
        """Root directory of the skill (parent of SKILL.md)."""
        return self.skill_path.parent.resolve()

    @classmethod
    def from_metadata(cls, meta: SkillMetadata, skill_path: Path) -> "SkillInfo":
        return cls(
            name=meta.name,
            description=meta.description,
            license=meta.license,
            compatibility=meta.compatibility,
            metadata=meta.metadata,
            allowed_tools=meta.allowed_tools,
            user_invocable=meta.user_invocable,
            skill_path=skill_path.resolve(),
        )
