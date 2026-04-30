"""Async context for sharing state with tools."""

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent13.skills import SkillManager

skill_manager_ctx: ContextVar["SkillManager | None"] = ContextVar(
    "skill_manager", default=None
)
