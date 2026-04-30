"""Load a skill's instructions and bundled resources."""

from tools import tool
from agent13.context import skill_manager_ctx


@tool(is_async=True)
async def skill(name: str) -> str:
    """Load a specialized skill by name. Returns instructions and bundled resources.

    Args:
        name: Skill name (e.g., "code-review", "git-workflow")
    """
    sm = skill_manager_ctx.get()
    if sm is None:
        return "Skills system not available."

    result = sm.format_skill_content(name)
    if result is None:
        return f"Skill '{name}' not found."
    return result
