"""Skill discovery and management."""

from __future__ import annotations
from logging import getLogger
from pathlib import Path
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from agent13.config import Config

from agent13.skills.models import SkillInfo, SkillMetadata
from agent13.skills.parser import parse_frontmatter
from agent13.config_paths import get_skills_dir

logger = getLogger(__name__)

# Default search paths
PROJECT_SKILLS_DIR = Path.cwd() / ".agent13" / "skills"
GLOBAL_SKILLS_DIR = get_skills_dir()

# Default skills bundled with the package
DEFAULT_SKILLS_DIR = Path(__file__).parent.parent / "default_skills"


def ensure_default_skills() -> None:
    """Copy default skills to user's skills directory if it's empty.

    This provides starter skills for new users.
    """
    # Check if user's skills directory exists and has content
    if GLOBAL_SKILLS_DIR.exists():
        existing_skills = list(GLOBAL_SKILLS_DIR.glob("*/SKILL.md"))
        if existing_skills:
            return  # User already has skills

    # Check if we have default skills to copy
    if not DEFAULT_SKILLS_DIR.exists():
        return

    # Create user's skills directory
    GLOBAL_SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    # Copy each skill directory
    for skill_dir in DEFAULT_SKILLS_DIR.iterdir():
        if skill_dir.is_dir():
            dest = GLOBAL_SKILLS_DIR / skill_dir.name
            if not dest.exists():
                try:
                    import shutil

                    shutil.copytree(skill_dir, dest)
                    logger.info("Copied default skill: %s", skill_dir.name)
                except OSError as e:
                    logger.warning(
                        "Failed to copy default skill %s: %s", skill_dir.name, e
                    )


# Files/directories to exclude from bundled files listing
EXCLUDED_FILES = {
    # VCS
    ".git",
    ".gitignore",
    ".gitmodules",
    ".gitattributes",
    # IDE/Editor
    ".idea",
    ".vscode",
    ".sublime-*",
    # OS files
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    # Python
    "__pycache__",
    ".pytest_cache",
    # Node
    "node_modules",
    "package-lock.json",
    # Misc
    ".env",
}


def _should_exclude(path: Path) -> bool:
    """Check if a file/directory should be excluded from listing."""
    name = path.name
    # Check exact matches
    if name in EXCLUDED_FILES:
        return True
    # Check common patterns
    if name.startswith(".") and name.endswith(".pyc"):
        return True
    if name.endswith(".pyc") or name.endswith(".pyo"):
        return True
    if name.endswith(".log") or name.endswith(".tmp"):
        return True
    if name.endswith(".egg-info"):
        return True
    if name.startswith(".env"):
        return True
    return False


class SkillManager:
    """Discovers and manages available skills."""

    def __init__(self, config_getter: Callable[[], "Config"]) -> None:
        self._config_getter = config_getter
        self._search_paths = self._compute_search_paths()
        self._skills: dict[str, SkillInfo] = self._discover_skills()

        if self._skills:
            logger.info(
                "Discovered %d skill(s) from %d path(s)",
                len(self._skills),
                len(self._search_paths),
            )

    @property
    def skills(self) -> dict[str, SkillInfo]:
        """All discovered skills keyed by name."""
        return self._skills

    def get_skill(self, name: str) -> SkillInfo | None:
        """Get a skill by name, or None if not found."""
        return self._skills.get(name)

    def load_skill_content(self, name: str) -> str | None:
        """Load the full SKILL.md content for a skill."""
        skill = self.get_skill(name)
        if not skill:
            return None
        try:
            return skill.skill_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.error("Failed to read skill %s: %s", name, e)
            return None

    def format_skill_content(self, name: str) -> str | None:
        """Format skill content for injection into conversation.

        Returns a formatted string with the skill content and bundled files,
        or None if the skill doesn't exist or can't be loaded.
        """
        info = self.get_skill(name)
        if not info:
            available = ", ".join(sorted(self.skills.keys()))
            return f"Skill '{name}' not found. Available: {available or 'none'}"

        content = self.load_skill_content(name)
        if not content:
            return f"Skill '{name}' found but content could not be loaded."

        # List bundled files (up to 10), excluding junk files
        files = []
        try:
            for entry in sorted(info.skill_dir.rglob("*")):
                if (
                    entry.is_file()
                    and entry.name != "SKILL.md"
                    and not _should_exclude(entry)
                ):
                    files.append(str(entry.relative_to(info.skill_dir)))
                    if len(files) >= 10:
                        break
        except OSError:
            pass

        files_xml = "\n".join(f"<file>{f}</file>" for f in files)

        return f"""<skill_content name="{name}">
# Skill: {name}

{content}

Base directory: {info.skill_dir}
Relative paths in this skill are relative to this directory.

<skill_files>
{files_xml}
</skill_files>
</skill_content>

Briefly acknowledge the skill is loaded. Do not summarize or explain it."""

    def _resolve_skill_path(self, path: Path) -> Path | None:
        """
        Resolve a skill directory path, following symlinks.

        Returns the resolved path if it exists and is a directory, None otherwise.
        Symlinks are followed to their target.
        """
        try:
            # Follow symlinks to their target
            resolved = path.resolve()
            if resolved.is_dir():
                return resolved
        except (OSError, RuntimeError) as e:
            logger.warning("Could not resolve skill path %s: %s", path, e)
        return None

    def _compute_search_paths(self) -> list[Path]:
        """Build ordered list of directories to search for skills."""
        paths: list[Path] = []
        config = self._config_getter()

        # Config-specified paths (highest priority)
        for path in getattr(config, "skill_paths", []):
            resolved = self._resolve_skill_path(path)
            if resolved:
                paths.append(resolved)

        # Project-local skills
        resolved = self._resolve_skill_path(PROJECT_SKILLS_DIR)
        if resolved:
            paths.append(resolved)

        # User-global skills (lowest priority)
        resolved = self._resolve_skill_path(GLOBAL_SKILLS_DIR)
        if resolved:
            paths.append(resolved)

        # Deduplicate while preserving order
        seen: set[Path] = set()
        unique: list[Path] = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                unique.append(p)

        return unique

    def _discover_skills(self) -> dict[str, SkillInfo]:
        """Scan all search paths for skills."""
        skills: dict[str, SkillInfo] = {}

        for base in self._search_paths:
            for name, info in self._discover_in_dir(base).items():
                if name in skills:
                    logger.warning(
                        "Skill '%s' at %s shadowed by earlier discovery at %s",
                        name,
                        info.skill_path,
                        skills[name].skill_path,
                    )
                else:
                    skills[name] = info
        return skills

    def _discover_in_dir(self, base: Path) -> dict[str, SkillInfo]:
        """Find all skills in a single directory."""
        skills: dict[str, SkillInfo] = {}

        try:
            for skill_dir in base.iterdir():
                if not skill_dir.is_dir():
                    continue

                skill_file = skill_dir / "SKILL.md"
                if not skill_file.is_file():
                    continue

                skill_info = self._try_load_skill(skill_file)
                if skill_info:
                    skills[skill_info.name] = skill_info
        except OSError as e:
            logger.debug("Could not read skill directory %s: %s", base, e)

        return skills

    def _try_load_skill(self, skill_path: Path) -> SkillInfo | None:
        """Parse a SKILL.md file, returning None on error."""
        try:
            content = skill_path.read_text(encoding="utf-8")
            frontmatter, _ = parse_frontmatter(content)
            metadata = SkillMetadata.model_validate(frontmatter)

            # Warn if name doesn't match directory
            dir_name = skill_path.parent.name
            if metadata.name != dir_name:
                logger.warning(
                    "Skill name '%s' doesn't match directory '%s' at %s",
                    metadata.name,
                    dir_name,
                    skill_path,
                )

            return SkillInfo.from_metadata(metadata, skill_path)
        except Exception as e:
            logger.warning("Failed to parse skill at %s: %s", skill_path, e)
            return None
