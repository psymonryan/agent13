"""Default-skills sync: hash-manifest tracking + /skills refresh.

The package bundles starter skills (agent13/default_skills/). This module
keeps a user's installed copies in sync with the bundle WITHOUT ever
clobbering local edits:

- A manifest (``.skills-defaults.json`` at the TOP LEVEL of the config dir,
  not inside ``skills/``) records what WE shipped:
  ``{name: {"hash": <dir sha256>, "version": <frontmatter version>}}``.
- ``sync_default_skills()`` (run at startup, and from ``/skills refresh``):
    - copies in bundled skills that are not installed, even when the user
      already has other skills (the old seed-once-into-empty-dir rule is
      gone - that rule meant existing users never received new defaults),
    - overwrites installed skills whose bytes still match the manifest when
      the bundled copy changed (the hash is the trigger, so a content fix
      ships even without a version bump; the version is display only),
    - leaves anything the user modified alone and reports it,
    - adopts pre-manifest installs into the manifest only when they are
      byte-identical to the current bundle; anything else stays
      user-managed and is reported the same way as a local edit.
- ``/skills refresh`` re-runs the sync from the front-end; ``-y`` overwrites
  locally-modified skills too (same convention as ``/save <name> -y``).

The sync RETURNS a report instead of printing: it runs in cli.py before any
front-end exists, so the caller decides the surface (REPL print, TUI notify,
batch = the debug log the module logger already writes).
"""

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from agent13.config_paths import get_config_dir, get_skills_dir
from agent13.skills.parser import parse_frontmatter

logger = logging.getLogger(__name__)

# Bundled starter skills shipped with the package (one home; manager.py
# re-exports this name for compatibility).
DEFAULT_SKILLS_DIR = Path(__file__).parent.parent / "default_skills"

# Manifest lives at the top level of the config dir, NOT inside skills/ -
# the skills dir stays purely user content.
MANIFEST_NAME = ".skills-defaults.json"

# Platform noise that must not change a skill's identity hash.
_HASH_EXCLUDE_DIRS = {"__pycache__", ".git", ".pytest_cache", "node_modules"}
_HASH_EXCLUDE_FILES = {".DS_Store", "Thumbs.db", "desktop.ini"}


@dataclass
class DefaultSkillsReport:
    """What a sync did (or would have done). Empty means "nothing to say"."""

    copied: list[str] = field(default_factory=list)
    # (name, old_version, new_version) - versions may be "" (not in frontmatter).
    updated: list[tuple[str, str, str]] = field(default_factory=list)
    skipped_modified: list[str] = field(default_factory=list)
    adopted: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(
            self.copied or self.updated or self.skipped_modified or self.adopted
        )

    def render(self) -> str:
        """User-facing text; "" when nothing happened."""
        lines: list[str] = []
        if self.copied:
            lines.append(f"New default skills installed: {', '.join(self.copied)}")
        if self.updated:
            parts = [
                f"{name} ({old or '?'} -> {new})" for name, old, new in self.updated
            ]
            lines.append(f"Default skills updated: {', '.join(parts)}")
        if self.skipped_modified:
            lines.append(
                "NOT updated (local changes): "
                f"{', '.join(self.skipped_modified)}"
                " - use /skills refresh -y to overwrite"
            )
        if self.adopted:
            lines.append(f"Now tracking {len(self.adopted)} existing default skill(s)")
        return "\n".join(lines)


@dataclass(frozen=True)
class SyncPaths:
    """The three locations the sync reads/writes. Explicit so tests can
    point every part at a tmp dir without monkeypatching module constants."""

    skills_dir: Path  # user's global skills dir
    bundled_dir: Path  # package defaults
    manifest_path: Path


def default_paths() -> SyncPaths:
    return SyncPaths(
        skills_dir=get_skills_dir(),
        bundled_dir=DEFAULT_SKILLS_DIR,
        manifest_path=get_config_dir() / MANIFEST_NAME,
    )


def dir_hash(path: Path) -> str:
    """Deterministic sha256 over a directory tree (relative paths + bytes)."""
    h = hashlib.sha256()
    for f in sorted(_hash_files(path)):
        h.update(f.relative_to(path).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _hash_files(root: Path) -> Iterator[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in _HASH_EXCLUDE_DIRS for part in rel.parts[:-1]):
            continue
        if p.name in _HASH_EXCLUDE_FILES:
            continue
        yield p


def read_version(skill_dir: Path) -> str:
    """Best-effort version from SKILL.md frontmatter (metadata.version, then
    version). Display only - the hash decides updates, so a missing version
    is never an error."""
    try:
        content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        fm, _ = parse_frontmatter(content)
    except Exception:  # noqa: BLE001 - unreadable/unparseable = no version
        return ""
    meta = fm.get("metadata")
    if isinstance(meta, dict) and isinstance(meta.get("version"), str):
        return meta["version"]
    return fm["version"] if isinstance(fm.get("version"), str) else ""


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Read the manifest; missing or corrupt reads as empty (legacy rules
    then apply - adopt only what is byte-identical to the bundle)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.warning("Skills manifest %s unreadable (%s); treating as empty", path, e)
        return {}
    if not isinstance(data, dict):
        logger.warning("Skills manifest %s is not an object; treating as empty", path)
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def save_manifest(path: Path, manifest: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def plan_sync(
    bundled: dict[str, dict],
    manifest: dict[str, dict[str, str]],
    installed: dict[str, str],
) -> dict[str, str]:
    """Pure decision: bundled skill name -> action.

    copy          not installed - install it
    update        installed bytes match the manifest, bundled copy changed
    skip_modified installed bytes differ from the manifest (user edits, or a
                  pre-manifest install from an older release)
    adopt         pre-manifest install byte-identical to the bundled copy -
                  record it, change nothing
    ok            installed, tracked, in sync
    """
    actions: dict[str, str] = {}
    for name, b in bundled.items():
        bh = b["hash"]
        if name not in installed:
            actions[name] = "copy"
            continue
        entry = manifest.get(name)
        if entry is None:
            actions[name] = "adopt" if installed[name] == bh else "skip_modified"
        elif installed[name] != entry.get("hash"):
            actions[name] = "skip_modified"
        elif entry.get("hash") != bh:
            actions[name] = "update"
        else:
            actions[name] = "ok"
    return actions


def sync_default_skills(
    paths: SyncPaths | None = None, force: bool = False
) -> DefaultSkillsReport:
    """Bring the user's default skills in line with the bundle.

    force (the /skills refresh -y path) also overwrites locally-modified
    skills. Never touches skills that are not bundled defaults, and never
    deletes anything (a removed default stays installed, untracked).
    """
    p = paths or default_paths()
    report = DefaultSkillsReport()
    bundled = _scan_bundled(p.bundled_dir)
    if not bundled:
        return report

    manifest = load_manifest(p.manifest_path)
    installed = {
        name: dir_hash(p.skills_dir / name)
        for name in bundled
        if (p.skills_dir / name).is_dir()
    }
    actions = plan_sync(bundled, manifest, installed)

    changed = False
    for name in sorted(actions):
        action = actions[name]
        if action == "skip_modified" and force:
            action = "update"  # -y: the user chose to overwrite local edits
        b = bundled[name]
        dest = p.skills_dir / name

        if action == "copy":
            if _copy_tree(b["path"], dest):
                manifest[name] = {"hash": b["hash"], "version": b["version"]}
                report.copied.append(name)
                logger.info("Installed default skill: %s", name)
        elif action == "update":
            old_version = manifest.get(name, {}).get("version", "")
            if _replace_tree(b["path"], dest):
                manifest[name] = {"hash": b["hash"], "version": b["version"]}
                report.updated.append((name, old_version, b["version"]))
                logger.info(
                    "Updated default skill: %s (%s -> %s)",
                    name,
                    old_version or "?",
                    b["version"] or "?",
                )
        elif action == "adopt":
            manifest[name] = {"hash": b["hash"], "version": b["version"]}
            report.adopted.append(name)
            logger.info("Tracking existing default skill: %s", name)
        elif action == "skip_modified":
            report.skipped_modified.append(name)
            logger.info(
                "Default skill %s has local changes; not updated "
                "(use /skills refresh -y to overwrite)",
                name,
            )
        # "ok": nothing to do, manifest entry stays as-is.

        if action in ("copy", "update", "adopt"):
            changed = True

    if changed:
        save_manifest(p.manifest_path, manifest)
    return report


def _scan_bundled(bundled_dir: Path) -> dict[str, dict]:
    """name -> {hash, version, path} for every bundled skill (has SKILL.md)."""
    out: dict[str, dict] = {}
    try:
        entries = sorted(bundled_dir.iterdir())
    except OSError:
        return out
    for d in entries:
        if d.is_dir() and (d / "SKILL.md").is_file():
            out[d.name] = {
                "hash": dir_hash(d),
                "version": read_version(d),
                "path": d,
            }
    return out


def _copy_tree(src: Path, dest: Path) -> bool:
    """copytree with the error handling the report depends on: a failure is
    logged and reported as NOT installed rather than raising out of startup."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest)
        return True
    except OSError as e:
        logger.warning("Failed to install default skill %s: %s", src.name, e)
        return False


def _replace_tree(src: Path, dest: Path) -> bool:
    """Update an installed skill: wipe the old copy, copy the bundle fresh.

    Safe to wipe because the caller only gets here when the installed bytes
    matched the manifest (we shipped it, the user didn't touch it) - or when
    the user explicitly passed -y. If the copy fails after the wipe, the
    next startup sees the skill missing and installs it fresh (self-healing).
    """
    try:
        shutil.rmtree(dest)
        shutil.copytree(src, dest)
        return True
    except OSError as e:
        logger.warning("Failed to update default skill %s: %s", src.name, e)
        return False
