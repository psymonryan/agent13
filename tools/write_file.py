"""Write file tool - create new files for AI agents."""

from pathlib import Path

from tools import tool
from tools.security import validate_path_for_write, get_current_sandbox_mode
from tools.edit_file import _snapshots, _snapshot_counter, MAX_SNAPSHOTS_PER_FILE


@tool
def write_file(filepath: str, content: str, overwrite: bool = False) -> dict:
    """Write content to a file. Fails if file exists unless overwrite=True. Use edit_file for modifications.

    Args:
        filepath: Path to file
        content: Content to write
        overwrite: Overwrite existing file (default: False)

    Returns:
        Dict with success status and details. When overwriting, includes
        snapshot_id for potential rollback via edit_file(mode="rollback").
    """
    # Expand ~ and resolve to absolute path — consistent with edit_file
    filepath = str(Path(filepath).expanduser().resolve())

    # Validate path with sandbox enforcement
    is_valid, error = validate_path_for_write(filepath)
    if not is_valid:
        return {
            "error": error,
            "sandbox_mode": get_current_sandbox_mode().value
            if get_current_sandbox_mode()
            else "default",
        }

    path = Path(filepath)

    # Check if file exists
    if path.exists() and not overwrite:
        return {
            "error": f"File already exists: {filepath}\n\nUse overwrite=True to replace the existing file, or use edit_file to modify it."
        }

    # Snapshot existing content before overwriting — enables rollback via edit_file
    snapshot_id = None
    if overwrite and path.exists():
        original_content = path.read_text("utf-8")
        if filepath not in _snapshots:
            _snapshots[filepath] = {}
            _snapshot_counter[filepath] = 0
        _snapshots[filepath][_snapshot_counter[filepath]] = original_content
        snapshot_id = _snapshot_counter[filepath]
        _snapshot_counter[filepath] += 1
        # Evict oldest snapshots if over cap
        while len(_snapshots[filepath]) > MAX_SNAPSHOTS_PER_FILE:
            oldest = min(_snapshots[filepath].keys())
            del _snapshots[filepath][oldest]

    # Ensure parent directory exists
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return {"error": f"Failed to create parent directory: {e}"}

    # Write content
    try:
        path.write_text(content, "utf-8")
    except Exception as e:
        # Write failed — undo snapshot so state is consistent
        if snapshot_id is not None:
            del _snapshots[filepath][snapshot_id]
            _snapshot_counter[filepath] -= 1
        return {"error": f"Failed to write file: {e}"}

    result = {
        "success": True,
        "message": f"Created file: {filepath}"
        if not overwrite
        else f"Overwrote file: {filepath}",
        "filepath": filepath,
        "sandbox_mode": get_current_sandbox_mode().value
        if get_current_sandbox_mode()
        else "default",
    }
    if snapshot_id is not None:
        result["snapshot_id"] = snapshot_id

    return result
