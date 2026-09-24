"""File transfer tools: binary-exact copy to/from remote hosts over sftp."""

from typing import Optional

from tools import tool
from agent13.remote_transfer import remote_put as _remote_put
from agent13.remote_transfer import remote_get as _remote_get

__all__ = ["remote_put", "remote_get"]


@tool(is_async=True)
async def remote_put(
    local_path: str,
    remote_path: str,
    remote: str,
    remote_shell: Optional[str] = None,
) -> dict:
    """Copy a file from THIS machine to a remote host. Binary-exact (sftp), SHA-256 verified on both ends.

    Use this instead of base64-piping a file through command output -
    binary files (exe, pax, images) survive intact. Missing remote parent
    directories are created automatically. Fails loudly (success=false,
    non-zero exit_code) on any hash mismatch or truncated transfer; both
    hashes are returned in the result.

    Paths: pass the path as given for the target OS (Windows: C:\folder\
    file.zip; Linux/macOS: /abs/path). Equivalent spellings are accepted
    and normalized; the result echoes the canonical form.

    Args:
        local_path: Path to the file on THIS machine
        remote_path: Destination path on the remote host
        remote: ssh target (user@host or host)
        remote_shell: Override auto-detect: 'posix' or 'powershell'.

    Returns: Dict with success, exit_code, bytes, local_sha256, remote_sha256,
             verified, stderr
    """
    return await _remote_put(
        host=remote,
        local_path=local_path,
        remote_path=remote_path,
        remote_shell=remote_shell,
    )


@tool(is_async=True)
async def remote_get(
    remote_path: str,
    local_path: str,
    remote: str,
    remote_shell: Optional[str] = None,
) -> dict:
    """Copy a file from a remote host to THIS machine. Binary-exact (sftp), SHA-256 verified on both ends.

    Use this instead of base64-piping a file through command output -
    binary files (exe, pax, images) survive intact. Missing local parent
    directories are created automatically. Fails loudly (success=false,
    non-zero exit_code) on any hash mismatch or truncated transfer; both
    hashes are returned in the result.

    Paths: pass the path as given for the target OS (Windows: C:\folder\
    file.zip; Linux/macOS: /abs/path). Equivalent spellings are accepted
    and normalized; the result echoes the canonical form.

    Args:
        remote_path: Path to the file on the remote host
        local_path: Destination path on THIS machine
        remote: ssh target (user@host or host)
        remote_shell: Override auto-detect: 'posix' or 'powershell'.

    Returns: Dict with success, exit_code, bytes, local_sha256, remote_sha256,
             verified, stderr
    """
    return await _remote_get(
        host=remote,
        remote_path=remote_path,
        local_path=local_path,
        remote_shell=remote_shell,
    )
