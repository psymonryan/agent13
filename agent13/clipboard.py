"""Clipboard support for agent13.

Two methods:
  - "osc52": Terminal escape sequence (default). Works over SSH.
    Requires terminal support (Alacritty, Ghostty, Kitty, iTerm2,
    Windows Terminal v1.18+). Does NOT work in tmux (without config),
    GNU screen, or conhost/PowerShell.
  - "system": OS-level subprocess (pbcopy/xclip/wl-copy/clip.exe).
    Works locally regardless of terminal, including tmux and screen.
    Does NOT work over SSH (writes to remote clipboard).

Config (in ~/.agent13/config.toml):
    [clipboard]
    method = "osc52"    # "osc52" (default) or "system"
"""

import logging
import platform
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

VALID_METHODS = ("osc52", "system")


def copy_via_system(text: str) -> bool:
    """Copy text to clipboard using OS-level commands.

    Uses pbcopy (macOS), xclip/xsel (Linux X11), wl-copy (Linux Wayland),
    or clip.exe (Windows).

    Returns True on success, False on failure.
    """
    system = platform.system()
    try:
        if system == "Darwin":
            proc = subprocess.run(
                ["pbcopy"], input=text, text=True, timeout=5,
            )
        elif system == "Linux":
            # Try xclip, then xsel, then wl-copy (Wayland)
            if shutil.which("xclip"):
                proc = subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=text, text=True, timeout=5,
                )
            elif shutil.which("xsel"):
                proc = subprocess.run(
                    ["xsel", "--clipboard", "--input"],
                    input=text, text=True, timeout=5,
                )
            elif shutil.which("wl-copy"):
                proc = subprocess.run(
                    ["wl-copy"], input=text, text=True, timeout=5,
                )
            else:
                return False
        elif system == "Windows":
            proc = subprocess.run(
                ["clip.exe"], input=text, text=True, timeout=5,
            )
        else:
            return False
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def copy_to_clipboard(
    text: str,
    method: str = "osc52",
    osc52_handler: Optional[callable] = None,
) -> bool:
    """Copy text to clipboard using the configured method.

    Args:
        text: The text to copy.
        method: "osc52" (terminal escape) or "system" (OS subprocess).
        osc52_handler: Callable that performs OSC 52 copy (typically
            Textual's App.copy_to_clipboard). Required when method="osc52".

    Returns:
        True if copy succeeded (or assumed succeeded for OSC 52),
        False if copy failed.
    """
    if method == "system":
        return copy_via_system(text)

    # OSC 52 — delegate to the handler (Textual's built-in)
    if osc52_handler is not None:
        try:
            osc52_handler(text)
            return True  # optimistic — can't detect if terminal processed it
        except Exception as e:
            logger.debug("OSC 52 copy failed: %s", e)
            return False

    # No handler available and not system method
    return False
