"""Robust reading of user-editable config files.

Users edit config.toml / .env / prompts.yaml in whatever editor they have.
On Windows that is often Notepad, which can save files as UTF-8 *with BOM*,
UTF-16 ("Unicode"), or ANSI (cp1252). Python's tomllib only accepts plain
UTF-8 (it even rejects the BOM that the TOML spec says must be ignored),
so a perfectly valid edit can make the config unreadable.

read_text_robust() transparently handles all of these. If nothing works it
raises ConfigFileError with an actionable message (what to do in Notepad),
and the CLI turns that into a clean one-line error instead of a traceback.
"""

import locale
from pathlib import Path


class ConfigFileError(Exception):
    """A user-editable config file could not be read or parsed.

    The message is user-facing: it says what went wrong AND what to try
    next. The CLI catches this and exits cleanly (fail-fast, no traceback).
    """


# BOM -> codec. Codec names with BOM handling ('utf-8-sig', 'utf-16',
# 'utf-32') consume the BOM so it never reaches the parser.
# Order matters: UTF-32 LE BOM (ff fe 00 00) starts with the UTF-16 LE BOM.
_BOMS = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe\x00\x00", "utf-32"),
    (b"\x00\x00\xfe\xff", "utf-32"),
    (b"\xff\xfe", "utf-16"),
    (b"\xfe\xff", "utf-16"),
)

_RESAVE_HINT = (
    "If you edited it in a text editor, re-save it as UTF-8 "
    "(in Notepad: Save As -> Encoding -> UTF-8, not 'UTF-8 with BOM' "
    "or 'Unicode')."
)


def read_text_robust(path) -> str:
    """Read a text file, tolerating BOMs and Windows editor encodings.

    Handles: UTF-8 (with or without BOM), UTF-16 LE/BE, UTF-32 LE/BE,
    and falls back to the OS-default encoding (cp1252 on Windows) for
    ANSI files. CRLF line endings pass through untouched - parsers
    (TOML/YAML) handle them natively.

    Args:
        path: Path to the file.

    Returns:
        Decoded file content (BOM stripped).

    Raises:
        ConfigFileError: If the bytes cannot be decoded by any known
            encoding. Message tells the user how to re-save the file.
        OSError: If the file cannot be opened (missing, permissions).
    """
    raw = Path(path).read_bytes()

    for bom, codec in _BOMS:
        if raw.startswith(bom):
            return raw.decode(codec)

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = None
    else:
        if "\x00" not in text:
            return text
        # NUL is legal in UTF-8 but never appears in real text files. NULs
        # here almost certainly mean BOM-less UTF-16 - re-decode both ways
        # and keep the endianness that yields more ASCII (config files are
        # overwhelmingly ASCII; the wrong endianness yields CJK garbage).
        le = raw.decode("utf-16-le", errors="replace")
        be = raw.decode("utf-16-be", errors="replace")
        le_ascii = sum(1 for c in le if ord(c) < 0x80)
        be_ascii = sum(1 for c in be if ord(c) < 0x80)
        return le if le_ascii >= be_ascii else be

    # ANSI/legacy encoding (cp1252 on Windows, locale default elsewhere)
    try:
        return raw.decode(locale.getpreferredencoding(False))
    except (UnicodeDecodeError, LookupError):
        raise ConfigFileError(
            f"Could not read {path}: unrecognized file encoding. {_RESAVE_HINT}"
        ) from None
