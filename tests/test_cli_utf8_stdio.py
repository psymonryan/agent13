"""Unit tests for _force_utf8_piped_stdio (Windows piped-stdio UTF-8 fix).

On Windows, piped stdout/stderr use the locale code page (e.g. cp1252),
so non-ASCII output (→, —, •) is mojibake'd or raises UnicodeEncodeError.
main() reconfigures piped streams to UTF-8 on Windows only.
"""

import io
import os
import sys
from unittest import mock

from agent13.cli import _force_utf8_piped_stdio


def _piped_text_stream(encoding="cp1252"):
    """TextIOWrapper over an os.pipe — simulates a piped stdout (isatty False)."""
    r, w = os.pipe()
    stream = io.TextIOWrapper(io.FileIO(w, "w"), encoding=encoding)
    return r, stream


def test_noop_off_windows():
    r, stream = _piped_text_stream("cp1252")
    try:
        with (
            mock.patch.object(sys, "platform", "darwin"),
            mock.patch.object(sys, "stdout", stream),
        ):
            _force_utf8_piped_stdio()
        assert stream.encoding == "cp1252"
    finally:
        stream.close()
        os.close(r)


def test_reconfigures_piped_stdout_on_windows():
    r, stream = _piped_text_stream("cp1252")
    try:
        with (
            mock.patch.object(sys, "platform", "win32"),
            mock.patch.object(sys, "stdout", stream),
            mock.patch.object(sys, "stderr", None),
        ):
            _force_utf8_piped_stdio()
        assert stream.encoding == "utf-8"
        # The character that raised UnicodeEncodeError under cp1252
        stream.write("Compacted 12→3 words")
    finally:
        stream.close()
        os.close(r)


def test_tty_stream_untouched_on_windows():
    r, stream = _piped_text_stream("cp1252")
    try:
        with (
            mock.patch.object(stream, "isatty", return_value=True),
            mock.patch.object(sys, "platform", "win32"),
            mock.patch.object(sys, "stdout", stream),
            mock.patch.object(sys, "stderr", None),
        ):
            _force_utf8_piped_stdio()
        assert stream.encoding == "cp1252"
    finally:
        stream.close()
        os.close(r)
