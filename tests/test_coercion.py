"""Tests for reverse coercion — when the LLM sends int/float/bool where str is expected."""

import json
import os
import tempfile
import pytest
from agent13 import execute_tool


class TestReverseCoercionStrExpected:
    """When the LLM sends non-string types for str parameters,
    _coerce_arguments should convert them to str rather than
    letting them through to cause TypeErrors in tool code."""

    @pytest.mark.asyncio
    async def test_edit_file_find_as_int(self):
        """find param is Optional[str]; sending int should be converted to str,
        not cause a TypeError in str.find()."""
        # Use .txt suffix to avoid edit_file's Python syntax checker
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line1\nline2\nline3\n")
            f.flush()
            path = f.name
        try:
            # LLM sends find: 42 instead of find: "42"
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": path,
                    "find": 42,
                    "content": "replaced",
                    "mode": "replace",
                },
            )
            result_dict = json.loads(result)
            # Should NOT be an unhandled TypeError
            assert (
                "error" not in result_dict
                or "find() argument" not in result_dict.get("error", "")
            ), f"Got unhandled TypeError: {result_dict}"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_edit_file_content_as_int_in_replace_mode(self):
        """content param (in replace mode) is Optional[str]; sending int should convert to str."""
        # Use .txt suffix to avoid edit_file's Python syntax checker
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world\n")
            f.flush()
            path = f.name
        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": path,
                    "find": "hello",
                    "content": 42,
                    "mode": "replace",
                },
            )
            result_dict = json.loads(result)
            # If content was left as int, the edit would silently work
            # (str concatenation with int), but the file content would be wrong
            # The value should have been converted to "42"
            assert "error" not in result_dict, f"Unexpected error: {result_dict}"
            # Read back the file — should contain "42" not 42
            with open(path) as f2:
                content = f2.read()
            assert "42" in content
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_edit_file_content_as_int(self):
        """content param is Optional[str]; sending int should convert to str."""
        # Use .txt suffix to avoid edit_file's Python syntax checker
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("old line\n")
            f.flush()
            path = f.name
        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": path,
                    "content": 99,
                    "mode": "replace_range",
                    "start_line": 1,
                    "end_line": 1,
                },
            )
            result_dict = json.loads(result)
            # Should succeed — content should be "99" not 99
            assert "error" not in result_dict or "TypeError" not in result_dict.get(
                "error", ""
            ), f"Got unhandled type error: {result_dict}"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_bash_command_as_int(self):
        """command param is str; sending int should convert to str,
        not cause a TypeError in subprocess.Popen()."""
        # LLM sends command: 42 instead of command: "42"
        result = await execute_tool("command", {"command": 42})
        result_dict = json.loads(result)
        # Should NOT be an unhandled TypeError from subprocess
        assert "error" not in result_dict or "TypeError" not in result_dict.get(
            "error", ""
        ), f"Got unhandled TypeError from subprocess: {result_dict}"

    @pytest.mark.asyncio
    async def test_bash_command_as_list(self):
        """command param is str; sending list should produce a clear error,
        not pass through to subprocess.Popen (security risk)."""
        result = await execute_tool("command", {"command": ["ls", "-la"]})
        result_dict = json.loads(result)
        # subprocess.Popen(list) actually WORKS on Python (treated as argv)
        # but it bypasses shell=True safety — this is a security concern.
        # Either it should be converted to "ls -la" or produce a clear error.
        assert "error" in result_dict or isinstance(result_dict.get("output"), str), (
            "List command should be rejected or converted, not passed through silently"
        )

    @pytest.mark.asyncio
    async def test_edit_file_filepath_as_int(self):
        """filepath param is str; sending int should convert to str
        rather than causing Path() errors."""
        result = await execute_tool(
            "edit_file",
            {
                "filepath": 12345,
                "mode": "replace",
                "find": "x",
                "content": "y",
            },
        )
        result_dict = json.loads(result)
        # Should get a file-not-found error, not a TypeError
        assert "error" in result_dict
        assert "TypeError" not in result_dict.get("error", ""), (
            f"Got unhandled TypeError: {result_dict}"
        )

    @pytest.mark.asyncio
    async def test_read_file_filepath_as_int(self):
        """filepath param is str; sending int should convert to str."""
        result = await execute_tool("read_file", {"filepath": 999})
        result_dict = json.loads(result)
        # Should get a file-not-found or path error, not a TypeError
        assert "error" in result_dict
        assert "TypeError" not in result_dict.get("error", ""), (
            f"Got unhandled TypeError: {result_dict}"
        )


class TestReverseCoercionFloatToInt:
    """When the LLM sends float where int is expected,
    non-integer floats should be rejected clearly,
    and integer-valued floats should be safely truncated."""

    @pytest.mark.asyncio
    async def test_edit_file_start_line_as_non_integer_float(self):
        """start_line is Optional[int]; sending 8.5 should be rejected,
        not silently truncated to 8."""
        # Use .txt suffix to avoid edit_file's Python syntax checker
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "line1\nline2\nline3\nline4\nline5\n"
                "line6\nline7\nline8\nline9\nline10\n"
            )
            f.flush()
            path = f.name
        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": path,
                    "content": "new",
                    "mode": "replace_range",
                    "start_line": 8.5,
                    "end_line": 10,
                },
            )
            result_dict = json.loads(result)
            # Should get a clear error about non-integer float,
            # NOT silently truncate 8.5 to 8
            assert "error" in result_dict, (
                "Non-integer float 8.5 should not be silently accepted"
            )
            assert (
                "8.5" in result_dict["error"]
                or "integer" in result_dict["error"].lower()
                or "int" in result_dict["error"].lower()
            ), f"Error should mention the float value or type: {result_dict}"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_edit_file_start_line_as_integer_float(self):
        """start_line is Optional[int]; sending 8.0 should be accepted
        (it's a whole number, equivalent to 8)."""
        # Use .txt suffix to avoid edit_file's Python syntax checker
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "line1\nline2\nline3\nline4\nline5\n"
                "line6\nline7\nline8\nline9\nline10\n"
            )
            f.flush()
            path = f.name
        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": path,
                    "content": "new line",
                    "mode": "replace_range",
                    "start_line": 8.0,
                    "end_line": 8.0,
                },
            )
            result_dict = json.loads(result)
            # Should succeed — 8.0 is a whole number
            assert "error" not in result_dict, (
                f"Integer-valued float 8.0 should be accepted: {result_dict}"
            )
        finally:
            os.unlink(path)


class TestReverseCoercionBoolToInt:
    """When the LLM sends bool where int is expected,
    it should be converted (True→1, False→0) or rejected clearly."""

    @pytest.mark.asyncio
    async def test_edit_file_start_line_as_bool(self):
        """start_line is Optional[int]; sending True should convert to 1
        or produce a clear error, not cause mysterious behavior."""
        # Use .txt suffix to avoid edit_file's Python syntax checker
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line1\nline2\nline3\n")
            f.flush()
            path = f.name
        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": path,
                    "content": "new",
                    "mode": "replace_range",
                    "start_line": True,
                    "end_line": True,
                },
            )
            result_dict = json.loads(result)
            # Should either succeed (True→1) or give a clear type error
            # Should NOT crash with a mysterious error
            if "error" in result_dict:
                assert "TypeError" not in result_dict["error"], (
                    f"Got unhandled TypeError: {result_dict}"
                )
        finally:
            os.unlink(path)
