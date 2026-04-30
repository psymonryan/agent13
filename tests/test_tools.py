"""Tests for agent.tools module."""

import os
import tempfile
import pytest
from agent13 import execute_tool, get_tools, get_tool_names, TOOLS


class TestExecuteTool:
    """Tests for execute_tool function."""

    @pytest.mark.asyncio
    async def test_execute_square_number(self):
        """Should execute square_number tool."""
        result = await execute_tool("square_number", {"x": 5})
        assert result == "25"

    @pytest.mark.asyncio
    async def test_execute_square_number_float(self):
        """Should handle float input."""
        result = await execute_tool("square_number", {"x": 3.5})
        assert result == "12.25"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Should return error for unknown tool."""
        result = await execute_tool("unknown_tool", {})
        assert "error" in result.lower()
        assert "unknown" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_with_invalid_args(self):
        """Should handle invalid arguments gracefully."""
        # Missing required argument
        result = await execute_tool("square_number", {})
        assert "error" in result.lower()


class TestGetTools:
    """Tests for get_tools function."""

    @pytest.mark.asyncio
    async def test_get_tools_returns_list(self):
        """Should return a list of tool schemas."""
        tools = get_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    @pytest.mark.asyncio
    async def test_tool_schema_structure(self):
        """Each tool should have correct schema structure."""
        tools = get_tools()

        for tool in tools:
            assert "type" in tool
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    @pytest.mark.asyncio
    async def test_square_number_tool_exists(self):
        """square_number tool should be registered."""
        tools = get_tools()
        tool_names = [t["function"]["name"] for t in tools]

        assert "square_number" in tool_names


class TestGetToolNames:
    """Tests for get_tool_names function."""

    @pytest.mark.asyncio
    async def test_get_tool_names_returns_list(self):
        """Should return a list of tool names."""
        names = get_tool_names()
        assert isinstance(names, list)
        assert len(names) > 0

    @pytest.mark.asyncio
    async def test_square_number_in_names(self):
        """square_number should be in tool names."""
        names = get_tool_names()
        assert "square_number" in names


class TestTOOLS:
    """Tests for TOOLS constant."""

    @pytest.mark.asyncio
    async def test_tools_is_list(self):
        """TOOLS should be a list."""
        assert isinstance(TOOLS, list)

    @pytest.mark.asyncio
    async def test_tools_matches_get_tools(self):
        """TOOLS should match get_tools()."""
        assert TOOLS == get_tools()


class TestReadFile:
    """Tests for read_file tool."""

    @pytest.mark.asyncio
    async def test_read_file_exists(self):
        """read_file tool should be registered."""
        names = get_tool_names()
        assert "read_file" in names

    @pytest.mark.asyncio
    async def test_read_file_basic(self):
        """Should read file with fallback view for unknown file types."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("line one\nline two\nline three\n")
            filepath = f.name

        try:
            result = await execute_tool("read_file", {"filepath": filepath})
            assert "filepath" in result
            assert "content" in result
            assert "total_lines" in result
            # Check fallback view format (line numbers)
            assert "line one" in result
            assert "line two" in result
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_read_file_not_found(self):
        """Should return error for missing file."""
        result = await execute_tool(
            "read_file", {"filepath": "nonexistent_file_xyz.txt"}
        )
        assert "error" in result
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_read_file_path_traversal(self):
        """Should reject path traversal."""
        result = await execute_tool("read_file", {"filepath": "../secret.txt"})
        assert "error" in result
        assert "traversal" in result.lower()

    @pytest.mark.asyncio
    async def test_read_file_offset_limit(self):
        """Should support offset and limit for fallback view."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("line 1\nline 2\nline 3\nline 4\nline 5\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "read_file", {"filepath": filepath, "offset": 2, "limit": 2}
            )
            assert "offset" in result
            # Should show lines 2-3
            assert "line 2" in result
            assert "line 3" in result
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_read_file_python_skim(self):
        """Should show skim view for Python files."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir="."
        ) as f:
            f.write(
                '"""Module docstring."""\n\ndef hello():\n    """Say hello."""\n    print("hello")\n'
            )
            filepath = f.name

        try:
            result = await execute_tool("read_file", {"filepath": filepath})
            assert "filepath" in result
            assert "view" in result
            # Should show skim view with function
            assert "hello" in result
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_read_file_binary_file(self):
        """Should return error for binary files."""
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".bin", delete=False, dir="."
        ) as f:
            f.write(b"\x00\x01\x02\x03\x04")
            filepath = f.name

        try:
            result = await execute_tool("read_file", {"filepath": filepath})
            assert "error" in result
            assert "binary" in result.lower()
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_read_file_directory(self):
        """Should return error when path is a directory."""
        import tempfile as tf

        dirpath = tf.mkdtemp(dir=".")
        try:
            result = await execute_tool("read_file", {"filepath": dirpath})
            assert "error" in result
            assert "not a file" in result.lower()
        finally:
            os.rmdir(dirpath)

    # NOTE: Symbol-related tests removed - symbol parameter was removed from read_file


class TestEditFile:
    """Tests for edit_file tool."""

    @pytest.mark.asyncio
    async def test_edit_file_exists(self):
        """edit_file tool should be registered."""
        names = get_tool_names()
        assert "edit_file" in names

    @pytest.mark.asyncio
    async def test_edit_file_replace_global(self):
        """Should replace text globally."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("line one\nline two\nline three\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {"filepath": filepath, "find": "line two", "content": "NEW LINE"},
            )
            import json

            data = json.loads(result)
            assert data.get("success") is True

            # Verify change
            with open(filepath) as f:
                content = f.read()
            assert "NEW LINE" in content
            assert "line two" not in content
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_edit_file_append_to_file(self):
        """Should append content to file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("line one\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {"filepath": filepath, "mode": "append", "content": "line two\n"},
            )
            import json

            data = json.loads(result)
            assert data.get("success") is True

            # Verify
            with open(filepath) as f:
                content = f.read()
            assert "line one" in content
            assert "line two" in content
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_edit_file_prepend_to_file(self):
        """Should prepend content to file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir="."
        ) as f:
            f.write("def hello():\n    pass\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {"filepath": filepath, "mode": "prepend", "content": "import sys\n"},
            )
            import json

            data = json.loads(result)
            assert data.get("success") is True

            # Verify import is at top
            with open(filepath) as f:
                content = f.read()
            assert content.startswith("import sys")
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_edit_file_find_not_found(self):
        """Should return error if find text not found."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("line one\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {"filepath": filepath, "find": "nonexistent", "content": "something"},
            )
            import json

            data = json.loads(result)
            assert data.get("success") is False
            assert "error" in data or "not found" in str(data).lower()
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_edit_file_multiple_matches(self):
        """Should return error if multiple matches found."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("x = 0\nx = 0\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file", {"filepath": filepath, "find": "x = 0", "content": "y = 1"}
            )
            import json

            data = json.loads(result)
            assert data.get("success") is False
            assert "multiple" in str(data).lower()
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_edit_file_not_found(self):
        """Should return error for missing file."""
        result = await execute_tool(
            "edit_file",
            {"filepath": "nonexistent_xyz.txt", "find": "old", "content": "new"},
        )
        assert "error" in result
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_edit_file_path_traversal(self):
        """Should reject path traversal."""
        result = await execute_tool(
            "edit_file", {"filepath": "../secret.txt", "find": "old", "content": "new"}
        )
        assert "error" in result
        assert "traversal" in result.lower()

    @pytest.mark.asyncio
    async def test_edit_file_missing_params(self):
        """Should return error if required params missing."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("content\n")
            filepath = f.name

        try:
            # Missing replace
            result = await execute_tool(
                "edit_file", {"filepath": filepath, "find": "content"}
            )
            assert "error" in result
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_edit_file_invalid_mode(self):
        """Should return error for invalid mode."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("content\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {"filepath": filepath, "mode": "invalid_mode", "content": "test"},
            )
            import json

            data = json.loads(result)
            assert data.get("success") is False or "error" in data
            assert "invalid" in str(data).lower()
        finally:
            os.unlink(filepath)

    # Line-number-based editing tests

    @pytest.mark.asyncio
    async def test_edit_file_replace_range_basic(self):
        """Should replace a range of lines."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("line 1\nline 2\nline 3\nline 4\nline 5\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": filepath,
                    "mode": "replace_range",
                    "start_line": 2,
                    "end_line": 4,
                    "content": "new line A\nnew line B",
                },
            )
            import json

            data = json.loads(result)
            assert data.get("success") is True, f"Expected success, got: {data}"

            with open(filepath) as f:
                content = f.read()
            # Lines 2-4 should be replaced
            assert "line 1" in content
            assert "new line A" in content
            assert "new line B" in content
            assert "line 2" not in content
            assert "line 3" not in content
            assert "line 4" not in content
            assert "line 5" in content
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_edit_file_replace_range_single_line(self):
        """Should replace a single line."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("line 1\nline 2\nline 3\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": filepath,
                    "mode": "replace_range",
                    "start_line": 2,
                    "end_line": 2,
                    "content": "replaced line",
                },
            )
            import json

            data = json.loads(result)
            assert data.get("success") is True

            with open(filepath) as f:
                content = f.read()
            assert "line 1" in content
            assert "replaced line" in content
            assert "line 2" not in content
            assert "line 3" in content
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_edit_file_replace_range_requires_both_params(self):
        """replace_range mode should require both start_line and end_line."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("content\n")
            filepath = f.name

        try:
            # Missing end_line
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": filepath,
                    "mode": "replace_range",
                    "start_line": 1,
                    "content": "new content",
                },
            )
            import json

            data = json.loads(result)
            assert data.get("success") is False or "error" in data
            assert "start_line" in str(data).lower() or "end_line" in str(data).lower()
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_edit_file_replace_range_invalid_range(self):
        """Should return error if start_line > end_line."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("content\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": filepath,
                    "mode": "replace_range",
                    "start_line": 5,
                    "end_line": 3,
                    "content": "new content",
                },
            )
            import json

            data = json.loads(result)
            assert data.get("success") is False or "error" in data
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_edit_file_replace_range_exceeds_file_length(self):
        """Should return error if line numbers exceed file length."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("line 1\nline 2\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": filepath,
                    "mode": "replace_range",
                    "start_line": 1,
                    "end_line": 100,
                    "content": "new content",
                },
            )
            import json

            data = json.loads(result)
            assert data.get("success") is False or "error" in data
            assert "exceeds" in str(data).lower()
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_edit_file_replace_with_line_range(self):
        """Should scope find/replace to a line range."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("x = 0\nx = 1\nx = 2\nx = 3\n")
            filepath = f.name

        try:
            # Replace only in lines 2-3
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": filepath,
                    "find": "x = 1",
                    "content": "y = 1",
                    "start_line": 2,
                    "end_line": 3,
                },
            )
            import json

            data = json.loads(result)
            assert data.get("success") is True

            with open(filepath) as f:
                content = f.read()
            assert "x = 0" in content  # Line 1 unchanged
            assert "y = 1" in content  # Line 2 changed
            assert "x = 2" in content  # Line 3 unchanged
            assert "x = 3" in content  # Line 4 unchanged
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_edit_file_append_with_line_range(self):
        """Should append content after a specific line."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("line 1\nline 2\nline 3\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": filepath,
                    "mode": "append",
                    "start_line": 2,
                    "end_line": 2,
                    "content": "inserted line",
                },
            )
            import json

            data = json.loads(result)
            assert data.get("success") is True

            with open(filepath) as f:
                content = f.read()
            # Inserted after line 2
            lines = content.splitlines()
            assert lines[0] == "line 1"
            assert lines[1] == "line 2"
            assert lines[2] == "inserted line"
            assert lines[3] == "line 3"
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_edit_file_prepend_with_line_number(self):
        """Should prepend content at a specific line."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("line 1\nline 2\nline 3\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": filepath,
                    "mode": "prepend",
                    "start_line": 2,
                    "end_line": 2,
                    "content": "inserted line",
                },
            )
            import json

            data = json.loads(result)
            assert data.get("success") is True

            with open(filepath) as f:
                content = f.read()
            # Inserted at line 2 (before original line 2)
            lines = content.splitlines()
            assert lines[0] == "line 1"
            assert lines[1] == "inserted line"
            assert lines[2] == "line 2"
            assert lines[3] == "line 3"
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_edit_file_line_numbers_1_indexed(self):
        """Line numbers should be 1-indexed (first line is line 1)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("first line\nsecond line\n")
            filepath = f.name

        try:
            # Replace line 1 (the first line)
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": filepath,
                    "mode": "replace_range",
                    "start_line": 1,
                    "end_line": 1,
                    "content": "new first",
                },
            )
            import json

            data = json.loads(result)
            assert data.get("success") is True

            with open(filepath) as f:
                content = f.read()
            assert content.startswith("new first")
            assert "second line" in content
            assert "first line" not in content
        finally:
            os.unlink(filepath)

    # NOTE: Symbol-related tests removed - symbol parameter and replace_symbol mode were removed from edit_file


class TestDeleteMode:
    """Tests for edit_file delete mode."""

    @pytest.mark.asyncio
    async def test_delete_lines_by_range(self):
        """Delete a range of lines."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("line1\nline2\nline3\nline4\nline5\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": filepath,
                    "mode": "delete",
                    "start_line": 2,
                    "end_line": 3,
                },
            )
            import json

            data = json.loads(result)
            assert data.get("success") is True
            assert "Deleted lines 2-3" in data.get("message", "")
            assert data.get("original_lines") == ["line2", "line3"]

            with open(filepath) as f:
                content = f.read()
            assert content == "line1\nline4\nline5\n"
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_delete_single_line(self):
        """Delete a single line by range."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("line1\nline2\nline3\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": filepath,
                    "mode": "delete",
                    "start_line": 2,
                    "end_line": 2,
                },
            )
            import json

            data = json.loads(result)
            assert data.get("success") is True

            with open(filepath) as f:
                content = f.read()
            assert content == "line1\nline3\n"
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_delete_text_single_match(self):
        """Delete entire line containing text pattern (single match)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("hello world\nfoo bar\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file", {"filepath": filepath, "mode": "delete", "find": "hello"}
            )
            import json

            data = json.loads(result)
            assert data.get("success") is True
            assert data.get("deletions") == 1
            assert "Deleted line" in data.get("message", "")

            with open(filepath) as f:
                content = f.read()
            assert content == "foo bar\n"  # entire line deleted
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_delete_text_all_occurrences(self):
        """Delete all lines containing text pattern."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("foo bar\nkeep me\nfoo baz\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": filepath,
                    "mode": "delete",
                    "find": "foo",
                    "replace_all": True,
                },
            )
            import json

            data = json.loads(result)
            assert data.get("success") is True
            assert data.get("deletions") == 2

            with open(filepath) as f:
                content = f.read()
            assert content == "keep me\n"  # only non-matching line remains
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_delete_text_not_found(self):
        """Error when text to delete is not found."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("hello world\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {"filepath": filepath, "mode": "delete", "find": "notfound"},
            )
            import json

            data = json.loads(result)
            assert data.get("success") is False
            assert "not found" in data.get("error", "").lower()
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_delete_multiple_matches_without_replace_all(self):
        """Error when multiple lines contain pattern and replace_all not set."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("hello world\nkeep me\nhello again\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file", {"filepath": filepath, "mode": "delete", "find": "hello"}
            )
            import json

            data = json.loads(result)
            assert data.get("success") is False
            assert "multiple lines contain" in data.get("error", "").lower()
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_delete_requires_find_or_lines(self):
        """Error when neither find nor start_line/end_line provided."""
        result = await execute_tool(
            "edit_file", {"filepath": "dummy.txt", "mode": "delete"}
        )
        import json

        data = json.loads(result)
        assert "error" in data
        assert "requires either" in data["error"].lower()


class TestApplyIndentation:
    """Tests for _apply_indentation helper function."""

    @pytest.mark.asyncio
    async def test_single_line_no_indent(self):
        """Single line with no existing indent should get target indent."""
        from tools.edit_file import _apply_indentation

        result = _apply_indentation("def foo():", "    ")
        assert result == "    def foo():"

    @pytest.mark.asyncio
    async def test_single_line_with_indent(self):
        """Single line with existing indent should be replaced with target."""
        from tools.edit_file import _apply_indentation

        result = _apply_indentation("    def foo():", "        ")
        assert result == "        def foo():"

    @pytest.mark.asyncio
    async def test_multiline_preserves_relative_indent(self):
        """Multi-line content should preserve relative indentation offsets."""
        from tools.edit_file import _apply_indentation

        # Content with 0, 4, 4 space indents
        content = "def foo():\n    x = 1\n    y = 2"
        # Apply 4-space target indent
        result = _apply_indentation(content, "    ")
        # Should become 4, 8, 8 space indents
        expected = "    def foo():\n        x = 1\n        y = 2"
        assert result == expected

    @pytest.mark.asyncio
    async def test_multiline_nested_indent(self):
        """Deeply nested content should preserve relative structure."""
        from tools.edit_file import _apply_indentation

        # Content: for loop with nested if
        content = "for item in items:\n    if valid(item):\n        process(item)"
        # Apply 4-space target indent (inside a function)
        result = _apply_indentation(content, "    ")
        expected = (
            "    for item in items:\n        if valid(item):\n            process(item)"
        )
        assert result == expected

    @pytest.mark.asyncio
    async def test_docstring_insertion(self):
        """Original bug case: inserting docstring should preserve indentation."""
        from tools.edit_file import _apply_indentation

        # Content: function with docstring
        content = (
            'def create_sbar(risk_history):\n    """Docstring."""\n    _dates = []'
        )
        # Apply 4-space target indent
        result = _apply_indentation(content, "    ")
        expected = '    def create_sbar(risk_history):\n        """Docstring."""\n        _dates = []'
        assert result == expected

    @pytest.mark.asyncio
    async def test_empty_lines_preserved(self):
        """Empty lines should be preserved as-is."""
        from tools.edit_file import _apply_indentation

        content = "x = 1\n\ny = 2"
        result = _apply_indentation(content, "    ")
        expected = "    x = 1\n\n    y = 2"
        assert result == expected

    @pytest.mark.asyncio
    async def test_empty_content(self):
        """Empty content should return empty string."""
        from tools.edit_file import _apply_indentation

        result = _apply_indentation("", "    ")
        assert result == ""

    @pytest.mark.asyncio
    async def test_content_with_no_base_indent(self):
        """Content with no base indent should just get target indent on all lines."""
        from tools.edit_file import _apply_indentation

        content = "line1\nline2\nline3"
        result = _apply_indentation(content, "    ")
        expected = "    line1\n    line2\n    line3"
        assert result == expected

    @pytest.mark.asyncio
    async def test_mixed_indent_levels(self):
        """Content with varying indent levels should preserve relative offsets."""
        from tools.edit_file import _apply_indentation

        # if/elif/else block
        content = "if x:\n    do_x()\nelif y:\n    do_y()\nelse:\n    do_default()"
        result = _apply_indentation(content, "    ")
        expected = "    if x:\n        do_x()\n    elif y:\n        do_y()\n    else:\n        do_default()"
        assert result == expected

    @pytest.mark.asyncio
    async def test_first_line_empty(self):
        """Should find base indent from first non-empty line."""
        from tools.edit_file import _apply_indentation

        content = "\n    def foo():\n        pass"
        result = _apply_indentation(content, "    ")
        # Base indent is 4 spaces, target is 4 spaces
        # Line 1: 4 spaces (base) -> 4 spaces (target), relative = 0
        # Line 2: 8 spaces -> 4 (target) + 4 (relative) = 8 spaces
        expected = "\n    def foo():\n        pass"
        assert result == expected


class TestAppendPrependWithFind:
    """Tests for edit_file append/prepend modes with find parameter."""

    @pytest.mark.asyncio
    async def test_append_with_find(self):
        """Should append content after the line containing find text."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("line one\nline two\nline three\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": filepath,
                    "mode": "append",
                    "find": "line two",
                    "content": "inserted after line two",
                },
            )
            import json

            data = json.loads(result)
            assert data.get("success") is True
            assert "line 2" in data.get("message", "")

            with open(filepath) as f:
                content = f.read()
            lines = content.splitlines()
            assert lines[0] == "line one"
            assert lines[1] == "line two"
            assert lines[2] == "inserted after line two"
            assert lines[3] == "line three"
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_prepend_with_find(self):
        """Should prepend content before the line containing find text."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("line one\nline two\nline three\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": filepath,
                    "mode": "prepend",
                    "find": "line two",
                    "content": "inserted before line two",
                },
            )
            import json

            data = json.loads(result)
            assert data.get("success") is True
            assert "line 2" in data.get("message", "")

            with open(filepath) as f:
                content = f.read()
            lines = content.splitlines()
            assert lines[0] == "line one"
            assert lines[1] == "inserted before line two"
            assert lines[2] == "line two"
            assert lines[3] == "line three"
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_append_with_find_not_found(self):
        """Should return error if find text not found in append mode."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("line one\nline two\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": filepath,
                    "mode": "append",
                    "find": "nonexistent",
                    "content": "should not appear",
                },
            )
            import json

            data = json.loads(result)
            assert data.get("success") is False
            assert "not found" in str(data).lower()
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_prepend_with_find_not_found(self):
        """Should return error if find text not found in prepend mode."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("line one\nline two\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": filepath,
                    "mode": "prepend",
                    "find": "nonexistent",
                    "content": "should not appear",
                },
            )
            import json

            data = json.loads(result)
            assert data.get("success") is False
            assert "not found" in str(data).lower()
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_append_with_find_multiple_matches(self):
        """Should return error if multiple matches found in append mode."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("x = 0\nx = 0\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": filepath,
                    "mode": "append",
                    "find": "x = 0",
                    "content": "inserted",
                },
            )
            import json

            data = json.loads(result)
            assert data.get("success") is False
            assert "multiple" in str(data).lower()
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_prepend_with_find_multiple_matches(self):
        """Should return error if multiple matches found in prepend mode."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("x = 0\nx = 0\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": filepath,
                    "mode": "prepend",
                    "find": "x = 0",
                    "content": "inserted",
                },
            )
            import json

            data = json.loads(result)
            assert data.get("success") is False
            assert "multiple" in str(data).lower()
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_append_with_replace_all(self):
        """Should append after all matching lines with replace_all=True."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("x = 0\ny = 1\nx = 0\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": filepath,
                    "mode": "append",
                    "find": "x = 0",
                    "content": "# after x",
                    "replace_all": True,
                },
            )
            import json

            data = json.loads(result)
            assert data.get("success") is True
            assert "2" in data.get("message", "")

            with open(filepath) as f:
                content = f.read()
            lines = content.splitlines()
            assert lines[0] == "x = 0"
            assert lines[1] == "# after x"
            assert lines[2] == "y = 1"
            assert lines[3] == "x = 0"
            assert lines[4] == "# after x"
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_prepend_with_replace_all(self):
        """Should prepend before all matching lines with replace_all=True."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("x = 0\ny = 1\nx = 0\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {
                    "filepath": filepath,
                    "mode": "prepend",
                    "find": "x = 0",
                    "content": "# before x",
                    "replace_all": True,
                },
            )
            import json

            data = json.loads(result)
            assert data.get("success") is True
            assert "2" in data.get("message", "")

            with open(filepath) as f:
                content = f.read()
            lines = content.splitlines()
            assert lines[0] == "# before x"
            assert lines[1] == "x = 0"
            assert lines[2] == "y = 1"
            assert lines[3] == "# before x"
            assert lines[4] == "x = 0"
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_append_without_find_defaults_to_end(self):
        """Append without find should default to end of file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("line one\nline two\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {"filepath": filepath, "mode": "append", "content": "appended line"},
            )
            import json

            data = json.loads(result)
            assert data.get("success") is True
            assert "end of file" in data.get("message", "").lower()

            with open(filepath) as f:
                content = f.read()
            lines = content.splitlines()
            assert lines[0] == "line one"
            assert lines[1] == "line two"
            assert lines[2] == "appended line"
        finally:
            os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_prepend_without_find_defaults_to_start(self):
        """Prepend without find should default to start of file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("line one\nline two\n")
            filepath = f.name

        try:
            result = await execute_tool(
                "edit_file",
                {"filepath": filepath, "mode": "prepend", "content": "prepended line"},
            )
            import json

            data = json.loads(result)
            assert data.get("success") is True
            assert "start of file" in data.get("message", "").lower()

            with open(filepath) as f:
                content = f.read()
            lines = content.splitlines()
            assert lines[0] == "prepended line"
            assert lines[1] == "line one"
            assert lines[2] == "line two"
        finally:
            os.unlink(filepath)
