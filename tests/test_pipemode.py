"""Unit tests for pipe mode — input parsing and event emission."""

import io
import json
import sys

import pytest

from agent13.pipemode import _parse_turn, _emit, _usage_zeros


# ── _parse_turn ────────────────────────────────────────────────────────────


class TestParseTurn:
    """Test NDJSON turn prompt parsing."""

    def test_valid_turn(self):
        line = json.dumps(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello world"}],
                },
            }
        )
        assert _parse_turn(line) == "Hello world"

    def test_multiple_text_blocks(self):
        line = json.dumps(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello "},
                        {"type": "text", "text": "world"},
                    ],
                },
            }
        )
        assert _parse_turn(line) == "Hello world"

    def test_empty_text(self):
        line = json.dumps(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": ""}],
                },
            }
        )
        assert _parse_turn(line) == ""

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            _parse_turn("not valid json")

    def test_json_array_not_object(self):
        with pytest.raises(ValueError, match="must be a JSON object"):
            _parse_turn("[1, 2, 3]")

    def test_missing_type(self):
        with pytest.raises(ValueError, match="type must be"):
            _parse_turn(json.dumps({"message": {}}))

    def test_wrong_type(self):
        with pytest.raises(ValueError, match="type must be 'user'"):
            _parse_turn(
                json.dumps(
                    {"type": "assistant", "message": {"role": "user", "content": []}}
                )
            )

    def test_missing_message(self):
        with pytest.raises(ValueError, match="missing 'message'"):
            _parse_turn(json.dumps({"type": "user"}))

    def test_missing_role(self):
        with pytest.raises(ValueError, match="message.role must be 'user'"):
            _parse_turn(
                json.dumps({"type": "user", "message": {"content": []}})
            )

    def test_wrong_role(self):
        with pytest.raises(ValueError, match="message.role must be 'user'"):
            _parse_turn(
                json.dumps(
                    {
                        "type": "user",
                        "message": {"role": "assistant", "content": []},
                    }
                )
            )

    def test_missing_content(self):
        with pytest.raises(ValueError, match="message.content must be"):
            _parse_turn(
                json.dumps({"type": "user", "message": {"role": "user"}})
            )

    def test_empty_content(self):
        with pytest.raises(ValueError, match="message.content must be"):
            _parse_turn(
                json.dumps(
                    {
                        "type": "user",
                        "message": {"role": "user", "content": []},
                    }
                )
            )

    def test_non_text_content_block(self):
        with pytest.raises(ValueError, match="unsupported content block type"):
            _parse_turn(
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": [{"type": "image", "url": "http://..."}],
                        },
                    }
                )
            )

    def test_text_block_missing_text_field(self):
        with pytest.raises(ValueError, match="requires a string 'text' field"):
            _parse_turn(
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text"}],
                        },
                    }
                )
            )

    def test_text_block_non_string_text(self):
        with pytest.raises(ValueError, match="requires a string 'text' field"):
            _parse_turn(
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": 42}],
                        },
                    }
                )
            )

    def test_content_block_not_object(self):
        with pytest.raises(ValueError, match="content block must be an object"):
            _parse_turn(
                json.dumps(
                    {
                        "type": "user",
                        "message": {"role": "user", "content": ["just a string"]},
                    }
                )
            )


# ── _emit ──────────────────────────────────────────────────────────────────


class TestEmit:
    """Test JSON event emission to stdout."""

    def test_emit_valid_json(self, capsys):
        obj = {"type": "system", "session_id": "abc-123", "data": {}}
        _emit(obj)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())
        assert parsed == obj

    def test_emit_flushes(self):
        """_emit must flush so the Go adapter sees events in real time."""
        old_stdout = sys.stdout
        try:
            buf = io.StringIO()
            sys.stdout = buf
            _emit({"type": "test"})
            # If _emit didn't flush, buf.tell() would be 0
            assert buf.tell() > 0
        finally:
            sys.stdout = old_stdout

    def test_emit_nested_structure(self, capsys):
        obj = {
            "type": "assistant",
            "session_id": "abc",
            "message": {
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "tool_use", "id": "t1", "name": "read_file"},
                ]
            },
        }
        _emit(obj)
        captured = capsys.readouterr()
        assert json.loads(captured.out.strip()) == obj

    def test_emit_non_serializable_uses_default(self, capsys):
        """default=str handles non-JSON-serializable values gracefully."""
        _emit({"type": "test", "value": object()})
        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())
        assert parsed["type"] == "test"
        assert isinstance(parsed["value"], str)


# ── _usage_zeros ───────────────────────────────────────────────────────────


class TestUsageZeros:
    def test_all_fields_zero(self):
        u = _usage_zeros()
        assert u == {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

    def test_is_independent(self):
        a = _usage_zeros()
        b = _usage_zeros()
        a["input_tokens"] = 100
        assert b["input_tokens"] == 0
