"""The fold machinery itself: CollapsibleMessage, ReasoningMessage, ToolWidget.

The settings that decide the initial state live in tests/test_widget_collapse.py;
this file is about what a widget DOES - header, body, buffering while folded,
the click that opens it, and the pending -> call upgrade a tool widget goes
through while the model is still streaming its arguments.
"""

import asyncio

import pytest
from textual.app import App, ComposeResult


class _Host(App):
    """Minimal host app - a widget needs to be mounted to compose its parts."""

    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def compose(self) -> ComposeResult:
        yield self.widget


class _FakeHeader:
    """Stands in for the mounted header in tests that don't need a screen."""

    def __init__(self):
        self.content = ""

    def update(self, content):
        self.content = content


async def _wait_for(pilot, condition, *, steps: int = 200) -> None:
    for _ in range(steps):
        if condition():
            return
        await pilot.pause()
    raise AssertionError("condition not met within polling budget")


def _body_text(widget) -> str:
    """What the body widget is currently showing (Static markup / Markdown source)."""
    body = widget._body
    return str(getattr(body, "content", None) or getattr(body, "source", ""))


# =============================================================================
# ReasoningMessage
# =============================================================================


def test_reasoning_message_constructor_defaults_to_expanded():
    """The widget starts expanded; the caller passes the setting in.

    The collapsed-by-default rule is the app's (flag > pin > config), not the
    widget's - which is what lets /rebuild mount history afresh under the
    current setting.
    """
    from ui.tui import ReasoningMessage

    widget = ReasoningMessage("Thinking")
    assert widget.collapsed is False
    assert widget._title == "Thinking"


def test_reasoning_message_mounts_folded_when_asked():
    from ui.tui import ReasoningMessage

    widget = ReasoningMessage("Reflecting", collapsed=True)
    assert widget.collapsed is True
    assert widget._title == "Reflecting"


def test_header_shows_the_fold_indicator():
    """Collapsed = a closed triangle, expanded = an open one."""
    from ui.tui import ReasoningMessage

    assert ReasoningMessage("T", collapsed=True)._header_markup().endswith("▶")
    assert ReasoningMessage("T", collapsed=False)._header_markup().endswith("▼")


def test_append_stores_content_while_folded():
    """Content is buffered, not written, while the body is hidden."""
    from ui.tui import ReasoningMessage

    widget = ReasoningMessage("Test", collapsed=True)

    async def append_content():
        await widget.append("Hello ")
        await widget.append("World")

    asyncio.run(append_content())

    assert widget._content == "Hello World"
    assert widget.collapsed is True  # Still collapsed


def test_set_title_updates_the_header():
    from ui.tui import ReasoningMessage

    widget = ReasoningMessage("Thinking")
    widget._header = _FakeHeader()
    widget.set_title("Thoughts")
    assert "Thoughts" in widget._header.content


# =============================================================================
# Folding in a mounted widget
# =============================================================================


class TestMountedFolding:
    @pytest.mark.asyncio
    async def test_folded_widget_hides_the_body(self):
        from ui.tui import ReasoningMessage

        widget = ReasoningMessage("Thinking", collapsed=True)
        async with _Host(widget).run_test():
            assert widget._body.display is False
            assert "▶" in widget._header.content

    @pytest.mark.asyncio
    async def test_expanded_widget_shows_the_body(self):
        from ui.tui import ReasoningMessage

        widget = ReasoningMessage("Thinking", collapsed=False)
        async with _Host(widget).run_test():
            assert widget._body.display is True
            assert "▼" in widget._header.content

    @pytest.mark.asyncio
    async def test_clicking_the_header_opens_a_folded_widget(self):
        from ui.tui import ReasoningMessage

        widget = ReasoningMessage("Thinking", collapsed=True)
        widget.set_body("the argument list")
        async with _Host(widget).run_test() as pilot:
            await pilot.click(widget._header)
            await _wait_for(pilot, lambda: widget.collapsed is False)
            assert widget._body.display is True
            # Content buffered while folded is rendered on expand
            assert "the argument list" in _body_text(widget)

    @pytest.mark.asyncio
    async def test_clicking_the_body_does_not_toggle(self):
        from ui.tui import ReasoningMessage

        widget = ReasoningMessage("Thinking", collapsed=True)
        async with _Host(widget).run_test() as pilot:
            await pilot.click(widget._body)
            await pilot.pause()
            assert widget.collapsed is True

    @pytest.mark.asyncio
    async def test_set_collapsed_renders_buffered_content(self):
        from ui.tui import ReasoningMessage

        widget = ReasoningMessage("Thinking", collapsed=True)
        widget.set_body("hidden while folded")
        async with _Host(widget).run_test():
            assert _body_text(widget) == ""
            await widget.set_collapsed(False)
            assert "hidden while folded" in _body_text(widget)


# =============================================================================
# Completed tense: the live path and /rebuild must label the same block alike
# =============================================================================


def test_finalize_puts_the_header_in_its_completed_tense():
    """A finished block reads "Thoughts" / "Reflection", not the live title."""
    from ui.tui import ReasoningMessage

    for live, done in (("Thinking", "Thoughts"), ("Reflecting", "Reflection")):
        widget = ReasoningMessage(live)
        widget._header = _FakeHeader()

        async def finish(w=widget):
            await w.append("some reasoning")
            await w.finalize()

        asyncio.run(finish())
        assert widget._title == done
        assert done in widget._header.content


def test_finalize_leaves_an_untracked_title_alone():
    """The rename is a lookup, not a blanket rewrite of every header."""
    from ui.tui import ReasoningMessage

    widget = ReasoningMessage("Custom")

    async def finish():
        await widget.finalize()

    asyncio.run(finish())
    assert widget._title == "Custom"


@pytest.mark.asyncio
async def test_rebuild_labels_a_completed_reasoning_block_thoughts():
    """/rebuild mounts history afresh, and history only holds finished blocks.

    The rebuild path used to hard-code the live title, so a rebuilt turn showed
    "Thinking" where the live turn had already renamed itself to "Thoughts".
    """
    from unittest.mock import AsyncMock, MagicMock

    from ui.tui import ReasoningMessage

    tui = MagicMock()
    tui._reasoning_collapsed = True
    tui._is_at_bottom = MagicMock(return_value=False)
    tui._scroll_if_at_bottom = AsyncMock()
    mounted = []

    async def capture(widget):
        mounted.append(widget)
        return True

    tui._safe_mount = capture

    from ui.tui import AgentTUI

    await AgentTUI._rebuild_from_messages(
        tui, [{"role": "assistant", "content": "answer", "reasoning_content": "think"}]
    )

    reasoning = [w for w in mounted if isinstance(w, ReasoningMessage)]
    assert len(reasoning) == 1
    assert reasoning[0]._title == "Thoughts"
    assert reasoning[0]._content == "think"
    # Fold state still comes from the setting, not from the rename
    assert reasoning[0].collapsed is True


@pytest.mark.asyncio
async def test_rebuild_numbers_tool_calls_and_results_per_turn():
    """/rebuild assigns the same per-user-turn seq as the live path.

    The seq is the call's 1-based number within the user turn - a running
    counter that resets on each user message and continues across the
    turn's assistant messages. The paired result carries the same number,
    so a rebuilt chat pairs results with calls exactly like the one that
    streamed.
    """
    from unittest.mock import AsyncMock, MagicMock

    from ui.tui import AgentTUI

    tui = MagicMock()
    tui._reasoning_collapsed = True
    tui._toolview_collapsed = True
    tui._is_at_bottom = MagicMock(return_value=False)
    tui._scroll_if_at_bottom = AsyncMock()
    tui._safe_mount = AsyncMock(return_value=True)
    tui._write_user = AsyncMock()
    tui._write_tool_call = AsyncMock()
    tui._write_tool_result = AsyncMock()

    messages = [
        {"role": "user", "content": "turn 1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "function": {
                        "name": "command",
                        "arguments": '{"command": "ls"}',
                    },
                },
                {
                    "id": "tc_2",
                    "function": {
                        "name": "command",
                        "arguments": '{"command": "pwd"}',
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "tc_1", "content": "a"},
        {"role": "tool", "tool_call_id": "tc_2", "content": "b"},
        {"role": "user", "content": "turn 2"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tc_3",
                    "function": {
                        "name": "command",
                        "arguments": '{"command": "whoami"}',
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "tc_3", "content": "c"},
    ]

    await AgentTUI._rebuild_from_messages(tui, messages)

    call_seqs = [c.args[2] for c in tui._write_tool_call.call_args_list]
    result_seqs = [c.args[2] for c in tui._write_tool_result.call_args_list]
    # Turn 1's batch numbers 1, 2; turn 2 restarts at 1.
    assert call_seqs == [1, 2, 1], f"call seqs: {call_seqs}"
    assert result_seqs == [1, 2, 1], f"result seqs: {result_seqs}"


# =============================================================================
# The folded one-line "what": tool calls and reasoning
# =============================================================================


class TestToolWhatHeader:
    def test_call_header_carries_the_command(self):
        from ui.tui import ToolWidget

        w = ToolWidget("command")
        w.set_call("command", {"command": "git log --oneline -5"})
        assert "git log --oneline -5" in w._title_markup()
        assert "⚙ command" in w._title_markup()

    def test_call_header_carries_the_filepath(self):
        from ui.tui import ToolWidget

        w = ToolWidget("edit_file")
        w.set_call("edit_file", {"filepath": "src/foo.py", "content": "x" * 200})
        assert "src/foo.py" in w._title_markup()

    def test_call_header_is_name_only_without_args(self):
        from ui.tui import ToolWidget

        w = ToolWidget("some_tool")
        w.set_call("some_tool", {})
        assert w._title_markup() == "[bold yellow]⚙ some_tool[/]"

    def test_call_header_truncates_to_the_available_width(self):
        from rich.cells import cell_len

        from ui.tui import ToolWidget

        w = ToolWidget("command")
        w.set_call("command", {"command": "a" * 200})
        w._header_width = lambda: 60  # constrain the header's columns
        m = w._title_markup()
        visible = m.replace("[bold yellow]", "").replace("[/]", "")
        assert cell_len(visible) <= 60
        assert "…" in m


class TestReasoningWhatHeader:
    def test_streaming_header_has_no_hint(self):
        from ui.tui import ReasoningMessage

        w = ReasoningMessage("Thinking")
        w._content = "some thought here"
        # Not finished: no hint, just the in-progress title.
        assert w._title_markup() == "[dim]Thinking[/]"

    def test_finalized_header_carries_the_first_line(self):
        from ui.tui import ReasoningMessage

        w = ReasoningMessage("Thinking")
        w._content = "let me check the log\nthen the failing test"

        async def run():
            await w.finalize()

        asyncio.run(run())
        m = w._title_markup()
        assert "Thoughts: let me check the log" in m
        assert "failing test" not in m


# =============================================================================
# ToolWidget
# =============================================================================


class TestToolWidgetHeaders:
    def test_call_header_is_the_bright_gear(self):
        from ui.tui import ToolWidget

        assert ToolWidget("write_file")._title_markup() == (
            "[bold yellow]⚙ write_file[/]"
        )

    def test_result_header_names_its_tool(self):
        from ui.tui import ToolWidget

        markup = ToolWidget("write_file", kind="result")._title_markup()
        assert "Result" in markup
        assert "write_file" in markup

    def test_markup_in_a_tool_name_is_escaped(self):
        """A tool name is provider data - it must not parse as markup."""
        from ui.tui import ToolWidget

        assert "\\[" in ToolWidget("[bold]evil")._title_markup()


class TestToolWidgetLifecycle:
    def test_pending_to_call_upgrade_in_place(self):
        """One widget: placeholder -> named -> complete, never a duplicate."""
        from ui.tui import ToolWidget

        widget = ToolWidget("", pending=True, pending_unknown=True)
        assert widget._title_markup() == "[dim]⚙ …[/]"

        widget.set_pending_name("write_file")
        assert "write_file" in widget._title_markup()
        assert "…" in widget._title_markup()

        widget.set_call("write_file", {"filepath": "a.py"})
        assert widget.pending is False
        assert widget.pending_unknown is False
        assert "…" not in widget._title_markup()
        assert "a.py" in widget._content

    def test_set_result_fills_the_body(self):
        from ui.tui import ToolWidget

        widget = ToolWidget("write_file", kind="result")
        widget.set_result('{"success": true}')
        assert '"success": true' in widget._content

    @pytest.mark.asyncio
    async def test_tool_widget_mounts_folded_by_default_in_the_app(self):
        """The folded header keeps the "what is it doing" signal."""
        from ui.tui import ToolWidget

        widget = ToolWidget("write_file", collapsed=True)
        widget.set_call("write_file", {"filepath": "a.py"})
        async with _Host(widget).run_test():
            assert widget._body.display is False
            assert "write_file" in widget._header.content
            # The args are known, just not on screen
            assert "a.py" in widget._content

    @pytest.mark.asyncio
    async def test_clicking_a_tool_header_reveals_the_args(self):
        from ui.tui import ToolWidget

        widget = ToolWidget("write_file", collapsed=True)
        widget.set_call("write_file", {"filepath": "a.py"})
        async with _Host(widget).run_test() as pilot:
            await pilot.click(widget._header)
            await _wait_for(pilot, lambda: widget.collapsed is False)
            assert "a.py" in _body_text(widget)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
