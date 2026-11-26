"""Command registry - the one index of every slash command.

The registry is the seam both front-ends cross: a shared command has a handler
here, a front-end-local one does not. These tests pin the catalogue's integrity
and the parity between the two front-ends, so a command can't be added to one
and forgotten in the other.
"""

import os
import time

from agent13 import commands as C


class TestCatalogue:
    def test_names_are_unique(self):
        names = [c.name for c in C._COMMAND_LIST]
        assert len(names) == len(set(names))

    def test_aliases_do_not_shadow_names(self):
        names = {c.name for c in C._COMMAND_LIST}
        for cmd in C._COMMAND_LIST:
            for alias in cmd.aliases:
                assert alias not in names, f"alias {alias} shadows a command"
                assert alias != cmd.name

    def test_aliases_are_unique(self):
        seen = {}
        for cmd in C._COMMAND_LIST:
            for alias in cmd.aliases:
                assert alias not in seen, f"alias {alias} claimed twice"
                seen[alias] = cmd.name

    def test_every_entry_has_a_known_group(self):
        for cmd in C._COMMAND_LIST:
            assert cmd.group in C.HELP_GROUPS, f"{cmd.name} has group {cmd.group}"

    def test_every_entry_has_a_description(self):
        for cmd in C._COMMAND_LIST:
            assert cmd.description.strip(), f"{cmd.name} has no description"

    def test_every_entry_appears_in_exactly_one_help_section(self):
        seen = [e.name for s in C.format_help_sections() for e in s.entries]
        assert sorted(seen) == sorted(c.name for c in C._COMMAND_LIST)


class TestLookup:
    def test_get_command_by_name(self):
        assert C.get_command("save").name == "save"

    def test_get_command_is_case_insensitive(self):
        assert C.get_command("SAVE").name == "save"

    def test_get_command_resolves_aliases(self):
        assert C.get_command("quit").name == "exit"
        assert C.get_command("prioritize").name == "prioritise"
        assert C.get_command("deprioritize").name == "deprioritise"

    def test_get_command_unknown_is_none(self):
        assert C.get_command("nope") is None

    def test_command_names_excludes_aliases(self):
        assert "quit" not in C.command_names()
        assert "exit" in C.command_names()


class TestRunCommand:
    def test_shared_command_runs_its_handler(self, tmp_path):
        class Agent:
            messages = []

        result = C.run_command("cwd", Agent(), "")
        assert result is not None
        assert str(os.getcwd()) in result.message

    def test_local_command_returns_none(self):
        """A front-end-local command has no handler - the caller handles it."""
        assert C.run_command("help", None, "") is None

    def test_unknown_command_returns_none(self):
        assert C.run_command("nope", None, "") is None

    def test_alias_runs_the_same_handler(self):
        class Agent:
            messages = []

        assert C.run_command("prioritize", Agent(), "") is not None


class TestBellHandlers:
    """The bell commands are shared - both front-ends reach these handlers.

    The manager is attached to the agent by each front-end (it supplies the
    fallback ring), so the handler works the same in both.
    """

    def _agent(self, **kwargs):
        from agent13.bell import BellManager

        class Agent:
            pass

        agent = Agent()
        agent.bell = BellManager(**kwargs)
        return agent

    def test_status_query_reports_the_threshold(self):
        result = C.run_command("bell", self._agent(threshold=30), "")
        assert result.success
        assert result.message == "Bell: on (30s)"

    def test_status_query_when_disabled(self):
        result = C.run_command("bell", self._agent(enabled=False), "")
        assert result.message == "Bell: off"

    def test_set_threshold(self):
        agent = self._agent()
        result = C.run_command("bell", agent, "45")
        assert result.success
        assert result.message == "Bell: on (45s)"
        assert agent.bell.threshold == 45

    def test_zero_means_always_ring(self):
        agent = self._agent(threshold=30)
        result = C.run_command("bell", agent, "0")
        assert result.message == "Bell: on (always)"
        assert agent.bell.threshold == 0

    def test_off_disables(self):
        agent = self._agent(threshold=30)
        result = C.run_command("bell", agent, "off")
        assert result.message == "Bell: off"
        assert agent.bell.enabled is False

    def test_invalid_value_shows_usage(self):
        result = C.run_command("bell", self._agent(), "soon")
        assert not result.success
        assert "Usage: /bell" in result.message

    def test_negative_value_shows_usage(self):
        result = C.run_command("bell", self._agent(), "-5")
        assert not result.success
        assert "Usage: /bell" in result.message

    def test_bell_command_set_and_clear(self):
        import sys

        agent = self._agent()
        set_result = C.run_command("bell-command", agent, sys.executable)
        assert set_result.success
        assert agent.bell.command == sys.executable

        query = C.run_command("bell-command", agent, "")
        assert query.message == f"Bell command: {sys.executable}"

        cleared = C.run_command("bell-command", agent, "off")
        assert cleared.message == "Bell command: cleared (terminal bell)"
        assert agent.bell.command == ""

    def test_bell_command_query_without_command(self):
        result = C.run_command("bell-command", self._agent(), "")
        assert result.message == "Bell command: (terminal bell)"

    def test_bell_command_rejects_non_executable(self):
        result = C.run_command("bell-command", self._agent(), "no-such-binary-xyz")
        assert not result.success
        assert "not executable" in result.message

    def test_missing_manager_is_an_actionable_error(self):
        class Agent:
            pass

        for name in ("bell", "bell-command"):
            result = C.run_command(name, Agent(), "")
            assert not result.success
            assert "not available in this mode" in result.message


class TestModelHandler:
    """`/model` lists or selects; the TUI turns the listing into a picker."""

    def _agent(self, models=None, model="m1"):
        class Agent:
            pass

        agent = Agent()
        agent.available_models = models if models is not None else ["m1", "m2"]
        agent.model = model
        agent.set_model = lambda name: setattr(agent, "model", name)
        return agent

    def test_no_args_lists_models_with_current_marked(self):
        result = C.run_command("model", self._agent(), "")
        assert result.success
        assert "1. m1 *" in result.message
        assert "2. m2" in result.message
        assert result.data["models"] == ["m1", "m2"]
        assert result.data["current"] == "m1"

    def test_no_args_without_models_explains_the_fix(self):
        result = C.run_command("model", self._agent(models=[]), "")
        assert not result.success
        assert "/provider" in result.message

    def test_select_by_number(self):
        agent = self._agent()
        result = C.run_command("model", agent, "2")
        assert result.success
        assert result.message == "Model set to: m2"
        assert agent.model == "m2"
        assert result.refresh_status is True

    def test_select_by_name(self):
        agent = self._agent()
        assert C.run_command("model", agent, "m2").success
        assert agent.model == "m2"

    def test_unknown_model_reports_the_reason(self):
        result = C.run_command("model", self._agent(), "nope")
        assert not result.success
        assert "nope" in result.message


class TestProviderHandler:
    """`/provider` swaps the client; the model fetch is the front-end's tail."""

    def _agent(self):
        class Client:
            class base_url:
                pass

        class Agent:
            pass

        agent = Agent()
        agent.client = None
        agent.set_client = lambda c: setattr(agent, "client", c)
        return agent

    def test_no_args_lists_providers_with_current_marked(self):
        from unittest.mock import patch

        with patch(
            "agent13.config.get_provider_names", return_value=["alpha", "beta"]
        ), patch("agent13.config.current_provider_name", return_value="alpha"):
            result = C.run_command("provider", self._agent(), "")

        assert result.success
        assert "1. alpha *" in result.message
        assert result.data["providers"] == ["alpha", "beta"]
        assert result.data["current"] == "alpha"

    def test_no_args_without_providers_reports_it(self):
        from unittest.mock import patch

        with patch("agent13.config.get_provider_names", return_value=[]):
            result = C.run_command("provider", self._agent(), "")
        assert not result.success
        assert "No providers configured" in result.message

    def test_switch_returns_the_new_base_url_for_the_async_tail(self):
        from unittest.mock import patch

        agent = self._agent()
        with patch(
            "agent13.config.resolve_provider_selection", return_value="alpha"
        ), patch(
            "agent13.config.resolve_provider_arg",
            return_value=("http://alpha:8000/v1", "key", 10.0, 5.0),
        ), patch("agent13.config.create_client", return_value="CLIENT"):
            result = C.run_command("provider", agent, "alpha")

        assert result.success
        assert result.message == "Provider changed to: http://alpha:8000/v1"
        assert result.data["base_url"] == "http://alpha:8000/v1"
        assert agent.client == "CLIENT"
        assert result.refresh_status is True

    def test_missing_api_key_is_an_error_not_a_crash(self):
        from unittest.mock import patch

        agent = self._agent()
        with patch(
            "agent13.config.resolve_provider_selection", return_value="alpha"
        ), patch(
            "agent13.config.resolve_provider_arg",
            side_effect=ValueError("OPENAI_API_KEY not found in environment"),
        ):
            result = C.run_command("provider", agent, "alpha")

        assert not result.success
        assert "OPENAI_API_KEY" in result.message
        assert agent.client is None

    def test_unknown_provider_reports_the_reason(self):
        from unittest.mock import patch

        with patch("agent13.config.resolve_provider_selection", return_value=None):
            result = C.run_command("provider", self._agent(), "nope")
        assert not result.success
        assert "nope" in result.message


class TestAgentSettingHandlers:
    """Two agent flags that were TUI-only: the REPL now reaches them too."""

    def _agent(self, remove_reasoning=False, response_format=None):
        class Agent:
            pass

        agent = Agent()
        agent.remove_reasoning = remove_reasoning
        agent.response_format = response_format
        agent.set_response_format = lambda fmt: setattr(agent, "response_format", fmt)
        return agent

    def test_remove_reasoning_on(self):
        agent = self._agent()
        result = C.run_command("remove-api-reasoning", agent, "on")
        assert result.success
        assert agent.remove_reasoning is True
        assert "stripped between turns" in result.message

    def test_remove_reasoning_off(self):
        agent = self._agent(remove_reasoning=True)
        result = C.run_command("remove-api-reasoning", agent, "off")
        assert result.success
        assert agent.remove_reasoning is False
        assert "preserved between turns" in result.message

    def test_remove_reasoning_no_args_toggles(self):
        agent = self._agent()
        assert C.run_command("remove-api-reasoning", agent, "").message.endswith("on")
        assert agent.remove_reasoning is True
        assert C.run_command("remove-api-reasoning", agent, "").message.endswith("off")
        assert agent.remove_reasoning is False

    def test_remove_reasoning_invalid_shows_usage_and_current(self):
        agent = self._agent(remove_reasoning=True)
        result = C.run_command("remove-api-reasoning", agent, "maybe")
        assert not result.success
        assert "Usage: /remove-api-reasoning" in result.message
        assert "Current: on" in result.message

    def test_tool_response_query(self):
        assert C.run_command("tool-response", self._agent(), "").message.startswith(
            "Tool response format: raw"
        )
        json_agent = self._agent(response_format={"type": "json_object"})
        assert C.run_command("tool-response", json_agent, "").message.startswith(
            "Tool response format: json"
        )

    def test_tool_response_sets_the_format(self):
        agent = self._agent()
        result = C.run_command("tool-response", agent, "json")
        assert result.success
        assert agent.response_format == {"type": "json_object"}

        C.run_command("tool-response", agent, "raw")
        assert agent.response_format is None

    def test_tool_response_invalid(self):
        agent = self._agent()
        result = C.run_command("tool-response", agent, "xml")
        assert not result.success
        assert "raw" in result.message and "json" in result.message


class TestQueueHandler:
    """/queue: one listing both front-ends render, plus the pane request."""

    def _agent(self, *items, running=False):
        from agent13.queue import AgentQueue

        class _Agent:
            queue = AgentQueue()

        agent = _Agent()
        for text in items:
            agent.queue.add(text)
        if running:
            agent.queue.get_next()
        return agent

    def test_empty_queue(self):
        result = C.run_command("queue", self._agent(), "")
        assert result.success
        assert result.message == "Queue is empty"
        assert result.pane == "queue"

    def test_pending_items_are_numbered_from_one(self):
        result = C.run_command("queue", self._agent("alpha", "beta"), "")
        assert result.message.splitlines() == [
            "Queue (2 pending):",
            "     1. alpha",
            "     2. beta",
        ]
        assert result.pane == "queue"

    def test_running_item_is_unnumbered_and_pending_restart_at_one(self):
        result = C.run_command(
            "queue", self._agent("alpha", "beta", running=True), ""
        )
        assert result.message.splitlines() == [
            "Queue (1 pending):",
            "     → alpha (running)",
            "     1. beta",
        ]

    def test_priority_and_interrupt_markers(self):
        """A priority item jumps to the front, so it prints as item 1."""
        agent = self._agent("plain", "urgent", "stop")
        agent.queue.set_priority_at(2, True)
        agent.queue.list_items()[2].interrupt = True
        result = C.run_command("queue", agent, "")
        lines = result.message.splitlines()
        assert lines[1] == "  !  1. urgent"
        assert lines[2] == "     2. plain"
        assert lines[3] == "  !! 3. stop"

    def test_polite_waiting_header_is_not_a_command_form(self):
        """The TUI's polite tick passes it; /queue itself never does."""
        agent = self._agent("alpha")
        assert C.execute_queue(agent, "", polite_waiting=True).message.splitlines()[0] == (
            "Waiting for lock"
        )
        assert C.run_command("queue", agent, "").message.splitlines()[0] == (
            "Queue (1 pending):"
        )

    def test_item_text_is_whitespace_collapsed(self):
        result = C.run_command("queue", self._agent("two\nlines   here"), "")
        assert "two lines here" in result.message


class TestJournalHandler:
    """/journal: the flag flips here, the queued journal item is the tail."""

    def _agent(self, journal_mode=False):
        class _Agent:
            pass

        agent = _Agent()
        agent.journal_mode = journal_mode
        return agent

    def test_on_sets_the_flag_and_refreshes_the_status_bar(self):
        agent = self._agent()
        result = C.run_command("journal", agent, "on")
        assert result.success
        assert agent.journal_mode is True
        assert result.refresh_status is True
        assert "Journal mode enabled" in result.message

    def test_off_clears_the_flag(self):
        agent = self._agent(journal_mode=True)
        result = C.run_command("journal", agent, "off")
        assert result.success
        assert agent.journal_mode is False
        assert result.refresh_status is True

    def test_status_query(self):
        assert C.run_command("journal", self._agent(), "").message == "Journal mode: off"
        assert (
            C.run_command("journal", self._agent(True), "status").message
            == "Journal mode: on"
        )

    def test_last_and_all_ask_the_front_end_to_queue_an_agent_message(self):
        last = C.run_command("journal", self._agent(), "last")
        assert last.data["agent_message"] == ("/journal last", "journal_last", {})

        every = C.run_command("journal", self._agent(), "all")
        assert every.data["agent_message"] == ("/journal all", "journal_all", {})

    def test_invalid_form_shows_usage_and_current(self):
        result = C.run_command("journal", self._agent(True), "bogus")
        assert not result.success
        assert "Usage: /journal" in result.message
        assert "Current: on" in result.message


class TestInfoPaneColour:
    """A no-arg invocation is a status query, so it reads yellow in the pane."""

    class _Renderer:
        def __init__(self):
            self.calls = []
            self._info_pane_mode = None

        def _update_info_content(self, text):
            self.calls.append(text)

        def update_status(self):
            pass

    def _render(self, result, args):
        from ui.tui import AgentTUI

        renderer = self._Renderer()
        AgentTUI._render_command_result(renderer, result, args)
        return renderer.calls[0]

    def test_status_query_reads_yellow(self):
        text = self._render(C.CommandResult(True, "Bell: off"), "")
        assert text.startswith("[yellow]")

    def test_change_reads_green(self):
        text = self._render(C.CommandResult(True, "Bell: off"), "off")
        assert text.startswith("[green]")

    def test_failure_reads_red(self):
        text = self._render(C.CommandResult(False, "Usage: /bell"), "soon")
        assert text.startswith("[red]")


class TestUnimplemented:
    def test_reports_commands_with_no_handler_and_no_local_body(self):
        gaps = C.unimplemented([])
        assert "help" in gaps
        assert "save" not in gaps  # has a handler

    def test_local_names_close_the_gap(self):
        assert C.unimplemented(C.command_names()) == []


class TestCompletions:
    def test_static_list_is_prefix_filtered(self):
        assert C.complete_param("pretty", None, "") == ["on", "off"]
        assert C.complete_param("pretty", None, "o") == ["on", "off"]
        assert C.complete_param("pretty", None, "on") == ["on"]

    def test_callable_completer_is_used(self):
        class Agent:
            available_models = ["alpha", "beta"]

        assert C.complete_param("model", Agent(), "") == ["alpha", "beta"]
        assert C.complete_param("model", Agent(), "b") == ["beta"]

    def test_number_selects_the_model(self):
        class Agent:
            available_models = ["alpha", "beta"]

        assert C.complete_param("model", Agent(), "2") == ["beta"]
        assert C.complete_param("model", Agent(), "9") == []

    def test_command_without_completions_returns_empty(self):
        assert C.complete_param("status", None, "") == []

    def test_unknown_command_returns_empty(self):
        assert C.complete_param("nope", None, "") == []


class TestHelpSections:
    def test_sections_follow_group_order(self):
        titles = [s.title for s in C.format_help_sections()]
        assert titles == [g for g in C.HELP_GROUPS if g in titles]

    def test_filtering_by_local_names_drops_unimplemented_commands(self):
        sections = C.format_help_sections({"help", "status"})
        names = {e.name for s in sections for e in s.entries}
        assert "help" in names
        assert "spinner" not in names

    def test_no_filter_shows_everything(self):
        names = {e.name for s in C.format_help_sections() for e in s.entries}
        assert "snippet" in names


class TestCommandResult:
    def test_side_effect_hints_default_to_off(self):
        result = C.CommandResult(True, "ok")
        assert result.refresh_status is False
        assert result.pane is None

    def test_delete_history_asks_for_a_status_refresh(self):
        """The TUI's status bar shows the message count - /delete h changes it."""
        result = C._delete_history(_fake_agent_with_groups(2), "1")
        assert result.success
        assert result.refresh_status is True

    def test_polite_asks_for_a_status_refresh_when_it_changes(self):
        """The status bar carries the polite interval - a change must redraw."""

        class _PoliteAgent:
            polite_lock = None

            def set_polite(self, interval):
                self.polite_lock = type("Lock", (), {"interval": interval})()

            def disable_polite(self):
                self.polite_lock = None

        agent = _PoliteAgent()

        assert C.execute_polite(agent, "0.5").refresh_status is True
        assert C.execute_polite(agent, "off").refresh_status is True
        # A status query changes nothing, so nothing needs redrawing.
        assert C.execute_polite(agent, "").refresh_status is False


class TestFrontEndParity:
    """Both front-ends must cover the catalogue, gaps declared explicitly."""

    def test_tui_covers_the_catalogue(self):
        from ui.tui import AgentTUI

        assert set(C.unimplemented(AgentTUI._LOCAL_COMMANDS)) == set(
            AgentTUI._NOT_IN_TUI
        )

    def test_repl_covers_the_catalogue(self):
        from agent13.repl import _NOT_IN_REPL, _REPL_LOCAL_COMMANDS

        assert set(C.unimplemented(_REPL_LOCAL_COMMANDS)) == set(_NOT_IN_REPL)

    def test_tui_slash_commands_come_from_the_catalogue(self):
        from ui.tui import AgentTUI

        catalogue = set(C.command_names())
        for name in AgentTUI.SLASH_COMMANDS:
            assert name.lstrip("/") in catalogue, f"{name} is not in the catalogue"

    def test_repl_commands_come_from_the_catalogue(self):
        from agent13.repl import COMMANDS

        catalogue = set(C.command_names())
        for name in COMMANDS:
            assert name.lstrip("/") in catalogue, f"{name} is not in the catalogue"

    def test_tui_and_repl_agree_on_shared_commands(self):
        """A command with a handler must be offered by both front-ends."""
        from agent13.repl import COMMANDS
        from ui.tui import AgentTUI

        shared = {f"/{c.name}" for c in C._COMMAND_LIST if c.handler is not None}
        assert shared <= set(AgentTUI.SLASH_COMMANDS)
        assert shared <= set(COMMANDS)


def _fake_agent_with_groups(n: int):
    """Minimal agent stub: n history groups of one message each."""

    class History:
        def get_message_groups(self):
            return [[i] for i in range(n)]

    class Agent:
        history = History()
        messages = [{"role": "user", "content": f"m{i}"} for i in range(n)]

    return Agent()


# ============== /history ==============
# The listing is plain text: both front-ends render the same message, and the
# TUI keeps the info pane open in "history" mode.  Tool call arguments and
# tool result messages (especially edit_file's "Replaced ... of '<find>' with
# '<content>'") embed raw user content that frequently contains newlines; if
# those reach the rendered output they corrupt the one-entry-per-line layout
# - see samples/corrupted_history.md.  These tests pin both guarantees.


def _fake_agent_with_messages(messages, groups):
    """Agent stub: explicit messages plus the group index lists over them."""

    class History:
        def get_message_groups(self):
            return groups

    class Agent:
        history = History()

    agent = Agent()
    agent.messages = messages
    return agent


class TestHistoryCommand:
    def test_empty_history_says_so_and_keeps_the_pane(self):
        result = C.execute_history(_fake_agent_with_messages([], []))

        assert result.success
        assert "No messages in history" in result.message
        assert result.pane == "history"

    def test_groups_are_numbered_with_role_and_content(self):
        agent = _fake_agent_with_messages(
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second question"},
            ],
            [[0, 1], [2]],
        )

        result = C.execute_history(agent)

        assert "Message history (2 groups)" in result.message
        assert "1. user: first question" in result.message
        assert "assistant: first answer" in result.message
        assert "2. user: second question" in result.message
        assert result.pane == "history"

    def test_tool_result_line_uses_the_parsed_preview(self):
        agent = _fake_agent_with_messages(
            [
                {"role": "user", "content": "run it"},
                {"role": "tool", "content": '{"success": true, "exit_code": 0, "stdout": "ok"}'},
            ],
            [[0, 1]],
        )

        result = C.execute_history(agent)

        assert "tool result: \u2713 exit 0: ok" in result.message

    def test_tool_call_line_shows_the_key_argument(self):
        agent = _fake_agent_with_messages(
            [
                {"role": "user", "content": "read it"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": '{"filepath": "docs/x.md"}',
                            }
                        }
                    ],
                },
            ],
            [[0, 1]],
        )

        result = C.execute_history(agent)

        assert "tool call: read_file \u2192 docs/x.md" in result.message

    def test_interrupt_and_injected_entries_are_labelled(self):
        agent = _fake_agent_with_messages(
            [
                {"role": "user", "content": "go"},
                {"role": "user", "content": "stop that", "interrupt": True},
                {"role": "user", "content": "a screenshot"},
            ],
            [[0, 1, 2]],
        )
        # Third user message is a native-vision injection - marked with the
        # local flag message_history.is_injected() reads.
        agent.messages[2]["injected"] = True
        agent.messages[2]["content"] = [
            {"type": "text", "text": "a screenshot"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
        ]

        result = C.execute_history(agent)

        assert "interrupt: stop that" in result.message
        assert "injected: a screenshot" in result.message

    def test_single_group_reads_singular(self):
        agent = _fake_agent_with_messages(
            [{"role": "user", "content": "hello"}], [[0]]
        )

        assert "Message history (1 group)" in C.execute_history(agent).message

    def test_assistant_with_only_tool_calls_gets_no_empty_line(self):
        """A tool-calling assistant message usually carries no text."""
        agent = _fake_agent_with_messages(
            [
                {"role": "user", "content": "read it"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": '{"filepath": "a.md"}',
                            }
                        }
                    ],
                },
            ],
            [[0, 1]],
        )

        lines = C.execute_history(agent).message.splitlines()

        assert not [ln for ln in lines if ln.rstrip().endswith("assistant:")]
        assert any("tool call: read_file" in ln for ln in lines)

    def test_long_content_is_truncated(self):
        agent = _fake_agent_with_messages(
            [{"role": "user", "content": "x" * 300}], [[0]]
        )

        result = C.execute_history(agent)

        assert "..." in result.message
        assert "x" * 300 not in result.message

    def test_message_is_plain_text(self):
        """No Rich markup - each front-end renders it its own way."""
        agent = _fake_agent_with_messages(
            [{"role": "user", "content": "hello"}], [[0]]
        )

        result = C.execute_history(agent)

        assert "[" not in result.message
        assert "]" not in result.message

    def test_points_at_delete_for_removal(self):
        agent = _fake_agent_with_messages(
            [{"role": "user", "content": "hello"}], [[0]]
        )

        assert "/delete h N" in C.execute_history(agent).message


class TestHistoryPreviews:
    """The single-line guarantee for tool calls and tool results."""

    def test_tool_result_edit_file_multiline_message(self):
        import json

        msg = "Replaced 1 occurrence of '---\n\n## 6. Open Questions' with '---\n\n## 6. iPad Safar"
        content = json.dumps({"success": True, "message": msg, "filepath": "doc.md"})

        result = C.format_tool_result_preview(content)

        assert "\n" not in result, f"newline leaked into preview: {repr(result)}"
        assert "\u2713" in result

    def test_tool_result_error_with_newlines(self):
        import json

        content = json.dumps({"success": False, "error": "line1\nline2\n## bad"})

        result = C.format_tool_result_preview(content)

        assert "\n" not in result
        assert "\u2717" in result

    def test_tool_result_command_multiline_stdout(self):
        import json

        content = json.dumps(
            {
                "success": True,
                "exit_code": 0,
                "stdout": "line1\nline2\n## out",
                "stderr": "",
            }
        )

        assert "\n" not in C.format_tool_result_preview(content)

    def test_tool_result_non_json_multiline(self):
        assert "\n" not in C.format_tool_result_preview("raw\ntext\n## heading")

    def test_tool_result_fallback_dict_multiline_value(self):
        import json

        content = json.dumps({"custom_key": "val1\nval2\n## x"})

        assert "\n" not in C.format_tool_result_preview(content)

    def test_tool_result_normal_still_works(self):
        import json

        content = json.dumps(
            {"success": True, "filepath": "doc.md", "total_lines": 406, "view": "raw"}
        )

        result = C.format_tool_result_preview(content)

        assert "\n" not in result
        assert "doc.md" in result
        assert "406" in result

    def test_tool_result_long_message_truncated(self):
        import json

        content = json.dumps({"success": True, "message": "x" * 200})

        result = C.format_tool_result_preview(content)

        assert result.endswith("...")
        assert "\n" not in result

    def test_tool_result_empty(self):
        assert C.format_tool_result_preview("") == "tool result (empty)"

    def test_tool_call_multiline_content_arg(self):
        """Tool not in the key-arg table falls back to the first arg."""
        import json

        args = json.dumps({"content": "line1\nline2\n## Heading", "filepath": "x.md"})

        result = C.format_tool_call_preview("some_tool", args)

        assert "\n" not in result, f"newline leaked into preview: {repr(result)}"

    def test_tool_call_multiline_command_arg(self):
        import json

        args = json.dumps({"command": "echo a\necho b\n## Heading"})

        assert "\n" not in C.format_tool_call_preview("command", args)

    def test_tool_call_normal_still_works(self):
        import json

        args = json.dumps({"filepath": "doc.md", "find": "x"})

        result = C.format_tool_call_preview("edit_file", args)

        assert "\n" not in result
        assert "doc.md" in result

    def test_tool_call_without_arguments_is_just_the_name(self):
        assert C.format_tool_call_preview("square_number", "") == "square_number"


# ============== /tools ==============
# The listing is plain text and keeps the info pane open in "tools" mode.
# The per-tool mode breakdown and the argument-combination breakdown are the
# detail the TUI used to render itself; the combo rows only appear when a tool
# was called with more than one combination (a single row is noise).


def _fake_agent_with_tool_stats(
    calls=None,
    successes=None,
    modes=None,
    mode_successes=None,
    param_combos=None,
    param_combo_successes=None,
):
    from agent13.core import ToolStats

    class Agent:
        pass

    agent = Agent()
    agent.tool_stats = ToolStats(
        calls=calls or {},
        successes=successes or {},
        modes=modes or {},
        mode_successes=mode_successes or {},
        param_combos=param_combos or {},
        param_combo_successes=param_combo_successes or {},
    )
    return agent


class TestToolsCommand:
    def test_no_calls_says_so_and_keeps_the_pane(self):
        result = C.execute_tools(_fake_agent_with_tool_stats())

        assert result.success
        assert "No tool calls yet this session" in result.message
        assert result.pane == "tools"

    def test_totals_and_rate(self):
        agent = _fake_agent_with_tool_stats(
            calls={"read_file": 7, "edit_file": 3}, successes={"read_file": 6, "edit_file": 2}
        )

        result = C.execute_tools(agent)

        assert "8/10 successful (80%)" in result.message

    def test_tools_are_sorted_by_call_count(self):
        agent = _fake_agent_with_tool_stats(
            calls={"edit_file": 1, "read_file": 9}, successes={"edit_file": 1, "read_file": 9}
        )

        lines = C.execute_tools(agent).message.splitlines()

        assert lines[2].startswith("  read_file")
        assert lines[3].startswith("  edit_file")

    def test_mode_breakdown_is_inline(self):
        agent = _fake_agent_with_tool_stats(
            calls={"read_file": 5},
            successes={"read_file": 4},
            modes={"read_file": {"raw": 3, "lines": 2}},
            mode_successes={"read_file": {"raw": 2, "lines": 2}},
        )

        result = C.execute_tools(agent)

        assert "modes: lines=2/2, raw=2/3" in result.message

    def test_single_param_combo_is_not_listed(self):
        """One combination is the tool's normal shape - a row adds nothing."""
        agent = _fake_agent_with_tool_stats(
            calls={"read_file": 2},
            successes={"read_file": 2},
            param_combos={"read_file": {"filepath": 2}},
            param_combo_successes={"read_file": {"filepath": 2}},
        )

        result = C.execute_tools(agent)

        assert "filepath=" not in result.message

    def test_multiple_param_combos_are_listed_most_used_first(self):
        agent = _fake_agent_with_tool_stats(
            calls={"edit_file": 6},
            successes={"edit_file": 5},
            param_combos={"edit_file": {"filepath,find": 2, "filepath,find,content": 4}},
            param_combo_successes={
                "edit_file": {"filepath,find": 1, "filepath,find,content": 4}
            },
        )

        lines = C.execute_tools(agent).message.splitlines()

        assert "    filepath,find,content=4/4" in lines
        assert "    filepath,find=1/2" in lines
        assert lines.index("    filepath,find,content=4/4") < lines.index(
            "    filepath,find=1/2"
        )

    def test_message_is_plain_text(self):
        agent = _fake_agent_with_tool_stats(
            calls={"read_file": 1}, successes={"read_file": 1}
        )

        result = C.execute_tools(agent)

        assert "[" not in result.message
        assert "]" not in result.message


# ============== /wrapup, /chain ==============
# Both queue one agent turn. add_message is async, so the handler leaves the
# call in data["agent_message"] as (text, kind, data) and each front-end runs
# it - the same tail /journal last|all uses.


class TestWrapupCommand:
    def test_bare_queues_the_wrapup_turn(self):
        result = C.execute_wrapup(_fake_agent_with_groups(0), "")

        assert result.success
        assert result.data["agent_message"] == ("/wrapup", "wrapup", {})
        assert "Wrap-up queued" in result.message

    def test_focus_name_is_stripped_and_passed_as_data(self):
        result = C.execute_wrapup(_fake_agent_with_groups(0), "  my cool game  ")

        assert result.data["agent_message"] == (
            "/wrapup my cool game",
            "wrapup",
            {"focus_name": "my cool game"},
        )
        assert "my cool game" in result.message

    def test_whitespace_only_arg_is_bare(self):
        result = C.execute_wrapup(_fake_agent_with_groups(0), "   ")

        assert result.data["agent_message"] == ("/wrapup", "wrapup", {})


class TestChainCommand:
    def test_bare_queues_the_restart(self):
        result = C.execute_chain(_fake_agent_with_groups(0), "")

        assert result.success
        assert result.data["agent_message"] == ("/chain", "chain", {})
        assert "Chain restart queued" in result.message

    def test_no_skills_queues_the_flag(self):
        result = C.execute_chain(_fake_agent_with_groups(0), "no-skills")

        assert result.data["agent_message"] == (
            "/chain",
            "chain",
            {"no_skills": True},
        )

    def test_no_skills_spelling_variants(self):
        for spelling in ("--no-skills", "no_skills", "NO-SKILLS", " -no-skills "):
            result = C.execute_chain(_fake_agent_with_groups(0), spelling)
            assert result.data["agent_message"][2] == {"no_skills": True}, spelling

    def test_other_args_queue_a_plain_restart(self):
        result = C.execute_chain(_fake_agent_with_groups(0), "bogus")

        assert result.data["agent_message"][2] == {}


class TestQueuedTurnShape:
    """One key, one shape: every turn-queuing command emits a 3-tuple."""

    def test_all_queuing_commands_agree(self):
        agent = _fake_agent_with_groups(0)
        results = [
            C.execute_wrapup(agent, ""),
            C.execute_chain(agent, ""),
            C.run_command("journal", agent, "last"),
            C.run_command("journal", agent, "all"),
        ]

        for result in results:
            queued = result.data["agent_message"]
            assert len(queued) == 3
            text, kind, data = queued
            assert isinstance(text, str) and text.startswith("/")
            assert isinstance(kind, str) and kind
            assert isinstance(data, dict)


# ============== /mcp ==============
# The listing is synchronous and shared; connect/disconnect/reload are async,
# so the handler does the pre-checks and hands the action to the front-end via
# data["mcp_action"], which awaits run_mcp_action() for the result message.


class _FakeMCP:
    def __init__(self, servers=None, fail=False):
        self._servers = servers or {}
        self._fail = fail
        self.reloaded = False

    def get_server_info(self):
        return self._servers

    async def connect_all(self):
        if self._fail:
            raise RuntimeError("boom")
        return self._servers

    async def reload(self):
        if self._fail:
            raise RuntimeError("boom")
        self.reloaded = True
        return self._servers


def _fake_agent_for_mcp(configs=None, mcp=None, ensure=None):
    class Agent:
        pass

    agent = Agent()
    agent._mcp_server_configs = configs or {}
    agent.mcp = mcp

    async def _ensure_mcp():
        return ensure

    agent._ensure_mcp = _ensure_mcp

    async def disconnect_mcp():
        agent.mcp = None

    agent.disconnect_mcp = disconnect_mcp
    return agent


class TestLoadCommand:
    """`/load` validates the name and asks for a deferred load."""

    def _agent(self):
        from unittest.mock import MagicMock

        return MagicMock()

    def test_no_arg_lists_the_saves(self, monkeypatch):
        monkeypatch.setattr(C, "list_save_names", lambda: ["alpha", "beta"])

        result = C.execute_load(self._agent(), "")

        assert result.success
        assert "Available saves:" in result.message
        assert "  alpha" in result.message
        assert "  beta" in result.message
        assert "Usage: /load <name>" in result.message

    def test_no_arg_with_no_saves(self, monkeypatch):
        monkeypatch.setattr(C, "list_save_names", lambda: [])

        result = C.execute_load(self._agent(), "")

        assert "No saves found" in result.message
        assert "Usage: /load <name>" in result.message

    def test_missing_save_fails_with_the_save_hint(self, tmp_path, monkeypatch):
        import agent13.persistence as P

        monkeypatch.setattr(P, "resolve_save_path", lambda n: tmp_path / f"{n}.ctx")

        result = C.execute_load(self._agent(), "nope")

        assert result.success is False
        assert "Not found" in result.message
        assert "/save nope" in result.message
        assert "load_path" not in result.data

    def test_existing_save_asks_for_a_deferred_load(self, tmp_path, monkeypatch):
        import agent13.persistence as P

        target = tmp_path / "mine.ctx"
        target.write_text("{}")
        monkeypatch.setattr(P, "resolve_save_path", lambda n: target)

        result = C.execute_load(self._agent(), "mine")

        assert result.success
        assert result.data["load_path"] == str(target)
        assert "Load queued" in result.message

    def test_never_loads_inline(self, tmp_path, monkeypatch):
        # The load must land at a safe boundary behind queued work, so the
        # handler only reports the path.
        import agent13.persistence as P

        target = tmp_path / "mine.ctx"
        target.write_text("{}")
        monkeypatch.setattr(P, "resolve_save_path", lambda n: target)
        monkeypatch.setattr(
            P,
            "load_context",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("loaded inline")),
        )

        C.execute_load(self._agent(), "mine")

    def test_name_is_stripped(self, tmp_path, monkeypatch):
        import agent13.persistence as P

        seen = {}

        def _resolve(name):
            seen["name"] = name
            return tmp_path / "x.ctx"

        (tmp_path / "x.ctx").write_text("{}")
        monkeypatch.setattr(P, "resolve_save_path", _resolve)

        C.execute_load(self._agent(), "  mine  ")

        assert seen["name"] == "mine"


class TestNewSessionCommand:
    """`/new_session` asks for a deferred clear; it never wipes inline."""

    def _agent(self, idle: bool = True):
        from unittest.mock import MagicMock

        agent = MagicMock()
        agent.is_idle = idle
        return agent

    def test_asks_for_a_deferred_clear(self):
        result = C.execute_new_session(self._agent(), "")

        assert result.success
        assert result.data["clear_mode"] == "all"
        assert "New session queued" in result.message

    def test_never_clears_inline_even_when_idle(self):
        # is_idle looks at the running turn, not at the queue, so an inline
        # wipe could jump ahead of work already waiting.
        agent = self._agent(idle=True)

        C.execute_new_session(agent, "")

        agent.clear_messages.assert_not_called()
        agent.reset_session_counters.assert_not_called()

    def test_args_are_ignored(self):
        result = C.execute_new_session(self._agent(), "bogus")

        assert result.data["clear_mode"] == "all"

    def test_asks_for_no_ui_side_effects(self):
        result = C.execute_new_session(self._agent(), "")

        assert result.pane is None
        assert result.refresh_status is False

    def test_message_is_plain_text(self):
        result = C.execute_new_session(self._agent(), "")

        assert "[" not in result.message


class TestMcpListing:
    def test_not_initialized_names_the_configured_count(self):
        result = C.execute_mcp(_fake_agent_for_mcp(configs={"a": {}, "b": {}}), "")

        assert result.success
        assert "Status: Not initialized" in result.message
        assert "Configured servers: 2" in result.message
        assert "/mcp connect" in result.message
        assert result.pane == "mcp"

    def test_connected_but_no_servers(self):
        result = C.execute_mcp(_fake_agent_for_mcp(mcp=_FakeMCP({})), "")

        assert "Configured servers: 0" in result.message
        assert "Not initialized" not in result.message
        assert result.pane == "mcp"

    def test_servers_are_listed_plainly(self):
        mcp = _FakeMCP({"alpha": ["mcp://alpha/read", "mcp://alpha/write"]})
        result = C.execute_mcp(_fake_agent_for_mcp(mcp=mcp), "")

        assert "MCP Servers:" in result.message
        assert "alpha: 2 tools" in result.message
        assert "- read" in result.message
        assert "[" not in result.message, "shared messages are plain text"
        assert result.pane == "mcp"


class TestMcpActions:
    def test_connect_without_configs_fails_without_a_tail(self):
        result = C.execute_mcp(_fake_agent_for_mcp(configs={}), "connect")

        assert not result.success
        assert "No MCP servers configured" in result.message
        assert "mcp_action" not in result.data

    def test_connect_asks_the_front_end_to_run_it(self):
        result = C.execute_mcp(_fake_agent_for_mcp(configs={"a": {}}), "connect")

        assert result.success
        assert result.data["mcp_action"] == "connect"
        assert "Connecting" in result.message

    def test_disconnect_when_not_connected(self):
        result = C.execute_mcp(_fake_agent_for_mcp(mcp=None), "disconnect")

        assert not result.success
        assert "MCP not connected" in result.message
        assert "mcp_action" not in result.data

    def test_disconnect_asks_the_front_end_to_run_it(self):
        result = C.execute_mcp(_fake_agent_for_mcp(mcp=_FakeMCP()), "disconnect")

        assert result.data["mcp_action"] == "disconnect"

    def test_reload_when_not_initialized(self):
        result = C.execute_mcp(_fake_agent_for_mcp(mcp=None), "reload")

        assert not result.success
        assert "not initialized" in result.message
        assert "mcp_action" not in result.data

    def test_reload_asks_the_front_end_to_run_it(self):
        result = C.execute_mcp(_fake_agent_for_mcp(mcp=_FakeMCP()), "reload")

        assert result.data["mcp_action"] == "reload"

    def test_unknown_subcommand_shows_usage(self):
        result = C.execute_mcp(_fake_agent_for_mcp(), "bogus")

        assert not result.success
        assert "Usage: /mcp" in result.message
        assert "mcp_action" not in result.data

    def test_subcommand_is_case_insensitive(self):
        result = C.execute_mcp(_fake_agent_for_mcp(configs={"a": {}}), "CONNECT")

        assert result.data["mcp_action"] == "connect"


class TestRunMcpAction:
    async def test_connect_reports_the_connected_servers(self):
        mcp = _FakeMCP({"alpha": [], "beta": []})
        agent = _fake_agent_for_mcp(configs={"alpha": {}, "beta": {}}, ensure=mcp)

        ok, message = await C.run_mcp_action(agent, "connect")

        assert ok
        assert message == "MCP connected: ['alpha', 'beta']"

    async def test_connect_without_a_client_fails(self):
        agent = _fake_agent_for_mcp(configs={"a": {}}, ensure=None)

        ok, message = await C.run_mcp_action(agent, "connect")

        assert not ok
        assert "no MCP client" in message

    async def test_disconnect_reports_success(self):
        agent = _fake_agent_for_mcp(mcp=_FakeMCP())

        ok, message = await C.run_mcp_action(agent, "disconnect")

        assert ok
        assert message == "MCP servers disconnected"

    async def test_reload_reports_the_servers(self):
        mcp = _FakeMCP({"alpha": []})
        agent = _fake_agent_for_mcp(mcp=mcp)

        ok, message = await C.run_mcp_action(agent, "reload")

        assert ok
        assert mcp.reloaded
        assert message == "MCP reconnected: ['alpha']"

    async def test_an_error_is_reported_not_raised(self):
        agent = _fake_agent_for_mcp(mcp=_FakeMCP(fail=True))

        ok, message = await C.run_mcp_action(agent, "reload")

        assert not ok
        assert "MCP reload error: boom" in message

    async def test_unknown_action(self):
        ok, message = await C.run_mcp_action(_fake_agent_for_mcp(), "bogus")

        assert not ok
        assert "Unknown MCP action" in message


# ---------------------------------------------------------------------------
# /prompt, /skills, /snippet, /skillname - the manager-backed commands
# ---------------------------------------------------------------------------


class _FakePromptManager:
    def __init__(self, names=("default", "concise"), active="default"):
        self.prompts = {n: f"prompt text for {n}" for n in names}
        self.active_prompt = active

    def set_active(self, name):
        if name not in self.prompts:
            return False
        self.active_prompt = name
        return True

    def get_prompt(self, name=None):
        # The real PromptManager falls back to the default prompt for an
        # unknown name; resolve_compact_prompt depends on that.
        name = name or self.active_prompt
        if name not in self.prompts:
            name = "default"
        return self.prompts[name]


class _FakeSkillInfo:
    def __init__(self, description="A skill.", user_invoked=False):
        self.description = description
        self.disable_model_invocation = user_invoked


class _FakeSkillManager:
    def __init__(self, skills=None):
        self.skills = skills if skills is not None else {}

    def get_skill(self, name):
        return self.skills.get(name)

    def format_skill_content(self, name):
        return f"<skill_content name=\"{name}\">body</skill_content>"


class _FakeSnippetManager:
    def __init__(self, snippets=None):
        self.snippets = dict(snippets or {})

    def list_snippets(self):
        return [
            {"name": n, "preview": t[:20], "collision": False}
            for n, t in self.snippets.items()
        ]

    def get_snippet(self, name):
        return self.snippets.get(name)

    def validate_name(self, name):
        return None if name.isidentifier() else "Invalid snippet name"

    def add_snippet(self, name, content):
        self.snippets[name] = content
        return None

    def delete_snippet(self, name):
        return self.snippets.pop(name, None) is not None

    def rename_snippet(self, old, new):
        if old not in self.snippets:
            return f"Snippet not found: {old}"
        self.snippets[new] = self.snippets.pop(old)
        return None


def _manager_agent(prompt_manager=None, skill_manager=None, snippet_manager=None):
    """A stub agent carrying the three managers and their state."""

    class Agent:
        pass

    agent = Agent()
    agent.prompt_manager = prompt_manager
    agent.skill_manager = skill_manager
    agent.snippet_manager = snippet_manager
    agent.skills_mode = False
    agent.skills_list_enabled = False
    agent.system_prompt = "base prompt"
    agent.system_prompt_set = []
    agent.set_system_prompt = agent.system_prompt_set.append
    agent.reload_agents_md = lambda: None
    agent.set_skills_mode = lambda on: setattr(agent, "skills_mode", on)
    agent.note_skill_loaded = lambda name, content: None
    return agent


class TestPromptHandler:
    """/prompt: switch the system prompt, list them on no args."""

    def test_switch_sets_the_prompt_and_reloads_agents_md(self):
        pm = _FakePromptManager()
        agent = _manager_agent(prompt_manager=pm)
        reloaded = []
        agent.reload_agents_md = lambda: reloaded.append(True)

        result = C.run_command("prompt", agent, "concise")

        assert result.success
        assert pm.active_prompt == "concise"
        assert agent.system_prompt_set == ["prompt text for concise"]
        assert reloaded == [True]
        assert result.refresh_status is True

    def test_unknown_prompt_lists_the_available_ones(self):
        agent = _manager_agent(prompt_manager=_FakePromptManager())

        result = C.run_command("prompt", agent, "nope")

        assert not result.success
        assert "Prompt not found: nope" in result.message
        assert "default" in result.message and "concise" in result.message

    def test_no_args_lists_and_asks_for_the_picker(self):
        agent = _manager_agent(prompt_manager=_FakePromptManager(active="concise"))

        result = C.run_command("prompt", agent, "")

        assert result.success
        assert "Available prompts:" in result.message
        assert "* concise" in result.message
        names, active = result.data["prompt_picker"]
        assert names == ["default", "concise"]
        assert active == "concise"

    def test_without_a_manager_it_explains_itself(self):
        result = C.run_command("prompt", _manager_agent(), "default")

        assert not result.success
        assert "No prompt manager configured" in result.message


class TestSkillsHandler:
    """/skills: toggle the list, report both switches, list the skills."""

    def _agent(self, skills=None, skills_mode=False, list_enabled=False):
        sm = _FakeSkillManager(
            skills
            if skills is not None
            else {"alpha": _FakeSkillInfo("First skill.")}
        )
        agent = _manager_agent(prompt_manager=_FakePromptManager(), skill_manager=sm)
        agent.skills_mode = skills_mode
        agent.skills_list_enabled = list_enabled
        return agent

    def test_on_enables_the_tool_and_the_list(self):
        agent = self._agent()

        result = C.run_command("skills", agent, "on")

        assert result.success
        assert agent.skills_mode is True
        assert agent.skills_list_enabled is True
        assert "alpha" in agent.system_prompt_set[0], "list goes into the prompt"

    def test_off_drops_the_list_but_not_the_tool(self):
        agent = self._agent(skills_mode=True, list_enabled=True)

        result = C.run_command("skills", agent, "off")

        assert result.success
        assert agent.skills_mode is True, "the tool has no off-switch"
        assert agent.skills_list_enabled is False
        assert "alpha" not in agent.system_prompt_set[0], "no skills section"

    def test_status_reports_both_switches(self):
        agent = self._agent(skills_mode=True, list_enabled=False)

        result = C.run_command("skills", agent, "status")

        assert "List in prompt: off" in result.message
        assert "Skill tool:     on" in result.message
        assert "1 skill(s) available" in result.message

    def test_list_separates_user_invoked_skills(self):
        agent = self._agent(
            skills={
                "alpha": _FakeSkillInfo("First skill."),
                "beta": _FakeSkillInfo("Hidden one.", user_invoked=True),
            }
        )

        result = C.run_command("skills", agent, "")

        assert "/alpha - First skill." in result.message
        assert "User-invoked only" in result.message
        assert "/beta - Hidden one." in result.message

    def test_no_skills(self):
        agent = self._agent(skills={})

        assert C.run_command("skills", agent, "").message == "No skills available"

    def test_unknown_subcommand_shows_usage(self):
        result = C.run_command("skills", self._agent(), "maybe")

        assert not result.success
        assert "Usage: /skills" in result.message

    def test_without_a_manager_it_explains_itself(self):
        result = C.run_command("skills", _manager_agent(), "on")

        assert not result.success
        assert "No skill manager configured" in result.message


class TestSnippetHandler:
    """/snippet: list, add last, delete, rename, use."""

    def _agent(self, snippets=None, last_message="the last thing I typed"):
        sm = _FakeSnippetManager(snippets)
        agent = _manager_agent(snippet_manager=sm)

        class History:
            @staticmethod
            def find_last_user_idx():
                return 0 if last_message else None

        agent.history = History()
        agent.messages = [{"content": last_message}]
        return agent

    def test_no_args_shows_usage(self):
        result = C.run_command("snippet", self._agent(), "")

        assert result.success
        assert "Usage:" in result.message
        assert "/snippet use <name>" in result.message

    def test_list(self):
        agent = self._agent({"one": "first snippet text"})

        result = C.run_command("snippet", agent, "list")

        assert "Available snippets:" in result.message
        assert "one" in result.message

    def test_list_empty(self):
        result = C.run_command("snippet", self._agent(), "list")

        assert result.message == "No snippets defined."

    def test_add_last_saves_the_last_user_message(self):
        agent = self._agent()

        result = C.run_command("snippet", agent, "add last mine")

        assert result.success
        assert agent.snippet_manager.snippets["mine"] == "the last thing I typed"
        assert result.data["snippet_added"] == "mine"

    def test_add_last_needs_the_last_keyword(self):
        result = C.run_command("snippet", self._agent(), "add mine")

        assert not result.success
        assert "Usage: /snippet add last" in result.message

    def test_add_last_refuses_to_overwrite_without_force(self):
        agent = self._agent({"mine": "old"})

        result = C.run_command("snippet", agent, "add last mine")

        assert not result.success
        assert "already exists" in result.message
        assert agent.snippet_manager.snippets["mine"] == "old"

        forced = C.run_command("snippet", agent, "add last mine -y")
        assert forced.success
        assert agent.snippet_manager.snippets["mine"] == "the last thing I typed"

    def test_add_last_with_no_history(self):
        agent = self._agent(last_message=None)

        result = C.run_command("snippet", agent, "add last mine")

        assert not result.success
        assert "No user message to save" in result.message

    def test_add_last_rejects_an_invalid_name(self):
        result = C.run_command("snippet", self._agent(), "add last 1bad")

        assert not result.success
        assert "Invalid snippet name" in result.message

    def test_delete(self):
        agent = self._agent({"mine": "text"})

        result = C.run_command("snippet", agent, "delete mine")

        assert result.success
        assert "mine" not in agent.snippet_manager.snippets
        assert result.data["snippet_removed"] == "mine"

    def test_delete_missing(self):
        result = C.run_command("snippet", self._agent(), "delete nope")

        assert not result.success
        assert "Snippet not found: nope" in result.message

    def test_rename_moves_the_slash_command(self):
        agent = self._agent({"old": "text"})

        result = C.run_command("snippet", agent, "rename old new")

        assert result.success
        assert "new" in agent.snippet_manager.snippets
        assert result.data["snippet_removed"] == "old"
        assert result.data["snippet_added"] == "new"

    def test_rename_missing(self):
        result = C.run_command("snippet", self._agent(), "rename old new")

        assert not result.success
        assert "Snippet not found: old" in result.message

    def test_use_carries_the_text_for_the_front_end(self):
        agent = self._agent({"mine": "the snippet body"})

        result = C.run_command("snippet", agent, "use mine")

        assert result.success
        assert result.data["snippet_text"] == "the snippet body"
        assert "the snippet body" in result.message

    def test_use_missing(self):
        result = C.run_command("snippet", self._agent(), "use nope")

        assert not result.success
        assert "Snippet not found: nope" in result.message

    def test_unknown_subcommand_shows_usage(self):
        result = C.run_command("snippet", self._agent(), "bogus")

        assert not result.success
        assert "Unknown snippet command: bogus" in result.message

    def test_without_a_manager_it_explains_itself(self):
        result = C.run_command("snippet", _manager_agent(), "list")

        assert not result.success
        assert "No snippet manager configured" in result.message


class TestSkillInvocation:
    """/skillname: the dynamic dispatch both front-ends share."""

    def _agent(self, skills=None, skills_mode=False):
        sm = _FakeSkillManager(
            skills if skills is not None else {"alpha": _FakeSkillInfo()}
        )
        agent = _manager_agent(skill_manager=sm)
        agent.skills_mode = skills_mode
        agent.loaded = []
        agent.note_skill_loaded = lambda name, content: agent.loaded.append(
            (name, content)
        )
        return agent

    def test_unknown_name_returns_none(self):
        assert C.execute_skill_invocation(self._agent(), "nope") is None

    def test_no_manager_returns_none(self):
        assert C.execute_skill_invocation(_manager_agent(), "alpha") is None

    def test_loads_and_queues_one_turn(self):
        agent = self._agent()

        result = C.execute_skill_invocation(agent, "alpha")

        assert result.success
        text, kind, data = result.data["agent_message"]
        assert text.startswith('<skill_content name="alpha">')
        assert kind == "prompt"
        assert agent.loaded == [("alpha", text)]

    def test_ratchets_the_tool_on(self):
        agent = self._agent(skills_mode=False)

        C.execute_skill_invocation(agent, "alpha")

        assert agent.skills_mode is True

    def test_does_not_re_set_an_already_enabled_tool(self):
        agent = self._agent(skills_mode=True)
        calls = []
        agent.set_skills_mode = lambda on: calls.append(on)

        C.execute_skill_invocation(agent, "alpha")

        assert calls == []

    def test_args_are_appended_to_the_content(self):
        agent = self._agent()

        result = C.execute_skill_invocation(agent, "alpha", "do the other thing")

        text = result.data["agent_message"][0]
        assert text.endswith("body</skill_content>\n\ndo the other thing")
        assert agent.loaded[0][1].endswith("do the other thing")

    def test_unloadable_content_fails(self):
        agent = self._agent()
        agent.skill_manager.format_skill_content = lambda name: ""

        result = C.execute_skill_invocation(agent, "alpha")

        assert not result.success
        assert "Failed to load skill: alpha" in result.message


class TestCompactHandler:
    """/compact: queue the compaction turn with the resolved prompt."""

    def _agent(self, prompt_manager=None):
        agent = _manager_agent(prompt_manager=prompt_manager or _FakePromptManager())
        return agent

    def test_no_args_queues_the_default_prompt(self):
        result = C.run_command("compact", self._agent(), "")

        assert result.success
        text, kind, data = result.data["agent_message"]
        assert text == "/compact"
        assert kind == "compact"
        assert data["compact_prompt"], "a prompt is always resolved"
        assert data["user_specified_prompt"] is False

    def test_focus_text_is_the_queued_message(self):
        result = C.run_command("compact", self._agent(), "finish the registry work")

        text, _kind, data = result.data["agent_message"]
        assert text == "/compact finish the registry work"
        assert data["user_specified_prompt"] is True

    def test_prompt_flag_selects_the_prompt(self):
        pm = _FakePromptManager()
        result = C.run_command("compact", self._agent(pm), "--prompt concise")

        _text, _kind, data = result.data["agent_message"]
        assert data["compact_prompt"] == "prompt text for concise"

    def test_unknown_prompt_is_an_error_without_a_queue(self):
        result = C.run_command("compact", self._agent(), "--prompt nope")

        assert not result.success
        assert "nope" in result.message
        assert "agent_message" not in result.data


class TestStatusHandler:
    """/status: one plain-text report built entirely from the agent."""

    def _agent(self, **overrides):
        agent = _manager_agent(prompt_manager=_FakePromptManager())
        agent.provider_name = "studio"
        agent.model = "GLM-5.2-mlx"
        agent.session_start_time = time.time()
        agent.tracker = None
        agent.spinner_speed = "fast"
        agent.clipboard_method = "osc52"
        agent.reasoning_collapsed = True
        agent.toolview_collapsed = False
        agent.last_turn_duration = None
        agent.last_turn_end_time = None
        agent.resumed_turn_count = 0
        agent.resumed_saved_at = None
        agent.resumed_prompt_tokens = 0
        agent.resumed_completion_tokens = 0
        agent.journal_mode = False
        agent.devel_mode = False
        agent.remove_reasoning = False
        agent.auto_context_threshold = 220000
        agent.auto_context_action = "report_and_compact"
        agent.auto_context_chain = 3
        agent.auto_context_chain_used = 0
        agent.auto_save_idle_seconds = 300
        agent.messages = []
        agent.mcp = None
        agent.tool_stats = type(
            "S",
            (),
            {"total_successes": 0, "total_calls": 0, "bounded": 0},
        )()
        agent.queue = type("Q", (), {"pending_count": 0})()
        agent.is_pausing = False
        agent.is_paused = False
        agent.status = type("St", (), {"value": "IDLE"})()
        for k, v in overrides.items():
            setattr(agent, k, v)
        return agent

    def test_sections_and_identity(self):
        result = C.run_command("status", self._agent(), "")

        assert result.success
        assert result.pane == "status"
        for section in ("Session", "Provider", "Context", "Tools", "Settings"):
            assert section in result.message
        assert "provider: studio" in result.message
        assert "model: GLM-5.2-mlx" in result.message
        assert "prompt: default" in result.message

    def test_message_is_plain_text(self):
        result = C.run_command("status", self._agent(), "")

        assert "[" not in result.message
        assert "[/]" not in result.message

    def test_display_flags_come_from_the_agent(self):
        agent = self._agent(reasoning_collapsed=True, toolview_collapsed=False)

        text = C.run_command("status", agent, "").message

        assert "reasoning widgets: collapsed" in text
        assert "tool widgets: expanded" in text
        assert "spinner: fast" in text
        assert "clipboard: osc52" in text

    def test_last_turn_is_reported(self):
        agent = self._agent(last_turn_duration=3.0, last_turn_end_time=time.time())

        text = C.run_command("status", agent, "").message

        assert "last turn: 3s" in text

    def test_resumed_section_only_when_resumed(self):
        assert "Resumed session" not in C.run_command(
            "status", self._agent(), ""
        ).message

        agent = self._agent(
            resumed_turn_count=4, resumed_prompt_tokens=1000, resumed_completion_tokens=500
        )
        text = C.run_command("status", agent, "").message
        assert "Resumed session" in text
        assert "previous: 4 turns" in text

    def test_no_bell_manager_reports_off(self):
        text = C.run_command("status", self._agent(), "").message

        assert "bell: off" in text
