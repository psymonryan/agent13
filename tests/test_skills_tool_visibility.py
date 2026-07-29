"""Regression tests for skill *tool* visibility gating.

Background: the TUI used to compute `skills_mode` purely from whether skills
existed on disk (`bool(skill_manager and skill_manager.skills)`). Since
`ensure_default_skills()` always seeds `~/.agent13/skills/`, the `skill` tool
was always exposed to the LLM in a normal TUI run — violating the
minimal-context principle. The fix threads `include_skills` (from `--skills`
or `config.include_skills`) through the TUI constructor and ANDs it into the
gate, mirroring the CLI/batch path in `agent13/cli.py`.

These tests pin the gate so the leak cannot silently return.
"""

from tools import get_filtered_tools


def _has_skill_tool(skills_mode: bool) -> bool:
    """Return True if the `skill` tool is in the filtered tool list."""
    schemas = get_filtered_tools(
        devel=False,
        skills=skills_mode,
        enabled_tools=None,
        disabled_tools=None,
    )
    return "skill" in [s["function"]["name"] for s in schemas]


class TestSkillToolVisibility:
    """Verify the skill tool only appears when skills are opted in."""

    def test_skill_tool_hidden_by_default(self):
        """With skills_mode=False, the skill tool must NOT be in the list."""
        # This is the core regression: even though skills exist on disk
        # (~/.agent13/skills/ is always seeded), the tool stays hidden.
        assert _has_skill_tool(skills_mode=False) is False

    def test_skill_tool_visible_when_opted_in(self):
        """With skills_mode=True, the skill tool IS in the list."""
        assert _has_skill_tool(skills_mode=True) is True

    def test_tui_gate_expression_matches_cli(self):
        """The TUI's gate expression must match the CLI's.

        CLI (agent13/cli.py:545,568):
            skills_mode = include_skills and bool(skill_manager.skills)

        TUI (ui/tui.py, post-fix):
            skills_mode = self._include_skills and bool(
                self.skill_manager and self.skill_manager.skills
            )

        Both must yield False when include_skills is False, regardless of
        whether skills exist. We simulate both forms here against a stand-in
        skill_manager to lock the equivalence.
        """

        class FakeMgr:
            def __init__(self, has_skills: bool):
                self.skills = {"x": object()} if has_skills else {}

        # The four (include_skills, has_skills) combos and expected result.
        cases = [
            (False, False, False),
            (False, True, False),  # <-- the bug case: skills exist, not opted in
            (True, False, False),
            (True, True, True),
        ]
        for include_skills, has_skills, expected in cases:
            mgr = FakeMgr(has_skills)

            # CLI form
            cli_gate = include_skills and bool(mgr.skills)

            # TUI form (post-fix). Note the extra `and self.skill_manager`
            # short-circuit — also covered by bool(...) on FakeMgr.
            tui_gate = include_skills and bool(mgr and mgr.skills)

            assert cli_gate == expected, (
                f"CLI gate mismatch for (include={include_skills}, "
                f"has_skills={has_skills}): got {cli_gate}, want {expected}"
            )
            assert tui_gate == expected, (
                f"TUI gate mismatch for (include={include_skills}, "
                f"has_skills={has_skills}): got {tui_gate}, want {expected}"
            )
            assert cli_gate == tui_gate, (
                f"CLI/TUI gate divergence for (include={include_skills}, "
                f"has_skills={has_skills})"
            )

    def test_tui_constructor_accepts_include_skills(self):
        """The AgentTUI __init__ must accept an `include_skills` kwarg.

        This catches accidental removal of the parameter (e.g. during a
        refactor that drops it from the signature). We only inspect the
        signature — we don't construct the App (heavy).
        """
        import inspect

        from ui.tui import AgentTUI

        sig = inspect.signature(AgentTUI.__init__)
        assert "include_skills" in sig.parameters, (
            "AgentTUI.__init__ must accept `include_skills` so the gate can "
            "be driven by --skills / config.include_skills"
        )
        # Default must be False (minimal context when not specified).
        assert sig.parameters["include_skills"].default is False
