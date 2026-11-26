"""The shipped prompts.yaml and the Python prompt constants must not diverge.

There is exactly ONE copy of each prompt in the repo. The shipped yaml carries
the hand-written prompts; `default`, `report_and_compact` and `compaction` are
written into the USER's file from the constants on first run
(GENERATED_PROMPTS). This file pins both halves of that arrangement:

- the shipped yaml must not grow a second copy of a generated prompt (the
  drift this replaced was behavioural, not cosmetic: core.compact_history
  compares the resolved compaction prompt to DEFAULT_COMPACT_PROMPT BY VALUE
  to decide whether to append the durable-record pointer block);
- the writer must emit every generated prompt, byte-identical to its constant.

The YAML only reaches fresh installs (ensure_default_prompts never overwrites
an existing file), so a silent divergence means new users get a stale prompt
with no error anywhere.
"""

import yaml

from agent13.prompts import (
    DEFAULT_COMPACT_PROMPT,
    DEFAULT_PROMPT,
    DEFAULT_PROMPTS_FILE,
    DEFAULT_REPORT_AND_COMPACT_PROMPT,
    GENERATED_PROMPTS,
    _render_generated_prompts,
    ensure_default_prompts,
)


def _shipped() -> dict:
    return yaml.safe_load(DEFAULT_PROMPTS_FILE.read_text(encoding="utf-8"))


class TestShippedYamlHasNoSecondCopy:
    def test_no_generated_prompt_appears_in_the_shipped_file(self):
        """A second copy is what drifts. Do not add these keys back."""
        shipped = _shipped()
        for key in GENERATED_PROMPTS:
            assert key not in shipped, (
                f"'{key}' is generated from a Python constant; a copy in "
                f"default_prompts.yaml is a second home that drifts silently. "
                f"See GENERATED_PROMPTS in agent13/prompts.py."
            )

    def test_the_hand_written_prompts_are_still_there(self):
        shipped = _shipped()
        assert {"concise", "redteam", "code-reviewer", "mentor"} <= set(shipped)

    def test_generated_prompts_are_the_constants(self):
        """The mapping is the single source; keep it pointing at the constants."""
        assert GENERATED_PROMPTS == {
            "default": DEFAULT_PROMPT,
            "report_and_compact": DEFAULT_REPORT_AND_COMPACT_PROMPT,
            "compaction": DEFAULT_COMPACT_PROMPT,
        }


class TestGeneratedBlockRoundTrips:
    def test_rendered_block_parses_back_to_the_mapping(self):
        """The YAML emitter must not mangle a 9.5k prompt (indent, unicode)."""
        assert yaml.safe_load(_render_generated_prompts()) == GENERATED_PROMPTS


class TestEnsureDefaultPrompts:
    def test_writer_emits_every_prompt(self, monkeypatch, tmp_path):
        """A fresh install gets the hand-written prompts AND the generated ones."""
        monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
        ensure_default_prompts()

        written = yaml.safe_load((tmp_path / "prompts.yaml").read_text("utf-8"))
        assert set(written) == set(_shipped()) | set(GENERATED_PROMPTS)
        for key, text in GENERATED_PROMPTS.items():
            assert written[key] == text, f"{key} differs from its constant"

    def test_writer_preserves_the_hand_written_prompts(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
        ensure_default_prompts()

        written = yaml.safe_load((tmp_path / "prompts.yaml").read_text("utf-8"))
        shipped = _shipped()
        for key, text in shipped.items():
            assert written[key] == text

    def test_writer_keeps_the_header_comment(self, monkeypatch, tmp_path):
        """The header is the only place a user learns the file is editable."""
        monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
        ensure_default_prompts()

        text = (tmp_path / "prompts.yaml").read_text("utf-8")
        assert text.startswith("# Default system prompts for agent13")
        assert "Edit that file to customize" in text

    def test_writer_never_overwrites_an_existing_file(self, monkeypatch, tmp_path):
        """A user's edits always win - the generated keys are not re-injected."""
        monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
        target = tmp_path / "prompts.yaml"
        target.write_text("default: |\n  mine\n", encoding="utf-8")

        ensure_default_prompts()

        assert target.read_text(encoding="utf-8") == "default: |\n  mine\n"
