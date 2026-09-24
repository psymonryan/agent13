"""Pin the shipped default_prompts.yaml to the Python prompt constants.

The YAML only reaches fresh installs (ensure_default_prompts never overwrites
an existing prompts.yaml), so a silent drift means new users get a stale
wrap-up/compaction prompt with no error anywhere. This test is the drift
alarm: change a constant, forget the YAML, and CI fails here.
"""

import yaml

from agent13.prompts import (
    DEFAULT_PROMPTS_FILE,
    DEFAULT_REPORT_AND_COMPACT_PROMPT,
    DEFAULT_COMPACT_PROMPT,
)


class TestDefaultPromptsYamlPinned:
    def test_report_and_compact_matches_constant(self):
        shipped = yaml.safe_load(DEFAULT_PROMPTS_FILE.read_text(encoding="utf-8"))
        assert (
            shipped["report_and_compact"].strip()
            == DEFAULT_REPORT_AND_COMPACT_PROMPT.strip()
        ), (
            "default_prompts.yaml 'report_and_compact' has drifted from "
            "DEFAULT_REPORT_AND_COMPACT_PROMPT. Regenerate the YAML block "
            "from the constant (see docs_archive/chain_prompting_design.md §9)."
        )

    def test_compaction_matches_constant(self):
        shipped = yaml.safe_load(DEFAULT_PROMPTS_FILE.read_text(encoding="utf-8"))
        assert shipped["compaction"].strip() == DEFAULT_COMPACT_PROMPT.strip(), (
            "default_prompts.yaml 'compaction' has drifted from "
            "DEFAULT_COMPACT_PROMPT. Regenerate the YAML block from the "
            "constant (see docs_archive/chain_prompting_design.md §9)."
        )
