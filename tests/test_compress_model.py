"""Tests for the status-bar model name compression (ui.tui.compress_model)."""

from ui.tui import compress_model


def test_truncates_long_name_at_dash():
    """A long model name is cut at a dash boundary with a __ marker."""
    assert (
        compress_model("Qwen3.8-27B-MTPLX-Optimized-Quality")
        == "Qwen3.8-27B__"
    )


def test_keeps_short_name_unchanged():
    """A name that fits the budget is returned unchanged."""
    assert compress_model("GLM-5.1") == "GLM-5.1"


def test_keeps_name_that_fits_budget():
    """A 16-char name fits exactly and is not truncated."""
    assert compress_model("gpt-oss-120b-oQ8") == "gpt-oss-120b-oQ8"


def test_distinguishable_variants_stay_distinct():
    """A fixed pair-count would collapse these; the budget keeps them apart."""
    assert compress_model("gpt-oss-120b-oQ8") != compress_model("gpt-oss-20b-oQ8")


def test_keeps_size_when_it_fits():
    """The size token is preserved when the whole name fits."""
    assert compress_model("Qwen-3.6-27B") == "Qwen-3.6-27B"


def test_shortens_thinking_suffix_to_three():
    """A trailing :value thinking suffix is shortened to its first 3 letters."""
    assert (
        compress_model("Qwen3.8-27B-MTPLX-Optimized-Quality:medium")
        == "Qwen3.8-27B__:med"
    )


def test_thinking_suffix_on_fitting_name():
    """The suffix is shortened even when the base name fits."""
    assert compress_model("GLM-5.1:medium") == "GLM-5.1:med"


def test_all_thinking_verbs_shorten():
    """Every thinking verb is reduced to three letters."""
    expected = {
        "nothink": "not",
        "none": "non",
        "low": "low",
        "medium": "med",
        "high": "hig",
        "xhigh": "xhi",
        "max": "max",
    }
    for verb, short in expected.items():
        assert compress_model(f"GLM-5.1:{verb}") == f"GLM-5.1:{short}", verb


def test_no_suffix_no_dash():
    """A simple name with no dash or suffix is unchanged."""
    assert compress_model("devstral2") == "devstral2"


def test_long_no_dash_name_hard_truncates():
    """A name with no dash that overruns the budget is hard-truncated."""
    result = compress_model("a" * 24)
    assert result == "a" * 16 + "__"
    assert len(result) == 18  # 16 budget + 2 marker


def test_custom_budget():
    """The budget is configurable."""
    # budget 7: "GLM-5.1" fits (7), "GLM-5.1-oQ4e" would not.
    assert compress_model("GLM-5.1-oQ4e-mtp", budget=7) == "GLM-5.1__"
