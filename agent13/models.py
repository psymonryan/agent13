"""Shared model selection utilities.

These functions are used by CLI, TUI, and headless mode.
"""

import sys
from openai import AsyncOpenAI


async def fetch_models(client: AsyncOpenAI) -> list[str]:
    """Fetch available models from the API.

    Args:
        client: AsyncOpenAI client

    Returns:
        Sorted list of model names

    Raises:
        RuntimeError: If models cannot be fetched
    """
    try:
        models = await client.models.list()
        return sorted([m.id for m in models.data])
    except Exception as e:
        raise RuntimeError(f"Failed to fetch models: {e}")


def resolve_from_list(
    items: list[str],
    choice: str,
    label: str = "item",
    output=None,
) -> str | None:
    """Resolve a selection from a numbered list by index or name.

    Shared logic for /model, /provider, and any future numbered-list selection.

    Args:
        items: List of available items
        choice: User's choice (number like "1" or name/partial name)
        label: Human-readable label for error messages (e.g. "model", "provider")
        output: File object for error output (default: sys.stdout)

    Returns:
        Selected item, or None if ambiguous/not found
    """
    if output is None:
        output = sys.stdout

    if not choice:
        return None

    # Numeric selection
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(items):
            return items[idx]
        else:
            print(
                f"{label.capitalize()} number {choice} out of range (1-{len(items)})",
                file=output,
            )
            return None

    # Exact match
    if choice in items:
        return choice

    # Partial match (case-insensitive)
    matches = [m for m in items if choice.lower() in m.lower()]

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"Ambiguous {label} '{choice}'. Matches:", file=output)
        for i, m in enumerate(matches, 1):
            print(f"  {i}. {m}", file=output)
        if len(matches) <= 10:
            print("Use a more specific name or number.", file=output)
        else:
            print(
                f"({len(matches)} total matches. Use a more specific name or number.)",
                file=output,
            )
        return None
    else:
        print(f"No {label} matching '{choice}'", file=output)
        return None


def resolve_model_selection(
    model_names: list[str], choice: str, use_stderr: bool = False
) -> str | None:
    """Resolve a model selection by number or name, with optional alias.

    An alias may be appended after a colon, e.g. "GLM:nothink" or
    "3:nothink".  The portion before the last colon is resolved against
    the model list (by number, exact name, or partial name); the alias
    is then re-attached to the resolved name with a colon separator.

    Model names that themselves contain colons (e.g.
    "meta-llama/llama-3.1:free" on OpenRouter) are matched exactly
    first, so they work unchanged when typed in full.

    Args:
        model_names: List of available model names
        choice: User's choice (number, name, or name:alias)
        use_stderr: If True, print errors to stderr instead of stdout

    Returns:
        Selected model name (with alias re-attached if given), or None
        if ambiguous/not found
    """
    output = sys.stderr if use_stderr else sys.stdout

    # No colon — resolve as before (unchanged behaviour)
    if ":" not in choice:
        return resolve_from_list(model_names, choice, label="model", output=output)

    # Colon present — first try an exact match on the full string.
    # This handles provider model names that contain colons (e.g.
    # "meta-llama/llama-3.1:free" on OpenRouter) so they work unchanged.
    if choice in model_names:
        return choice

    # No exact match — split on the LAST colon.  Everything before is
    # the model selector; everything after is the alias.
    model_part, _, alias = choice.rpartition(":")
    if not model_part:
        # Leading colon (e.g. ":nothink") — nothing to resolve
        print(f"No model matching '{choice}'", file=output)
        return None

    resolved = resolve_from_list(model_names, model_part, label="model", output=output)
    if resolved is None:
        return None

    # Re-attach alias (drop trailing colon if alias is empty)
    return f"{resolved}:{alias}" if alias else resolved


async def select_model(model_names: list[str], model_arg: str = None) -> str:
    """Select a model, exiting if not found.

    Args:
        model_names: List of available model names
        model_arg: Optional model selection (number or name)

    Returns:
        Selected model name

    Raises:
        SystemExit: If model cannot be resolved
    """
    if model_arg:
        model = resolve_model_selection(model_names, model_arg, use_stderr=True)
        if model:
            return model
        sys.exit(1)

    # No argument - list models and prompt
    print("\nAvailable models:")
    for i, name in enumerate(model_names, 1):
        print(f"  {i}. {name}")
    print()

    while True:
        try:
            choice = input("Select model (number or name, or 'q' to quit): ").strip()
            if choice.lower() == "q":
                sys.exit(0)
            model = resolve_model_selection(model_names, choice)
            if model:
                return model
        except EOFError:
            sys.exit(0)


def print_model_list(model_names: list[str], current: str = None):
    """Print a list of available models.

    Args:
        model_names: List of model names
        current: Currently selected model (shown with *)
    """
    print("\nAvailable models:")
    for i, name in enumerate(model_names, 1):
        marker = " *" if name == current else ""
        print(f"  {i}. {name}{marker}")
    print()
