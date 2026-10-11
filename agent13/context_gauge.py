"""Context gauge — grounded token accounting for context-size decisions.

The gauge owns two facts about the most recent LLM call:

- ``grounded``: the prompt_tokens the backend reported for it — the true
  size of the request that went out (tokenizer-specific).
- ``cursor``: how many messages that request contained. The grounded value
  covers exactly ``messages[:cursor]``.

``estimate()`` returns grounded + a chars/4 estimate of the messages
appended after the cursor (the tool results a mid-turn safe point sees
before the next LLM call). A shrink guard handles the one case where the
grounded value is dead: if the history has since SHRUNK below the cursor
(a compact rewrote the messages to a summary pair), the number describes a
history that no longer exists, and the gauge falls back to a fresh chars/4
estimate of the current history (system prompt + messages).

The gauge is pure state: every method takes what it needs (no constructor
bindings), so no closure can go stale, and tests feed it plain data.
"""


def estimate_message_tokens(messages: list) -> int:
    """Roughly estimate the token count of a list of messages.

    Uses a chars/4 heuristic (no tokenizer dependency). Adequate for a
    threshold check given the headroom between the auto-context threshold
    and the model's real context limit.

    Image content blocks are counted as a fixed 170 tokens (the actual
    API cost for a medium-resolution image), not as the length of the
    base64 data URI.
    """
    IMAGE_BLOCK_TOKENS = 170  # ~token cost of one image block
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += len(content) // 4
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type")
                    if btype == "text":
                        total += len(block.get("text", "")) // 4
                    elif btype == "image_url":
                        total += IMAGE_BLOCK_TOKENS
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            total += len(fn.get("arguments", "") or "") // 4
    return total


class ContextGauge:
    """Grounded prompt-token state for context-size decisions.

    Single store for the grounded value and the stream cursor. The Agent
    feeds it at four sites (stream start, stream usage, history rewrite,
    tokenizer change) and reads ``estimate()`` at the two decision points
    (pre-turn check, mid-turn safe point).
    """

    def __init__(self) -> None:
        self._grounded = 0
        self._cursor = 0

    @property
    def grounded(self) -> int:
        """The grounded prompt-token value (0 = ungrounded)."""
        return self._grounded

    @property
    def cursor(self) -> int:
        """Messages covered by the grounded value (messages[:cursor])."""
        return self._cursor

    def record_stream_start(self, msg_count: int) -> None:
        """The next LLM request will contain exactly messages[:msg_count].

        Called before each stream; after the stream's token_usage arrives
        (record_usage), grounded + cursor together describe that request.
        """
        self._cursor = msg_count

    def record_usage(self, tokens: int) -> None:
        """Ground the gauge to the last LLM call's reported prompt tokens.

        Only the grounded value changes; the cursor stays where the last
        record_stream_start put it. A 0 here (token-usage reset) ungrounds
        the value without moving the cursor — exactly what the plain
        attribute assignment did.
        """
        self._grounded = tokens

    def reground(
        self, messages: list, system_len: int, tools_len: int | None = None
    ) -> int:
        """Re-ground to a fresh chars/4 estimate of the current history.

        System prompt + messages + optional serialized tools schema. Called
        after the history is rewritten below the cursor (compact,
        mechanical restart) or the tokenizer changes (model swap) — the
        grounded value then describes a request that will not be repeated.
        Returns the new grounded value (for CONTEXT_ESTIMATE emission).
        """
        value = system_len // 4 + estimate_message_tokens(messages)
        if tools_len:
            value += tools_len
        self._grounded = value
        self._cursor = len(messages)
        return value

    def estimate(self, messages: list, system_len: int) -> int:
        """Estimate the true current context size at the safe point.

        grounded + chars/4 of the messages appended since the grounded
        request. Stale-grounding guard: if the history has since SHRUNK
        below the cursor (a compact rewrote the messages to the summary
        pair), the grounded value describes a history that no longer
        exists and must not drive decisions — fall back to a fresh chars/4
        estimate of the current history (system prompt + messages). If the
        history grew — the normal flow — the grounded request is a prefix
        of the live history and the count is trusted regardless of size.

        A chars/4 "ceiling" test was used here instead and misjudged
        tool-heavy history, where the same content tokenizes up to ~3.5x
        worse than chars/4 (observed 2026-09-29: a genuine 237k grounded
        count was vetoed as stale all day and the 180k threshold never
        fired — the ceiling was retired in 87678c8).
        """
        added = messages[self._cursor :]
        grounded = self._grounded + estimate_message_tokens(added)
        if len(messages) < self._cursor:
            # History rewritten below the grounding point (compact) — the
            # grounded count is dead; estimate the current history from
            # scratch.
            return system_len // 4 + estimate_message_tokens(messages)
        return grounded
