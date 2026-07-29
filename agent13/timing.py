"""Token timing and TPS computation.

Extracted from TUI's _update_token_usage and REPL's inline TPS logic.
Single source of truth for token-per-second calculation, turn timing,
and session elapsed tracking.

Design notes:
- Pure computation, no UI or logging dependencies
- TUI thresholds adopted as canonical (MIN_ELAPSED=1.5 vs REPL's 1.0)
- Stale TPS preserved across short responses (REPL lacked this)
- Race-condition safe: caller passes captured timestamps as arguments
"""

from dataclasses import dataclass


@dataclass
class TPSResult:
    """Result of a TPS computation.

    Attributes:
        tps: Tokens per second (completion_tokens / elapsed).
        completion_tokens: Number of completion tokens in this response.
        elapsed: Wall-clock seconds between first and last token.
        turn_count: Total turns completed so far.
        total_processing_time: Cumulative seconds spent in all turns.
    """

    tps: float
    completion_tokens: int
    elapsed: float
    turn_count: int
    total_processing_time: float


class TokenTimingTracker:
    """Tracks token timing across turns and computes TPS.

    Lifecycle per turn:
        1. reset_stream() — called on STREAM_START
        2. record_token() — called on each ASSISTANT_TOKEN
        3. compute_tps(data, first_token_time, last_token_time) — called on TOKEN_USAGE

    Lifecycle across turns:
        turn_start() / turn_end() bracket each turn; session_elapsed
        increases monotonically from the first call.
    """

    # Thresholds — suppress TPS on short responses where measurement is
    # unreliable. Short responses (e.g. skill acknowledgments ~30 tokens)
    # produce inaccurate TPS because prompt evaluation time dominates
    # elapsed. When a large tool result is in the context, prompt_eval can
    # be 4-5s of a 7s total elapsed, making TPS appear 3x lower than the
    # backend's reported generation speed.
    MIN_TOKENS = 50
    MIN_ELAPSED = 1.5       # seconds — minimum elapsed to show TPS
    MIN_ELAPSED_CALC = 0.1  # floor to prevent near-zero division

    def __init__(self):
        # Token counts (updated from TOKEN_USAGE event data)
        self.completion_tokens: int = 0
        self.prompt_tokens: int = 0
        self.total_tokens: int = 0

        # Stream timing (reset per stream)
        self._first_token_time: float | None = None
        self._last_token_time: float | None = None
        self._token_count: int = 0

        # TPS persistence — stale value from prior turn
        self._last_tps: float | None = None

        # Turn timing
        self._turn_count: int = 0
        self._total_processing_time: float = 0.0
        self._session_elapsed_start: float | None = None

    @property
    def is_first_token(self) -> bool:
        """True if no token has been recorded since last reset_stream."""
        return self._first_token_time is None

    @property
    def last_tps(self) -> float | None:
        """Most recent valid TPS, preserved across short responses."""
        return self._last_tps

    def reset_stream(self) -> None:
        """Reset per-stream timing state. Called on STREAM_START."""
        self._first_token_time = None
        self._last_token_time = None
        self._token_count = 0

    def record_token(self, timestamp: float) -> None:
        """Record a token timestamp. Called on each ASSISTANT_TOKEN.

        Args:
            timestamp: Wall-clock time when the token arrived (time.time()).
        """
        if self._first_token_time is None:
            self._first_token_time = timestamp
        self._last_token_time = timestamp
        self._token_count += 1

    def compute_tps(
        self,
        data: dict,
        first_token_time: float | None = None,
        last_token_time: float | None = None,
    ) -> TPSResult | None:
        """Compute tokens-per-second for the current response.

        TPS elapsed is measured from first_token to last_token, measuring
        generation speed (excludes prompt evaluation time). Thresholds
        suppress TPS on short responses where measurement is unreliable.

        Args:
            data: Token usage dict from TOKEN_USAGE event with keys
                  prompt_tokens, completion_tokens, total_tokens.
            first_token_time: When first token arrived (caller-captured).
                Takes precedence over internal state.
            last_token_time: When last token arrived (caller-captured).
                Takes precedence over internal state.

        Returns:
            TPSResult if thresholds met, None otherwise.
        """
        self.prompt_tokens = data.get("prompt_tokens", 0)
        self.completion_tokens = data.get("completion_tokens", 0)
        self.total_tokens = data.get("total_tokens", 0)

        # Prefer caller-captured timestamps (race-condition safe),
        # fall back to internal state
        first_time = (
            first_token_time
            if first_token_time is not None
            else self._first_token_time
        )
        last_time = (
            last_token_time
            if last_token_time is not None
            else self._last_token_time
        )

        if not (first_time and last_time):
            return None

        elapsed = last_time - first_time

        # Sanity check: reject absurdly short elapsed (likely race condition)
        if elapsed < self.MIN_ELAPSED_CALC:
            return None

        # Threshold gating — suppress noisy short responses
        if not (
            elapsed >= self.MIN_ELAPSED
            and self.completion_tokens >= self.MIN_TOKENS
        ):
            return None  # _last_tps preserved — stale value is better signal

        tps_value = self.completion_tokens / elapsed
        if tps_value > 0:
            self._last_tps = tps_value

        return TPSResult(
            tps=tps_value,
            completion_tokens=self.completion_tokens,
            elapsed=elapsed,
            turn_count=self._turn_count,
            total_processing_time=self._total_processing_time,
        )

    def turn_start(self) -> None:
        """Mark the start of a new turn. Call before first LLM request."""
        import time

        self._session_elapsed_start = time.time()

    def turn_end(self) -> float | None:
        """Mark the end of a turn. Increments turn count and accumulates time.

        Returns the duration of the turn in seconds, or None if turn_start
        was not called.
        """
        import time

        if self._session_elapsed_start is None:
            return None  # turn_start not called — no-op

        now = time.time()
        duration = now - self._session_elapsed_start
        self._total_processing_time += duration
        self._turn_count += 1
        self._session_elapsed_start = None
        return duration

    @property
    def session_elapsed(self) -> float:
        """Seconds since first turn_start(). Returns 0.0 if no turn started."""
        import time

        if self._session_elapsed_start is None:
            return 0.0
        return time.time() - self._session_elapsed_start
