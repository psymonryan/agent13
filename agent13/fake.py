"""Built-in offline "fake" provider.

A deterministic, no-network ``AsyncOpenAI`` stand-in that lets the full
agent pipeline (Agent -> LLM seam -> NDJSON pipe) run and be tested
offline. Invoked like a named provider::

    agent13.py fake --model anything --io-format json

Design: the fake is a *drop-in client*. It implements exactly the
``client.*`` surface the pipeline touches -- ``chat.completions.create``
(streaming and non-streaming), ``models.list``, ``close`` -- and returns
the same chunk/usage structure the real client yields, so nothing
downstream (``llm.py``, ``core.py``, ``pipemode.py``) needs to change.

v1 scope: one fixed, deterministic text reply, no tool calls, no
scripting. Override the reply with the ``AGENT13_FAKE_RESPONSE`` env var.
"""

import os
from types import SimpleNamespace

#: Default deterministic reply. Plain (single chunk, not tokenised) so tests
#: can assert an exact match; self-identifying so it's obvious in the pipe
#: output that the fake - not a real model - answered.
DEFAULT_RESPONSE = "Hello from the fake provider - no network was used."

#: Env var to override the reply (handy for manual pipe poking).
ENV_OVERRIDE = "AGENT13_FAKE_RESPONSE"


def _response_text() -> str:
    """Current reply text (env override wins, else the default)."""
    return os.environ.get(ENV_OVERRIDE, DEFAULT_RESPONSE)


def _usage_for(text: str):
    """Deterministic token usage derived from the reply (not real counts)."""
    completion = max(1, len(text.split()))
    prompt = 1
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


def _content_chunk(text: str):
    """A stream chunk carrying the reply text (finish_reason not set yet)."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text),
                finish_reason=None,
            )
        ],
        usage=None,
    )


def _finish_chunk(usage):
    """Terminal stream chunk: empty delta, finish_reason set, usage present.

    Mirrors how real providers end a stream - a payload-less delta with a
    finish_reason, carrying the usage block (via stream_options.include_usage).
    """
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=None),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )


class _FakeStream:
    """Async-iterable wrapper matching the real ``AsyncStream`` surface.

    The pipeline does ``async for chunk in stream`` and, in a ``finally``,
    ``await stream.close()``. We mirror both so the drop-in is faithful.
    """

    def __init__(self, chunks):
        self._it = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration

    async def close(self):
        return None


class _FakeCompletions:
    async def create(self, **params):
        text = _response_text()
        if params.get("stream", False):
            return _FakeStream([_content_chunk(text), _finish_chunk(_usage_for(text))])
        # Non-streaming (batch / get_initial_response path): a single
        # completion object with .choices[0].message and .usage.
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content=text,
                        tool_calls=None,
                        reasoning_content=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=_usage_for(text),
        )


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeModels:
    async def list(self):
        # Empty on purpose: when the fetched list is empty the CLI falls back
        # to using the --model value as-is (see cli.py), so any model passes
        # without a network round-trip.
        return SimpleNamespace(data=[])


class FakeOpenAIClient:
    """Drop-in ``AsyncOpenAI`` replacement for the offline fake provider."""

    def __init__(self):
        self.chat = _FakeChat()
        self.models = _FakeModels()
        # base_url is read off the client as a fallback polite-lock key
        # (core.set_polite); give it a stable sentinel so that path works
        # without a real endpoint.
        self.base_url = "fake"

    async def close(self):
        return None
