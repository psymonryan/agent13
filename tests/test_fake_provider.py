"""Unit / wiring tests for the built-in offline fake provider (agent13/fake.py).

These exercise ``FakeOpenAIClient`` directly - no subprocess, no network.
The crown-jewel test proves the fake is a faithful *drop-in* for the real LLM
seam (``llm.stream_response_with_tools``), which is what lets the whole
downstream pipeline (core -> pipemode -> NDJSON) run unchanged on top of it.
"""

from agent13.fake import DEFAULT_RESPONSE, ENV_OVERRIDE, FakeOpenAIClient


async def _collect_stream(client) -> list:
    """Drive one streaming ``create()`` and return the yielded chunks."""
    stream = await client.chat.completions.create(
        model="m", messages=[], stream=True
    )
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    await stream.close()
    return chunks


class TestFakeClientSurface:
    """The fake must expose exactly the client surface the pipeline touches."""

    def test_has_chat_models_and_base_url(self):
        client = FakeOpenAIClient()
        assert client.chat.completions is not None
        assert client.models is not None
        assert client.base_url == "fake"

    async def test_close_is_awaitable(self):
        assert await FakeOpenAIClient().close() is None

    async def test_models_list_returns_empty(self):
        """Empty on purpose: lets the CLI fall back to using --model as-is."""
        client = FakeOpenAIClient()
        assert (await client.models.list()).data == []


class TestStreaming:
    """The streaming path (llm.py:602) is what pipe mode drives."""

    async def test_yields_content_then_finish(self):
        chunks = await _collect_stream(FakeOpenAIClient())
        assert len(chunks) == 2
        first, last = chunks
        assert first.choices[0].delta.content == DEFAULT_RESPONSE
        assert first.usage is None
        assert last.choices[0].delta.content is None
        assert last.choices[0].finish_reason == "stop"
        assert last.usage is not None

    async def test_usage_is_deterministic(self):
        chunks = await _collect_stream(FakeOpenAIClient())
        usage = chunks[-1].usage
        words = len(DEFAULT_RESPONSE.split())
        assert usage.completion_tokens == words
        assert usage.prompt_tokens == 1
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

    async def test_env_override_changes_reply(self, monkeypatch):
        monkeypatch.setenv(ENV_OVERRIDE, "override reply text")
        chunks = await _collect_stream(FakeOpenAIClient())
        assert chunks[0].choices[0].delta.content == "override reply text"


class TestNonStreaming:
    """The non-streaming path (llm.py:294 get_initial_response) must not crash."""

    async def test_returns_message_and_usage(self):
        client = FakeOpenAIClient()
        response = await client.chat.completions.create(model="m", messages=[])
        assert response.choices[0].message.content == DEFAULT_RESPONSE
        assert response.choices[0].finish_reason == "stop"
        assert response.usage is not None


class TestDropInAgainstRealSeam:
    """Feed the fake through the real LLM seam.

    ``llm.stream_response_with_tools`` is the single seam the Agent runs on.
    If the fake satisfies it, the entire pipeline executes unchanged - which is
    exactly the property that makes an offline provider useful for testing.
    """

    async def test_stream_response_with_tools_yields_content_and_usage(self):
        from agent13.llm import stream_response_with_tools

        client = FakeOpenAIClient()
        messages = [{"role": "user", "content": "hello"}]
        events = []
        async for event_type, data in stream_response_with_tools(
            client, "anything", messages
        ):
            events.append((event_type, data))

        types = [t for t, _ in events]
        assert types == ["content", "token_usage"], f"Unexpected events: {types}"

        content_data = next(d for t, d in events if t == "content")
        assert content_data == DEFAULT_RESPONSE

        usage = next(d for t, d in events if t == "token_usage")
        assert usage["total_tokens"] == (
            usage["prompt_tokens"] + usage["completion_tokens"]
        )
        assert usage["completion_tokens"] == len(DEFAULT_RESPONSE.split())
