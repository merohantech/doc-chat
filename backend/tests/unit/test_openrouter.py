import json
from decimal import Decimal
from typing import Any

import httpx2
import pytest
from openai import AsyncOpenAI

from app.llm.base import (
    ChatRequest,
    Completion,
    LLMRateLimitError,
    LLMRequestError,
    LLMUnavailableError,
    StreamEvent,
    TextDelta,
)
from app.llm.openrouter import OpenRouterProvider, merge_reasoning_details, parse_usage

BASE_URL = "https://openrouter.test/api/v1"
COMPLETIONS_URL = f"{BASE_URL}/chat/completions"

REQUEST = ChatRequest(
    model="anthropic/claude-sonnet-5.5",
    messages=[{"role": "user", "content": "Hi"}],
    max_tokens=256,
    session_id="thread-1",
    extra_body={"cache_control": {"type": "ephemeral"}},
)


def _chunk(**fields: Any) -> dict[str, Any]:
    return {
        "id": "gen-123",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "anthropic/claude-sonnet-5.5",
        "choices": [],
        **fields,
    }


def _delta(finish_reason: str | None = None, **delta: Any) -> dict[str, Any]:
    return {"index": 0, "delta": delta, "finish_reason": finish_reason}


def _sse(*chunks: dict[str, Any]) -> httpx2.Response:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})


class FakeOpenRouter:
    """Serves queued responses through the SDK's own HTTP client (no network, no patching).

    The OpenAI SDK uses `httpx2`, so it is faked via `http_client` + `MockTransport`.
    """

    def __init__(self) -> None:
        self.responses: list[httpx2.Response] = []
        self.requests: list[httpx2.Request] = []

    def reply(self, *responses: httpx2.Response) -> None:
        self.responses.extend(responses)

    def provider(self, max_retries: int = 0) -> OpenRouterProvider:
        def handler(request: httpx2.Request) -> httpx2.Response:
            assert str(request.url) == COMPLETIONS_URL
            self.requests.append(request)
            return self.responses.pop(0)

        client = AsyncOpenAI(
            api_key="test-key",
            base_url=BASE_URL,
            max_retries=max_retries,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        )
        return OpenRouterProvider(client)

    def last_body(self) -> dict[str, Any]:
        body: dict[str, Any] = json.loads(self.requests[-1].content)
        return body


@pytest.fixture
def server() -> FakeOpenRouter:
    return FakeOpenRouter()


@pytest.fixture
def provider(server: FakeOpenRouter) -> OpenRouterProvider:
    return server.provider()


async def _collect(provider: OpenRouterProvider, request: ChatRequest) -> list[StreamEvent]:
    return [event async for event in provider.stream_chat(request)]


async def test_streams_text_then_one_completion(
    server: FakeOpenRouter, provider: OpenRouterProvider
) -> None:
    server.reply(
        _sse(
            _chunk(choices=[_delta(role="assistant", content="Hel")]),
            _chunk(choices=[_delta(content="lo")]),
            _chunk(choices=[_delta(finish_reason="stop", content="")], provider="Anthropic"),
            _chunk(
                usage={
                    "prompt_tokens": 1200,
                    "completion_tokens": 2,
                    "total_tokens": 1202,
                    "cost": 0.0042,
                    "prompt_tokens_details": {"cached_tokens": 1000, "cache_write_tokens": 150},
                    "completion_tokens_details": {"reasoning_tokens": 0},
                }
            ),
        )
    )

    events = await _collect(provider, REQUEST)

    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Hel", "lo"]
    assert sum(isinstance(e, Completion) for e in events) == 1
    completion = events[-1]
    assert isinstance(completion, Completion)
    assert completion.message == {"role": "assistant", "content": "Hello"}
    assert completion.finish_reason == "stop"
    assert completion.provider == "Anthropic"
    assert completion.generation_id == "gen-123"
    assert completion.usage is not None
    assert completion.usage.cached_tokens == 1000
    assert completion.usage.cache_write_tokens == 150
    assert completion.usage.cost == Decimal("0.0042")

    sent = server.last_body()
    assert sent["session_id"] == "thread-1"
    assert sent["cache_control"] == {"type": "ephemeral"}
    assert sent["stream"] is True
    assert sent["max_tokens"] == 256


async def test_reasoning_details_are_reassembled_into_the_message(
    server: FakeOpenRouter, provider: OpenRouterProvider
) -> None:
    fragment = {"type": "reasoning.text", "format": "anthropic-claude-v1", "index": 0}
    server.reply(
        _sse(
            _chunk(choices=[_delta(reasoning_details=[{**fragment, "text": "Think"}])]),
            _chunk(choices=[_delta(reasoning_details=[{**fragment, "text": "ing."}])]),
            _chunk(choices=[_delta(reasoning_details=[{**fragment, "signature": "sig"}])]),
            _chunk(choices=[_delta(content="Answer", finish_reason="stop")]),
        )
    )

    completion = (await _collect(provider, REQUEST))[-1]

    assert isinstance(completion, Completion)
    assert completion.message["reasoning_details"] == [
        {**fragment, "text": "Thinking.", "signature": "sig"}
    ]


async def test_mid_stream_error_raises_unavailable(
    server: FakeOpenRouter, provider: OpenRouterProvider
) -> None:
    server.reply(
        _sse(
            _chunk(choices=[_delta(content="partial")]),
            _chunk(
                error={"code": 502, "message": "provider disconnected"},
                choices=[_delta(finish_reason="error")],
            ),
        )
    )

    with pytest.raises(LLMUnavailableError):
        await _collect(provider, REQUEST)


@pytest.mark.parametrize(
    ("status", "expected"),
    [(400, LLMRequestError), (401, LLMRequestError), (429, LLMRateLimitError)],
)
async def test_http_errors_map_to_domain_errors(
    server: FakeOpenRouter, provider: OpenRouterProvider, status: int, expected: type[Exception]
) -> None:
    server.reply(httpx2.Response(status, json={"error": {"message": "nope", "code": status}}))

    with pytest.raises(expected):
        await _collect(provider, REQUEST)


async def test_server_errors_are_retried_by_the_sdk(server: FakeOpenRouter) -> None:
    server.reply(
        httpx2.Response(
            503, json={"error": {"message": "overloaded"}}, headers={"retry-after-ms": "1"}
        ),
        _sse(_chunk(choices=[_delta(content="ok", finish_reason="stop")])),
    )

    completion = (await _collect(server.provider(max_retries=2), REQUEST))[-1]

    assert isinstance(completion, Completion)
    assert completion.message["content"] == "ok"
    assert len(server.requests) == 2


def test_merge_keeps_separate_blocks_in_order() -> None:
    merged: list[dict[str, Any]] = []
    merge_reasoning_details(
        merged,
        [
            {"type": "reasoning.text", "index": 0, "text": "a"},
            {"type": "reasoning.encrypted", "index": 1, "data": "x"},
        ],
    )
    merge_reasoning_details(merged, [{"type": "reasoning.encrypted", "index": 1, "data": "y"}])

    assert merged == [
        {"type": "reasoning.text", "index": 0, "text": "a"},
        {"type": "reasoning.encrypted", "index": 1, "data": "xy"},
    ]


def test_parse_usage_tolerates_missing_details() -> None:
    usage = parse_usage({"prompt_tokens": 5, "completion_tokens": 1})

    assert usage.cached_tokens == 0
    assert usage.cache_write_tokens == 0
    assert usage.cost is None
