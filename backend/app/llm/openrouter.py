"""OpenRouter implementation of `LLMProvider`, using the OpenAI SDK.

Retries on 408/409/429/5xx and connection errors are done by the SDK itself
(exponential backoff, honours `Retry-After`), so there is no retry loop here.
Retries only cover establishing the stream; a failure mid-stream is surfaced
as `LLMUnavailableError` for the caller to handle.
"""

from collections.abc import AsyncIterator, Iterable, Mapping
from decimal import Decimal
from typing import Any, cast

import openai
import structlog
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam

from app.config import Settings
from app.llm.base import (
    ChatRequest,
    Completion,
    LLMError,
    LLMRateLimitError,
    LLMRequestError,
    LLMUnavailableError,
    StreamEvent,
    TextDelta,
    Usage,
)

log = structlog.get_logger(__name__)

_CONCATENATED_REASONING_FIELDS = ("text", "summary", "data")


class OpenRouterProvider:
    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenRouterProvider":
        if settings.openrouter_api_key is None:
            raise ValueError("OPENROUTER_API_KEY is not set")
        headers = {"X-Title": settings.openrouter_app_name}
        if settings.openrouter_app_url:
            headers["HTTP-Referer"] = settings.openrouter_app_url
        client = AsyncOpenAI(
            api_key=settings.openrouter_api_key.get_secret_value(),
            base_url=settings.openrouter_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            default_headers=headers,
        )
        return cls(client)

    async def aclose(self) -> None:
        await self._client.close()

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        extra_body: dict[str, Any] = dict(request.extra_body)
        if request.session_id:
            extra_body["session_id"] = request.session_id

        # Messages are already in OpenAI wire format (see ChatMessage); the SDK's
        # TypedDicts are stricter than our persisted JSON, hence the cast.
        messages = cast(list[ChatCompletionMessageParam], list(request.messages))
        try:
            stream = await self._client.chat.completions.create(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens,
                stream=True,
                extra_body=extra_body or None,
            )
        except openai.APIError as exc:
            raise _map_error(exc) from exc

        accumulator = _StreamAccumulator()
        try:
            async for chunk in stream:
                text = accumulator.add(chunk)
                if text:
                    yield TextDelta(text)
        except openai.APIError as exc:
            raise _map_error(exc) from exc
        finally:
            await stream.close()

        completion = accumulator.result()
        log.debug(
            "llm.completion",
            model=completion.model,
            provider=completion.provider,
            finish_reason=completion.finish_reason,
            usage=completion.usage,
        )
        yield completion


class _StreamAccumulator:
    """Folds streamed chunks into one assistant message plus usage."""

    def __init__(self) -> None:
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._reasoning_details: list[dict[str, Any]] = []
        self._finish_reason: str | None = None
        self._model: str | None = None
        self._provider: str | None = None
        self._generation_id: str | None = None
        self._usage: Usage | None = None

    def add(self, chunk: ChatCompletionChunk) -> str:
        extra = chunk.model_extra or {}
        if error := extra.get("error"):
            # OpenRouter reports upstream failures after the stream has started
            # as a chunk carrying an `error` object (HTTP status is already 200).
            raise LLMUnavailableError(f"upstream error mid-stream: {error}")

        self._generation_id = self._generation_id or chunk.id
        self._model = chunk.model or self._model
        self._provider = extra.get("provider") or self._provider
        if chunk.usage is not None:
            self._usage = parse_usage(chunk.usage.model_dump())

        text = ""
        for choice in chunk.choices:
            if choice.finish_reason:
                self._finish_reason = choice.finish_reason
            delta = choice.delta
            if delta.content:
                text += delta.content
            delta_extra = delta.model_extra or {}
            if reasoning := delta_extra.get("reasoning"):
                self._reasoning.append(reasoning)
            if details := delta_extra.get("reasoning_details"):
                merge_reasoning_details(self._reasoning_details, details)

        if self._finish_reason == "error":
            raise LLMUnavailableError("upstream finished the stream with an error")
        self._content.append(text)
        return text

    def result(self) -> Completion:
        message: dict[str, Any] = {"role": "assistant", "content": "".join(self._content)}
        if self._reasoning_details:
            message["reasoning_details"] = self._reasoning_details
        elif self._reasoning:
            message["reasoning"] = "".join(self._reasoning)
        return Completion(
            message=message,
            finish_reason=self._finish_reason,
            model=self._model,
            provider=self._provider,
            generation_id=self._generation_id,
            usage=self._usage,
        )


def merge_reasoning_details(
    merged: list[dict[str, Any]], fragments: Iterable[Mapping[str, Any]]
) -> None:
    """Rebuild `reasoning_details` from streamed fragments, in place.

    Fragments that share an `index` are pieces of one block: their string
    payloads are concatenated and any other field takes the latest non-null
    value. Block order must be preserved exactly, since providers reject
    reasoning that is reordered or modified when it is sent back.
    """
    for fragment in fragments:
        index = fragment.get("index")
        target = next(
            (block for block in merged if index is not None and block.get("index") == index),
            None,
        )
        if target is None:
            merged.append(dict(fragment))
            continue
        for key, value in fragment.items():
            if value is None:
                continue
            if key in _CONCATENATED_REASONING_FIELDS and isinstance(value, str):
                target[key] = (target.get(key) or "") + value
            else:
                target[key] = value


def parse_usage(raw: Mapping[str, Any]) -> Usage:
    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    cost = raw.get("cost")
    return Usage(
        prompt_tokens=raw.get("prompt_tokens") or 0,
        completion_tokens=raw.get("completion_tokens") or 0,
        cached_tokens=prompt_details.get("cached_tokens") or 0,
        cache_write_tokens=prompt_details.get("cache_write_tokens") or 0,
        reasoning_tokens=completion_details.get("reasoning_tokens") or 0,
        # Via str() so a float like 0.1 doesn't become 0.1000000000000000055...
        cost=Decimal(str(cost)) if cost is not None else None,
    )


def _map_error(exc: openai.APIError) -> LLMError:
    if isinstance(exc, openai.RateLimitError):
        return LLMRateLimitError(str(exc))
    if isinstance(exc, openai.APIConnectionError | openai.InternalServerError):
        return LLMUnavailableError(str(exc))
    if isinstance(exc, openai.APIStatusError):
        return LLMRequestError(f"{exc.status_code}: {exc.message}")
    return LLMUnavailableError(str(exc))
