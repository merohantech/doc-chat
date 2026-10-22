"""Provider-neutral LLM interface.

Services depend on `LLMProvider`, never on a vendor SDK, so the provider can be
swapped (or faked in tests) without touching business logic. Messages use the
OpenAI chat format as the common wire shape, since that is what OpenRouter
speaks and what we persist in `messages.content_json`.
"""

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol

ChatMessage = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ChatRequest:
    model: str
    messages: Sequence[ChatMessage]
    max_tokens: int | None = None
    # Routes every request in a thread to the same upstream provider (cache stays warm).
    session_id: str | None = None
    # Provider-specific request fields, e.g. top-level `cache_control` from a CachePolicy.
    extra_body: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost: Decimal | None = None


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class Completion:
    """Final event of a stream: the assistant message to persist, plus accounting."""

    message: dict[str, Any]
    finish_reason: str | None
    model: str | None
    provider: str | None
    generation_id: str | None
    usage: Usage | None


StreamEvent = TextDelta | Completion


class LLMError(Exception):
    """Base class for provider failures surfaced to the service layer."""


class LLMRateLimitError(LLMError):
    """Rate limited after the client's own retries were exhausted."""


class LLMUnavailableError(LLMError):
    """Transient upstream failure (timeout, connection, 5xx, mid-stream error)."""


class LLMRequestError(LLMError):
    """The request itself was rejected (4xx); retrying will not help."""


class LLMProvider(Protocol):
    def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        """Yield `TextDelta`s as text arrives, then exactly one `Completion`."""
        ...

    async def aclose(self) -> None: ...
