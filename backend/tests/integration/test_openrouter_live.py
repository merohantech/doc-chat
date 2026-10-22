"""Phase 0 smoke test against the real OpenRouter API. Costs a few tokens.

Run with: pytest -m live
"""

import pytest

from app.config import Settings
from app.llm.base import ChatRequest, Completion, TextDelta
from app.llm.openrouter import OpenRouterProvider

pytestmark = pytest.mark.live


async def test_streams_a_reply_from_openrouter() -> None:
    settings = Settings()
    if settings.openrouter_api_key is None:
        pytest.skip("OPENROUTER_API_KEY is not set")
    provider = OpenRouterProvider.from_settings(settings)

    try:
        events = [
            event
            async for event in provider.stream_chat(
                ChatRequest(
                    model=settings.utility_model,
                    messages=[{"role": "user", "content": "Reply with exactly: pong"}],
                    max_tokens=16,
                )
            )
        ]
    finally:
        await provider.aclose()

    assert any(isinstance(e, TextDelta) for e in events)
    completion = events[-1]
    assert isinstance(completion, Completion)
    assert "pong" in completion.message["content"].lower()
    assert completion.usage is not None and completion.usage.prompt_tokens > 0
