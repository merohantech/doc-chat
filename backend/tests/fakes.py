from collections.abc import AsyncIterator

from app.llm.base import ChatRequest, Completion, StreamEvent, TextDelta, Usage


class FakeLLM:
    """Deterministic `LLMProvider` for tests; records every request it receives.

    `fail_before` raises before any text is streamed; `fail_after` raises after
    the first delta (a mid-stream failure).
    """

    def __init__(self, reply: str = "Hello from the fake.") -> None:
        self.reply = reply
        self.requests: list[ChatRequest] = []
        self.closed = False
        self.fail_before: Exception | None = None
        self.fail_after: Exception | None = None

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        if self.fail_before:
            raise self.fail_before
        words = self.reply.split(" ")
        for i, word in enumerate(words):
            yield TextDelta(word if i == len(words) - 1 else word + " ")
            if self.fail_after:
                raise self.fail_after
        yield Completion(
            message={"role": "assistant", "content": self.reply},
            finish_reason="stop",
            model=request.model,
            provider="fake",
            generation_id="gen-fake",
            usage=Usage(prompt_tokens=100 * len(self.requests), completion_tokens=5),
        )

    async def aclose(self) -> None:
        self.closed = True
