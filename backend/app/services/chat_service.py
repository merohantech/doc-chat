"""Orchestrates a chat turn: lock the thread, build the prompt, stream, persist.

Database sessions are short and never held open while the LLM streams, so a
slow reply does not pin a pooled connection. A turn is persisted atomically
(user message + assistant message + usage) only after the reply completes;
a failed or abandoned turn leaves the history untouched.
"""

import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import structlog
from redis.asyncio import Redis
from redis.exceptions import LockError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.db.models import Message, Thread
from app.db.models import Usage as UsageRow
from app.db.repositories.files import FileRepository
from app.db.repositories.threads import ThreadRepository
from app.llm.base import ChatRequest, Completion, LLMProvider, TextDelta, Usage
from app.services.context_builder import Attachment, build_prompt, user_message

log = structlog.get_logger(__name__)


class ThreadNotFoundError(Exception):
    pass


class ThreadBusyError(Exception):
    """Another turn is already streaming in this thread."""


class EmptyCompletionError(Exception):
    """The provider closed the stream without a final completion event."""


@dataclass(frozen=True, slots=True)
class TurnCompleted:
    user_message_id: uuid.UUID
    assistant_message_id: uuid.UUID
    finish_reason: str | None
    usage: Usage | None


TurnEvent = TextDelta | TurnCompleted


class ChatService:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        redis: Redis,
        llm: LLMProvider,
        settings: Settings,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._redis = redis
        self._llm = llm
        self._settings = settings

    async def create_thread(
        self, user_id: uuid.UUID, model: str | None, title: str | None
    ) -> Thread:
        async with self._sessionmaker.begin() as session:
            return await ThreadRepository(session).create(
                user_id, model or self._settings.default_chat_model, title
            )

    async def get_thread(self, user_id: uuid.UUID, thread_id: uuid.UUID) -> Thread:
        async with self._sessionmaker() as session:
            thread = await ThreadRepository(session).get_owned(user_id, thread_id)
        if thread is None:
            raise ThreadNotFoundError(thread_id)
        return thread

    async def list_messages(self, user_id: uuid.UUID, thread_id: uuid.UUID) -> list[Message]:
        async with self._sessionmaker() as session:
            repo = ThreadRepository(session)
            if await repo.get_owned(user_id, thread_id) is None:
                raise ThreadNotFoundError(thread_id)
            return await repo.messages_with_usage(thread_id)

    async def stream_turn(
        self, user_id: uuid.UUID, thread_id: uuid.UUID, text: str
    ) -> AsyncGenerator[TurnEvent]:
        """Yield `TextDelta`s as the reply streams, then one `TurnCompleted`."""
        lock = self._redis.lock(
            f"thread-turn:{thread_id}",
            timeout=self._settings.turn_lock_seconds,
            blocking=False,
        )
        if not await lock.acquire():
            raise ThreadBusyError(thread_id)
        try:
            async with self._sessionmaker() as session:
                repo = ThreadRepository(session)
                thread = await repo.get_owned(user_id, thread_id)
                if thread is None:
                    raise ThreadNotFoundError(thread_id)
                history = await repo.history(thread_id)
                # Files uploaded since the last turn ride along with this message.
                pending = await FileRepository(session).pending_with_sent_predecessor(thread_id)

            attachments = [
                Attachment(
                    filename=file.filename,
                    version=file.version,
                    text=file.extracted_text or "",
                    replaces_sent_version=replaces_sent,
                )
                for file, replaces_sent in pending
            ]
            new_message = user_message(text, attachments)
            prompt = build_prompt(
                thread.model, history, new_message, self._settings.prompt_cache_ttl
            )
            request = ChatRequest(
                model=thread.model,
                messages=prompt.messages,
                max_tokens=self._settings.chat_max_tokens,
                session_id=str(thread_id),
                extra_body=prompt.request_fields,
            )

            completion: Completion | None = None
            async for event in self._llm.stream_chat(request):
                if isinstance(event, TextDelta):
                    yield event
                else:
                    completion = event
            if completion is None:
                raise EmptyCompletionError(thread_id)

            async with self._sessionmaker.begin() as session:
                repo = ThreadRepository(session)
                thread = await repo.get_owned(user_id, thread_id)
                if thread is None:  # deleted while the reply was streaming
                    raise ThreadNotFoundError(thread_id)
                user_row, assistant_row = await repo.append_turn(
                    thread,
                    new_message,
                    completion.message,
                    completion.model,
                    _usage_row(completion),
                )
                # Only the files actually sent; uploads that arrived mid-turn stay pending.
                await FileRepository(session).attach_to_message(
                    [file.id for file, _ in pending], user_row.id
                )

            log.info(
                "chat.turn_completed",
                thread_id=str(thread_id),
                model=completion.model,
                provider=completion.provider,
                finish_reason=completion.finish_reason,
                usage=completion.usage,
            )
            yield TurnCompleted(
                user_message_id=user_row.id,
                assistant_message_id=assistant_row.id,
                finish_reason=completion.finish_reason,
                usage=completion.usage,
            )
        finally:
            try:
                await lock.release()
            except LockError:
                # The lock expired (turn exceeded turn_lock_seconds); nothing to release.
                log.warning("chat.turn_lock_expired", thread_id=str(thread_id))


def _usage_row(completion: Completion) -> UsageRow | None:
    usage = completion.usage
    if usage is None:
        return None
    return UsageRow(
        prompt_tokens=usage.prompt_tokens,
        cached_tokens=usage.cached_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        completion_tokens=usage.completion_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        cost=usage.cost,
        provider=completion.provider,
    )
