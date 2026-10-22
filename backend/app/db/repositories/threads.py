import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Message, Thread, Usage


class ThreadRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, user_id: uuid.UUID, model: str, title: str | None) -> Thread:
        thread = Thread(user_id=user_id, model=model, title=title)
        self._session.add(thread)
        await self._session.flush()
        await self._session.refresh(thread)
        return thread

    async def get_owned(self, user_id: uuid.UUID, thread_id: uuid.UUID) -> Thread | None:
        """Ownership is part of the lookup, so another user's thread is indistinguishable
        from a missing one."""
        return await self._session.scalar(
            select(Thread).where(Thread.id == thread_id, Thread.user_id == user_id)
        )

    async def messages_with_usage(self, thread_id: uuid.UUID) -> list[Message]:
        result = await self._session.scalars(
            select(Message)
            .where(Message.thread_id == thread_id)
            .options(selectinload(Message.usage), selectinload(Message.files))
            .order_by(Message.seq)
        )
        return list(result)

    async def history(self, thread_id: uuid.UUID) -> list[dict[str, Any]]:
        """Stored messages in order, exactly as they go back to the LLM."""
        result = await self._session.scalars(
            select(Message.content_json).where(Message.thread_id == thread_id).order_by(Message.seq)
        )
        return list(result)

    async def next_seq(self, thread_id: uuid.UUID) -> int:
        current = await self._session.scalar(
            select(func.max(Message.seq)).where(Message.thread_id == thread_id)
        )
        return (current or 0) + 1

    async def append_turn(
        self,
        thread: Thread,
        user_content: dict[str, Any],
        assistant_content: dict[str, Any],
        assistant_model: str | None,
        usage: Usage | None,
    ) -> tuple[Message, Message]:
        seq = await self.next_seq(thread.id)
        user = Message(thread_id=thread.id, seq=seq, role="user", content_json=user_content)
        assistant = Message(
            thread_id=thread.id,
            seq=seq + 1,
            role="assistant",
            content_json=assistant_content,
            model=assistant_model,
        )
        self._session.add_all([user, assistant])
        await self._session.flush()
        if usage is not None:
            usage.message_id = assistant.id
            self._session.add(usage)
            # The last request's prompt + reply is the best estimate of the
            # context the next turn starts from.
            thread.est_context_tokens = usage.prompt_tokens + usage.completion_tokens
        return user, assistant
