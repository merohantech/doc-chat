import uuid
from collections.abc import Sequence

from sqlalchemy import exists, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.db.models import File


class FileRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, file: File) -> File:
        self._session.add(file)
        await self._session.flush()
        await self._session.refresh(file)
        return file

    async def find_by_sha256(self, thread_id: uuid.UUID, sha256: str) -> File | None:
        return await self._session.scalar(
            select(File).where(File.thread_id == thread_id, File.sha256 == sha256)
        )

    async def current_by_name(self, thread_id: uuid.UUID, filename: str) -> File | None:
        return await self._session.scalar(
            select(File).where(
                File.thread_id == thread_id,
                File.filename == filename,
                File.superseded_by_id.is_(None),
            )
        )

    async def current_names(self, thread_id: uuid.UUID) -> set[str]:
        result = await self._session.scalars(
            select(File.filename).where(
                File.thread_id == thread_id, File.superseded_by_id.is_(None)
            )
        )
        return set(result)

    async def list_for_thread(self, thread_id: uuid.UUID) -> list[File]:
        result = await self._session.scalars(
            select(File).where(File.thread_id == thread_id).order_by(File.created_at)
        )
        return list(result)

    async def pending_with_sent_predecessor(self, thread_id: uuid.UUID) -> list[tuple[File, bool]]:
        """Current files not yet sent to the model, oldest first.

        The flag says whether the version this file replaces was already part
        of the conversation (so the model must be told to disregard it).
        """
        previous = aliased(File)
        replaces_sent = exists().where(
            previous.superseded_by_id == File.id, previous.message_id.is_not(None)
        )
        result = await self._session.execute(
            select(File, replaces_sent)
            .where(
                File.thread_id == thread_id,
                File.message_id.is_(None),
                File.superseded_by_id.is_(None),
                File.status == "ready",
            )
            .order_by(File.created_at)
        )
        return [(file, bool(flag)) for file, flag in result]

    async def attach_to_message(self, file_ids: Sequence[uuid.UUID], message_id: uuid.UUID) -> None:
        if file_ids:
            await self._session.execute(
                update(File).where(File.id.in_(file_ids)).values(message_id=message_id)
            )
