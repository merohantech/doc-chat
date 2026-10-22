"""File uploads: validate, extract text, detect duplicates and versions, store.

An uploaded file is *pending* until the user's next message, which carries it
into the conversation (see ChatService). Re-uploading a file whose name
matches a current file but whose content differs needs a decision:

- "replace": the upload becomes the next version; the old one is superseded
  (kept for "what changed?" questions, but no longer sent as current).
- "keep_both": the upload is stored as a separate file under a unique name.
"""

import asyncio
import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import PurePath
from typing import Literal

import structlog
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.db.models import File
from app.db.repositories.files import FileRepository
from app.db.repositories.threads import ThreadRepository
from app.ingestion.extractors import extract_text
from app.llm.tokens import estimate_tokens
from app.services.chat_service import ThreadNotFoundError
from app.storage.base import Storage

log = structlog.get_logger(__name__)

ConflictResolution = Literal["replace", "keep_both"]

_UNSAFE_FILENAME_CHARS = re.compile(r"[\x00-\x1f\x7f<>:\"|?*\\/]")


class FileTooLargeError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True, slots=True)
class ExistingFile:
    """Plain snapshot of a file row. Errors carry this rather than the ORM object,
    which is expired when the transaction rolls back."""

    id: uuid.UUID
    filename: str
    version: int

    @classmethod
    def of(cls, file: File) -> "ExistingFile":
        return cls(id=file.id, filename=file.filename, version=file.version)


class DuplicateFileError(Exception):
    def __init__(self, existing: ExistingFile) -> None:
        super().__init__(existing.filename)
        self.existing = existing


class FileNameConflictError(Exception):
    """A different file with this name exists; the client must choose how to resolve it."""

    def __init__(self, existing: ExistingFile) -> None:
        super().__init__(existing.filename)
        self.existing = existing


class FileService:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        storage: Storage,
        settings: Settings,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._storage = storage
        self._settings = settings

    async def list_files(self, user_id: uuid.UUID, thread_id: uuid.UUID) -> list[File]:
        async with self._sessionmaker() as session:
            if await ThreadRepository(session).get_owned(user_id, thread_id) is None:
                raise ThreadNotFoundError(thread_id)
            return await FileRepository(session).list_for_thread(thread_id)

    async def upload(
        self,
        user_id: uuid.UUID,
        thread_id: uuid.UUID,
        raw_filename: str,
        data: bytes,
        on_conflict: ConflictResolution | None = None,
    ) -> File:
        filename = clean_filename(raw_filename)
        if len(data) > self._settings.max_upload_bytes:
            limit_mb = self._settings.max_upload_bytes / 1024 / 1024
            raise FileTooLargeError(f"Files are limited to {limit_mb:g} MB.")

        # CPU-bound parsing runs off the event loop.
        extracted = await asyncio.to_thread(extract_text, filename, data)
        token_count = estimate_tokens(extracted.text)
        if token_count > self._settings.max_file_tokens:
            raise FileTooLargeError(
                f"This file is about {token_count:,} tokens; the limit is "
                f"{self._settings.max_file_tokens:,} (roughly 25 pages)."
            )
        sha256 = hashlib.sha256(data).hexdigest()

        async with self._sessionmaker.begin() as session:
            if await ThreadRepository(session).get_owned(user_id, thread_id) is None:
                raise ThreadNotFoundError(thread_id)
            files = FileRepository(session)

            if duplicate := await files.find_by_sha256(thread_id, sha256):
                raise DuplicateFileError(ExistingFile.of(duplicate))

            version = 1
            previous = await files.current_by_name(thread_id, filename)
            if previous is not None:
                if on_conflict is None:
                    raise FileNameConflictError(ExistingFile.of(previous))
                if on_conflict == "replace":
                    version = previous.version + 1
                else:
                    filename = unique_filename(filename, await files.current_names(thread_id))
                    previous = None

            file_id = uuid.uuid4()
            storage_key = f"{user_id}/{thread_id}/{file_id}{PurePath(filename).suffix.lower()}"
            await self._storage.save(storage_key, data)
            try:
                if previous is not None:
                    # Supersede first, or the "one current file per name" index rejects
                    # the insert. The FK to the not-yet-inserted row is checked at commit.
                    previous.superseded_by_id = file_id
                    await session.flush()
                new_file = File(
                    id=file_id,
                    thread_id=thread_id,
                    user_id=user_id,
                    filename=filename,
                    storage_key=storage_key,
                    mime=extracted.mime,
                    size=len(data),
                    sha256=sha256,
                    token_count=token_count,
                    mode="inline",
                    status="ready",
                    extracted_text=extracted.text,
                    version=version,
                )
                await files.add(new_file)
            except IntegrityError:
                # A concurrent upload won the race for this name or content.
                await self._storage.delete(storage_key)
                raise

        log.info(
            "file.uploaded",
            thread_id=str(thread_id),
            file_id=str(file_id),
            version=version,
            tokens=token_count,
        )
        return new_file


def clean_filename(raw: str) -> str:
    """Display name only (storage keys are generated): strip paths and unsafe characters."""
    name = PurePath(raw.replace("\\", "/")).name
    name = _UNSAFE_FILENAME_CHARS.sub("_", name).strip(" .")
    if not name:
        return "file"
    stem, suffix = PurePath(name).stem, PurePath(name).suffix
    return stem[: 255 - len(suffix)] + suffix


def unique_filename(filename: str, taken: set[str]) -> str:
    """'report.pdf' -> 'report (2).pdf', 'report (3).pdf', ... avoiding names in use."""
    stem, suffix = PurePath(filename).stem, PurePath(filename).suffix
    n = 2
    while f"{stem} ({n}){suffix}" in taken:
        n += 1
    return f"{stem} ({n}){suffix}"
