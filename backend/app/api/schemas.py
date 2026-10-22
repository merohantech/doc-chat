import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import Message
from app.db.models import Usage as UsageRow
from app.llm.base import Usage


class ThreadCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    model: str | None = Field(default=None, description="Defaults to DEFAULT_CHAT_MODEL.")


class ThreadOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    model: str
    est_context_tokens: int
    created_at: datetime


class MessageCreate(BaseModel):
    # Upper bound is enforced in the route from settings.max_message_chars.
    content: str = Field(min_length=1)


class UsageOut(BaseModel):
    prompt_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost: Decimal | None

    @classmethod
    def from_usage(cls, usage: Usage | UsageRow) -> "UsageOut":
        return cls(
            prompt_tokens=usage.prompt_tokens,
            cached_tokens=usage.cached_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            completion_tokens=usage.completion_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cost=usage.cost,
        )


class FileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    mime: str
    size: int
    token_count: int | None
    version: int
    status: str
    superseded_by_id: uuid.UUID | None
    message_id: uuid.UUID | None = Field(
        description="The message that carried this file to the model; null while pending."
    )
    created_at: datetime


class AttachmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    version: int


class MessageOut(BaseModel):
    id: uuid.UUID
    seq: int
    role: Literal["system", "user", "assistant"]
    content: str
    attachments: list[AttachmentOut]
    model: str | None
    created_at: datetime
    usage: UsageOut | None

    @classmethod
    def from_row(cls, row: Message) -> "MessageOut":
        return cls(
            id=row.id,
            seq=row.seq,
            role=row.role,  # values enforced by a DB check constraint
            content=_display_text(row.content_json.get("content")),
            attachments=[AttachmentOut.model_validate(f) for f in row.files],
            model=row.model,
            created_at=row.created_at,
            usage=UsageOut.from_usage(row.usage) if row.usage else None,
        )


def _display_text(content: Any) -> str:
    """Content is a string, or a list of text parts: file blocks first, the user's text last.

    File contents are shown as `attachments`, not repeated in the message text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content and isinstance(content[-1], dict):
        text = content[-1].get("text", "")
        return text if isinstance(text, str) else ""
    return ""


class TurnCompletedEvent(BaseModel):
    """Payload of the final `done` SSE event."""

    user_message_id: uuid.UUID
    assistant_message_id: uuid.UUID
    finish_reason: str | None
    usage: UsageOut | None


class ErrorEvent(BaseModel):
    code: str
    message: str
