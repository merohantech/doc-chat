"""Database schema. See docs/IMPLEMENTATION_PLAN.md section 4.

Files are small (a few pages) and always sent to the model in full, so there
is no `file_chunks` / pgvector table yet; it is added if retrieval is needed.
"""

import uuid
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

MESSAGE_ROLES = ("system", "user", "assistant")
FILE_STATUSES = ("pending", "processing", "ready", "failed")
FILE_MODES = ("inline", "rag")


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True)

    threads: Mapped[list["Thread"]] = relationship(back_populates="user")


class Thread(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "threads"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str | None] = mapped_column(String(200))
    model: Mapped[str] = mapped_column(String(200))
    # Latest compaction checkpoint; context is rebuilt from here onward (Phase 3).
    summary_checkpoint_msg_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL", use_alter=True)
    )
    est_context_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    user: Mapped[User] = relationship(back_populates="threads")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="thread",
        order_by="Message.seq",
        foreign_keys="Message.thread_id",
    )


class Message(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("thread_id", "seq"),
        CheckConstraint(_in("role", MESSAGE_ROLES), name="role"),
    )

    thread_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("threads.id", ondelete="CASCADE"))
    seq: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))
    # The full message exactly as sent to / received from the provider, including
    # `reasoning_details`. It is replayed byte-for-byte to keep the prompt cache warm.
    content_json: Mapped[dict[str, Any]] = mapped_column(JSONB)
    model: Mapped[str | None] = mapped_column(String(200))

    thread: Mapped[Thread] = relationship(back_populates="messages", foreign_keys=[thread_id])
    usage: Mapped["Usage | None"] = relationship(back_populates="message")
    files: Mapped[list["File"]] = relationship(order_by="File.created_at")


class File(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "files"
    __table_args__ = (
        UniqueConstraint("thread_id", "sha256"),
        CheckConstraint(_in("status", FILE_STATUSES), name="status"),
        CheckConstraint(f"mode IS NULL OR {_in('mode', FILE_MODES)}", name="mode"),
        CheckConstraint("version >= 1", name="version"),
        # At most one *current* file per name in a thread; older versions are superseded.
        Index(
            "uq_files_thread_id_filename_current",
            "thread_id",
            "filename",
            unique=True,
            postgresql_where=text("superseded_by_id IS NULL"),
        ),
    )

    thread_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("threads.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(255))
    storage_key: Mapped[str] = mapped_column(String(512))
    mime: Mapped[str] = mapped_column(String(127))
    size: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64))
    token_count: Mapped[int | None] = mapped_column(Integer)
    # Decided after extraction: small files go inline, large ones through retrieval.
    mode: Mapped[str | None] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="pending", server_default="pending")
    error: Mapped[str | None] = mapped_column(Text)
    extracted_text: Mapped[str | None] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    # Set when a newer version of this file (same name, different content) is uploaded.
    # Deferred FK: the old row is marked superseded *before* the new row is inserted
    # (the current-name unique index requires that order), so the check runs at commit.
    superseded_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("files.id", ondelete="SET NULL", deferrable=True, initially="DEFERRED")
    )
    # The user message that carried this file into the conversation. NULL means the
    # file is still pending and will be attached to the thread's next message.
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL"), index=True
    )


class Usage(Base):
    __tablename__ = "usage"

    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), primary_key=True
    )
    prompt_tokens: Mapped[int] = mapped_column(Integer)
    cached_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cache_write_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    completion_tokens: Mapped[int] = mapped_column(Integer)
    reasoning_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cost: Mapped[Decimal | None] = mapped_column(Numeric(14, 8))
    provider: Mapped[str | None] = mapped_column(String(100))

    message: Mapped[Message] = relationship(back_populates="usage")
