"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-10-22

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("email", name=op.f("uq_users_email")),
    )
    op.create_table(
        "threads",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=True),
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("summary_checkpoint_msg_id", sa.Uuid(), nullable=True),
        sa.Column("est_context_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_threads_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_threads")),
    )
    op.create_index(op.f("ix_threads_user_id"), "threads", ["user_id"], unique=False)
    op.create_table(
        "messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("thread_id", sa.Uuid(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("model", sa.String(length=200), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "role IN ('system', 'user', 'assistant')", name=op.f("ck_messages_role")
        ),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            ["threads.id"],
            name=op.f("fk_messages_thread_id_threads"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_messages")),
        sa.UniqueConstraint("thread_id", "seq", name=op.f("uq_messages_thread_id_seq")),
    )
    # threads <-> messages is circular, so this FK is added once both tables exist.
    op.create_foreign_key(
        op.f("fk_threads_summary_checkpoint_msg_id_messages"),
        "threads",
        "messages",
        ["summary_checkpoint_msg_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_table(
        "files",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("thread_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("storage_key", sa.String(length=512), nullable=False),
        sa.Column("mime", sa.String(length=127), nullable=False),
        sa.Column("size", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("mode", sa.String(length=16), nullable=True),
        sa.Column("status", sa.String(length=16), server_default="pending", nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("mode IS NULL OR mode IN ('inline', 'rag')", name=op.f("ck_files_mode")),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'failed')", name=op.f("ck_files_status")
        ),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            ["threads.id"],
            name=op.f("fk_files_thread_id_threads"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_files_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_files")),
        sa.UniqueConstraint("thread_id", "sha256", name=op.f("uq_files_thread_id_sha256")),
    )
    op.create_index(op.f("ix_files_thread_id"), "files", ["thread_id"], unique=False)
    op.create_index(op.f("ix_files_user_id"), "files", ["user_id"], unique=False)
    op.create_table(
        "usage",
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("cached_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("cache_write_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("reasoning_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("cost", sa.Numeric(precision=14, scale=8), nullable=True),
        sa.Column("provider", sa.String(length=100), nullable=True),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["messages.id"],
            name=op.f("fk_usage_message_id_messages"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("message_id", name=op.f("pk_usage")),
    )


def downgrade() -> None:
    op.drop_table("usage")
    op.drop_index(op.f("ix_files_user_id"), table_name="files")
    op.drop_index(op.f("ix_files_thread_id"), table_name="files")
    op.drop_table("files")
    op.drop_constraint(
        op.f("fk_threads_summary_checkpoint_msg_id_messages"), "threads", type_="foreignkey"
    )
    op.drop_table("messages")
    op.drop_index(op.f("ix_threads_user_id"), table_name="threads")
    op.drop_table("threads")
    op.drop_table("users")
