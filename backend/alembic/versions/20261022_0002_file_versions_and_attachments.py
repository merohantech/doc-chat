"""file versions and attachments

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-22

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("files", sa.Column("extracted_text", sa.Text(), nullable=True))
    op.add_column("files", sa.Column("version", sa.Integer(), server_default="1", nullable=False))
    op.add_column("files", sa.Column("superseded_by_id", sa.Uuid(), nullable=True))
    op.add_column("files", sa.Column("message_id", sa.Uuid(), nullable=True))
    op.create_check_constraint(op.f("ck_files_version"), "files", "version >= 1")
    op.create_index(op.f("ix_files_message_id"), "files", ["message_id"], unique=False)
    op.create_index(
        "uq_files_thread_id_filename_current",
        "files",
        ["thread_id", "filename"],
        unique=True,
        postgresql_where=sa.text("superseded_by_id IS NULL"),
    )
    op.create_foreign_key(
        op.f("fk_files_superseded_by_id_files"),
        "files",
        "files",
        ["superseded_by_id"],
        ["id"],
        ondelete="SET NULL",
        deferrable=True,
        initially="DEFERRED",
    )
    op.create_foreign_key(
        op.f("fk_files_message_id_messages"),
        "files",
        "messages",
        ["message_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(op.f("fk_files_message_id_messages"), "files", type_="foreignkey")
    op.drop_constraint(op.f("fk_files_superseded_by_id_files"), "files", type_="foreignkey")
    op.drop_index(
        "uq_files_thread_id_filename_current",
        table_name="files",
        postgresql_where=sa.text("superseded_by_id IS NULL"),
    )
    op.drop_index(op.f("ix_files_message_id"), table_name="files")
    op.drop_constraint(op.f("ck_files_version"), "files", type_="check")
    op.drop_column("files", "message_id")
    op.drop_column("files", "superseded_by_id")
    op.drop_column("files", "version")
    op.drop_column("files", "extracted_text")
