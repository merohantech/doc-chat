"""File upload, versioning and attachment flow over HTTP (fake LLM, real Postgres + Redis)."""

import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm.base import LLMUnavailableError
from app.main import create_app
from tests.api_helpers import create_thread, send, upload
from tests.fakes import FakeLLM
from tests.unit.test_extractors import make_pdf

pytestmark = pytest.mark.integration


def _file_blocks(request_messages: Sequence[Mapping[str, Any]]) -> list[str]:
    """Text of every <file> block in the last user message of a request."""
    content = request_messages[-1]["content"]
    if isinstance(content, str):
        return []
    return [part["text"] for part in content[:-1]]


def test_upload_returns_a_pending_file_and_stores_it(api: TestClient, settings: Settings) -> None:
    thread = create_thread(api)
    pdf = make_pdf("The launch date is the fourteenth of March.")

    response = upload(api, thread["id"], "plan.pdf", pdf)

    assert response.status_code == 201, response.text
    file = response.json()
    assert file["filename"] == "plan.pdf"
    assert file["mime"] == "application/pdf"
    assert file["version"] == 1
    assert file["status"] == "ready"
    assert file["message_id"] is None  # pending until the next message
    assert file["token_count"] > 0
    stored = list(Path(settings.storage_dir).rglob(f"{file['id']}.pdf"))
    assert len(stored) == 1 and stored[0].read_bytes() == pdf
    assert api.get(f"/threads/{thread['id']}/files").json() == [file]


def test_pending_file_rides_along_with_the_next_message_only(
    api: TestClient, fake_llm: FakeLLM
) -> None:
    thread = create_thread(api)
    upload(api, thread["id"], "notes.txt", b"The secret word is marmalade.")

    send(api, thread["id"], "What is the secret word?")
    send(api, thread["id"], "Thanks!")

    first, second = fake_llm.requests
    assert _file_blocks(first.messages) == [
        '<file name="notes.txt" version="1">\nThe secret word is marmalade.\n</file>'
    ]
    assert first.messages[-1]["content"][-1] == {
        "type": "text",
        "text": "What is the secret word?",
    }
    # The file is now part of history, not re-sent as a new attachment.
    assert _file_blocks(second.messages) == []
    assert second.messages[: len(first.messages)] == first.messages

    messages = api.get(f"/threads/{thread['id']}/messages").json()
    assert messages[0]["content"] == "What is the secret word?"
    assert [a["filename"] for a in messages[0]["attachments"]] == ["notes.txt"]
    files = api.get(f"/threads/{thread['id']}/files").json()
    assert files[0]["message_id"] == messages[0]["id"]


def test_identical_content_is_rejected_even_under_another_name(api: TestClient) -> None:
    thread = create_thread(api)
    upload(api, thread["id"], "a.txt", b"same bytes")

    response = upload(api, thread["id"], "b.txt", b"same bytes")

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "file_duplicate"


def test_same_name_different_content_asks_how_to_resolve(api: TestClient) -> None:
    thread = create_thread(api)
    first = upload(api, thread["id"], "report.txt", b"Version one.").json()

    response = upload(api, thread["id"], "report.txt", b"Version two.")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "file_name_conflict"
    assert detail["existing_file_id"] == first["id"]
    assert detail["existing_version"] == 1


def test_replace_after_sending_creates_v2_and_tells_the_model(
    api: TestClient, fake_llm: FakeLLM
) -> None:
    thread = create_thread(api)
    v1 = upload(api, thread["id"], "report.txt", b"Revenue: 10.").json()
    send(api, thread["id"], "What is revenue?")

    response = upload(api, thread["id"], "report.txt", b"Revenue: 12.", on_conflict="replace")
    send(api, thread["id"], "And now?")

    assert response.status_code == 201
    v2 = response.json()
    assert v2["version"] == 2
    first, second = fake_llm.requests
    # Earlier turns are untouched (cache-safe); v2 is appended and marked as a replacement.
    assert second.messages[: len(first.messages)] == first.messages
    assert _file_blocks(second.messages) == [
        '<file name="report.txt" version="2" replaces="version 1">\nRevenue: 12.\n</file>'
    ]
    files = {f["id"]: f for f in api.get(f"/threads/{thread['id']}/files").json()}
    assert files[v1["id"]]["superseded_by_id"] == v2["id"]
    assert files[v2["id"]]["superseded_by_id"] is None


def test_replacing_an_unsent_file_sends_only_the_new_version(
    api: TestClient, fake_llm: FakeLLM
) -> None:
    thread = create_thread(api)
    upload(api, thread["id"], "draft.txt", b"Old draft.")
    upload(api, thread["id"], "draft.txt", b"New draft.", on_conflict="replace")

    send(api, thread["id"], "Review it.")

    # The model never saw v1, so there is nothing to tell it to disregard.
    assert _file_blocks(fake_llm.requests[0].messages) == [
        '<file name="draft.txt" version="2">\nNew draft.\n</file>'
    ]


def test_keep_both_stores_a_separately_named_file(api: TestClient, fake_llm: FakeLLM) -> None:
    thread = create_thread(api)
    upload(api, thread["id"], "notes.txt", b"First notes.")

    response = upload(api, thread["id"], "notes.txt", b"Other notes.", on_conflict="keep_both")
    send(api, thread["id"], "Compare them.")

    assert response.status_code == 201
    assert response.json()["filename"] == "notes (2).txt"
    assert response.json()["version"] == 1
    assert _file_blocks(fake_llm.requests[0].messages) == [
        '<file name="notes.txt" version="1">\nFirst notes.\n</file>',
        '<file name="notes (2).txt" version="1">\nOther notes.\n</file>',
    ]


def test_file_stays_pending_when_the_turn_fails(api: TestClient, fake_llm: FakeLLM) -> None:
    thread = create_thread(api)
    upload(api, thread["id"], "notes.txt", b"Remember this.")
    fake_llm.fail_before = LLMUnavailableError("down")
    assert api.post(f"/threads/{thread['id']}/messages", json={"content": "Hi"}).status_code == 503

    fake_llm.fail_before = None
    send(api, thread["id"], "Hi again")

    assert len(_file_blocks(fake_llm.requests[-1].messages)) == 1
    assert api.get(f"/threads/{thread['id']}/files").json()[0]["message_id"] is not None


@pytest.mark.parametrize(
    ("filename", "content", "status_code", "code"),
    [
        ("photo.png", b"\x89PNG", 422, "file_type_unsupported"),
        ("scan.pdf", make_pdf("", ""), 422, "file_unreadable"),
        ("huge.txt", b"word " * 20_000, 413, "file_too_large"),
    ],
    ids=["unsupported-type", "scanned-pdf", "too-many-tokens"],
)
def test_invalid_uploads_are_rejected(
    api: TestClient, filename: str, content: bytes, status_code: int, code: str
) -> None:
    thread = create_thread(api)

    response = upload(api, thread["id"], filename, content)

    assert response.status_code == status_code
    assert response.json()["detail"]["code"] == code
    assert api.get(f"/threads/{thread['id']}/files").json() == []


def test_byte_size_limit(settings: Settings, migrated_db: None) -> None:
    small = settings.model_copy(update={"max_upload_bytes": 10})
    with TestClient(create_app(small, llm=FakeLLM())) as client:
        thread = create_thread(client)
        response = upload(client, thread["id"], "a.txt", b"more than ten bytes")

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "file_too_large"


def test_upload_to_unknown_thread_is_404(api: TestClient) -> None:
    response = upload(api, str(uuid.uuid4()), "a.txt", b"hello")

    assert response.status_code == 404
