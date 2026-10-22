"""HTTP helpers shared by the integration tests."""

import json
from typing import Any

from fastapi.testclient import TestClient
from httpx2 import Response


def create_thread(api: TestClient, **body: Any) -> dict[str, Any]:
    response = api.post("/threads", json=body)
    assert response.status_code == 201, response.text
    thread: dict[str, Any] = response.json()
    return thread


def send(api: TestClient, thread_id: str, content: str) -> list[tuple[str, dict[str, Any]]]:
    """POST a message and return the parsed SSE events as (event, data) pairs."""
    events: list[tuple[str, dict[str, Any]]] = []
    with api.stream("POST", f"/threads/{thread_id}/messages", json={"content": content}) as r:
        assert r.status_code == 200, r.read()
        assert r.headers["content-type"].startswith("text/event-stream")
        name = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                assert name is not None
                events.append((name, json.loads(line.removeprefix("data: "))))
    return events


def upload(
    api: TestClient,
    thread_id: str,
    filename: str,
    content: bytes,
    on_conflict: str | None = None,
) -> Response:
    data = {"on_conflict": on_conflict} if on_conflict else None
    return api.post(f"/threads/{thread_id}/files", files={"file": (filename, content)}, data=data)
