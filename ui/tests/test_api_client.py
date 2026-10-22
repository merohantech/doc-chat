import json

import httpx
import pytest

from api_client import ApiError, ChatApi, Delta, Done, parse_sse


def _api(handler) -> ChatApi:  # type: ignore[no-untyped-def]
    return ChatApi("http://api.test", transport=httpx.MockTransport(handler))


def test_parse_sse_yields_events_with_json_data() -> None:
    lines = ["event: delta", 'data: {"text": "Hi"}', "", "event: done", 'data: {"ok": true}', ""]

    assert list(parse_sse(lines)) == [("delta", {"text": "Hi"}), ("done", {"ok": True})]


def test_stream_reply_yields_deltas_then_done() -> None:
    body = (
        'event: delta\ndata: {"text": "Hel"}\n\n'
        'event: delta\ndata: {"text": "lo"}\n\n'
        'event: done\ndata: {"usage": null}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"content": "Hi"}
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    events = list(_api(handler).stream_reply("t1", "Hi"))

    assert events == [Delta("Hel"), Delta("lo"), Done({"usage": None})]


def test_stream_reply_raises_on_error_event() -> None:
    body = 'event: delta\ndata: {"text": "x"}\n\nevent: error\ndata: {"code": "llm_unavailable", "message": "Down"}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    with pytest.raises(ApiError) as excinfo:
        list(_api(handler).stream_reply("t1", "Hi"))
    assert excinfo.value.code == "llm_unavailable"


def test_http_errors_carry_the_api_error_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": {"code": "thread_busy", "message": "Busy"}})

    with pytest.raises(ApiError) as excinfo:
        list(_api(handler).stream_reply("t1", "Hi"))
    assert (excinfo.value.status, excinfo.value.code, excinfo.value.message) == (409, "thread_busy", "Busy")


def test_upload_sends_multipart_and_conflict_choice() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content
        assert request.url.path == "/threads/t1/files"
        assert b'filename="notes.txt"' in body and b"hello" in body
        assert b'name="on_conflict"' in body and b"replace" in body
        return httpx.Response(201, json={"id": "f1", "version": 2})

    assert _api(handler).upload_file("t1", "notes.txt", b"hello", "replace")["version"] == 2


def test_conflict_error_keeps_extra_details() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "detail": {
                    "code": "file_name_conflict",
                    "message": "Replace or keep both?",
                    "existing_version": 1,
                }
            },
        )

    with pytest.raises(ApiError) as excinfo:
        _api(handler).upload_file("t1", "notes.txt", b"hello")
    assert excinfo.value.code == "file_name_conflict"
    assert excinfo.value.extra == {"existing_version": 1}


def test_validation_errors_are_readable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": [{"msg": "String should have at least 1 character"}]})

    with pytest.raises(ApiError) as excinfo:
        _api(handler).create_thread()
    assert excinfo.value.code == "invalid_request"
    assert "at least 1 character" in excinfo.value.message
