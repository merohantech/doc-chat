"""End-to-end chat flow over HTTP, with a fake LLM and real Postgres + Redis."""

import uuid

import pytest
from fastapi.testclient import TestClient
from redis import Redis

from app.config import Settings
from app.llm.base import LLMRateLimitError, LLMUnavailableError
from tests.api_helpers import create_thread, send
from tests.conftest import run_sql
from tests.fakes import FakeLLM

pytestmark = pytest.mark.integration


def test_create_thread_uses_default_model(api: TestClient, settings: Settings) -> None:
    thread = create_thread(api, title="My chat")

    assert thread["model"] == settings.default_chat_model
    assert thread["title"] == "My chat"
    assert api.get(f"/threads/{thread['id']}").json()["id"] == thread["id"]


def test_create_thread_rejects_models_outside_the_allow_list(api: TestClient) -> None:
    response = api.post("/threads", json={"model": "openai/some-expensive-model"})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "model_not_allowed"


def test_a_turn_streams_deltas_then_done_and_is_persisted(
    api: TestClient, fake_llm: FakeLLM
) -> None:
    thread = create_thread(api)

    events = send(api, thread["id"], "Hi there")

    deltas = [data["text"] for name, data in events if name == "delta"]
    assert "".join(deltas) == fake_llm.reply
    name, done = events[-1]
    assert name == "done"
    assert done["usage"]["prompt_tokens"] == 100

    messages = api.get(f"/threads/{thread['id']}/messages").json()
    assert [(m["seq"], m["role"], m["content"]) for m in messages] == [
        (1, "user", "Hi there"),
        (2, "assistant", fake_llm.reply),
    ]
    assert messages[0]["id"] == done["user_message_id"]
    assert messages[1]["id"] == done["assistant_message_id"]
    assert messages[1]["usage"]["completion_tokens"] == 5
    assert api.get(f"/threads/{thread['id']}").json()["est_context_tokens"] == 105


def test_multi_turn_resends_full_history_with_a_stable_prefix(
    api: TestClient, fake_llm: FakeLLM
) -> None:
    thread = create_thread(api)

    for question in ("My name is Ada.", "What is my name?", "Spell it backwards."):
        send(api, thread["id"], question)

    first, second, third = fake_llm.requests
    # Each request is the previous one plus (assistant reply, new user message):
    # an append-only prefix, which is what lets the provider cache it.
    assistant = {"role": "assistant", "content": fake_llm.reply}
    assert second.messages == [
        *first.messages,
        assistant,
        {"role": "user", "content": "What is my name?"},
    ]
    assert third.messages == [
        *second.messages,
        assistant,
        {"role": "user", "content": "Spell it backwards."},
    ]
    # Same session id on every turn keeps OpenRouter on the same (warm-cache) provider.
    assert {r.session_id for r in fake_llm.requests} == {thread["id"]}
    assert len(api.get(f"/threads/{thread['id']}/messages").json()) == 6


def test_anthropic_threads_send_cache_markers(api: TestClient, fake_llm: FakeLLM) -> None:
    thread = create_thread(api, model="anthropic/claude-haiku-5.5")

    send(api, thread["id"], "Hi")

    request = fake_llm.requests[0]
    assert request.extra_body == {"cache_control": {"type": "ephemeral"}}
    assert request.messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_unknown_thread_is_404(api: TestClient) -> None:
    unknown = uuid.uuid4()

    assert api.get(f"/threads/{unknown}/messages").status_code == 404
    response = api.post(f"/threads/{unknown}/messages", json={"content": "Hi"})
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "thread_not_found"


def test_another_users_thread_is_404(api: TestClient, settings: Settings) -> None:
    other_user, other_thread = uuid.uuid4(), uuid.uuid4()
    run_sql(settings, "INSERT INTO users (id, email) VALUES (:id, 'other@x')", id=other_user)
    run_sql(
        settings,
        "INSERT INTO threads (id, user_id, model) VALUES (:id, :uid, 'm')",
        id=other_thread,
        uid=other_user,
    )

    assert api.get(f"/threads/{other_thread}").status_code == 404
    assert api.post(f"/threads/{other_thread}/messages", json={"content": "Hi"}).status_code == 404


def test_concurrent_turn_in_the_same_thread_is_409(api: TestClient, settings: Settings) -> None:
    thread = create_thread(api)
    redis = Redis.from_url(settings.redis_url)
    redis.set(f"thread-turn:{thread['id']}", "held-by-another-request", ex=30)
    try:
        response = api.post(f"/threads/{thread['id']}/messages", json={"content": "Hi"})
    finally:
        redis.delete(f"thread-turn:{thread['id']}")
        redis.close()

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "thread_busy"


@pytest.mark.parametrize(
    ("error", "status_code", "code"),
    [
        (LLMRateLimitError("slow down"), 429, "llm_rate_limited"),
        (LLMUnavailableError("down"), 503, "llm_unavailable"),
    ],
)
def test_upfront_llm_failure_is_an_http_error_and_persists_nothing(
    api: TestClient, fake_llm: FakeLLM, error: Exception, status_code: int, code: str
) -> None:
    thread = create_thread(api)
    fake_llm.fail_before = error

    response = api.post(f"/threads/{thread['id']}/messages", json={"content": "Hi"})

    assert response.status_code == status_code
    assert response.json()["detail"]["code"] == code
    assert api.get(f"/threads/{thread['id']}/messages").json() == []


def test_mid_stream_failure_is_an_error_event_and_persists_nothing(
    api: TestClient, fake_llm: FakeLLM
) -> None:
    thread = create_thread(api)
    fake_llm.fail_after = LLMUnavailableError("provider disconnected")

    events = send(api, thread["id"], "Hi")

    assert [name for name, _ in events] == ["delta", "error"]
    assert events[-1][1]["code"] == "llm_unavailable"
    assert api.get(f"/threads/{thread['id']}/messages").json() == []


def test_lock_is_released_after_a_failed_turn(api: TestClient, fake_llm: FakeLLM) -> None:
    thread = create_thread(api)
    fake_llm.fail_after = LLMUnavailableError("boom")
    send(api, thread["id"], "Hi")

    fake_llm.fail_after = None
    events = send(api, thread["id"], "Hi again")

    assert events[-1][0] == "done"


def test_message_validation(api: TestClient, settings: Settings) -> None:
    thread = create_thread(api)
    url = f"/threads/{thread['id']}/messages"

    assert api.post(url, json={"content": ""}).status_code == 422
    too_long = "x" * (settings.max_message_chars + 1)
    response = api.post(url, json={"content": too_long})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "message_too_long"
