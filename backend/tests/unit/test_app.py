import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm.openrouter import OpenRouterProvider
from app.main import create_app
from tests.fakes import FakeLLM


def test_liveness_needs_no_dependencies(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_request_id_is_generated_when_absent(client: TestClient) -> None:
    response = client.get("/health")

    assert len(response.headers["x-request-id"]) == 32


def test_request_id_is_propagated_when_given(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "abc-123"})

    assert response.headers["x-request-id"] == "abc-123"


def test_models_endpoint_lists_default_and_allowed(client: TestClient, settings: Settings) -> None:
    response = client.get("/models")

    assert response.json() == {
        "default": settings.default_chat_model,
        "allowed": settings.allowed_chat_models,
    }


def test_llm_provider_is_closed_on_shutdown(settings: Settings) -> None:
    llm = FakeLLM()
    with TestClient(create_app(settings, llm=llm)):
        assert not llm.closed
    assert llm.closed


def test_openrouter_requires_an_api_key(settings: Settings) -> None:
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider.from_settings(settings)
