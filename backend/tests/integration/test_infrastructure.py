"""Needs `docker compose up -d postgres redis`."""

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.fakes import FakeLLM

pytestmark = pytest.mark.integration


def test_models_match_migrations(alembic_config: Config, migrated_db: None) -> None:
    # Fails if a model was changed without a matching migration.
    command.check(alembic_config)


def test_readiness_ok_when_dependencies_are_up(client: TestClient) -> None:
    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "checks": {"database": "ok", "redis": "ok"}}


def test_readiness_503_when_a_dependency_is_down(settings: Settings) -> None:
    broken = settings.model_copy(update={"redis_url": "redis://localhost:6390/0"})

    with TestClient(create_app(broken, llm=FakeLLM())) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["checks"] == {"database": "ok", "redis": "error"}
