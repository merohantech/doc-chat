import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import Settings
from app.main import create_app
from tests.fakes import FakeLLM

BACKEND_DIR = Path(__file__).resolve().parent.parent


class TestEnv(Settings):
    """Reads TEST_DATABASE_URL in addition to the normal settings."""

    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    test_database_url: str = "postgresql+asyncpg://chat:chat@localhost:5433/chat_test"


@pytest.fixture(scope="session")
def test_env() -> TestEnv:
    return TestEnv()


@pytest.fixture(scope="session")
def settings(test_env: TestEnv, tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """App settings for tests: always the test database, never the dev one."""
    return Settings(
        app_env="test",
        database_url=test_env.test_database_url,
        # Separate Redis database so test keys never mix with dev data.
        redis_url=test_env.redis_url.rsplit("/", 1)[0] + "/15",
        openrouter_api_key=None,
        db_pool_size=2,
        db_max_overflow=0,
        storage_dir=tmp_path_factory.mktemp("uploads"),
    )


def _alembic_config(database_url: str) -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    cfg.attributes["configure_logger"] = False
    return cfg


@pytest.fixture(scope="session")
def alembic_config(settings: Settings) -> Config:
    return _alembic_config(settings.database_url)


@pytest.fixture(scope="session")
def migrated_db(alembic_config: Config) -> Iterator[None]:
    """Schema at head for the whole session; torn down afterwards."""
    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")
    yield
    command.downgrade(alembic_config, "base")


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(settings: Settings, fake_llm: FakeLLM) -> Iterator[TestClient]:
    with TestClient(create_app(settings, llm=fake_llm)) as test_client:
        yield test_client


def run_sql(settings: Settings, statement: str, **params: object) -> None:
    """Run one SQL statement against the test database from synchronous test code."""

    async def _run() -> None:
        engine = create_async_engine(settings.database_url)
        try:
            async with engine.begin() as conn:
                await conn.execute(text(statement), params)
        finally:
            await engine.dispose()

    asyncio.run(_run())


@pytest.fixture
def api(settings: Settings, fake_llm: FakeLLM, migrated_db: None) -> Iterator[TestClient]:
    """Client backed by the migrated test database; every table is emptied afterwards."""
    with TestClient(create_app(settings, llm=fake_llm)) as test_client:
        yield test_client
    run_sql(settings, "TRUNCATE users, threads, messages, files, usage CASCADE")
