from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from redis.asyncio import Redis

from app.api import files, health, models, threads
from app.api.errors import register_error_handlers
from app.config import Settings, get_settings
from app.db.session import create_engine, create_sessionmaker
from app.llm.base import LLMProvider
from app.llm.openrouter import OpenRouterProvider
from app.logging_setup import configure_logging
from app.middleware import RequestIdMiddleware
from app.storage.local import LocalStorage

log = structlog.get_logger(__name__)


def create_app(settings: Settings | None = None, llm: LLMProvider | None = None) -> FastAPI:
    """Build the app. `llm` can be injected (e.g. a fake in tests); otherwise OpenRouter."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, json=not settings.is_dev)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_engine(settings)
        redis = Redis.from_url(settings.redis_url)
        provider = llm or OpenRouterProvider.from_settings(settings)

        app.state.settings = settings
        app.state.engine = engine
        app.state.sessionmaker = create_sessionmaker(engine)
        app.state.redis = redis
        app.state.llm = provider
        app.state.storage = LocalStorage(settings.storage_dir)
        log.info("app.started", env=settings.app_env, chat_model=settings.default_chat_model)
        try:
            yield
        finally:
            await provider.aclose()
            await redis.aclose()
            await engine.dispose()
            log.info("app.stopped")

    app = FastAPI(title="Chat API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(RequestIdMiddleware)
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(threads.router)
    app.include_router(files.router)
    return app
