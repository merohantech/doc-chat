"""ARQ worker entry point: `arq app.workers.settings.WorkerSettings`.

Background jobs (file ingestion, compaction, titles) are registered in
`functions` as they are added in later phases.
"""

from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

import structlog
from arq.connections import RedisSettings

from app.config import get_settings
from app.logging_setup import configure_logging

log = structlog.get_logger(__name__)
_settings = get_settings()


async def ping(ctx: dict[str, Any]) -> str:
    """Trivial job used to verify the queue end to end."""
    return "pong"


async def on_startup(ctx: dict[str, Any]) -> None:
    configure_logging(_settings.log_level, json=not _settings.is_dev)
    log.info("worker.started", env=_settings.app_env)


class WorkerSettings:
    functions: ClassVar[list[Callable[..., Awaitable[Any]]]] = [ping]
    on_startup = on_startup
    redis_settings = RedisSettings.from_dsn(_settings.redis_url)
    max_jobs = 10
    job_timeout = 300
