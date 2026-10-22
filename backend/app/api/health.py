import asyncio
from typing import Literal

import structlog
from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from sqlalchemy import text

from app.api.deps import EngineDep, RedisDep

router = APIRouter(prefix="/health", tags=["health"])
log = structlog.get_logger(__name__)

CHECK_TIMEOUT_SECONDS = 2.0

CheckStatus = Literal["ok", "error"]


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    status: CheckStatus
    checks: dict[str, CheckStatus]


@router.get("", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    """The process is up. Deliberately checks no dependencies."""
    return LivenessResponse()


@router.get("/ready", response_model=ReadinessResponse)
async def readiness(engine: EngineDep, redis: RedisDep, response: Response) -> ReadinessResponse:
    """Dependencies are reachable; load balancers should only route here when this is 200."""

    async def check_db() -> None:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    async def check_redis() -> None:
        await redis.ping()

    checks: dict[str, CheckStatus] = {}
    for name, check in (("database", check_db), ("redis", check_redis)):
        try:
            await asyncio.wait_for(check(), timeout=CHECK_TIMEOUT_SECONDS)
            checks[name] = "ok"
        except Exception:
            log.warning("health.check_failed", check=name, exc_info=True)
            checks[name] = "error"

    healthy = all(result == "ok" for result in checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="ok" if healthy else "error", checks=checks)
