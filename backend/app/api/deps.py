"""FastAPI dependencies. Shared resources live on `app.state`, created in the lifespan."""

import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.config import Settings
from app.db.repositories.users import UserRepository
from app.db.session import session_scope
from app.llm.base import LLMProvider
from app.services.chat_service import ChatService
from app.services.file_service import FileService


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_engine(request: Request) -> AsyncEngine:
    engine: AsyncEngine = request.app.state.engine
    return engine


def get_redis(request: Request) -> Redis:
    redis: Redis = request.app.state.redis
    return redis


def get_llm(request: Request) -> LLMProvider:
    llm: LLMProvider = request.app.state.llm
    return llm


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    async for session in session_scope(request.app.state.sessionmaker):
        yield session


async def get_current_user_id(request: Request) -> uuid.UUID:
    """Placeholder identity until authentication lands (Phase 5).

    Every request acts as the configured dev user; the id is cached after the
    first lookup. Replace this dependency with token validation later; nothing
    else needs to change.
    """
    state = request.app.state
    user_id: uuid.UUID | None = getattr(state, "dev_user_id", None)
    if user_id is None:
        async with state.sessionmaker.begin() as session:
            user_id = await UserRepository(session).ensure(state.settings.dev_user_email)
        state.dev_user_id = user_id
    return user_id


def get_chat_service(request: Request) -> ChatService:
    state = request.app.state
    return ChatService(state.sessionmaker, state.redis, state.llm, state.settings)


def get_file_service(request: Request) -> FileService:
    state = request.app.state
    return FileService(state.sessionmaker, state.storage, state.settings)


SettingsDep = Annotated[Settings, Depends(get_settings)]
EngineDep = Annotated[AsyncEngine, Depends(get_engine)]
RedisDep = Annotated[Redis, Depends(get_redis)]
LLMDep = Annotated[LLMProvider, Depends(get_llm)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
CurrentUserDep = Annotated[uuid.UUID, Depends(get_current_user_id)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
FileServiceDep = Annotated[FileService, Depends(get_file_service)]
