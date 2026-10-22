import json
import uuid
from collections.abc import AsyncGenerator, AsyncIterator

import structlog
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.deps import ChatServiceDep, CurrentUserDep, SettingsDep
from app.api.errors import ApiError, describe
from app.api.schemas import (
    ErrorEvent,
    MessageCreate,
    MessageOut,
    ThreadCreate,
    ThreadOut,
    TurnCompletedEvent,
    UsageOut,
)
from app.llm.base import TextDelta
from app.services.chat_service import TurnCompleted, TurnEvent

router = APIRouter(prefix="/threads", tags=["threads"])
log = structlog.get_logger(__name__)


@router.post("", response_model=ThreadOut, status_code=status.HTTP_201_CREATED)
async def create_thread(
    body: ThreadCreate, user_id: CurrentUserDep, chat: ChatServiceDep, settings: SettingsDep
) -> ThreadOut:
    if body.model is not None and body.model not in settings.allowed_chat_models:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "model_not_allowed",
                "message": f"Model must be one of: {', '.join(settings.allowed_chat_models)}",
            },
        )
    thread = await chat.create_thread(user_id, body.model, body.title)
    return ThreadOut.model_validate(thread)


@router.get("/{thread_id}", response_model=ThreadOut)
async def get_thread(
    thread_id: uuid.UUID, user_id: CurrentUserDep, chat: ChatServiceDep
) -> ThreadOut:
    return ThreadOut.model_validate(await chat.get_thread(user_id, thread_id))


@router.get("/{thread_id}/messages", response_model=list[MessageOut])
async def list_messages(
    thread_id: uuid.UUID, user_id: CurrentUserDep, chat: ChatServiceDep
) -> list[MessageOut]:
    return [MessageOut.from_row(row) for row in await chat.list_messages(user_id, thread_id)]


@router.post(
    "/{thread_id}/messages",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": (
                "Server-sent events: `delta` {text} while the reply streams, then one "
                "`done` (TurnCompletedEvent) or `error` (ErrorEvent)."
            ),
        }
    },
)
async def send_message(
    thread_id: uuid.UUID,
    body: MessageCreate,
    user_id: CurrentUserDep,
    chat: ChatServiceDep,
    settings: SettingsDep,
) -> StreamingResponse:
    if len(body.content) > settings.max_message_chars:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "message_too_long",
                "message": f"Messages are limited to {settings.max_message_chars} characters.",
            },
        )

    events = chat.stream_turn(user_id, thread_id, body.content)
    # Pull the first event before committing to a 200 streaming response, so
    # failures that happen up front (unknown thread, busy thread, rate limit,
    # rejected request) become proper HTTP status codes via the error handlers.
    try:
        first = await anext(events)
    except StopAsyncIteration:  # pragma: no cover - stream_turn always yields a final event
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE) from None

    return StreamingResponse(
        _sse(first, events),
        media_type="text/event-stream",
        # Disable proxy buffering (nginx) so tokens reach the client as they arrive.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _sse(first: TurnEvent, rest: AsyncGenerator[TurnEvent]) -> AsyncIterator[str]:
    try:
        yield _encode(first)
        async for event in rest:
            yield _encode(event)
    except Exception as exc:
        # The 200 status is already sent; report the failure in-band.
        try:
            error = describe(exc)
        except Exception:
            error = ApiError(500, "internal_error", "Something went wrong. Please try again.")
        log.warning("chat.stream_failed", code=error.code, exc_info=True)
        yield _event("error", ErrorEvent(code=error.code, message=error.message))
    finally:
        await rest.aclose()


def _encode(event: TurnEvent) -> str:
    if isinstance(event, TextDelta):
        return f"event: delta\ndata: {json.dumps({'text': event.text})}\n\n"
    assert isinstance(event, TurnCompleted)
    return _event(
        "done",
        TurnCompletedEvent(
            user_message_id=event.user_message_id,
            assistant_message_id=event.assistant_message_id,
            finish_reason=event.finish_reason,
            usage=UsageOut.from_usage(event.usage) if event.usage else None,
        ),
    )


def _event(name: str, payload: BaseModel) -> str:
    return f"event: {name}\ndata: {payload.model_dump_json()}\n\n"
