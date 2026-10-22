"""Maps domain exceptions to HTTP responses, so routes and services stay HTTP-agnostic.

Error bodies are `{"detail": {"code": ..., "message": ..., **extra}}`; `code` is
stable and machine-readable, `message` is safe to show to a user.
"""

from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from app.ingestion.extractors import ExtractionError
from app.llm.base import LLMError, LLMRateLimitError, LLMRequestError, LLMUnavailableError
from app.services.chat_service import EmptyCompletionError, ThreadBusyError, ThreadNotFoundError
from app.services.file_service import (
    DuplicateFileError,
    FileNameConflictError,
    FileTooLargeError,
)


@dataclass(frozen=True, slots=True)
class ApiError:
    status: int
    code: str
    message: str
    extra: dict[str, Any] = field(default_factory=dict)

    def body(self) -> dict[str, Any]:
        return {"detail": {"code": self.code, "message": self.message, **self.extra}}


# (status code, code, message) for errors whose message never varies.
_STATIC: dict[type[Exception], tuple[int, str, str]] = {
    ThreadNotFoundError: (status.HTTP_404_NOT_FOUND, "thread_not_found", "Thread not found."),
    ThreadBusyError: (
        status.HTTP_409_CONFLICT,
        "thread_busy",
        "A reply is already being generated in this thread.",
    ),
    LLMRateLimitError: (
        status.HTTP_429_TOO_MANY_REQUESTS,
        "llm_rate_limited",
        "The model is rate limited. Try again shortly.",
    ),
    LLMUnavailableError: (
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "llm_unavailable",
        "The model is temporarily unavailable. Try again shortly.",
    ),
    EmptyCompletionError: (
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "llm_unavailable",
        "The model returned no reply. Try again.",
    ),
    LLMRequestError: (
        status.HTTP_502_BAD_GATEWAY,
        "llm_request_rejected",
        "The model provider rejected the request.",
    ),
    IntegrityError: (
        status.HTTP_409_CONFLICT,
        "conflict",
        "Another request changed this resource at the same time. Please retry.",
    ),
}


def describe(exc: Exception) -> ApiError:
    if isinstance(exc, FileNameConflictError):
        existing = exc.existing
        return ApiError(
            status.HTTP_409_CONFLICT,
            "file_name_conflict",
            f'A different file named "{existing.filename}" is already in this chat. '
            "Replace it with this version, or keep both?",
            {"existing_file_id": str(existing.id), "existing_version": existing.version},
        )
    if isinstance(exc, DuplicateFileError):
        return ApiError(
            status.HTTP_409_CONFLICT,
            "file_duplicate",
            f'This file is already in the chat as "{exc.existing.filename}".',
            {"existing_file_id": str(exc.existing.id)},
        )
    if isinstance(exc, FileTooLargeError):
        return ApiError(status.HTTP_413_CONTENT_TOO_LARGE, "file_too_large", exc.message)
    if isinstance(exc, ExtractionError):
        return ApiError(status.HTTP_422_UNPROCESSABLE_CONTENT, exc.code, exc.message)
    for exc_type, (status_code, code, message) in _STATIC.items():
        if isinstance(exc, exc_type):
            return ApiError(status_code, code, message)
    if isinstance(exc, LLMError):
        return describe(LLMUnavailableError(str(exc)))
    raise exc


HANDLED: tuple[type[Exception], ...] = (
    *_STATIC,
    LLMError,
    FileNameConflictError,
    DuplicateFileError,
    FileTooLargeError,
    ExtractionError,
)


def register_error_handlers(app: FastAPI) -> None:
    async def handle(request: Request, exc: Exception) -> JSONResponse:
        error = describe(exc)
        return JSONResponse(status_code=error.status, content=error.body())

    for exc_type in HANDLED:
        app.add_exception_handler(exc_type, handle)
