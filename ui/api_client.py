"""HTTP client for the chat backend. Kept free of Streamlit so it can be unit tested."""

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx


class ApiError(Exception):
    def __init__(
        self, status: int, code: str, message: str, extra: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.extra = extra or {}


@dataclass(frozen=True)
class Delta:
    text: str


@dataclass(frozen=True)
class Done:
    data: dict[str, Any]


def parse_sse(lines: Iterable[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (event, data) pairs from server-sent-event lines."""
    event = "message"
    data: list[str] = []
    for line in lines:
        if line.startswith("event:"):
            event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data.append(line.removeprefix("data:").strip())
        elif line == "" and data:
            yield event, json.loads("\n".join(data))
            event, data = "message", []
    if data:
        yield event, json.loads("\n".join(data))


class ChatApi:
    def __init__(self, base_url: str, transport: httpx.BaseTransport | None = None) -> None:
        # Long read timeout: a reply streams for as long as the model generates.
        self._http = httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(10.0, read=300.0),
            transport=transport,
        )

    def models(self) -> dict[str, Any]:
        return self._json("GET", "/models")

    def create_thread(self, model: str | None = None) -> dict[str, Any]:
        return self._json("POST", "/threads", json={"model": model})

    def get_thread(self, thread_id: str) -> dict[str, Any]:
        return self._json("GET", f"/threads/{thread_id}")

    def messages(self, thread_id: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._json("GET", f"/threads/{thread_id}/messages")
        return result

    def files(self, thread_id: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._json("GET", f"/threads/{thread_id}/files")
        return result

    def upload_file(
        self, thread_id: str, filename: str, data: bytes, on_conflict: str | None = None
    ) -> dict[str, Any]:
        """Upload a file; it is sent to the model with the thread's next message.

        Raises ApiError with code "file_name_conflict" when a different file with the
        same name exists; retry with on_conflict="replace" or "keep_both".
        """
        return self._json(
            "POST",
            f"/threads/{thread_id}/files",
            files={"file": (filename, data)},
            data={"on_conflict": on_conflict} if on_conflict else None,
        )

    def stream_reply(self, thread_id: str, content: str) -> Iterator[Delta | Done]:
        """Yield `Delta`s as the reply streams, then one `Done`. Raises `ApiError`."""
        with self._http.stream(
            "POST", f"/threads/{thread_id}/messages", json={"content": content}
        ) as response:
            if response.status_code != 200:
                response.read()
                raise _api_error(response)
            for event, data in parse_sse(response.iter_lines()):
                if event == "delta":
                    yield Delta(data["text"])
                elif event == "done":
                    yield Done(data)
                elif event == "error":
                    raise ApiError(200, data["code"], data["message"])

    def _json(self, method: str, url: str, **kwargs: Any) -> Any:
        response = self._http.request(method, url, **kwargs)
        if response.is_error:
            raise _api_error(response)
        return response.json()


def _api_error(response: httpx.Response) -> ApiError:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, dict):
        extra = {k: v for k, v in detail.items() if k not in ("code", "message")}
        return ApiError(
            response.status_code, detail.get("code", "error"), detail.get("message", ""), extra
        )
    if isinstance(detail, list):  # FastAPI request-validation errors
        return ApiError(response.status_code, "invalid_request", "; ".join(e.get("msg", "") for e in detail))
    return ApiError(response.status_code, "error", response.text or response.reason_phrase)
