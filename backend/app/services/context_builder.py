"""The single place where the prompt sent to the LLM is assembled.

Layout, most stable first (see docs/IMPLEMENTATION_PLAN.md section 2):

    [system prompt: frozen constant]       <- cache marker (explicit-cache models)
    [history: replayed exactly as stored]
    [new user turn: pending files + text]  <- automatic cache marker (request field)

Files enter the conversation inside the user turn that follows their upload,
never in the system prompt, so an upload doesn't invalidate the cached history.
A newer version is appended the same way (with a "replaces" attribute); the old
version stays in history untouched.

Rules that keep the prompt cache warm:
- SYSTEM_PROMPT is a constant. Never interpolate dates, IDs or user data into it.
- History is append-only and replayed exactly as persisted, including
  `reasoning_details`; nothing is edited, reordered or dropped.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import Any

from app.llm.cache_policy import CacheTTL, cache_policy_for

SYSTEM_PROMPT = """\
You are a helpful, knowledgeable assistant in a multi-turn chat. The user may ask \
follow-up questions that refer to earlier messages in this conversation; use the full \
conversation history to resolve references such as "it", "that" or "the previous one".

Guidelines:
- Answer accurately and directly. Lead with the answer, then add detail if it helps.
- If the question is ambiguous, state the interpretation you chose, or ask a brief \
clarifying question when the interpretations lead to very different answers.
- If you don't know something or the conversation doesn't contain it, say so plainly \
instead of guessing.
- Use Markdown when it aids readability: lists for steps, tables for comparisons, \
fenced code blocks with a language tag for code.
- Match the length of your answer to the question: short questions get short answers.

Files:
- The user can share files. Each appears inside a <file name="..." version="..."> tag \
in the message where it was shared; PDFs include "--- Page N ---" markers.
- When answering from a file, base the answer on the file's content and cite the file \
name, plus the page or section when you can (for example: report.pdf, page 3).
- A file tag with a "replaces" attribute is a newer version of a file shared earlier. \
Treat it as the current version and answer from it only, unless the user asks about \
an earlier version or what changed between versions.
- If a file doesn't contain the answer, say so rather than answering from general knowledge \
as if it came from the file.
"""


@dataclass(frozen=True, slots=True)
class Prompt:
    messages: list[dict[str, Any]]
    request_fields: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Attachment:
    filename: str
    version: int
    text: str
    # True when this file replaces a version the model has already seen.
    replaces_sent_version: bool = False


def user_message(text: str, attachments: Sequence[Attachment] = ()) -> dict[str, Any]:
    """A user turn. With files, content is a list of text parts: files first, question last."""
    if not attachments:
        return {"role": "user", "content": text}
    parts = [{"type": "text", "text": _file_block(a)} for a in attachments]
    parts.append({"type": "text", "text": text})
    return {"role": "user", "content": parts}


def _file_block(attachment: Attachment) -> str:
    attributes = f'name="{escape(attachment.filename)}" version="{attachment.version}"'
    if attachment.replaces_sent_version:
        attributes += f' replaces="version {attachment.version - 1}"'
    return f"<file {attributes}>\n{attachment.text}\n</file>"


def build_prompt(
    model: str,
    history: Sequence[Mapping[str, Any]],
    new_user_message: Mapping[str, Any],
    cache_ttl: CacheTTL = "5m",
) -> Prompt:
    policy = cache_policy_for(model, cache_ttl)
    messages = [
        policy.system_message(SYSTEM_PROMPT),
        *(dict(m) for m in history),
        dict(new_user_message),
    ]
    return Prompt(messages=messages, request_fields=policy.request_fields())
