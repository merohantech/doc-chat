"""Prompt-caching rules per model family on OpenRouter.

- Anthropic needs explicit `cache_control` markers. We mark the system prompt
  (a stable read point) and set the top-level automatic marker, which moves
  forward with the conversation so each turn reads the whole prior history.
- Qwen also needs explicit markers but has no automatic mode; we mark the
  system prompt only.
- Everything else (OpenAI, Gemini 2.5+, DeepSeek, Grok, ...) caches
  automatically, so we send nothing extra.

Caching is a prefix match: markers only help if everything before them is
byte-stable across requests, which `context_builder` guarantees.
"""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

CacheTTL = Literal["5m", "1h"]


class CachePolicy(Protocol):
    def system_message(self, text: str) -> dict[str, Any]: ...

    def request_fields(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ExplicitCachePolicy:
    ttl: CacheTTL
    automatic: bool

    def _marker(self) -> dict[str, Any]:
        marker: dict[str, Any] = {"type": "ephemeral"}
        if self.ttl != "5m":
            marker["ttl"] = self.ttl
        return marker

    def system_message(self, text: str) -> dict[str, Any]:
        return {
            "role": "system",
            "content": [{"type": "text", "text": text, "cache_control": self._marker()}],
        }

    def request_fields(self) -> dict[str, Any]:
        return {"cache_control": self._marker()} if self.automatic else {}


@dataclass(frozen=True, slots=True)
class ImplicitCachePolicy:
    def system_message(self, text: str) -> dict[str, Any]:
        return {"role": "system", "content": text}

    def request_fields(self) -> dict[str, Any]:
        return {}


def cache_policy_for(model: str, ttl: CacheTTL = "5m") -> CachePolicy:
    vendor = model.split("/", 1)[0]
    if vendor == "anthropic":
        return ExplicitCachePolicy(ttl=ttl, automatic=True)
    if vendor == "qwen":
        return ExplicitCachePolicy(ttl=ttl, automatic=False)
    return ImplicitCachePolicy()
