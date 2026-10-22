from typing import Any

from app.llm.cache_policy import cache_policy_for
from app.services.context_builder import SYSTEM_PROMPT, build_prompt, user_message

HISTORY: list[dict[str, Any]] = [
    {"role": "user", "content": "What is Python?"},
    {
        "role": "assistant",
        "content": "A programming language.",
        "reasoning_details": [{"type": "reasoning.text", "text": "...", "index": 0}],
    },
]


def test_anthropic_marks_system_prompt_and_sets_automatic_caching() -> None:
    prompt = build_prompt("anthropic/claude-haiku-5.5", HISTORY, user_message("Who made it?"))

    assert prompt.messages[0] == {
        "role": "system",
        "content": [
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
        ],
    }
    assert prompt.request_fields == {"cache_control": {"type": "ephemeral"}}


def test_one_hour_ttl_is_applied_to_every_marker() -> None:
    prompt = build_prompt("anthropic/claude-haiku-5.5", [], user_message("Hi"), cache_ttl="1h")

    marker = {"type": "ephemeral", "ttl": "1h"}
    assert prompt.messages[0]["content"][0]["cache_control"] == marker
    assert prompt.request_fields == {"cache_control": marker}


def test_qwen_marks_system_prompt_only() -> None:
    policy = cache_policy_for("qwen/qwen3-max")

    assert "cache_control" in policy.system_message("x")["content"][0]
    assert policy.request_fields() == {}


def test_automatically_cached_models_get_no_markers() -> None:
    prompt = build_prompt("openai/gpt-5.6-luna", HISTORY, user_message("Who made it?"))

    assert prompt.messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert prompt.request_fields == {}


def test_history_is_replayed_unchanged_and_new_message_goes_last() -> None:
    new = user_message("Who made it?")

    prompt = build_prompt("anthropic/claude-haiku-5.5", HISTORY, new)

    assert prompt.messages[1:] == [*HISTORY, new]


def test_prompt_is_deterministic_so_the_prefix_stays_cacheable() -> None:
    first = build_prompt("anthropic/claude-haiku-5.5", HISTORY, user_message("Q"))
    second = build_prompt("anthropic/claude-haiku-5.5", HISTORY, user_message("Q"))

    assert first == second
