from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, read from environment variables and `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://chat:chat@localhost:5432/chat"
    db_pool_size: int = Field(default=10, ge=1)
    db_max_overflow: int = Field(default=10, ge=0)

    redis_url: str = "redis://localhost:6379/0"

    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Sent as HTTP-Referer / X-Title so requests are attributed to this app on OpenRouter.
    openrouter_app_url: str | None = None
    openrouter_app_name: str = "chat-backend"

    default_chat_model: str = "anthropic/claude-haiku-5.5"
    # Models a thread may use. Keeps cost under control and rejects typos up front.
    allowed_chat_models: list[str] = [
        "anthropic/claude-haiku-5.5",
        "anthropic/claude-sonnet-5.5",
    ]
    # Cheap model for side tasks: titles, summaries, query rewrites.
    utility_model: str = "anthropic/claude-haiku-5.5"
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)

    chat_max_tokens: int = Field(default=4096, ge=1)
    max_message_chars: int = Field(default=32_000, ge=1)
    # "5m" suits continuous chat; "1h" pays off when users pause 5-60 min between turns.
    prompt_cache_ttl: Literal["5m", "1h"] = "5m"

    # Until authentication exists (Phase 5), every request acts as this user.
    dev_user_email: str = "dev@localhost"

    storage_dir: Path = Path("storage/uploads")
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    # Files are sent to the model in full; ~20k tokens is roughly 25 pages of prose.
    max_file_tokens: int = Field(default=20_000, ge=1)

    @field_validator("openrouter_api_key", "openrouter_app_url", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        # `KEY=` in .env means "not configured", not "configured as empty".
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def _default_model_is_allowed(self) -> "Settings":
        if self.default_chat_model not in self.allowed_chat_models:
            raise ValueError(
                f"DEFAULT_CHAT_MODEL {self.default_chat_model!r} is not in ALLOWED_CHAT_MODELS"
            )
        return self

    @property
    def is_dev(self) -> bool:
        return self.app_env == "dev"

    @property
    def turn_lock_seconds(self) -> float:
        """Upper bound for one chat turn: every attempt timing out, plus headroom."""
        return self.llm_timeout_seconds * (self.llm_max_retries + 1) + 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
