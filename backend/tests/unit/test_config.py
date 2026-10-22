from app.config import Settings


def test_blank_api_key_counts_as_unset() -> None:
    settings = Settings(_env_file=None, openrouter_api_key="  ")

    assert settings.openrouter_api_key is None


def test_api_key_is_kept_secret() -> None:
    settings = Settings(_env_file=None, openrouter_api_key="sk-or-123")

    assert settings.openrouter_api_key is not None
    assert settings.openrouter_api_key.get_secret_value() == "sk-or-123"
    assert "sk-or-123" not in repr(settings)
