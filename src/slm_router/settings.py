"""Environment-backed settings (secrets, paths). YAML research knobs live in config.py."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Loaded from environment / .env.

    Most knobs use the SLM_ prefix (SLM_REQUESTS_PER_SECOND, etc.); the API key
    keeps its conventional unprefixed name OPENROUTER_API_KEY.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    openrouter_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "SLM_OPENROUTER_API_KEY"),
    )
    requests_per_second: float = Field(
        default=4.0, validation_alias=AliasChoices("SLM_REQUESTS_PER_SECOND")
    )
    config_path: str = Field(
        default="configs/config.yaml", validation_alias=AliasChoices("SLM_CONFIG_PATH")
    )
    models_path: str = Field(
        default="configs/models.yaml", validation_alias=AliasChoices("SLM_MODELS_PATH")
    )
    cache_dir: str = Field(default=".cache", validation_alias=AliasChoices("SLM_CACHE_DIR"))
    obs_enabled: bool = Field(default=False, validation_alias=AliasChoices("SLM_OBS_ENABLED"))


def load_settings() -> Settings:
    return Settings()
