"""
Centralized application configuration.

Why this file exists (Point 12 - Configuration):
- Previously, env vars were read ad-hoc via os.getenv() scattered across main.py,
  with no validation, no documented defaults, and no distinction between what's
  required vs optional vs secret.
- pydantic-settings gives us: type validation at startup (fail fast, not mid-request),
  a single source of truth for every setting, and an easy path to per-environment
  .env files (.env.development, .env.production) without code changes.
- Secrets (API keys) are kept as SecretStr so they never get accidentally logged
  or printed in a traceback/debug dump.
"""
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Environment ---
    environment: Literal["development", "production", "test"] = Field(
        default="development", alias="APP_ENV"
    )

    # --- LLM provider (OpenRouter / OpenAI-compatible) ---
    openrouter_api_key: Optional[SecretStr] = Field(default=None, alias="OPENROUTER_API_KEY")
    openai_api_key: Optional[SecretStr] = Field(default=None, alias="OPENAI_API_KEY")
    grok_api_key: Optional[SecretStr] = Field(default=None, alias="GROK_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )
    model_name: str = Field(default="qwen/qwen3-8b", alias="MODEL_NAME")
    llm_timeout_seconds: float = Field(default=30.0, alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=2, alias="LLM_MAX_RETRIES")

    # --- Product source ---
    product_api_url: str = Field(
        default="http://localhost:3000/api/Products", alias="PRODUCT_API_URL"
    )

    # --- Chroma ---
    chroma_server_host: str = Field(default="chroma", alias="CHROMA_SERVER_HOST")
    chroma_server_http_port: int = Field(default=8000, alias="CHROMA_SERVER_HTTP_PORT")
    chroma_collection_name: str = Field(
        default="juice-shop-products", alias="CHROMA_COLLECTION_NAME"
    )

    # --- Retrieval tuning ---
    retrieval_candidate_count: int = Field(
        default=8, alias="RETRIEVAL_CANDIDATE_COUNT",
        description="How many candidates to pull from Chroma before reranking down to top-k",
    )
    retrieval_top_k: int = Field(default=3, alias="RETRIEVAL_TOP_K")
    fuzzy_match_threshold: float = Field(
        default=0.72, alias="FUZZY_MATCH_THRESHOLD",
        description="difflib SequenceMatcher ratio above which we treat a query token as naming a specific product",
    )
    retrieval_distance_threshold: float = Field(
        default=1.2, alias="RETRIEVAL_DISTANCE_THRESHOLD",
        description="L2 distance above which the best semantic-search match is treated as not "
                     "actually relevant, rather than being fed to the LLM as if it were real context. "
                     "Calibrated empirically: observed good matches ~0.68-0.99, observed genuinely "
                     "irrelevant matches ~1.47-1.50, on this catalog with all-MiniLM-L6-v2 embeddings.",
    )

    # --- Security ---
    ingest_api_key: Optional[SecretStr] = Field(
        default=None, alias="INGEST_API_KEY",
        description="Required in the X-API-Key header to call POST /assistant/ingest",
    )
    rate_limit_per_minute: int = Field(default=30, alias="RATE_LIMIT_PER_MINUTE")
    max_message_length: int = Field(default=2000, alias="MAX_MESSAGE_LENGTH")

    # --- Conversation memory ---
    conversation_max_turns: int = Field(
        default=6, alias="CONVERSATION_MAX_TURNS",
        description="How many prior user/assistant turn-pairs to keep per conversation",
    )
    conversation_ttl_seconds: int = Field(default=1800, alias="CONVERSATION_TTL_SECONDS")

    # --- Logging ---
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=True, alias="LOG_JSON")

    def resolve_provider(self) -> tuple[str, str, str, str]:
        """Returns (provider, api_key, model, base_url). Mirrors the original
        get_provider_settings() behavior but reads validated Settings instead
        of raw os.getenv() calls."""
        if self.openrouter_api_key:
            return "openrouter", self.openrouter_api_key.get_secret_value(), self.model_name, self.openrouter_base_url
        if self.grok_api_key:
            return "grok", self.grok_api_key.get_secret_value(), "grok-3-mini", "https://api.x.ai/v1"
        if self.openai_api_key:
            return "openai", self.openai_api_key.get_secret_value(), "gpt-4o-mini", "https://api.openai.com/v1"
        return "none", "", "gpt-4o-mini", "https://api.openai.com/v1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton - Settings is parsed from env once per process,
    not re-read on every request (Point 7 - Performance)."""
    return Settings()
