"""Central configuration for GuardianCX.

Values are read from environment variables (and a local .env). Every external
integration is optional; helper flags expose whether each is configured so the
rest of the app can pick a real client or a graceful fallback.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo layout: this file is guardiancx/config/settings.py
GUARDIANCX_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = GUARDIANCX_ROOT / "artifacts"
DATA_DIR = GUARDIANCX_ROOT / "data"
POLICY_DIR = DATA_DIR / "policies"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(GUARDIANCX_ROOT / ".env"),
        env_prefix="",
        extra="ignore",
    )

    # --- Claude ---
    anthropic_api_key: str = ""
    guardiancx_claude_model: str = "claude-opus-4-8"

    # --- OpenRouter (alternative LLM provider; OpenAI-compatible) ---
    openrouter_api_key: str = ""
    openrouter_model: str = "anthropic/claude-sonnet-4.5"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # --- Azure OpenAI embeddings ---
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_embedding_deployment: str = "text-embedding-3-small"
    azure_openai_api_version: str = "2024-06-01"

    # --- Azure Speech ---
    azure_speech_key: str = ""
    azure_speech_region: str = ""

    # --- Database ---
    guardiancx_database_url: str = ""

    # --- Vector DB ---
    guardiancx_chroma_dir: str = ""

    # --- Observability ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    mlflow_tracking_uri: str = ""

    # --- Guardrail thresholds ---
    guardiancx_confidence_threshold: float = 0.55
    guardiancx_high_risk_approval: bool = True

    # ------------------------------------------------------------------ #
    # Derived paths + capability flags
    # ------------------------------------------------------------------ #
    @property
    def database_url(self) -> str:
        if self.guardiancx_database_url:
            return self.guardiancx_database_url
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{(ARTIFACTS_DIR / 'guardiancx.db').as_posix()}"

    @property
    def chroma_dir(self) -> Path:
        p = Path(self.guardiancx_chroma_dir) if self.guardiancx_chroma_dir else ARTIFACTS_DIR / "chroma"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def claude_enabled(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def openrouter_enabled(self) -> bool:
        return bool(self.openrouter_api_key)

    @property
    def llm_enabled(self) -> bool:
        return self.claude_enabled or self.openrouter_enabled

    @property
    def azure_embeddings_enabled(self) -> bool:
        return bool(self.azure_openai_api_key and self.azure_openai_endpoint)

    @property
    def azure_speech_enabled(self) -> bool:
        return bool(self.azure_speech_key and self.azure_speech_region)

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def mlflow_enabled(self) -> bool:
        return bool(self.mlflow_tracking_uri)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
