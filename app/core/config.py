from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.core.env import load_dotenv


def _get_bool(raw_value: str, default: bool = False) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class AppConfig:
    app_env: str
    app_host: str
    app_port: int
    app_language: str
    llm_provider: str
    ollama_base_url: str
    ollama_default_model: str
    ollama_default_embed_model: str
    ollama_tags_path: str
    ollama_ps_path: str
    ollama_control_timeout_seconds: float
    ollama_inference_timeout_seconds: float
    openrouter_api_key: str
    openrouter_base_url: str
    openrouter_model: str
    openrouter_embed_model: str
    embedding_batch_size: int
    vector_backend: str
    vector_collection: str
    reranker_enabled: bool
    trace_agent_steps: bool

    @classmethod
    def load(cls, dotenv_path: str | Path = ".env") -> "AppConfig":
        values = load_dotenv(dotenv_path)
        merged = {**values, **os.environ}

        return cls(
            app_env=merged.get("APP_ENV", "dev"),
            app_host=merged.get("APP_HOST", "0.0.0.0"),
            app_port=int(merged.get("APP_PORT", "8501")),
            app_language=merged.get("APP_LANGUAGE", "ru").lower(),
            llm_provider=merged.get("LLM_PROVIDER", "openrouter"),
            ollama_base_url=merged.get("OLLAMA_BASE_URL", "http://10.32.2.36:11434").rstrip("/"),
            ollama_default_model=merged.get("OLLAMA_DEFAULT_MODEL", "gemma4:26b"),
            ollama_default_embed_model=merged.get("OLLAMA_DEFAULT_EMBED_MODEL", "qwen3-embedding:8b"),
            ollama_tags_path=merged.get("OLLAMA_TAGS_PATH", "/api/tags"),
            ollama_ps_path=merged.get("OLLAMA_PS_PATH", "/api/ps"),
            ollama_control_timeout_seconds=float(merged.get("OLLAMA_CONTROL_TIMEOUT_SECONDS", "2")),
            ollama_inference_timeout_seconds=float(merged.get("OLLAMA_INFERENCE_TIMEOUT_SECONDS", "300")),
            openrouter_api_key=merged.get("OPENROUTER_API_KEY", ""),
            openrouter_base_url=merged.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
            openrouter_model=merged.get("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it"),
            openrouter_embed_model=merged.get("OPENROUTER_EMBED_MODEL", "qwen/qwen3-embedding-8b"),
            embedding_batch_size=max(1, int(merged.get("EMBEDDING_BATCH_SIZE", "16"))),
            vector_backend=merged.get("VECTOR_BACKEND", "local"),
            vector_collection=merged.get("VECTOR_COLLECTION", "easyrag_chunks"),
            reranker_enabled=_get_bool(merged.get("RERANKER_ENABLED"), default=False),
            trace_agent_steps=_get_bool(merged.get("TRACE_AGENT_STEPS"), default=True),
        )
