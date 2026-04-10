from __future__ import annotations

from app.core.config import AppConfig
from app.core.models import EmbeddingResult, GenerationResult, ModelInfo, ProviderSelection
from app.providers.base import BaseProvider, ProviderError
from app.providers.http import HttpJsonClient


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.control_client = HttpJsonClient(
            base_url=config.ollama_base_url,
            timeout=config.ollama_control_timeout_seconds,
        )
        self.client = HttpJsonClient(
            base_url=config.ollama_base_url,
            timeout=config.ollama_inference_timeout_seconds,
        )

    def list_models(self) -> list[ModelInfo]:
        payload = self.control_client.get_json(self.config.ollama_tags_path)
        models = payload.get("models", [])

        result: list[ModelInfo] = []
        for item in models:
            details = item.get("details", {}) if isinstance(item, dict) else {}
            result.append(
                ModelInfo(
                    name=item.get("name", ""),
                    size=item.get("size"),
                    family=details.get("family"),
                    details=details,
                )
            )
        return result

    def get_active_model(self) -> str | None:
        payload = self.control_client.get_json(self.config.ollama_ps_path)
        models = payload.get("models", [])
        if not models:
            return None
        first = models[0]
        return first.get("name") if isinstance(first, dict) else None

    def resolve_selection(self) -> ProviderSelection:
        try:
            available_models = self.list_models()
            active_model = self.get_active_model()
        except Exception as exc:
            raise ProviderError(str(exc)) from exc

        available_names = {model.name for model in available_models}
        selected_chat_model = self._pick_chat_model(active_model, available_names)
        selected_embed_model = self._pick_embed_model(available_names)
        reason = "Active model from /api/ps is selected" if active_model in available_names else "Default model selection applied"

        return ProviderSelection(
            provider_name=self.name,
            reachable=True,
            reason=reason,
            chat_model=selected_chat_model,
            embed_model=selected_embed_model,
            available_models=available_models,
            active_model=active_model,
        )

    def generate(self, prompt: str, model: str | None = None) -> GenerationResult:
        resolved_model = model or self.resolve_selection().chat_model
        payload = {
            "model": resolved_model,
            "prompt": prompt,
            "stream": False,
        }
        response = self.client.post_json("/api/generate", payload)
        return GenerationResult(
            text=response.get("response", ""),
            model=resolved_model,
            raw=response,
        )

    def embed(self, text: str, model: str | None = None) -> EmbeddingResult:
        resolved_model = model or self.resolve_selection().embed_model
        payload = {
            "model": resolved_model,
            "input": text,
        }
        response = self.client.post_json("/api/embed", payload)
        embeddings = response.get("embeddings") or []
        vector = embeddings[0] if embeddings else []
        return EmbeddingResult(vector=vector, model=resolved_model, raw=response)

    def embed_many(self, texts: list[str], model: str | None = None) -> list[EmbeddingResult]:
        if not texts:
            return []

        resolved_model = model or self.resolve_selection().embed_model
        payload = {
            "model": resolved_model,
            "input": texts,
        }
        response = self.client.post_json("/api/embed", payload)
        embeddings = response.get("embeddings") or []
        return [EmbeddingResult(vector=vector, model=resolved_model, raw=response) for vector in embeddings]

    def _pick_chat_model(self, active_model: str | None, available_names: set[str]) -> str:
        if active_model and active_model in available_names:
            return active_model
        if self.config.ollama_default_model in available_names:
            return self.config.ollama_default_model
        if available_names:
            return sorted(available_names)[0]
        return self.config.ollama_default_model

    def _pick_embed_model(self, available_names: set[str]) -> str:
        if self.config.ollama_default_embed_model in available_names:
            return self.config.ollama_default_embed_model
        return self.config.ollama_default_embed_model
