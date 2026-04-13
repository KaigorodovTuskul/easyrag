from __future__ import annotations

import base64

from app.core.config import AppConfig
from app.core.models import EmbeddingResult, GenerationResult, ImageInput, ModelInfo, ProviderSelection
from app.providers.base import BaseProvider, ProviderError
from app.providers.http import HttpJsonClient


class OpenRouterProvider(BaseProvider):
    name = "openrouter"
    default_vision_model = "google/gemma-4-26b-a4b-it"
    fallback_vision_model = "qwen/qwen3-vl-32b-instruct"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        headers = {}
        if config.openrouter_api_key:
            headers["Authorization"] = f"Bearer {config.openrouter_api_key}"
        self.client = HttpJsonClient(base_url=config.openrouter_base_url, headers=headers)

    def list_models(self) -> list[ModelInfo]:
        if not self.config.openrouter_api_key:
            models = [self.config.openrouter_model, self.config.openrouter_embed_model]
            models.extend(self._vision_candidates())
            return [ModelInfo(name=name) for name in dict.fromkeys(models) if name]

        payload = self.client.get_json("/models")
        items = payload.get("data", [])
        return [
            ModelInfo(
                name=item.get("id", ""),
                family=item.get("architecture", {}).get("modality"),
                details=item,
            )
            for item in items
        ]

    def get_active_model(self) -> str | None:
        return None

    def resolve_selection(self) -> ProviderSelection:
        selected_vision_model = self._pick_vision_model(None if not self.config.openrouter_api_key else set())
        if not self.config.openrouter_api_key:
            return ProviderSelection(
                provider_name=self.name,
                reachable=False,
                reason="OPENROUTER_API_KEY is not set",
                chat_model=self.config.openrouter_model,
                embed_model=self.config.openrouter_embed_model,
                vision_model=selected_vision_model,
                available_models=self.list_models(),
                active_model=None,
            )

        try:
            models = self.list_models()
        except Exception as exc:
            raise ProviderError(str(exc)) from exc

        available_names = {model.name for model in models}
        return ProviderSelection(
            provider_name=self.name,
            reachable=True,
            reason="Configured OpenRouter fallback",
            chat_model=self.config.openrouter_model,
            embed_model=self.config.openrouter_embed_model,
            vision_model=self._pick_vision_model(available_names),
            available_models=models,
            active_model=None,
        )

    def generate(self, prompt: str, model: str | None = None) -> GenerationResult:
        if not self.config.openrouter_api_key:
            raise ProviderError("OPENROUTER_API_KEY is not set")

        resolved_model = model or self.config.openrouter_model
        payload = {
            "model": resolved_model,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = self.client.post_json("/chat/completions", payload)
        choices = response.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        return GenerationResult(
            text=message.get("content", ""),
            model=resolved_model,
            raw=response,
        )

    def generate_with_images(self, prompt: str, images: list[ImageInput], model: str | None = None) -> GenerationResult:
        if not self.config.openrouter_api_key:
            raise ProviderError("OPENROUTER_API_KEY is not set")
        if not images:
            return self.generate(prompt, model=model)

        resolved_model = model or self.config.openrouter_vision_model or self.config.openrouter_model
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        for image in images:
            encoded = base64.b64encode(image.data).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image.mime_type};base64,{encoded}"},
                }
            )

        payload = {
            "model": resolved_model,
            "messages": [{"role": "user", "content": content}],
        }
        response = self.client.post_json("/chat/completions", payload)
        choices = response.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        return GenerationResult(
            text=message.get("content", ""),
            model=resolved_model,
            raw=response,
        )

    def embed(self, text: str, model: str | None = None) -> EmbeddingResult:
        if not self.config.openrouter_api_key:
            raise ProviderError("OPENROUTER_API_KEY is not set")

        resolved_model = model or self.config.openrouter_embed_model
        payload = {
            "model": resolved_model,
            "input": text,
        }
        response = self.client.post_json("/embeddings", payload)
        data = response.get("data", [])
        vector = data[0].get("embedding", []) if data else []
        return EmbeddingResult(vector=vector, model=resolved_model, raw=response)

    def embed_many(self, texts: list[str], model: str | None = None) -> list[EmbeddingResult]:
        if not self.config.openrouter_api_key:
            raise ProviderError("OPENROUTER_API_KEY is not set")
        if not texts:
            return []

        resolved_model = model or self.config.openrouter_embed_model
        payload = {
            "model": resolved_model,
            "input": texts,
        }
        response = self.client.post_json("/embeddings", payload)
        data = response.get("data", [])
        if all(isinstance(item, dict) and "index" in item for item in data):
            data = sorted(data, key=lambda item: item["index"])
        return [
            EmbeddingResult(vector=item.get("embedding", []), model=resolved_model, raw=response)
            for item in data
            if isinstance(item, dict)
        ]

    def _pick_vision_model(self, available_names: set[str] | None) -> str | None:
        candidates = self._vision_candidates()
        if available_names is None or not available_names:
            return candidates[0] if candidates else None
        for candidate in candidates:
            if candidate in available_names:
                return candidate
        return candidates[0] if candidates else None

    def _vision_candidates(self) -> list[str]:
        candidates = [
            self.config.openrouter_vision_model,
            self.default_vision_model,
            self.fallback_vision_model,
        ]
        return [candidate for index, candidate in enumerate(candidates) if candidate and candidate not in candidates[:index]]
