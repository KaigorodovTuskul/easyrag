from __future__ import annotations

from app.core.config import AppConfig
from app.core.models import ProviderSelection
from app.providers.base import BaseProvider, ProviderError
from app.providers.ollama import OllamaProvider
from app.providers.openrouter import OpenRouterProvider


class ProviderRouter:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.ollama = OllamaProvider(config)
        self.openrouter = OpenRouterProvider(config)

    def resolve(self) -> tuple[BaseProvider, ProviderSelection]:
        try:
            selection = self.ollama.resolve_selection()
            return self.ollama, selection
        except ProviderError as ollama_error:
            fallback = self.openrouter.resolve_selection()
            fallback.reason = f"Ollama unavailable: {ollama_error}. {fallback.reason}"
            return self.openrouter, fallback
