from __future__ import annotations

from app.core.config import AppConfig
from app.core.models import ProviderSelection
from app.providers.base import BaseProvider, ProviderError
from app.providers.ollama import OllamaProvider
from app.providers.openrouter import OpenRouterProvider


_CACHED_SELECTION: tuple[BaseProvider, ProviderSelection] | None = None


class ProviderRouter:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.ollama = OllamaProvider(config)
        self.openrouter = OpenRouterProvider(config)

    def resolve(self) -> tuple[BaseProvider, ProviderSelection]:
        global _CACHED_SELECTION
        if _CACHED_SELECTION is not None:
            return _CACHED_SELECTION

        try:
            selection = self.ollama.resolve_selection()
            _CACHED_SELECTION = (self.ollama, selection)
            return _CACHED_SELECTION
        except ProviderError as ollama_error:
            fallback = self.openrouter.resolve_selection()
            fallback.reason = f"Ollama unavailable: {ollama_error}. {fallback.reason}"
            _CACHED_SELECTION = (self.openrouter, fallback)
            return _CACHED_SELECTION
