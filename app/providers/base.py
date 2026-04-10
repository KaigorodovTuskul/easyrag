from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.models import EmbeddingResult, GenerationResult, ModelInfo, ProviderSelection


class ProviderError(RuntimeError):
    """Raised when a provider cannot fulfill the request."""


class BaseProvider(ABC):
    name: str

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        raise NotImplementedError

    @abstractmethod
    def get_active_model(self) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def resolve_selection(self) -> ProviderSelection:
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str, model: str | None = None) -> GenerationResult:
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: str, model: str | None = None) -> EmbeddingResult:
        raise NotImplementedError

    def embed_many(self, texts: list[str], model: str | None = None) -> list[EmbeddingResult]:
        return [self.embed(text, model=model) for text in texts]
