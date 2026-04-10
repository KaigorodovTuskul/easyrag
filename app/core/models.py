from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelInfo:
    name: str
    size: int | None = None
    family: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderSelection:
    provider_name: str
    reachable: bool
    reason: str
    chat_model: str
    embed_model: str
    available_models: list[ModelInfo] = field(default_factory=list)
    active_model: str | None = None


@dataclass(slots=True)
class GenerationResult:
    text: str
    model: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EmbeddingResult:
    vector: list[float]
    model: str
    raw: dict[str, Any] = field(default_factory=dict)
