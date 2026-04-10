from __future__ import annotations

import math
from dataclasses import dataclass

from app.retrieval.exact import SearchResult
from app.retrieval.records import SearchRecord


@dataclass(slots=True)
class EmbeddingRecord:
    record: SearchRecord
    vector: list[float]
    model: str

    def to_dict(self) -> dict:
        return {
            "record": self.record.to_dict(),
            "vector": self.vector,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "EmbeddingRecord":
        return cls(
            record=SearchRecord.from_dict(payload["record"]),
            vector=[float(value) for value in payload["vector"]],
            model=payload["model"],
        )


def search_vector(
    embedding_records: list[EmbeddingRecord],
    query_vector: list[float],
    limit: int = 10,
) -> list[SearchResult]:
    if not query_vector:
        return []

    results: list[SearchResult] = []
    for item in embedding_records:
        score = cosine_similarity(query_vector, item.vector)
        if score <= 0:
            continue
        results.append(
            SearchResult(
                record=item.record,
                score=score,
                matched_terms=["vector"],
                snippet=item.record.text[:320],
            )
        )

    results.sort(key=lambda result: result.score, reverse=True)
    return results[:limit]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
