from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.exact import SearchResult, search_exact
from app.retrieval.records import SearchRecord
from app.retrieval.vector import EmbeddingRecord, search_vector


@dataclass(slots=True)
class HybridSearchTrace:
    exact_count: int
    vector_count: int
    fused_count: int


def search_hybrid(
    records: list[SearchRecord],
    embedding_records: list[EmbeddingRecord],
    query: str,
    query_vector: list[float] | None = None,
    limit: int = 20,
) -> tuple[list[SearchResult], HybridSearchTrace]:
    exact_results = search_exact(records, query=query, limit=limit)
    vector_results = search_vector(embedding_records, query_vector or [], limit=limit)
    fused = _fuse_results(exact_results, vector_results)

    return fused[:limit], HybridSearchTrace(
        exact_count=len(exact_results),
        vector_count=len(vector_results),
        fused_count=len(fused),
    )


def _fuse_results(exact_results: list[SearchResult], vector_results: list[SearchResult]) -> list[SearchResult]:
    by_id: dict[str, SearchResult] = {}

    for rank, result in enumerate(exact_results, start=1):
        normalized_score = 1 / (rank + 5)
        by_id[result.record.record_id] = SearchResult(
            record=result.record,
            score=result.score + normalized_score * 1000,
            matched_terms=result.matched_terms,
            snippet=result.snippet,
        )

    for rank, result in enumerate(vector_results, start=1):
        normalized_score = 1 / (rank + 20)
        existing = by_id.get(result.record.record_id)
        if existing is None:
            by_id[result.record.record_id] = SearchResult(
                record=result.record,
                score=normalized_score * 100,
                matched_terms=result.matched_terms,
                snippet=result.snippet,
            )
            continue

        existing.score += normalized_score * 100
        existing.matched_terms = list(dict.fromkeys([*existing.matched_terms, *result.matched_terms]))

    results = list(by_id.values())
    results.sort(key=lambda result: result.score, reverse=True)
    return results
