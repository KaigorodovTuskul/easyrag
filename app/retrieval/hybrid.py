from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.bm25 import search_bm25
from app.retrieval.exact import SearchResult, search_exact
from app.retrieval.records import SearchRecord
from app.retrieval.vector import EmbeddingRecord, search_vector


@dataclass(slots=True)
class HybridSearchTrace:
    exact_count: int
    bm25_count: int
    vector_count: int
    fused_count: int
    exact_contribution: int
    bm25_contribution: int
    vector_contribution: int
    top_sources: str


def search_hybrid(
    records: list[SearchRecord],
    embedding_records: list[EmbeddingRecord],
    query: str,
    query_vector: list[float] | None = None,
    limit: int = 20,
) -> tuple[list[SearchResult], HybridSearchTrace]:
    exact_results = search_exact(records, query=query, limit=limit)
    bm25_results = search_bm25(records, query=query, limit=limit)
    vector_results = search_vector(embedding_records, query_vector or [], limit=limit)
    fused = _fuse_results(exact_results, bm25_results, vector_results)
    exact_ids = {result.record.record_id for result in exact_results}
    bm25_ids = {result.record.record_id for result in bm25_results}
    vector_ids = {result.record.record_id for result in vector_results}
    fused_ids = {result.record.record_id for result in fused}

    return fused[:limit], HybridSearchTrace(
        exact_count=len(exact_results),
        bm25_count=len(bm25_results),
        vector_count=len(vector_results),
        fused_count=len(fused),
        exact_contribution=len(fused_ids & exact_ids),
        bm25_contribution=len(fused_ids & bm25_ids),
        vector_contribution=len(fused_ids & vector_ids),
        top_sources=_top_sources_summary(fused[:limit]),
    )


def _fuse_results(
    exact_results: list[SearchResult],
    bm25_results: list[SearchResult],
    vector_results: list[SearchResult],
) -> list[SearchResult]:
    by_id: dict[str, SearchResult] = {}

    for rank, result in enumerate(exact_results, start=1):
        normalized_score = 1 / (rank + 5)
        type_boost = _type_boost(result.record.record_type)
        by_id[result.record.record_id] = SearchResult(
            record=result.record,
            score=result.score + normalized_score * 1000 + type_boost,
            matched_terms=result.matched_terms,
            snippet=result.snippet,
        )

    for rank, result in enumerate(bm25_results, start=1):
        normalized_score = 1 / (rank + 10)
        existing = by_id.get(result.record.record_id)
        if existing is None:
            by_id[result.record.record_id] = SearchResult(
                record=result.record,
                score=normalized_score * 350 + result.score * 20 + _type_boost(result.record.record_type),
                matched_terms=result.matched_terms,
                snippet=result.snippet,
            )
            continue

        existing.score += normalized_score * 350 + result.score * 20
        existing.matched_terms = list(dict.fromkeys([*existing.matched_terms, *result.matched_terms]))

    for rank, result in enumerate(vector_results, start=1):
        normalized_score = 1 / (rank + 20)
        existing = by_id.get(result.record.record_id)
        if existing is None:
            by_id[result.record.record_id] = SearchResult(
                record=result.record,
                score=normalized_score * 100 + _type_boost(result.record.record_type),
                matched_terms=result.matched_terms,
                snippet=result.snippet,
            )
            continue

        existing.score += normalized_score * 100
        existing.matched_terms = list(dict.fromkeys([*existing.matched_terms, *result.matched_terms]))

    results = list(by_id.values())
    results.sort(key=lambda result: result.score, reverse=True)
    return results


def _type_boost(record_type: str) -> float:
    if record_type == "table_cell":
        return 30.0
    if record_type == "table_row":
        return 20.0
    if record_type == "table":
        return 5.0
    return 0.0


def _top_sources_summary(results: list[SearchResult], limit: int = 5) -> str:
    source_names: list[str] = []
    for result in results:
        name = result.record.source_name.strip()
        if not name or name in source_names:
            continue
        source_names.append(name)
        if len(source_names) >= limit:
            break
    return ", ".join(source_names)
