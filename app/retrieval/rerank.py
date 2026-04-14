from __future__ import annotations

import re

from app.retrieval.exact import SearchResult, normalize_text


def rerank_results(
    query: str,
    results: list[SearchResult],
    entities: list[str] | None = None,
    limit: int | None = None,
) -> list[SearchResult]:
    normalized_query = normalize_text(query)
    query_terms = _query_terms(normalized_query)
    normalized_entities = [normalize_text(entity) for entity in (entities or []) if entity]

    reranked: list[SearchResult] = []
    for result in results:
        text = normalize_text(f"{result.record.text} {' '.join(result.record.section_path)}")
        score = result.score

        if normalized_query and normalized_query in text:
            score += 40

        overlap = sum(1 for term in query_terms if term in text)
        score += overlap * 6

        if query_terms and all(term in text for term in query_terms[: min(len(query_terms), 3)]):
            score += 18

        entity_overlap = sum(1 for entity in normalized_entities if entity in text)
        score += entity_overlap * 14

        if len(normalized_entities) >= 2 and entity_overlap >= 2:
            score += 30

        if result.record.record_type == "paragraph" and len(result.record.text) > 1400:
            score -= 8

        if result.record.record_type == "table":
            score -= 10

        reranked.append(
            SearchResult(
                record=result.record,
                score=score,
                matched_terms=result.matched_terms,
                snippet=result.snippet,
            )
        )

    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked[:limit] if limit is not None else reranked


def _query_terms(normalized_query: str) -> list[str]:
    ignored = {"как", "что", "где", "это", "или", "для", "при", "про", "по", "между", "разница"}
    return [
        term
        for term in re.findall(r"[\w\.\-]+", normalized_query, flags=re.UNICODE)
        if len(term) > 1 and term not in ignored
    ]
