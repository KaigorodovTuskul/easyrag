from __future__ import annotations

import re
from dataclasses import dataclass

from app.retrieval.records import SearchRecord


@dataclass(slots=True)
class SearchResult:
    record: SearchRecord
    score: float
    matched_terms: list[str]
    snippet: str


def search_exact(records: list[SearchRecord], query: str, limit: int = 10) -> list[SearchResult]:
    normalized_query = normalize_text(query)
    if not normalized_query:
        return []

    query_terms = _terms(normalized_query)
    results: list[SearchResult] = []

    for record in records:
        normalized_text = normalize_text(record.text)
        section_text = normalize_text(" ".join(record.section_path))
        searchable_text = f"{normalized_text} {section_text}".strip()
        score, matched_terms = _score(searchable_text, normalized_query, query_terms, record.record_type)

        if score <= 0:
            continue

        results.append(
            SearchResult(
                record=record,
                score=score,
                matched_terms=matched_terms,
                snippet=_make_snippet(record.text, normalized_query),
            )
        )

    results.sort(key=lambda item: item.score, reverse=True)
    return results[:limit]


def normalize_text(value: str) -> str:
    lowered = value.lower().replace("\u0451", "\u0435")
    return re.sub(r"\s+", " ", lowered).strip()


def _terms(value: str) -> list[str]:
    return [term for term in re.findall(r"[\w\.\-]+", value, flags=re.UNICODE) if len(term) > 1]


def _score(text: str, query: str, query_terms: list[str], record_type: str) -> tuple[float, list[str]]:
    score = 0.0
    matched_terms: list[str] = []

    if query in text:
        score += 100.0 + min(len(query), 100) / 10
        matched_terms.append(query)

    for term in query_terms:
        if term in text:
            score += 5.0
            matched_terms.append(term)

    if query_terms and all(term in text for term in query_terms):
        score += 25.0

    if record_type == "table_row" and score > 0:
        score += 10.0
    elif record_type == "table" and score > 0:
        score += 3.0

    return score, list(dict.fromkeys(matched_terms))


def _make_snippet(text: str, normalized_query: str, width: int = 320) -> str:
    normalized = normalize_text(text)
    position = normalized.find(normalized_query)
    if position < 0:
        return text[:width]

    start = max(position - width // 3, 0)
    end = min(start + width, len(text))
    return text[start:end]
