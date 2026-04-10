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
    normalized = re.sub(r"\bnn\b", "n", lowered)
    normalized = re.sub(r"\bn\s+n\b", "n", normalized)
    normalized = re.sub(r"\b\u043d(?=\d)", "n", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _terms(value: str) -> list[str]:
    stopwords = {"как", "что", "где", "это", "или", "для", "при", "про", "по"}
    return [
        term
        for term in re.findall(r"[\w\.\-]+", value, flags=re.UNICODE)
        if len(term) > 1 and term not in stopwords
    ]


def _score(text: str, query: str, query_terms: list[str], record_type: str) -> tuple[float, list[str]]:
    score = 0.0
    matched_terms: list[str] = []

    if query in text:
        score += 100.0 + min(len(query), 100) / 10
        matched_terms.append(query)

    for term in query_terms:
        if term in text:
            score += _term_weight(term)
            matched_terms.append(term)

    if query_terms and all(term in text for term in query_terms):
        score += 25.0

    if record_type == "table_cell" and score > 0:
        score += 18.0
    elif record_type == "table_row" and score > 0:
        score += 12.0
    elif record_type == "table" and score > 0:
        score += 3.0

    return score, list(dict.fromkeys(matched_terms))


def _term_weight(term: str) -> float:
    if re.fullmatch(r"n\d+(?:\.\d+)?", term):
        return 55.0
    if re.fullmatch(r"\d{3,}(?:\.\d+)?", term):
        return 45.0
    if re.search(r"\d", term):
        return 20.0
    return 5.0


def _make_snippet(text: str, normalized_query: str, width: int = 320) -> str:
    normalized = normalize_text(text)
    position = normalized.find(normalized_query)
    if position < 0:
        return text[:width]

    start = max(position - width // 3, 0)
    end = min(start + width, len(text))
    return text[start:end]
