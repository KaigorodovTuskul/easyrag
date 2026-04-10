from __future__ import annotations

import math
import re
from collections import Counter

from app.retrieval.exact import SearchResult, normalize_text
from app.retrieval.records import SearchRecord

K1 = 1.5
B = 0.75

STOPWORDS = {
    "как",
    "что",
    "где",
    "это",
    "или",
    "для",
    "при",
    "про",
    "по",
    "the",
    "and",
    "for",
    "with",
    "what",
    "how",
    "is",
    "are",
}


def search_bm25(records: list[SearchRecord], query: str, limit: int = 20) -> list[SearchResult]:
    query_terms = _tokenize(query)
    if not records or not query_terms:
        return []

    docs = [_document_terms(record) for record in records]
    avg_doc_len = sum(len(doc) for doc in docs) / max(len(docs), 1)
    doc_freq = _document_frequencies(docs)
    query_term_counts = Counter(query_terms)

    results: list[SearchResult] = []
    for record, terms in zip(records, docs):
        if not terms:
            continue

        term_counts = Counter(terms)
        score = 0.0
        matched_terms: list[str] = []

        for term, query_count in query_term_counts.items():
            term_freq = term_counts.get(term, 0)
            if term_freq == 0:
                continue

            matched_terms.append(term)
            idf = math.log(1 + (len(records) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denominator = term_freq + K1 * (1 - B + B * len(terms) / max(avg_doc_len, 1))
            score += idf * (term_freq * (K1 + 1) / denominator) * query_count

        if score <= 0:
            continue

        results.append(
            SearchResult(
                record=record,
                score=score + _type_boost(record.record_type),
                matched_terms=[*matched_terms, "bm25"],
                snippet=record.text[:320],
            )
        )

    results.sort(key=lambda result: result.score, reverse=True)
    return results[:limit]


def _document_terms(record: SearchRecord) -> list[str]:
    return _tokenize(f"{record.text} {' '.join(record.section_path)}")


def _document_frequencies(docs: list[list[str]]) -> Counter[str]:
    frequencies: Counter[str] = Counter()
    for terms in docs:
        frequencies.update(set(terms))
    return frequencies


def _tokenize(value: str) -> list[str]:
    normalized = normalize_text(value)
    return [
        token
        for token in re.findall(r"[\w\.\-]+", normalized, flags=re.UNICODE)
        if len(token) > 1 and token not in STOPWORDS
    ]


def _type_boost(record_type: str) -> float:
    if record_type == "table_cell":
        return 0.4
    if record_type == "table_row":
        return 0.25
    if record_type == "table":
        return 0.1
    return 0.0
