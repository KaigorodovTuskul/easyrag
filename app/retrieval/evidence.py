from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.exact import SearchResult


@dataclass(slots=True)
class EvidenceReport:
    ok: bool
    confidence: str
    reason: str
    top_score: float
    result_count: int


def validate_evidence(results: list[SearchResult]) -> EvidenceReport:
    if not results:
        return EvidenceReport(False, "none", "no_context", 0.0, 0)

    top = results[0]
    if top.record.record_type in {"table_cell", "table_row"} and top.score >= 120:
        return EvidenceReport(True, "high", "strong_table_exact_match", top.score, len(results))

    if top.record.record_type in {"table_cell", "table_row"} and top.score >= 20:
        return EvidenceReport(True, "medium", "table_keyword_match", top.score, len(results))

    if top.score >= 120:
        return EvidenceReport(True, "high", "strong_exact_match", top.score, len(results))

    if top.score >= 40:
        return EvidenceReport(True, "medium", "partial_keyword_match", top.score, len(results))

    return EvidenceReport(False, "low", "weak_evidence", top.score, len(results))
