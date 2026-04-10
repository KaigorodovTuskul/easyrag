from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.providers.base import BaseProvider
from app.retrieval.evidence import EvidenceReport, validate_evidence
from app.retrieval.exact import SearchResult
from app.retrieval.hybrid import HybridSearchTrace, search_hybrid
from app.retrieval.records import SearchRecord
from app.retrieval.vector import EmbeddingRecord


@dataclass(slots=True)
class AgentStep:
    name: str
    details: dict[str, str | int | float | bool | None]


@dataclass(slots=True)
class AgentRetrievalResult:
    query: str
    query_type: str
    mode: str
    results: list[SearchResult]
    evidence: EvidenceReport
    steps: list[AgentStep] = field(default_factory=list)


def run_agent_retrieval(
    provider: BaseProvider,
    records: list[SearchRecord],
    embedding_records: list[EmbeddingRecord],
    query: str,
    embed_model: str,
    limit: int = 30,
) -> AgentRetrievalResult:
    steps: list[AgentStep] = []
    query_type = classify_query(query)
    mode = _select_mode(query_type, embedding_records)
    steps.append(AgentStep("classify_query", {"query_type": query_type, "selected_mode": mode}))

    normalized_query = normalize_query(query)
    if normalized_query != query:
        steps.append(AgentStep("normalize_query", {"from": query, "to": normalized_query}))

    candidate_records, candidate_embeddings = _filter_candidates(records, embedding_records, normalized_query, query_type, steps)
    results = _run_retrieval(provider, candidate_records, candidate_embeddings, normalized_query, embed_model, mode, limit, steps)
    results = _expand_table_context(records, results)
    results = _expand_paragraph_context(records, results, normalized_query)

    if not results and query_type != "exact":
        rewritten = rewrite_query(normalized_query)
        if rewritten != normalized_query:
            steps.append(AgentStep("rewrite_query", {"from": normalized_query, "to": rewritten}))
            candidate_records, candidate_embeddings = _filter_candidates(records, embedding_records, rewritten, query_type, steps)
            results = _run_retrieval(provider, candidate_records, candidate_embeddings, rewritten, embed_model, mode, limit, steps)
            results = _expand_table_context(records, results)
            results = _expand_paragraph_context(records, results, rewritten)

    evidence = validate_evidence(results)
    steps.append(
        AgentStep(
            "validate_evidence",
            {
                "ok": evidence.ok,
                "confidence": evidence.confidence,
                "reason": evidence.reason,
                "top_score": evidence.top_score,
            },
        )
    )
    steps.append(AgentStep("finalize", {"result_count": len(results)}))
    return AgentRetrievalResult(
        query=query,
        query_type=query_type,
        mode=mode,
        results=results,
        evidence=evidence,
        steps=steps,
    )


def classify_query(query: str) -> str:
    normalized = query.strip()
    lower = normalized.lower().replace("\u0451", "\u0435")

    if re.fullmatch(r"[\w\.\-]+", normalized, flags=re.UNICODE) and re.search(r"\d", normalized):
        return "exact"

    table_keywords = [
        "\u043a\u043e\u0434",
        "\u0441\u0447\u0435\u0442",
        "\u043d\u043e\u0440\u043c\u0430\u0442\u0438\u0432",
        "\u0442\u0430\u0431\u043b\u0438\u0446",
        "\u0441\u0442\u0440\u043e\u043a",
    ]
    if any(keyword in lower for keyword in table_keywords):
        return "table_lookup"

    if len(normalized.split()) >= 5:
        return "semantic"

    return "exact"


def normalize_query(query: str) -> str:
    normalized = re.sub(r"\bN\s+N\b", "N", query, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def rewrite_query(query: str) -> str:
    rewritten = normalize_query(query)
    rewritten = re.sub(r"[?!.]+$", "", rewritten).strip()
    return rewritten


def _select_mode(query_type: str, embedding_records: list[EmbeddingRecord]) -> str:
    if query_type in {"exact", "table_lookup"}:
        return "exact"
    if embedding_records:
        return "hybrid"
    return "exact"


def _run_retrieval(
    provider: BaseProvider,
    records: list[SearchRecord],
    embedding_records: list[EmbeddingRecord],
    query: str,
    embed_model: str,
    mode: str,
    limit: int,
    steps: list[AgentStep],
) -> list[SearchResult]:
    if mode == "hybrid":
        query_vector = []
        try:
            query_vector = provider.embed(query, model=embed_model).vector
            steps.append(AgentStep("embed_query", {"ok": True, "dimension": len(query_vector)}))
        except Exception as exc:
            steps.append(AgentStep("embed_query", {"ok": False, "error": str(exc)}))

        results, trace = search_hybrid(records, embedding_records, query=query, query_vector=query_vector, limit=limit)
        steps.append(AgentStep("search_hybrid", _hybrid_trace_to_dict(trace)))
        return results

    results, trace = search_hybrid(records, embedding_records, query=query, query_vector=[], limit=limit)
    steps.append(AgentStep("search_lexical", _hybrid_trace_to_dict(trace)))
    return results


def _hybrid_trace_to_dict(trace: HybridSearchTrace) -> dict[str, int]:
    return {
        "exact_count": trace.exact_count,
        "bm25_count": trace.bm25_count,
        "vector_count": trace.vector_count,
        "fused_count": trace.fused_count,
    }


def _filter_candidates(
    records: list[SearchRecord],
    embedding_records: list[EmbeddingRecord],
    query: str,
    query_type: str,
    steps: list[AgentStep],
) -> tuple[list[SearchRecord], list[EmbeddingRecord]]:
    target = _extract_target(query)
    table_intent = _has_table_intent(query)

    if not target and not table_intent:
        steps.append(AgentStep("filter_candidates", {"applied": False, "reason": "no_filter_target"}))
        return records, embedding_records

    candidates = records
    filters: list[str] = []

    if target:
        target_candidates = _filter_by_target(records, target)
        if target_candidates:
            candidates = target_candidates
            filters.append(f"target:{target}")

    if table_intent:
        table_candidates = [record for record in candidates if record.record_type in {"table", "table_row", "table_cell"}]
        if table_candidates:
            candidates = table_candidates
            filters.append("record_type:table")

    if not filters:
        steps.append(AgentStep("filter_candidates", {"applied": False, "reason": "no_matches"}))
        return records, embedding_records

    candidate_ids = {record.record_id for record in candidates}
    filtered_embeddings = [item for item in embedding_records if item.record.record_id in candidate_ids]
    steps.append(
        AgentStep(
            "filter_candidates",
            {
                "applied": True,
                "filters": ", ".join(filters),
                "records_before": len(records),
                "records_after": len(candidates),
                "embeddings_after": len(filtered_embeddings),
            },
        )
    )
    return candidates, filtered_embeddings


def _extract_target(query: str) -> str | None:
    norm_match = re.search(r"\b[НN]\s*\d+(?:\.\d+)?\b", query, flags=re.IGNORECASE)
    if norm_match:
        return norm_match.group(0).replace(" ", "")

    code_match = re.search(r"\b\d{3,}(?:\.\d+)?\b", query)
    if code_match:
        return code_match.group(0)

    return None


def _filter_by_target(records: list[SearchRecord], target: str) -> list[SearchRecord]:
    normalized_target = _normalize_target(target)
    return [
        record
        for record in records
        if normalized_target in _normalize_target(f"{record.text} {' '.join(record.section_path)}")
    ]


def _normalize_target(value: str) -> str:
    normalized = value.lower().replace("\u0451", "\u0435")
    normalized = re.sub(r"\bн(?=\d)", "n", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _has_table_intent(query: str) -> bool:
    normalized = query.lower().replace("\u0451", "\u0435")
    return any(term in normalized for term in ["код", "счет", "таблиц", "строк", "столб", "table", "code", "account", "row", "column"])


def _expand_table_context(records: list[SearchRecord], results: list[SearchResult]) -> list[SearchResult]:
    by_id = {record.record_id: record for record in records}
    expanded: list[SearchResult] = []
    seen: set[str] = set()

    for result in results:
        if result.record.record_id not in seen:
            expanded.append(result)
            seen.add(result.record.record_id)

        if result.record.record_type != "table_cell":
            continue

        table_id = result.record.metadata.get("table_id")
        row_index = result.record.metadata.get("row_index")
        row_id = f"{table_id}:r-{row_index}"
        row_record = by_id.get(row_id)
        if row_record is None or row_record.record_id in seen:
            continue

        expanded.append(
            SearchResult(
                record=row_record,
                score=max(result.score - 0.1, 0),
                matched_terms=[*result.matched_terms, "row_context"],
                snippet=row_record.text[:320],
            )
        )
        seen.add(row_record.record_id)

    return expanded


def _expand_paragraph_context(records: list[SearchRecord], results: list[SearchResult], query: str) -> list[SearchResult]:
    if not _needs_paragraph_context(query):
        return results

    by_id = {record.record_id: record for record in records}
    expanded: list[SearchResult] = []
    seen: set[str] = set()

    for result in results:
        if result.record.record_id not in seen:
            expanded.append(result)
            seen.add(result.record.record_id)

        if result.record.record_type != "paragraph" or not result.record.record_id.startswith("p-"):
            continue

        try:
            paragraph_number = int(result.record.record_id.split("-", 1)[1])
        except ValueError:
            continue

        for offset in range(1, 4):
            neighbor = by_id.get(f"p-{paragraph_number + offset}")
            if neighbor is None or neighbor.record_id in seen:
                continue
            expanded.append(
                SearchResult(
                    record=neighbor,
                    score=max(result.score - offset * 0.1, 0),
                    matched_terms=[*result.matched_terms, "paragraph_context"],
                    snippet=neighbor.text[:320],
                )
            )
            seen.add(neighbor.record_id)

    return expanded


def _needs_paragraph_context(query: str) -> bool:
    normalized = query.lower().replace("\u0451", "\u0435")
    return any(term in normalized for term in ["рассчитывается", "расчет", "формуле", "норматив"])
