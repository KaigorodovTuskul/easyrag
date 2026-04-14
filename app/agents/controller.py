from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.agents.query_understanding import classify_intent, extract_query_entity
from app.providers.base import BaseProvider
from app.retrieval.evidence import EvidenceReport, validate_evidence
from app.retrieval.exact import SearchResult, normalize_text
from app.retrieval.hybrid import HybridSearchTrace, search_hybrid
from app.retrieval.records import SearchRecord
from app.retrieval.rerank import rerank_results
from app.retrieval.vector import EmbeddingRecord


@dataclass(slots=True)
class AgentStep:
    name: str
    details: dict[str, str | int | float | bool | None]


@dataclass(slots=True)
class AgentRetrievalResult:
    query: str
    query_type: str
    entity: str | None
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
    entity_names: list[str] | None = None,
    limit: int = 30,
) -> AgentRetrievalResult:
    steps: list[AgentStep] = []
    query_type = classify_intent(query)
    entity = extract_query_entity(query, entity_names or [])
    mode = _select_mode(query_type, embedding_records)
    steps.append(AgentStep("classify_query", {"query_type": query_type, "entity": entity, "selected_mode": mode}))

    normalized_query = normalize_query(query)
    if normalized_query != query:
        steps.append(AgentStep("normalize_query", {"from": query, "to": normalized_query}))

    candidate_records, candidate_embeddings = _filter_candidates(
        records,
        embedding_records,
        normalized_query,
        query_type,
        entity,
        steps,
    )

    if entity and query_type in {"definition", "composition", "formula", "norm"}:
        search_records = records if query_type in {"formula", "norm"} else candidate_records
        results = _search_entity_records(search_records, entity, query_type, limit)
        steps.append(AgentStep("search_entity_records", {"entity": entity, "result_count": len(results)}))
    else:
        results = _run_retrieval(provider, candidate_records, candidate_embeddings, normalized_query, embed_model, mode, limit, steps)

    results = _expand_table_context(records, results)
    results = _expand_paragraph_context(records, results, normalized_query)
    rerank_entities = entities_for_rerank(query, entity)
    results = rerank_results(normalized_query, results, rerank_entities, limit=limit)
    steps.append(AgentStep("rerank_results", {"entity_count": len(rerank_entities), "result_count": len(results)}))

    if not results and query_type not in {"exact", "definition", "composition", "formula", "norm"}:
        rewritten = rewrite_query(normalized_query)
        if rewritten != normalized_query:
            steps.append(AgentStep("rewrite_query", {"from": normalized_query, "to": rewritten}))
            candidate_records, candidate_embeddings = _filter_candidates(
                records,
                embedding_records,
                rewritten,
                query_type,
                entity,
                steps,
            )
            results = _run_retrieval(provider, candidate_records, candidate_embeddings, rewritten, embed_model, mode, limit, steps)
            results = _expand_table_context(records, results)
            results = _expand_paragraph_context(records, results, rewritten)
            rerank_entities = entities_for_rerank(query, entity)
            results = rerank_results(rewritten, results, rerank_entities, limit=limit)
            steps.append(AgentStep("rerank_results", {"entity_count": len(rerank_entities), "result_count": len(results), "rewritten": True}))

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
        entity=entity,
        mode=mode,
        results=results,
        evidence=evidence,
        steps=steps,
    )


def entities_for_rerank(query: str, entity: str | None) -> list[str]:
    found: list[str] = []
    if entity:
        found.append(entity)
    for match in re.finditer(r"\b[РќN]\s*\d+(?:\.\d+)?\b", query, flags=re.IGNORECASE):
        value = match.group(0).replace(" ", "").upper()
        if value not in found:
            found.append(value)
    for match in re.finditer(r"\b\d{3,}(?:\.\d+)?\b", query):
        value = match.group(0)
        if value not in found:
            found.append(value)
    return found


def normalize_query(query: str) -> str:
    normalized = re.sub(r"\bN\s+N\b", "N", query, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def rewrite_query(query: str) -> str:
    rewritten = normalize_query(query)
    rewritten = re.sub(r"[?!.]+$", "", rewritten).strip()
    return rewritten


def _select_mode(query_type: str, embedding_records: list[EmbeddingRecord]) -> str:
    if query_type in {"exact", "table_lookup", "code_lookup", "definition", "composition", "formula", "norm"}:
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


def _hybrid_trace_to_dict(trace: HybridSearchTrace) -> dict[str, int | float | str]:
    return {
        "exact_count": trace.exact_count,
        "bm25_count": trace.bm25_count,
        "vector_count": trace.vector_count,
        "fused_count": trace.fused_count,
        "exact_contribution": trace.exact_contribution,
        "bm25_contribution": trace.bm25_contribution,
        "vector_contribution": trace.vector_contribution,
        "top_sources": trace.top_sources,
    }


def _filter_candidates(
    records: list[SearchRecord],
    embedding_records: list[EmbeddingRecord],
    query: str,
    query_type: str,
    entity: str | None,
    steps: list[AgentStep],
) -> tuple[list[SearchRecord], list[EmbeddingRecord]]:
    target = _extract_target(query)
    table_intent = _has_table_intent(query)
    entity_intent = entity is not None and query_type in {"definition", "composition", "formula", "norm"}

    if not target and not table_intent and not entity_intent:
        steps.append(AgentStep("filter_candidates", {"applied": False, "reason": "no_filter_target"}))
        return records, embedding_records

    candidates = records
    filters: list[str] = []

    if target:
        target_candidates = _filter_by_target(records, target)
        if target_candidates:
            candidates = target_candidates
            filters.append(f"target:{target}")

    if entity_intent and entity:
        entity_candidates = _filter_by_entity(candidates, entity)
        if entity_candidates:
            candidates = entity_candidates
            filters.append(f"entity:{entity}")

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

    named_norm = _extract_named_norm_target(query)
    if named_norm:
        return named_norm

    code_match = re.search(r"\b\d{3,}(?:\.\d+)?\b", query)
    if code_match:
        return code_match.group(0)

    return None


def _filter_by_target(records: list[SearchRecord], target: str) -> list[SearchRecord]:
    normalized_target = _normalize_target(target)
    return [
        record
        for record in records
        if _target_matches(_normalize_target(f"{record.text} {' '.join(record.section_path)}"), normalized_target)
    ]


def _filter_by_entity(records: list[SearchRecord], entity: str) -> list[SearchRecord]:
    return [
        record
        for record in records
        if _entity_matches_text(f"{record.text} {' '.join(record.section_path)}", entity)
    ]


def _normalize_target(value: str) -> str:
    normalized = value.lower().replace("\u0451", "\u0435")
    normalized = re.sub(r"\bн(?=\d)", "n", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _target_matches(text: str, target: str) -> bool:
    if re.fullmatch(r"n\d+(?:\.\d+)?", target):
        return re.search(rf"(?<![a-z0-9]){re.escape(target)}(?![\d\.])", text) is not None
    return target in text


def _extract_named_norm_target(query: str) -> str | None:
    normalized = query.lower().replace("\u0451", "\u0435")
    if "краткосрочн" in normalized and "ликвидност" in normalized:
        return "Н3"
    if "текущ" in normalized and "ликвидност" in normalized:
        return "Н3"
    if "мгновенн" in normalized and "ликвидност" in normalized:
        return "Н2"
    return None


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
    return any(term in normalized for term in ["рассчитывается", "расчет", "формуле", "норматив", "состав", "входит", "показатель", "что такое"])


def _search_entity_records(records: list[SearchRecord], entity: str, query_type: str, limit: int) -> list[SearchResult]:
    normalized_entity = _normalize_target(entity)
    results: list[SearchResult] = []

    for record in records:
        if not _entity_matches_text(f"{record.text} {' '.join(record.section_path)}", entity):
            continue
        haystack = _normalize_target(f"{record.text} {' '.join(record.section_path)}")

        score = 40.0
        matched_terms = [entity, "entity_lookup"]
        text = record.text.lower().replace("\u0451", "\u0435")

        if record.record_type == "paragraph":
            score += 10.0
        elif query_type in {"formula", "norm"} and record.record_type in {"table", "table_row", "table_cell"}:
            score -= 150.0

        if re.search(rf"^\s*{re.escape(entity.lower())}\s*-", text, flags=re.MULTILINE):
            score += 80.0
            matched_terms.append("definition")

        if f"показатель {normalized_entity}" in haystack and "рассчитывается как" in text:
            score += 120.0
            matched_terms.append("calculation")
        elif "рассчитывается как" in text:
            score += 60.0
            matched_terms.append("calculation")

        if query_type in {"formula", "norm"} and "рассчитывается по формуле" in text:
            score += 160.0
            matched_terms.append("formula_anchor")
        elif query_type in {"formula", "norm"} and "рассчитывается" in text:
            score += 90.0
            matched_terms.append("calculation")

        if query_type in {"formula", "norm"} and "норматив" in text:
            score += 20.0
            matched_terms.append("normative_context")
        if query_type in {"formula", "norm"} and record.record_type == "paragraph" and re.search(rf"норматив\s+{re.escape(entity.lower())}", text):
            score += 120.0
            matched_terms.append("norm_anchor")

        if query_type == "composition" and any(term in text for term in ["включаются", "сумма", "вычитаются", "уменьшенная на", "рассчитывается как"]):
            score += 50.0
            matched_terms.append("composition")

        if query_type == "formula" and "формул" in text:
            score += 40.0
            matched_terms.append("formula")

        results.append(
            SearchResult(
                record=record,
                score=score,
                matched_terms=list(dict.fromkeys(matched_terms)),
                snippet=record.text[:320],
            )
        )

    results.sort(key=lambda item: item.score, reverse=True)
    return results[:limit]


def _entity_matches_text(text: str, entity: str) -> bool:
    normalized_entity = normalize_text(entity).replace(" ", "")
    normalized_text = normalize_text(text)
    if re.fullmatch(r"n\d+(?:\.\d+)?", normalized_entity):
        return re.search(rf"(?<![a-z0-9]){re.escape(normalized_entity)}(?![\d\.])", normalized_text) is not None
    return re.search(rf"(?<![\w]){re.escape(normalized_entity)}(?![\w])", normalized_text, flags=re.UNICODE) is not None
