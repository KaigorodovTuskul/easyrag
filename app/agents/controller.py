from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.providers.base import BaseProvider
from app.retrieval.exact import SearchResult, search_exact
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

    results = _run_retrieval(provider, records, embedding_records, query, embed_model, mode, limit, steps)

    if not results and query_type != "exact":
        rewritten = rewrite_query(query)
        if rewritten != query:
            steps.append(AgentStep("rewrite_query", {"from": query, "to": rewritten}))
            results = _run_retrieval(provider, records, embedding_records, rewritten, embed_model, mode, limit, steps)

    steps.append(AgentStep("finalize", {"result_count": len(results)}))
    return AgentRetrievalResult(
        query=query,
        query_type=query_type,
        mode=mode,
        results=results,
        steps=steps,
    )


def classify_query(query: str) -> str:
    normalized = query.strip()
    if re.fullmatch(r"[\w\.\-А-Яа-яЁё]+", normalized) and re.search(r"\d", normalized):
        return "exact"
    if re.search(r"\b(код|счет|норматив|таблиц|строк)", normalized, flags=re.IGNORECASE):
        return "table_lookup"
    if len(normalized.split()) >= 5:
        return "semantic"
    return "exact"


def rewrite_query(query: str) -> str:
    rewritten = re.sub(r"\bN\s+N\b", "N", query, flags=re.IGNORECASE)
    rewritten = re.sub(r"\s+", " ", rewritten).strip()
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

    results = search_exact(records, query=query, limit=limit)
    steps.append(AgentStep("search_exact", {"result_count": len(results)}))
    return results


def _hybrid_trace_to_dict(trace: HybridSearchTrace) -> dict[str, int]:
    return {
        "exact_count": trace.exact_count,
        "vector_count": trace.vector_count,
        "fused_count": trace.fused_count,
    }
