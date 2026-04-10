from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.providers.base import BaseProvider
from app.retrieval.evidence import EvidenceReport, validate_evidence
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

    results = _run_retrieval(provider, records, embedding_records, normalized_query, embed_model, mode, limit, steps)

    if not results and query_type != "exact":
        rewritten = rewrite_query(normalized_query)
        if rewritten != normalized_query:
            steps.append(AgentStep("rewrite_query", {"from": normalized_query, "to": rewritten}))
            results = _run_retrieval(provider, records, embedding_records, rewritten, embed_model, mode, limit, steps)

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

    results = search_exact(records, query=query, limit=limit)
    steps.append(AgentStep("search_exact", {"result_count": len(results)}))
    return results


def _hybrid_trace_to_dict(trace: HybridSearchTrace) -> dict[str, int]:
    return {
        "exact_count": trace.exact_count,
        "vector_count": trace.vector_count,
        "fused_count": trace.fused_count,
    }
