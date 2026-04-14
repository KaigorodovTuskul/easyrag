from __future__ import annotations

from dataclasses import dataclass
import re

from app.retrieval.evidence import EvidenceReport
from app.retrieval.exact import SearchResult


@dataclass(slots=True)
class AnswerContext:
    prompt: str
    citations: list[dict[str, str | float]]


def build_answer_context(
    question: str,
    results: list[SearchResult],
    max_results: int = 6,
    evidence: EvidenceReport | None = None,
    language: str = "ru",
) -> AnswerContext:
    selected = results[:max_results]
    context_blocks: list[str] = []
    citations: list[dict[str, str | float]] = []

    for index, result in enumerate(selected, start=1):
        record = result.record
        section = " / ".join(record.section_path) if record.section_path else "Root"
        context_blocks.append(
            "\n".join(
                [
                    f"[{index}] source={record.source_name}",
                    f"type={record.record_type}",
                    f"id={record.record_id}",
                    f"section={section}",
                    f"text={record.text}",
                ]
            )
        )
        citations.append(
            {
                "ref": f"[{index}]",
                "source": record.source_name,
                "id": record.record_id,
                "type": record.record_type,
                "section": section,
                "score": round(result.score, 2),
            }
        )

    answer_mode = _select_answer_mode(results, evidence)
    answer_language = "Russian" if language == "ru" else "English"
    compare_note = _compare_instruction(question, language)
    prompt = "\n\n".join(
        [
            "You are a precise RAG answerer for corporate documents.",
            "Answer only using the provided context.",
            "If the context is insufficient, say that the answer was not found in the indexed documents.",
            "Keep exact codes, account numbers, regulatory references, and table values unchanged.",
            "Cite supporting context with bracket references like [1], [2].",
            f"Answer mode: {answer_mode}.",
            f"Evidence: {evidence.reason if evidence else 'not_checked'}.",
            compare_note,
            "",
            f"Question: {question}",
            "",
            "Context:",
            "\n\n".join(context_blocks) if context_blocks else "No context found.",
            "",
            f"Answer in {answer_language}.",
        ]
    )

    return AnswerContext(prompt=prompt, citations=citations)


def _select_answer_mode(results: list[SearchResult], evidence: EvidenceReport | None) -> str:
    if evidence and not evidence.ok:
        return "refuse_if_not_found"
    if results and results[0].record.record_type in {"table_cell", "table_row"}:
        return "extractive_table_answer"
    return "grounded_summary"


def _compare_instruction(question: str, language: str) -> str:
    normalized = question.lower().replace("ё", "е")
    explicit_entities = re.findall(r"\b[нn]\s*\d+(?:\.\d+)?\b|\b\d{3,}(?:\.\d+)?\b", normalized)
    asks_compare = len(explicit_entities) >= 2 and any(
        marker in normalized
        for marker in ["разница", "отлич", "сравни", "сравнение", "difference", "compare", "comparison"]
    )
    if not asks_compare:
        return "Comparison mode: off."
    if language == "en":
        return (
            "Comparison mode: on. Compare all requested entities explicitly. "
            "Do not answer about only the first one. Structure the answer as: entity A, entity B, key differences."
        )
    return (
        "Comparison mode: on. Сравни все запрошенные сущности явно. "
        "Не отвечай только про первую. Структурируй ответ так: первая сущность, вторая сущность, ключевые различия."
    )
