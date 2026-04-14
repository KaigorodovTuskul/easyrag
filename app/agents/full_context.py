from __future__ import annotations

from dataclasses import dataclass

from app.providers.base import BaseProvider
from app.retrieval.exact import normalize_text
from app.retrieval.records import SearchRecord


FULL_CONTEXT_TOKEN_THRESHOLD = 6500
FULL_CONTEXT_RECORD_TYPES = {"paragraph", "table_row", "formula_text"}


@dataclass(slots=True)
class FullContextAnswer:
    answer: str
    prompt: str
    citations: list[dict[str, str]]


def should_use_full_context(
    records: list[SearchRecord],
    requested_by_route: bool = False,
    action: str | None = None,
) -> bool:
    token_estimate = estimate_document_tokens(records)
    if token_estimate > FULL_CONTEXT_TOKEN_THRESHOLD:
        return False
    if requested_by_route:
        return True
    return action in {"compare_entities", "topic_query", "document_summary", "general_qa"}


def estimate_document_tokens(records: list[SearchRecord]) -> int:
    return sum(max(1, len(record.text) // 4) for record in records if record.text.strip())


def build_full_document_context(records: list[SearchRecord], max_records: int = 220) -> tuple[str, list[dict[str, str]]]:
    sections_seen: set[tuple[str, ...]] = set()
    blocks: list[str] = []
    citations: list[dict[str, str]] = []

    for record in records:
        if record.record_type not in FULL_CONTEXT_RECORD_TYPES:
            continue
        if len(citations) >= max_records:
            break

        section = tuple(record.section_path)
        if section and section not in sections_seen:
            sections_seen.add(section)
            blocks.append(f"## {' / '.join(section)}")

        blocks.append(_render_full_context_record(record))
        citations.append(
            {
                "id": record.record_id,
                "type": record.record_type,
                "section": " / ".join(record.section_path),
                "source": record.source_name,
            }
        )

    return "\n\n".join(blocks), citations


def answer_with_full_context(
    provider: BaseProvider,
    chat_model: str,
    records: list[SearchRecord],
    question: str,
    entities: list[str] | None = None,
    action: str | None = None,
    language: str = "ru",
) -> FullContextAnswer:
    context, citations = build_focused_full_context(records, question, entities=entities, action=action)
    answer_language = "Russian" if language == "ru" else "English"
    prompt = "\n\n".join(
        [
            "You are a precise document QA assistant.",
            "Read the provided focused excerpts and full document context together.",
            "Answer only from the document.",
            "If the answer is not present, say so explicitly.",
            "For comparisons, compare all requested entities directly and structure the answer clearly.",
            "Cite evidence with record ids in square brackets, for example [p-500] or [t-22:r-274].",
            "",
            f"Question: {question}",
            "",
            "Document context:",
            context or "No document context available.",
            "",
            f"Answer in {answer_language}.",
        ]
    )
    generated = provider.generate(prompt, model=chat_model)
    return FullContextAnswer(answer=generated.text or "", prompt=prompt, citations=citations)


def build_focused_full_context(
    records: list[SearchRecord],
    question: str,
    entities: list[str] | None = None,
    action: str | None = None,
) -> tuple[str, list[dict[str, str]]]:
    focused_records = _select_focused_records(records, question, entities=entities, action=action)
    focused_context, focused_citations = build_full_document_context(focused_records, max_records=48)
    full_context, full_citations = build_full_document_context(records, max_records=220)

    context = "\n\n".join(
        [
            "### Focused excerpts",
            focused_context or "No focused excerpts found.",
            "### Full document",
            full_context or "No full document context available.",
        ]
    )

    citations = []
    seen_ids: set[str] = set()
    for item in [*focused_citations, *full_citations]:
        record_id = item.get("id")
        if record_id in seen_ids:
            continue
        seen_ids.add(record_id)
        citations.append(item)
    return context, citations


def _render_full_context_record(record: SearchRecord) -> str:
    if record.record_type == "paragraph":
        return f"[{record.record_id}]\n{record.text}"

    if record.record_type == "table_row":
        return "\n".join(
            [
                f"[{record.record_id}] Table row",
                _pretty_table_row(record.text),
            ]
        )

    if record.record_type == "formula_text":
        return f"[{record.record_id}] Formula text\n{record.text}"

    return f"[{record.record_id}] {record.text}"


def _pretty_table_row(text: str) -> str:
    parts = [part.strip() for part in text.split(" | ") if part.strip()]
    if not parts:
        return text
    return "\n".join(f"- {part}" for part in parts)


def _select_focused_records(
    records: list[SearchRecord],
    question: str,
    entities: list[str] | None = None,
    action: str | None = None,
    limit: int = 32,
) -> list[SearchRecord]:
    normalized_question = normalize_text(question)
    question_terms = [term for term in normalized_question.split() if len(term) > 2]
    normalized_entities = [normalize_text(entity) for entity in (entities or []) if entity]

    scored: list[tuple[int, SearchRecord]] = []
    for record in records:
        if record.record_type not in FULL_CONTEXT_RECORD_TYPES:
            continue
        searchable = normalize_text(f"{record.text} {' '.join(record.section_path)}")
        score = 0
        if normalized_question and normalized_question in searchable:
            score += 80
        score += sum(6 for term in question_terms if term in searchable)
        entity_hits = sum(1 for entity in normalized_entities if entity in searchable)
        score += entity_hits * 18
        if len(normalized_entities) >= 2 and entity_hits >= 2:
            score += 24
        if action == "compare_entities" and entity_hits >= 1:
            score += 12
        if record.record_type == "formula_text":
            score += 4
        if score <= 0:
            continue
        scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    seen_ids: set[str] = set()
    selected: list[SearchRecord] = []
    for _, record in scored:
        if record.record_id in seen_ids:
            continue
        seen_ids.add(record.record_id)
        selected.append(record)
        if len(selected) >= limit:
            break

    if selected:
        return selected
    return [record for record in records if record.record_type in FULL_CONTEXT_RECORD_TYPES][:limit]
