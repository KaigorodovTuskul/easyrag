from __future__ import annotations

import re
from dataclasses import dataclass

from app.agents.query_understanding import QueryUnderstanding
from app.retrieval.exact import SearchResult, normalize_text
from app.retrieval.records import SearchRecord


@dataclass(slots=True)
class TermLookupAnswer:
    entity: str
    intent: str
    answer: str
    confidence: str


def try_build_term_lookup_answer(
    understanding: QueryUnderstanding,
    results: list[SearchResult],
    records: list[SearchRecord],
    language: str = "ru",
) -> TermLookupAnswer | None:
    entity = understanding.entity
    if not entity or understanding.intent not in {"definition", "composition"}:
        return None

    candidates = _find_entity_records(entity, results, records)
    if not candidates:
        return None

    if understanding.intent == "definition":
        definition = _extract_definition(entity, candidates)
        if definition:
            return TermLookupAnswer(
                entity=entity,
                intent=understanding.intent,
                answer=_format_definition(entity, definition, candidates[0], language),
                confidence="high",
            )

    if understanding.intent == "composition":
        composition = _extract_composition(entity, candidates)
        if composition:
            return TermLookupAnswer(
                entity=entity,
                intent=understanding.intent,
                answer=_format_composition(entity, composition, candidates[0], language),
                confidence="high",
            )

        definition = _extract_definition(entity, candidates)
        if definition:
            return TermLookupAnswer(
                entity=entity,
                intent=understanding.intent,
                answer=_format_definition(entity, definition, candidates[0], language),
                confidence="medium",
            )

    return None


def _find_entity_records(entity: str, results: list[SearchResult], records: list[SearchRecord]) -> list[SearchRecord]:
    normalized_entity = normalize_text(entity)
    candidates: list[SearchRecord] = []
    seen: set[str] = set()
    all_candidates = [*results, *[SearchResult(record=record, score=0.0, matched_terms=[], snippet=record.text[:320]) for record in records]]

    for result in all_candidates:
        record = result.record
        if record.record_id in seen:
            continue
        text = normalize_text(record.text)
        if re.search(rf"(?<![\w]){re.escape(normalized_entity)}(?![\w])", text):
            candidates.append(record)
            seen.add(record.record_id)

    candidates.sort(key=lambda record: _record_priority(record, normalized_entity))
    return candidates


def _record_priority(record: SearchRecord, normalized_entity: str) -> tuple[int, int]:
    text = normalize_text(record.text)
    if re.search(rf"^{re.escape(normalized_entity)}\s+-", text):
        return (0, len(text))
    if f"\u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u044c {normalized_entity}" in text and "\u0440\u0430\u0441\u0441\u0447\u0438\u0442\u044b\u0432\u0430\u0435\u0442\u0441\u044f \u043a\u0430\u043a" in text:
        return (1, len(text))
    if "\u0440\u0430\u0441\u0441\u0447\u0438\u0442\u044b\u0432\u0430\u0435\u0442\u0441\u044f \u043a\u0430\u043a" in text:
        return (2, len(text))
    return (3, len(text))


def _extract_definition(entity: str, records: list[SearchRecord]) -> str | None:
    normalized_entity = normalize_text(entity)
    for record in records:
        match = re.search(rf"^{re.escape(entity)}\s*-\s*(.+)", record.text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return _compact(match.group(1))
        text = normalize_text(record.text)
        if f"\u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u044c {normalized_entity}" in text:
            return _compact(record.text)
    return None


def _extract_composition(entity: str, records: list[SearchRecord]) -> str | None:
    patterns = [
        rf"\u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u044c\s+{re.escape(entity)}\s+\u0440\u0430\u0441\u0441\u0447\u0438\u0442\u044b\u0432\u0430\u0435\u0442\u0441\u044f\s+\u043a\u0430\u043a\s+(.+)",
        rf"{re.escape(entity)}\s+\u0440\u0430\u0441\u0441\u0447\u0438\u0442\u044b\u0432\u0430\u0435\u0442\u0441\u044f\s+\u043a\u0430\u043a\s+(.+)",
        rf"\u0432\s+\u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u044c\s+{re.escape(entity)}\s+\u0432\u043a\u043b\u044e\u0447\u0430\u044e\u0442\u0441\u044f\s+(.+)",
        rf"\u0438\u0437\s+\u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u044f\s+{re.escape(entity)}\s+\u0432\u044b\u0447\u0438\u0442\u0430\u044e\u0442\u0441\u044f\s+(.+)",
    ]
    normalized_entity = normalize_text(entity)
    for record in records:
        for pattern in patterns:
            match = re.search(pattern, record.text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return _compact(match.group(1))
        text = normalize_text(record.text)
        if normalized_entity in text and "\u0440\u0430\u0441\u0441\u0447\u0438\u0442\u044b\u0432\u0430\u0435\u0442\u0441\u044f \u043a\u0430\u043a" in text:
            return _compact(record.text)
    return None


def _format_definition(entity: str, definition: str, record: SearchRecord, language: str) -> str:
    source = _source_line(record, language)
    return f"{entity}: {definition}\n\n{source}"


def _format_composition(entity: str, composition: str, record: SearchRecord, language: str) -> str:
    source = _source_line(record, language)
    if language == "en":
        return f"{entity}: composition found in the document.\n\n{composition}\n\n{source}"
    return f"{entity}: \u043d\u0430\u0439\u0434\u0435\u043d\u043e \u043e\u043f\u0438\u0441\u0430\u043d\u0438\u0435 \u0441\u043e\u0441\u0442\u0430\u0432\u0430 \u0432 \u0442\u0435\u043a\u0441\u0442\u0435 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430.\n\n{composition}\n\n{source}"


def _source_line(record: SearchRecord, language: str) -> str:
    if language == "en":
        return f"Source: {record.source_name}, {record.record_id}."
    return f"\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a: {record.source_name}, {record.record_id}."


def _compact(text: str, limit: int = 1800) -> str:
    compacted = re.sub(r"\s+", " ", text).strip()
    if len(compacted) <= limit:
        return compacted
    return compacted[: limit - 3].rstrip() + "..."
