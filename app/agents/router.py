from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.providers.base import BaseProvider
from app.retrieval.records import SearchRecord


@dataclass(slots=True)
class QueryRoute:
    action: str
    entities: list[str]
    needs_formula: bool
    needs_tables: bool
    use_full_context: bool
    confidence: float
    reason: str
    source: str


def resolve_query_route(
    provider: BaseProvider,
    chat_model: str,
    prompt: str,
    records: list[SearchRecord],
    known_entities: list[str] | None = None,
    language: str = "ru",
) -> QueryRoute:
    fallback = _fallback_route(prompt, records, known_entities=known_entities)
    parsed = _request_router_json(provider, chat_model, prompt, records, known_entities, language)
    if parsed is None:
        return fallback

    route = QueryRoute(
        action=_normalized_action(parsed.get("action")),
        entities=_normalize_entities(parsed.get("entities"), known_entities),
        needs_formula=bool(parsed.get("needs_formula", False)),
        needs_tables=bool(parsed.get("needs_tables", False)),
        use_full_context=bool(parsed.get("use_full_context", False)),
        confidence=_normalize_confidence(parsed.get("confidence")),
        reason=str(parsed.get("reason", "")).strip() or "llm_router",
        source="llm",
    )
    if route.action == "unknown":
        return fallback
    if route.action == "compare_entities" and len(route.entities) < 2:
        return fallback
    if route.action == "lookup_entity" and not route.entities:
        return fallback
    if route.confidence < 0.35:
        return fallback
    return route


def _request_router_json(
    provider: BaseProvider,
    chat_model: str,
    prompt: str,
    records: list[SearchRecord],
    known_entities: list[str] | None,
    language: str,
) -> dict | None:
    base_prompt = _build_router_prompt(prompt, records, known_entities, language)
    try:
        generated = provider.generate(base_prompt, model=chat_model)
    except Exception:
        return None

    parsed = _parse_router_output(generated.text, known_entities=known_entities)
    if parsed is not None:
        return parsed

    repair_prompt = "\n\n".join(
        [
            base_prompt,
            "Your previous answer was not valid JSON.",
            "Return ONLY one JSON object and nothing else.",
        ]
    )
    try:
        repaired = provider.generate(repair_prompt, model=chat_model)
    except Exception:
        return None
    return _parse_router_output(repaired.text, known_entities=known_entities)


def _build_router_prompt(
    prompt: str,
    records: list[SearchRecord],
    known_entities: list[str] | None,
    language: str,
) -> str:
    stats = _document_router_summary(records)
    entity_examples = ", ".join((known_entities or [])[:16]) or "none"
    return "\n".join(
        [
            "You are a query router for a document QA system.",
            "Return only valid JSON.",
            "Do not use markdown fences.",
            "Do not add explanations before or after JSON.",
            'Allowed actions: "lookup_entity", "compare_entities", "topic_query", "formula_query", "document_summary", "general_qa".',
            "The route must be document-agnostic and based on user intent, not on hardcoded domain rules.",
            "If the user compares two or more entities, action must be compare_entities.",
            "If the user asks for a formula or how something is calculated, set needs_formula=true.",
            "If the user asks which codes or items belong to a topic, use topic_query.",
            "If the question likely requires reading the whole small document, set use_full_context=true.",
            "Only extract entities that are explicitly present in the user question or clearly referenced there.",
            "Schema:",
            '{"action":"general_qa","entities":[],"needs_formula":false,"needs_tables":false,"use_full_context":false,"confidence":0.0,"reason":"..."}',
            "",
            f"Language: {language}",
            f"Document stats: {stats}",
            f"Known document entities: {entity_examples}",
            f"Question: {prompt}",
        ]
    )


def _document_router_summary(records: list[SearchRecord]) -> str:
    paragraph_count = sum(1 for record in records if record.record_type == "paragraph")
    table_row_count = sum(1 for record in records if record.record_type == "table_row")
    formula_count = sum(1 for record in records if record.record_type == "formula_text")
    token_estimate = sum(max(1, len(record.text) // 4) for record in records)
    return (
        f"paragraphs={paragraph_count}, table_rows={table_row_count}, "
        f"formula_texts={formula_count}, token_estimate={token_estimate}"
    )


def _parse_router_output(text: str, known_entities: list[str] | None = None) -> dict | None:
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = text.strip().strip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return _parse_router_text_fallback(cleaned, known_entities=known_entities)


def _fallback_route(
    prompt: str,
    records: list[SearchRecord],
    known_entities: list[str] | None = None,
) -> QueryRoute:
    normalized = _normalize_text(prompt)
    entities = _extract_entities(prompt, known_entities=known_entities)
    action = "general_qa"
    needs_formula = any(token in normalized for token in ["формул", "рассчит", "calculate", "formula"])
    needs_tables = any(token in normalized for token in ["таблиц", "table", "код", "codes"])

    if _is_document_summary_query(normalized):
        action = "document_summary"
    elif len(entities) >= 2 and _has_comparison_intent(normalized):
        action = "compare_entities"
    elif entities and any(token in normalized for token in ["код", "code", "codes"]):
        action = "topic_query" if len(entities) > 1 or not _has_explicit_single_code(prompt) else "lookup_entity"
    elif entities:
        action = "formula_query" if needs_formula else "lookup_entity"
    elif any(token in normalized for token in ["какие коды", "which codes", "относятся к", "related codes"]):
        action = "topic_query"

    return QueryRoute(
        action=action,
        entities=entities,
        needs_formula=needs_formula,
        needs_tables=needs_tables,
        use_full_context=_estimated_document_tokens(records) <= 6500
        and action in {"compare_entities", "topic_query", "document_summary", "general_qa"},
        confidence=0.4,
        reason="fallback_router",
        source="fallback",
    )


def _parse_router_text_fallback(text: str, known_entities: list[str] | None = None) -> dict | None:
    lowered = _normalize_text(text)
    action = None
    for candidate in [
        "compare_entities",
        "lookup_entity",
        "topic_query",
        "formula_query",
        "document_summary",
        "general_qa",
    ]:
        if candidate in lowered:
            action = candidate
            break

    if action is None:
        if any(marker in lowered for marker in ["compare", "difference", "разниц", "сравн"]):
            action = "compare_entities"
        elif any(marker in lowered for marker in ["formula", "рассчит", "формул"]):
            action = "formula_query"
        elif any(marker in lowered for marker in ["summary", "о чем документ", "summarize"]):
            action = "document_summary"
        elif any(marker in lowered for marker in ["topic", "какие коды", "относятся к"]):
            action = "topic_query"
        elif any(marker in lowered for marker in ["lookup", "entity", "код", "норматив", "term"]):
            action = "lookup_entity"
        else:
            action = "general_qa"

    entities = _extract_entities(text, known_entities=known_entities)
    return {
        "action": action,
        "entities": entities,
        "needs_formula": any(marker in lowered for marker in ["formula", "формул", "рассчит", "calculate"]),
        "needs_tables": any(marker in lowered for marker in ["table", "таблиц", "код", "codes"]),
        "use_full_context": any(
            marker in lowered for marker in ["full_context", "full context", "whole document", "entire document"]
        ),
        "confidence": 0.45,
        "reason": "parsed_router_text",
    }


def _normalize_entities(value, known_entities: list[str] | None = None) -> list[str]:
    if not isinstance(value, list):
        return []
    aliases = {_normalize_for_entities(item): item for item in (known_entities or []) if isinstance(item, str)}
    entities: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if not normalized:
            continue
        canonical = aliases.get(_normalize_for_entities(normalized), normalized)
        if canonical not in entities:
            entities.append(canonical)
    return entities


def _normalized_action(value: object) -> str:
    if not isinstance(value, str):
        return "unknown"
    action = value.strip().lower()
    if action in {"lookup_entity", "compare_entities", "topic_query", "formula_query", "document_summary", "general_qa"}:
        return action
    return "unknown"


def _normalize_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except Exception:
        return 0.0
    return min(max(confidence, 0.0), 1.0)


def _extract_entities(prompt: str, known_entities: list[str] | None = None) -> list[str]:
    found: list[str] = []
    for match in re.finditer(r"\b[НN]\s*\d+(?:\.\d+)?\b", prompt, flags=re.IGNORECASE):
        entity = match.group(0).replace(" ", "").upper().replace("N", "Н")
        if entity not in found:
            found.append(entity)
    for match in re.finditer(r"\b\d{3,}(?:\.\d+)?\b", prompt):
        entity = match.group(0)
        if entity not in found:
            found.append(entity)

    normalized_prompt = _normalize_for_entities(prompt)
    prompt_words = set(_entity_words(normalized_prompt))
    for entity in sorted((known_entities or []), key=len, reverse=True):
        normalized_entity = _normalize_for_entities(entity)
        if len(normalized_entity) < 2:
            continue
        if _entity_present_in_prompt(normalized_prompt, prompt_words, normalized_entity) and entity not in found:
            found.append(entity)
    return found


def _normalize_text(value: str) -> str:
    return value.lower().replace("ё", "е").replace("С‘", "Рµ")


def _normalize_for_entities(value: str) -> str:
    lowered = _normalize_text(value)
    lowered = re.sub(r"[\"'`“”«»()\[\]{}:;,.!?/\\]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _entity_present_in_prompt(normalized_prompt: str, prompt_words: set[str], normalized_entity: str) -> bool:
    if " " not in normalized_entity:
        return normalized_entity in prompt_words
    return normalized_entity in normalized_prompt


def _entity_words(value: str) -> list[str]:
    return [token for token in value.split(" ") if token]


def _has_comparison_intent(normalized: str) -> bool:
    return any(
        marker in normalized
        for marker in [
            "в чем разница",
            "чем отличается",
            "отличается от",
            "сравни",
            "сравнение",
            "difference",
            "compare",
            "comparison",
        ]
    )


def _has_explicit_single_code(prompt: str) -> bool:
    return len(re.findall(r"\b\d{3,}(?:\.\d+)?\b", prompt)) == 1


def _is_document_summary_query(normalized: str) -> bool:
    return any(
        marker in normalized
        for marker in [
            "о чем документ",
            "о чем этот документ",
            "расскажи вкратце",
            "кратко о документе",
            "краткое содержание",
            "summary of the document",
            "summarize the document",
            "what is this document about",
        ]
    )


def _estimated_document_tokens(records: list[SearchRecord]) -> int:
    return sum(max(1, len(record.text) // 4) for record in records if record.text.strip())
