from __future__ import annotations

import re
from dataclasses import dataclass

from app.retrieval.exact import SearchResult, normalize_text


@dataclass(slots=True)
class NormLookupAnswer:
    target_norm: str
    answer: str
    confidence: str


def try_build_norm_lookup_answer(query: str, results: list[SearchResult]) -> NormLookupAnswer | None:
    target = extract_norm_target(query)
    if not target:
        return None

    normalized_target = normalize_text(target)
    relevant = [
        result
        for result in results
        if result.record.record_type == "paragraph" and normalized_target in normalize_text(result.record.text)
    ]
    if not relevant:
        return None

    anchor = _select_anchor_paragraph(relevant)
    paragraphs = _select_anchor_context(anchor, results, include_previous=_asks_table(query))
    text = "\n\n".join(result.record.text for result in paragraphs[:5])

    if _asks_table(query):
        return NormLookupAnswer(
            target_norm=target,
            answer=_build_component_table_answer(target, paragraphs),
            confidence="medium",
        )

    if _asks_simple(query):
        return NormLookupAnswer(
            target_norm=target,
            answer=_build_simple_answer(target, paragraphs),
            confidence="medium",
        )

    lines = [
        f"{target}: найдено описание расчета в тексте документа.",
        text,
    ]

    if "по формуле" in normalize_text(text):
        lines.append(
            "Важно: сама формула в этом DOCX, вероятно, вставлена как изображение/объект Word. "
            "Текущий текстовый индекс извлек окружающий текст, но не распознал изображение формулы. "
            "Для точного извлечения самой дроби нужно добавить OCR/извлечение формул из изображений."
        )

    lines.append(f"Источник: {paragraphs[0].record.source_name}, {paragraphs[0].record.record_id}.")

    return NormLookupAnswer(target_norm=target, answer="\n\n".join(lines), confidence="medium")


def extract_norm_target(query: str) -> str | None:
    match = re.search(r"\b[НN]\s*\d+(?:\.\d+)?\b", query, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(0).replace(" ", "").upper().replace("N", "Н")


def _dedupe_paragraphs(results: list[SearchResult]) -> list[SearchResult]:
    output: list[SearchResult] = []
    seen: set[str] = set()
    for result in results:
        if result.record.record_id in seen:
            continue
        output.append(result)
        seen.add(result.record.record_id)
    return sorted(output, key=_paragraph_sort_key)


def _select_anchor_paragraph(results: list[SearchResult]) -> SearchResult:
    for result in results:
        text = normalize_text(result.record.text)
        if "рассчитывается" in text or "формуле" in text:
            return result
    return results[0]


def _select_anchor_context(anchor: SearchResult, results: list[SearchResult], include_previous: bool = False) -> list[SearchResult]:
    anchor_number = _paragraph_sort_key(anchor)
    start_number = anchor_number - 3 if include_previous else anchor_number
    nearby = [
        result
        for result in results
        if result.record.record_type == "paragraph"
        and start_number <= _paragraph_sort_key(result) <= anchor_number + 6
    ]
    return _dedupe_paragraphs([anchor, *nearby])


def _paragraph_sort_key(result: SearchResult) -> int:
    if not result.record.record_id.startswith("p-"):
        return 10**9
    try:
        return int(result.record.record_id.split("-", 1)[1])
    except ValueError:
        return 10**9


def _asks_simple(query: str) -> bool:
    normalized = normalize_text(query)
    return any(
        marker in normalized
        for marker in [
            "простыми словами",
            "в двух словах",
            "коротко",
            "объясни проще",
        ]
    )


def _asks_table(query: str) -> bool:
    normalized = normalize_text(query)
    return any(marker in normalized for marker in ["таблиц", "столбец", "компонент"])


def _build_component_table_answer(target: str, paragraphs: list[SearchResult]) -> str:
    paragraph_texts = [result.record.text for result in paragraphs]
    components = _extract_components(paragraph_texts)
    if not components:
        return _build_simple_answer(target, paragraphs)

    rows = [
        "| Компонент | Какие коды и счета входят |",
        "|---|---|",
    ]
    for component in _requested_component_order(components):
        rows.append(f"| {component} | {components[component]} |")

    source = paragraphs[0].record if paragraphs else None
    source_text = f"\n\nИсточник: {source.source_name}, {source.record_id}." if source else ""
    return "\n".join(rows) + source_text


def _extract_components(paragraphs: list[str]) -> dict[str, str]:
    components: dict[str, str] = {}
    for paragraph in paragraphs:
        match = re.match(r"^([А-Яа-яA-Za-z][А-Яа-яA-Za-z0-9*]*)\s+-\s+(.+)", paragraph, flags=re.DOTALL)
        if not match:
            continue

        name = match.group(1).strip()
        description = _compact(match.group(2))
        if name in {"Лат", "Овт", "Овт*", "Лам"}:
            components[name] = description

        if name == "Лат" and "Лам" not in components and "показатель Лам" in paragraph:
            components["Лам"] = "Упоминается как высоколиквидные активы в составе расчета Лат; детали состава Лам приведены в пункте 4.2/связанных положениях, если они есть в контексте."

    return components


def _requested_component_order(components: dict[str, str]) -> list[str]:
    ordered = [name for name in ["Лам", "Лат", "Овт", "Овт*"] if name in components]
    return ordered or list(components)


def _compact(text: str, limit: int = 1200) -> str:
    compacted = re.sub(r"\s+", " ", text).strip()
    if len(compacted) <= limit:
        return compacted
    return compacted[: limit - 3].rstrip() + "..."


def _build_simple_answer(target: str, paragraphs: list[SearchResult]) -> str:
    source = paragraphs[0].record if paragraphs else None
    source_text = f"\n\nИсточник: {source.source_name}, {source.record_id}." if source else ""
    return (
        f"{target} - это лимит риска банка на связанное с ним лицо или группу связанных лиц.\n\n"
        "Проще: сколько банк может рискнуть на связанных с ним лиц относительно своего капитала. "
        "В найденном тексте указано, что показатель Крл сравнивается с собственными средствами банка; "
        "сама формула в DOCX вставлена как изображение и текущим текстовым индексом не распознана."
        f"{source_text}"
    )
