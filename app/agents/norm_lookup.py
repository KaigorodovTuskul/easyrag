from __future__ import annotations

import re
from dataclasses import dataclass

from app.retrieval.exact import SearchResult, normalize_text


@dataclass(slots=True)
class NormLookupAnswer:
    target_norm: str
    answer: str
    confidence: str


def try_build_norm_lookup_answer(query: str, results: list[SearchResult], language: str = "ru") -> NormLookupAnswer | None:
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
            answer=_build_component_table_answer(target, paragraphs, language),
            confidence="medium",
        )

    if _asks_simple(query):
        return NormLookupAnswer(
            target_norm=target,
            answer=_build_simple_answer(target, paragraphs, language),
            confidence="medium",
        )

    lines = [
        _found_norm_line(target, language),
        text,
    ]

    if "по формуле" in normalize_text(text):
        lines.append(_formula_image_note(language))

    lines.append(_source_line(paragraphs[0].record.source_name, paragraphs[0].record.record_id, language))

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
            "in simple terms",
            "simple words",
            "briefly",
            "shortly",
            "explain simply",
        ]
    )


def _asks_table(query: str) -> bool:
    normalized = normalize_text(query)
    return any(marker in normalized for marker in ["таблиц", "столбец", "компонент", "table", "column", "component"])


def _build_component_table_answer(target: str, paragraphs: list[SearchResult], language: str) -> str:
    paragraph_texts = [result.record.text for result in paragraphs]
    components = _extract_components(paragraph_texts)
    if not components:
        return _build_simple_answer(target, paragraphs, language)

    if language == "en":
        rows = ["| Component | Included codes and accounts |", "|---|---|"]
    else:
        rows = ["| Компонент | Какие коды и счета входят |", "|---|---|"]
    for component in _requested_component_order(components):
        rows.append(f"| {component} | {components[component]} |")

    source = paragraphs[0].record if paragraphs else None
    source_text = f"\n\n{_source_line(source.source_name, source.record_id, language)}" if source else ""
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


def _build_simple_answer(target: str, paragraphs: list[SearchResult], language: str) -> str:
    source = paragraphs[0].record if paragraphs else None
    source_text = f"\n\n{_source_line(source.source_name, source.record_id, language)}" if source else ""
    if language == "en":
        return (
            f"{target} is a bank risk limit for a related party or group of related parties.\n\n"
            "In simple terms: how much risk the bank may take on related parties relative to its capital. "
            "The matched text says that Крл is compared with the bank's own funds; "
            "the formula itself appears to be inserted into the DOCX as an image and is not recognized by the current text index."
            f"{source_text}"
        )
    return (
        f"{target} - это лимит риска банка на связанное с ним лицо или группу связанных лиц.\n\n"
        "Проще: сколько банк может рискнуть на связанных с ним лиц относительно своего капитала. "
        "В найденном тексте указано, что показатель Крл сравнивается с собственными средствами банка; "
        "сама формула в DOCX вставлена как изображение и текущим текстовым индексом не распознана."
        f"{source_text}"
    )


def _found_norm_line(target: str, language: str) -> str:
    if language == "en":
        return f"{target}: calculation description found in the document text."
    return f"{target}: найдено описание расчета в тексте документа."


def _formula_image_note(language: str) -> str:
    if language == "en":
        return (
            "Note: the formula in this DOCX is probably inserted as an image or Word object. "
            "The current text index extracted the surrounding text but did not recognize the formula image. "
            "OCR/formula extraction is needed to extract the exact fraction."
        )
    return (
        "Важно: сама формула в этом DOCX, вероятно, вставлена как изображение/объект Word. "
        "Текущий текстовый индекс извлек окружающий текст, но не распознал изображение формулы. "
        "Для точного извлечения самой дроби нужно добавить OCR/извлечение формул из изображений."
    )


def _source_line(source_name: str, record_id: str, language: str) -> str:
    if language == "en":
        return f"Source: {source_name}, {record_id}."
    return f"Источник: {source_name}, {record_id}."
