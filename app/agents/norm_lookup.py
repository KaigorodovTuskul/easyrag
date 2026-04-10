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
    paragraphs = _select_anchor_context(anchor, results)
    text = "\n\n".join(result.record.text for result in paragraphs[:5])

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


def _select_anchor_context(anchor: SearchResult, results: list[SearchResult]) -> list[SearchResult]:
    anchor_number = _paragraph_sort_key(anchor)
    nearby = [
        result
        for result in results
        if result.record.record_type == "paragraph"
        and anchor_number <= _paragraph_sort_key(result) <= anchor_number + 6
    ]
    return _dedupe_paragraphs([anchor, *nearby])


def _paragraph_sort_key(result: SearchResult) -> int:
    if not result.record.record_id.startswith("p-"):
        return 10**9
    try:
        return int(result.record.record_id.split("-", 1)[1])
    except ValueError:
        return 10**9
