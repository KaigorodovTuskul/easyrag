from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.formula_vision import is_plausible_formula_text
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

    normalized_target = _normalize_norm_target(target)
    relevant = [
        result
        for result in results
        if result.record.record_type == "paragraph" and _norm_target_matches(normalize_text(result.record.text), normalized_target)
    ]
    if not relevant:
        return None

    anchor = _select_anchor_paragraph(relevant, query)
    paragraphs = _select_anchor_context(anchor, results, include_previous=_asks_table(query))
    raw_text = "\n\n".join(result.record.text for result in paragraphs[:5])
    has_unrecognized_formula_image = "[FORMULA_IMAGE:" in raw_text
    text = _format_formula_markers(raw_text, language)

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

    inferred_formula = _infer_ratio_formula(target, paragraphs, language)
    if inferred_formula:
        lines.append(inferred_formula)

    if has_unrecognized_formula_image or "рассчитывается по формуле" in normalize_text(text):
        lines.append(_formula_image_note(language))

    lines.append(_source_line(paragraphs[0].record.source_name, paragraphs[0].record.record_id, language))

    return NormLookupAnswer(target_norm=target, answer="\n\n".join(lines), confidence="medium")


def extract_norm_target(query: str) -> str | None:
    match = re.search(r"\b[НN]\s*\d+(?:\.\d+)?\b", query, flags=re.IGNORECASE)
    if match:
        return match.group(0).replace(" ", "").upper().replace("N", "Н")

    normalized = normalize_text(query)
    if "краткосрочн" in normalized and "ликвидност" in normalized:
        return "Н3"
    if "мгновенн" in normalized and "ликвидност" in normalized:
        return "Н2"
    if "текущ" in normalized and "ликвидност" in normalized:
        return "Н3"
    return None


def _normalize_norm_target(target: str) -> str:
    return normalize_text(target).replace(" ", "")


def _norm_target_matches(text: str, target: str) -> bool:
    if re.fullmatch(r"n\d+(?:\.\d+)?", target):
        return re.search(rf"(?<![a-z0-9]){re.escape(target)}(?![\d\.])", text) is not None
    return target in text


def _dedupe_paragraphs(results: list[SearchResult]) -> list[SearchResult]:
    output: list[SearchResult] = []
    seen: set[str] = set()
    for result in results:
        if result.record.record_id in seen:
            continue
        output.append(result)
        seen.add(result.record.record_id)
    return sorted(output, key=_paragraph_sort_key)


def _select_anchor_paragraph(results: list[SearchResult], query: str) -> SearchResult:
    return max(results, key=lambda result: _anchor_score(result, query))


def _anchor_score(result: SearchResult, query: str) -> float:
    text = normalize_text(result.record.text)
    query_text = normalize_text(query)
    score = result.score

    asks_formula = any(term in query_text for term in ["формула", "расчет", "рассчитывается", "calculation", "formula"])
    if asks_formula and "рассчитывается по формуле" in text:
        score += 120
    elif asks_formula and "рассчитывается" in text:
        score += 40

    if "не рассчитывается" in text:
        score -= 80 if asks_formula else 20

    if re.match(r"^\d+(?:\.\d+)*\.", result.record.text.strip()):
        score += 5

    return score


def _select_anchor_context(anchor: SearchResult, results: list[SearchResult], include_previous: bool = False) -> list[SearchResult]:
    anchor_number = _paragraph_sort_key(anchor)
    start_number = anchor_number - 3 if include_previous else anchor_number
    nearby = [
        result
        for result in results
        if result.record.record_type == "paragraph" and start_number <= _paragraph_sort_key(result) <= anchor_number + 6
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
            components["Лам"] = "Упоминается как высоколиквидные активы в составе расчета Лат; детали состава Лам приведены в соседних положениях документа."

    return components


def _requested_component_order(components: dict[str, str]) -> list[str]:
    ordered = [name for name in ["Лам", "Лат", "Овт", "Овт*"] if name in components]
    return ordered or list(components)


def _compact(text: str, limit: int = 1200) -> str:
    compacted = re.sub(r"\s+", " ", text).strip()
    if len(compacted) <= limit:
        return compacted
    return compacted[: limit - 3].rstrip() + "..."


def _format_formula_markers(text: str, language: str) -> str:
    def replace_omml(match: re.Match[str]) -> str:
        formula = match.group(1).strip()
        if language == "en":
            return f"Formula extracted from DOCX equation: {formula}"
        return f"Формула извлечена из уравнения DOCX: {formula}"

    def replace_ocr(match: re.Match[str]) -> str:
        formula = match.group(1).strip()
        if not is_plausible_formula_text(formula):
            if language == "en":
                return "Formula is present as an image, but automatic extraction did not recognize it reliably."
            return "Формула есть как изображение, но автоматическое распознавание не извлекло ее надежно."
        if language == "en":
            return f"Formula recognized from image by the vision model, verify accuracy: {formula}"
        return f"Формула распознана vision-моделью из изображения, проверьте точность: {formula}"

    def replace_image(match: re.Match[str]) -> str:
        image_name = match.group(1).replace("; not recognized", "").strip()
        if language == "en":
            return f"Formula is present as an image but was not recognized: {image_name}."
        return f"Формула есть как изображение, но не распознана: {image_name}."

    text = re.sub(r"\[FORMULA_OMML:\s*(.*?)\]", replace_omml, text)
    text = re.sub(r"\[FORMULA_VISION:\s*(.*?)\]", replace_ocr, text)
    text = re.sub(r"\[FORMULA_OCR:\s*(.*?)\]", replace_ocr, text)
    text = re.sub(r"\[FORMULA_IMAGE:\s*(.*?)\]", replace_image, text)
    return text


def _infer_ratio_formula(target: str, paragraphs: list[SearchResult], language: str) -> str | None:
    if not paragraphs:
        return None

    combined = " ".join(result.record.text for result in paragraphs[:4])
    compacted = re.sub(r"\s+", " ", combined).strip()
    normalized = normalize_text(compacted)
    if "отношение" not in normalized or "рассчитывается по формуле" not in normalized:
        return None

    match = re.search(
        r"отношение\s+(.+?)\s+к\s+(.+?)(?:\s+и\s+регулирует|\.\s+Норматив|\.\s+[А-ЯA-Z]|$)",
        compacted,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    numerator = _compact(match.group(1), limit=700)
    denominator = _compact(match.group(2), limit=450)
    variable = _extract_first_formula_variable(paragraphs)

    if language == "en":
        prefix = f"From the text around the formula, {target} is a ratio:"
        variable_text = f" The numerator is denoted as {variable} in the following definition." if variable else ""
        return f"{prefix} numerator - {numerator}; denominator - {denominator}.{variable_text}"

    variable_text = f" В ближайшей расшифровке числитель обозначен как {variable}." if variable else ""
    return (
        f"По текстовому описанию рядом с формулой {target} - это отношение: "
        f"числитель - {numerator}; знаменатель - {denominator}.{variable_text}"
    )


def _extract_first_formula_variable(paragraphs: list[SearchResult]) -> str | None:
    for result in paragraphs[:6]:
        match = re.match(r"^([А-ЯA-Z][А-Яа-яA-Za-z0-9*\.]*)\s+-\s+", result.record.text.strip())
        if match:
            return match.group(1)
    return None


def _build_simple_answer(target: str, paragraphs: list[SearchResult], language: str) -> str:
    source = paragraphs[0].record if paragraphs else None
    source_text = f"\n\n{_source_line(source.source_name, source.record_id, language)}" if source else ""
    if language == "en":
        return (
            f"{target}: a calculation description is present in the document text."
            f"{source_text}"
        )
    return f"{target}: описание расчета есть в тексте документа.{source_text}"


def _found_norm_line(target: str, language: str) -> str:
    if language == "en":
        return f"{target}: calculation description found in the document text."
    return f"{target}: найдено описание расчета в тексте документа."


def _formula_image_note(language: str) -> str:
    if language == "en":
        return (
            "Note: the formula in this DOCX is probably inserted as an image or Word object. "
            "The current text index extracted the surrounding text but did not recognize the formula image. "
            "Vision-based formula extraction is needed to extract the exact fraction."
        )
    return (
        "Важно: сама формула в этом DOCX, вероятно, вставлена как изображение или объект Word. "
        "Текстовый индекс извлек окружающий текст, а точная дробь берется из изображения формулы."
    )


def _source_line(source_name: str, record_id: str, language: str) -> str:
    if language == "en":
        return f"Source: {source_name}, {record_id}."
    return f"Источник: {source_name}, {record_id}."
