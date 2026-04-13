from __future__ import annotations

import re
from dataclasses import dataclass

from app.retrieval.exact import SearchResult, normalize_text


@dataclass(slots=True)
class CodeLookupAnswer:
    target_code: str
    answer: str
    source_record_id: str
    related_norms: list[str]
    asks_calculation: bool
    confidence: str


def try_build_code_lookup_answer(query: str, results: list[SearchResult], language: str = "ru") -> CodeLookupAnswer | None:
    target_code = extract_code_target(query)
    if not target_code:
        return None

    row_result = _find_code_row(target_code, results)
    if row_result is None:
        return None

    row = _parse_code_row(row_result.record.text)
    if not row or row.get("code") != target_code:
        return None

    answer_lines = [
        _format_code_line(target_code, row.get("content", ""), language),
    ]

    normatives = row.get("normatives")
    if normatives:
        answer_lines.append(_format_normatives_line(normatives, language))

    asks_calculation = _asks_calculation(query)
    if asks_calculation:
        answer_lines.append(_format_missing_formula_line(language))

    answer_lines.append(_format_source_line(row_result.record.source_name, row_result.record.record_id, language))

    return CodeLookupAnswer(
        target_code=target_code,
        answer="\n\n".join(answer_lines),
        source_record_id=row_result.record.record_id,
        related_norms=_extract_norm_targets(normatives or ""),
        asks_calculation=asks_calculation,
        confidence="high",
    )


def extract_code_target(query: str) -> str | None:
    match = re.search(r"\b\d{3,}(?:\.\d+)?\b", query)
    return match.group(0) if match else None


def _find_code_row(target_code: str, results: list[SearchResult]) -> SearchResult | None:
    normalized_target = normalize_text(f"Код: {target_code}")
    for result in results:
        if result.record.record_type != "table_row":
            continue
        if normalized_target in normalize_text(result.record.text):
            return result
    return None


def _parse_code_row(row_text: str) -> dict[str, str] | None:
    parts = [part.strip() for part in row_text.split(" | ") if part.strip()]
    parsed: dict[str, str] = {}

    for part in parts:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key_normalized = key.lower().replace("ё", "е").strip()
        value = value.strip()

        if "содержание кода" in key_normalized:
            parsed["content"] = value
        elif key_normalized == "код":
            parsed["code"] = value
        elif "обязательные нормативы" in key_normalized:
            parsed["normatives"] = value

    return parsed or None


def _asks_calculation(query: str) -> bool:
    normalized = normalize_text(query)
    return any(
        term in normalized
        for term in [
            "рассчитывается",
            "расчет",
            "рассчитать",
            "как считается",
            "calculated",
            "calculation",
            "calculate",
            "how is",
        ]
    )


def _extract_norm_targets(value: str) -> list[str]:
    found: list[str] = []
    for match in re.finditer(r"\b[РќN]\s*\d+(?:\.\d+)?\b", value, flags=re.IGNORECASE):
        normalized = match.group(0).replace(" ", "").upper().replace("N", "Рќ")
        if normalized not in found:
            found.append(normalized)
    return found


def _format_code_line(target_code: str, content: str, language: str) -> str:
    if language == "en":
        return f"Code {target_code}: {content or 'description not found'}"
    return f"Код {target_code}: {content or 'описание не найдено'}"


def _format_normatives_line(normatives: str, language: str) -> str:
    if language == "en":
        return f"Used in calculation: {normatives}"
    return f"Используется при расчете: {normatives}"


def _format_missing_formula_line(language: str) -> str:
    if language == "en":
        return "A separate calculation formula is not provided in the matched table row."
    return "Отдельная формула расчета в найденной строке таблицы не приведена."


def _format_source_line(source_name: str, record_id: str, language: str) -> str:
    if language == "en":
        return f"Source: {source_name}, {record_id}."
    return f"Источник: {source_name}, {record_id}."
