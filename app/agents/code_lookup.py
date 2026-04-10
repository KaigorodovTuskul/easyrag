from __future__ import annotations

import re
from dataclasses import dataclass

from app.retrieval.exact import SearchResult, normalize_text


@dataclass(slots=True)
class CodeLookupAnswer:
    target_code: str
    answer: str
    source_record_id: str
    confidence: str


def try_build_code_lookup_answer(query: str, results: list[SearchResult]) -> CodeLookupAnswer | None:
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
        f"Код {target_code}: {row.get('content', 'описание не найдено')}",
    ]

    normatives = row.get("normatives")
    if normatives:
        answer_lines.append(f"Используется при расчете: {normatives}")

    if _asks_calculation(query):
        answer_lines.append("Отдельная формула расчета в найденной строке таблицы не приведена.")

    answer_lines.append(f"Источник: {row_result.record.source_name}, {row_result.record.record_id}.")

    return CodeLookupAnswer(
        target_code=target_code,
        answer="\n\n".join(answer_lines),
        source_record_id=row_result.record.record_id,
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
    return any(term in normalized for term in ["рассчитывается", "расчет", "рассчитать", "как считается"])
