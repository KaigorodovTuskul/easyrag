from __future__ import annotations

import re
from dataclasses import dataclass

from app.retrieval.exact import SearchResult, normalize_text
from app.retrieval.records import SearchRecord


@dataclass(slots=True)
class CodeLookupAnswer:
    target_code: str
    answer: str
    source_record_id: str
    related_norms: list[str]
    asks_calculation: bool
    confidence: str


@dataclass(slots=True)
class CodeTopicAnswer:
    answer: str
    source_record_ids: list[str]
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


def try_build_code_topic_answer(query: str, records: list[SearchRecord], language: str = "ru") -> CodeTopicAnswer | None:
    if not _looks_like_code_topic_query(query):
        return None

    target_code = extract_code_target(query)
    asks_calculation = _asks_calculation(query)
    topic_rows = _find_code_topic_rows(query, records, target_code=target_code)
    if not topic_rows:
        return None

    if target_code:
        answer = _format_code_group_answer(target_code, topic_rows, language, asks_calculation=asks_calculation)
    else:
        answer = _format_code_topic_answer(query, topic_rows, language)

    return CodeTopicAnswer(
        answer=answer,
        source_record_ids=[record.record_id for record in topic_rows],
        confidence="medium",
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


def _looks_like_code_topic_query(query: str) -> bool:
    normalized = normalize_text(query)
    if not any(term in normalized for term in ["код", "коды", "code", "codes"]):
        return False
    return extract_code_target(query) is None or any(term in normalized for term in ["рассчит", "calculation", "calculate"])


def _find_code_topic_rows(query: str, records: list[SearchRecord], target_code: str | None = None, limit: int = 8) -> list[SearchRecord]:
    row_candidates = [record for record in records if record.record_type == "table_row"]
    if target_code:
        prefixed = [
            record
            for record in row_candidates
            if any(code == target_code or code.startswith(f"{target_code}.") for code in _extract_codes_from_text(record.text))
        ]
        if prefixed:
            return prefixed[:limit]

    query_stems = _topic_stems(query)
    if not query_stems:
        return []

    scored: list[tuple[int, int, SearchRecord]] = []
    for record in row_candidates:
        record_stems = _topic_stems(record.text)
        overlap = [stem for stem in query_stems if stem in record_stems]
        if not overlap:
            continue
        score = len(overlap) * 10
        if len(overlap) == len(query_stems):
            score += 15
        if "Приложение 1" in record.section_path:
            score += 5
        scored.append((score, len(record.text), record))

    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [record for _, _, record in scored[:limit]]


def _topic_stems(value: str) -> set[str]:
    normalized = normalize_text(value)
    tokens = re.findall(r"[\w\.\-]+", normalized, flags=re.UNICODE)
    ignored = {
        "что", "как", "какие", "какой", "какая", "какие", "которые", "к", "по", "для", "из",
        "от", "на", "в", "и", "или", "код", "коды", "относятся", "относится", "расчет", "рассчитывается",
        "document", "code", "codes",
    }
    stems: set[str] = set()
    for token in tokens:
        if token in ignored or len(token) < 4:
            continue
        stems.add(_rough_stem(token))
    return stems


def _rough_stem(token: str) -> str:
    for suffix in ["иями", "ями", "ами", "ему", "ому", "его", "ого", "ыми", "ими", "иях", "иях", "иях", "иях", "ий", "ый", "ой", "ая", "яя", "ое", "ее", "ые", "ие", "ого", "ему", "ому", "ах", "ях", "ов", "ев", "ом", "ем", "ам", "ям", "ую", "юю", "а", "я", "у", "ю", "е", "о", "ы", "и"]:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _extract_codes_from_text(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\b\d{3,}(?:\.\d+)?\b", text)))


def _parse_code_topic_rows(records: list[SearchRecord]) -> list[dict[str, str]]:
    parsed_rows: list[dict[str, str]] = []
    for record in records:
        parsed = _parse_code_row(record.text)
        if not parsed:
            continue
        codes = _extract_codes_from_text(parsed.get("code", ""))
        if not codes:
            continue
        parsed_rows.append(
            {
                "record_id": record.record_id,
                "codes": ", ".join(codes),
                "content": parsed.get("content", ""),
                "normatives": parsed.get("normatives", ""),
                "source_name": record.source_name,
            }
        )
    return parsed_rows


def _format_code_group_answer(target_code: str, records: list[SearchRecord], language: str, asks_calculation: bool) -> str:
    parsed_rows = _parse_code_topic_rows(records)
    if not parsed_rows:
        return _format_missing_formula_line(language)

    if language == "en":
        lines = [f"Codes in the {target_code} group are calculated through these rows:"]
    else:
        lines = [f"Коды группы {target_code} рассчитываются по следующим строкам:"]

    for item in parsed_rows[:6]:
        lines.append(f"- {item['codes']}: {item['content']}")
        if item["normatives"]:
            if language == "en":
                lines.append(f"  Used in: {item['normatives']}")
            else:
                lines.append(f"  Используется при расчете: {item['normatives']}")

    if asks_calculation:
        lines.append(_format_missing_formula_line(language))

    source = parsed_rows[0]
    lines.append(_format_source_line(source["source_name"], source["record_id"], language))
    return "\n".join(lines)


def _format_code_topic_answer(query: str, records: list[SearchRecord], language: str) -> str:
    parsed_rows = _parse_code_topic_rows(records)
    if not parsed_rows:
        return ""

    if language == "en":
        lines = [f"Codes related to '{query}' found in the document:"]
    else:
        lines = [f"В документе найдены следующие коды по запросу '{query}':"]

    for item in parsed_rows[:8]:
        lines.append(f"- {item['codes']}: {item['content']}")

    source = parsed_rows[0]
    lines.append(_format_source_line(source["source_name"], source["record_id"], language))
    return "\n".join(lines)


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
