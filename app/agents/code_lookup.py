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


@dataclass(slots=True)
class TopicSignature:
    canonical: str
    direct_phrases: tuple[str, ...]
    related_phrases: tuple[str, ...]
    aliases: tuple[str, ...]


@dataclass(slots=True)
class CodeTopicMatches:
    topic: str
    direct: list[SearchRecord]
    related: list[SearchRecord]


_RISK_TOPICS: tuple[TopicSignature, ...] = (
    TopicSignature(
        canonical="операционный риск",
        direct_phrases=("величина операционного риска",),
        related_phrases=("операционный риск", "величина операционного риска"),
        aliases=("операционному риску", "операционного риска", "операционный риск"),
    ),
    TopicSignature(
        canonical="рыночный риск",
        direct_phrases=("величина рыночного риска",),
        related_phrases=("рыночный риск", "величина рыночного риска"),
        aliases=("рыночному риску", "рыночного риска", "рыночный риск"),
    ),
    TopicSignature(
        canonical="кредитный риск",
        direct_phrases=("величина кредитного риска", "кредитный риск"),
        related_phrases=("кредитный риск", "величина кредитного риска"),
        aliases=("кредитному риску", "кредитного риска", "кредитный риск"),
    ),
)


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

    answer_lines = [_format_code_line(target_code, row.get("content", ""), language)]
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
    if target_code:
        topic_rows = _find_code_topic_rows(query, records, target_code=target_code)
        if not topic_rows:
            return None
        return CodeTopicAnswer(
            answer=_format_code_group_answer(target_code, topic_rows, language, asks_calculation=_asks_calculation(query)),
            source_record_ids=[record.record_id for record in topic_rows],
            confidence="medium",
        )

    topic = _extract_code_topic(query)
    matches = _find_code_topic_matches(query, records, topic)
    if not matches.direct and not matches.related:
        return None

    used_records = [*matches.direct, *matches.related]
    return CodeTopicAnswer(
        answer=_format_code_topic_answer(query, matches, language),
        source_record_ids=[record.record_id for record in used_records],
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
    for match in re.finditer(r"\b[НN]\s*\d+(?:\.\d+)?\b", value, flags=re.IGNORECASE):
        normalized = match.group(0).replace(" ", "").upper().replace("N", "Н")
        if normalized not in found:
            found.append(normalized)
    return found


def _looks_like_code_topic_query(query: str) -> bool:
    normalized = normalize_text(query)
    if not any(term in normalized for term in ["код", "коды", "code", "codes"]):
        return False
    return extract_code_target(query) is None or _asks_calculation(query)


def _extract_code_topic(query: str) -> str:
    normalized = normalize_text(query)
    normalized = re.sub(
        r"\b(какие|какой|какая|код|коды|относятся|относится|к|по|для|есть|ли|расскажи|найди|покажи)\b",
        " ",
        normalized,
    )
    return re.sub(r"\s+", " ", normalized).strip(" ?.")


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
        overlap = query_stems & _topic_stems(record.text)
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


def _find_code_topic_matches(query: str, records: list[SearchRecord], topic: str, limit: int = 8) -> CodeTopicMatches:
    signature = _resolve_topic_signature(topic or query)
    topic_phrase = normalize_text(signature.canonical if signature else (topic or query))
    direct_scored: list[tuple[int, SearchRecord]] = []
    related_scored: list[tuple[int, SearchRecord]] = []

    for record in records:
        if record.record_type not in {"paragraph", "table_row"}:
            continue

        codes = _extract_record_codes(record)
        if not codes:
            continue

        text = normalize_text(record.text)
        if signature:
            if _is_direct_topic_match(record, text, signature):
                score = _direct_score(record, text, signature)
                direct_scored.append((score, record))
                continue
            if _is_related_topic_match(record, text, signature):
                score = _related_score(record, text, signature)
                related_scored.append((score, record))
            continue

        topic_stems = _topic_stems(topic_phrase)
        overlap = len(topic_stems & _topic_stems(record.text))
        has_phrase = bool(topic_phrase) and topic_phrase in text
        if overlap == 0 and not has_phrase:
            continue
        if _generic_direct_match(text, topic_phrase, overlap):
            score = 50 + overlap * 10 + (10 if record.record_type == "table_row" else 0)
            direct_scored.append((score, record))
            continue
        if _generic_related_match(text, topic_phrase, overlap):
            score = 20 + overlap * 10 + (5 if record.record_type == "table_row" else 0)
            related_scored.append((score, record))

    direct = _dedupe_records_by_codes(direct_scored, limit=limit)
    direct_ids = {record.record_id for record in direct}
    related = _dedupe_records_by_codes(
        [(score, record) for score, record in related_scored if record.record_id not in direct_ids],
        limit=limit,
    )
    return CodeTopicMatches(topic=signature.canonical if signature else (topic or query), direct=direct, related=related)


def _resolve_topic_signature(topic: str) -> TopicSignature | None:
    normalized = normalize_text(topic)
    for signature in _RISK_TOPICS:
        if normalize_text(signature.canonical) in normalized:
            return signature
        if any(normalize_text(alias) in normalized for alias in signature.aliases):
            return signature
    return None


def _extract_record_codes(record: SearchRecord) -> list[str]:
    if record.record_type == "table_row":
        parsed = _parse_code_row(record.text)
        if parsed is not None:
            return _extract_codes_from_text(parsed.get("code", ""))
    return _extract_explicit_code_mentions(record.text) or _extract_codes_from_text(record.text)


def _is_direct_topic_match(record: SearchRecord, text: str, signature: TopicSignature) -> bool:
    if record.record_type == "table_row":
        parsed = _parse_code_row(record.text)
        if parsed is None:
            return False
        content = normalize_text(parsed.get("content", ""))
        if any(_starts_with_direct_definition(content, phrase) for phrase in signature.direct_phrases):
            return True
        return False

    if record.record_type == "paragraph":
        head = text[:320]
        has_direct_phrase = any(_starts_with_direct_definition(head, phrase) for phrase in signature.direct_phrases)
        return "код " in text and has_direct_phrase

    return False


def _is_related_topic_match(record: SearchRecord, text: str, signature: TopicSignature) -> bool:
    if any(_is_false_topic_context(text, signature) for _ in (0,)):
        return False

    has_topic = any(phrase in text for phrase in signature.related_phrases)
    if not has_topic:
        return False

    indirect_markers = (
        "по которым рассчитывается",
        "включается в расчет",
        "уменьшаются на",
        "корректирующая",
        "в части",
    )
    return any(marker in text for marker in indirect_markers)


def _is_false_topic_context(text: str, signature: TopicSignature) -> bool:
    if signature.canonical == "операционный риск":
        return "операционного дня" in text or "операционный день" in text
    return False


def _direct_score(record: SearchRecord, text: str, signature: TopicSignature) -> int:
    score = 80
    if record.record_type == "table_row":
        score += 20
    if any(phrase in text[:220] for phrase in signature.direct_phrases):
        score += 20
    if "код " in text:
        score += 10
    return score


def _starts_with_direct_definition(text: str, phrase: str) -> bool:
    compact = text.strip(" -:;,.")
    if compact.startswith(phrase):
        return True
    return compact.startswith(f"ор - {phrase}") or compact.startswith(f"рр.i - {phrase}") or compact.startswith(f"рр - {phrase}")


def _related_score(record: SearchRecord, text: str, signature: TopicSignature) -> int:
    score = 30
    if record.record_type == "table_row":
        score += 10
    if "по которым рассчитывается" in text:
        score += 20
    if "включается в расчет" in text or "корректирующая" in text:
        score += 10
    if any(phrase in text for phrase in signature.related_phrases):
        score += 10
    return score


def _generic_direct_match(text: str, topic_phrase: str, overlap: int) -> bool:
    if not topic_phrase:
        return False
    return overlap >= 2 and "код " in text and f"величина {topic_phrase}" in text


def _generic_related_match(text: str, topic_phrase: str, overlap: int) -> bool:
    if not topic_phrase:
        return False
    if topic_phrase in text and any(marker in text for marker in ("по которым", "включается в расчет", "корректирующая")):
        return True
    return overlap >= 2 and any(marker in text for marker in ("по которым", "включается в расчет", "корректирующая"))


def _dedupe_records_by_codes(scored: list[tuple[int, SearchRecord]], limit: int) -> list[SearchRecord]:
    scored.sort(key=lambda item: item[0], reverse=True)
    selected: list[SearchRecord] = []
    seen_code_sets: set[tuple[str, ...]] = set()
    for _, record in scored:
        codes = tuple(_extract_record_codes(record))
        if not codes or codes in seen_code_sets:
            continue
        seen_code_sets.add(codes)
        selected.append(record)
        if len(selected) >= limit:
            break
    return selected


def _topic_stems(value: str) -> set[str]:
    normalized = normalize_text(value)
    tokens = re.findall(r"[\w\.\-]+", normalized, flags=re.UNICODE)
    ignored = {
        "что",
        "как",
        "какие",
        "какой",
        "какая",
        "которые",
        "к",
        "по",
        "для",
        "из",
        "от",
        "на",
        "в",
        "и",
        "или",
        "код",
        "коды",
        "относятся",
        "относится",
        "расчет",
        "рассчитывается",
        "document",
        "code",
        "codes",
    }
    stems: set[str] = set()
    for token in tokens:
        if token in ignored or len(token) < 4:
            continue
        stems.add(_rough_stem(token))
    return stems


def _rough_stem(token: str) -> str:
    suffixes = [
        "иями",
        "ями",
        "ами",
        "ему",
        "ому",
        "его",
        "ого",
        "ыми",
        "ими",
        "иях",
        "ий",
        "ый",
        "ой",
        "ая",
        "яя",
        "ое",
        "ее",
        "ые",
        "ие",
        "ах",
        "ях",
        "ов",
        "ев",
        "ом",
        "ем",
        "ам",
        "ям",
        "ую",
        "юю",
        "а",
        "я",
        "у",
        "ю",
        "е",
        "о",
        "ы",
        "и",
    ]
    for suffix in suffixes:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _extract_codes_from_text(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\b\d{3,}(?:\.\d+)?\b", text)))


def _extract_explicit_code_mentions(text: str) -> list[str]:
    patterns = [
        r"\bкод(?:ом|а|ы|у|е)?\s+(\d{3,}(?:\.\d+)?)\b",
        r"\(\s*код\s+(\d{3,}(?:\.\d+)?)\s*\)",
    ]
    found: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            code = match.group(1)
            if code not in found:
                found.append(code)
    return found


def _parse_code_topic_rows(records: list[SearchRecord]) -> list[dict[str, str]]:
    parsed_rows: list[dict[str, str]] = []
    for record in records:
        parsed = _parse_code_row(record.text)
        if parsed is None:
            codes = _extract_explicit_code_mentions(record.text)
            if not codes:
                continue
            parsed_rows.append(
                {
                    "record_id": record.record_id,
                    "codes": ", ".join(codes),
                    "content": record.text.strip(),
                    "normatives": "",
                    "source_name": record.source_name,
                }
            )
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


def _unique_parsed_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    unique: list[dict[str, str]] = []
    seen_codes: set[str] = set()
    for row in rows:
        codes = row["codes"]
        if codes in seen_codes:
            continue
        seen_codes.add(codes)
        unique.append(row)
    return unique


def _format_code_group_answer(target_code: str, records: list[SearchRecord], language: str, asks_calculation: bool) -> str:
    parsed_rows = _unique_parsed_rows(_parse_code_topic_rows(records))
    if not parsed_rows:
        return _format_missing_formula_line(language)

    if language == "en":
        lines = [f"Codes in the {target_code} group are described in these rows:"]
    else:
        lines = [f"Коды группы {target_code} описаны в следующих строках:"]

    for item in parsed_rows[:6]:
        lines.append(f"- {item['codes']}: {item['content']}")
        if item["normatives"]:
            lines.append(f"  {'Used in calculation' if language == 'en' else 'Используется при расчете'}: {item['normatives']}")

    if asks_calculation:
        lines.append(_format_missing_formula_line(language))

    source = parsed_rows[0]
    lines.append(_format_source_line(source["source_name"], source["record_id"], language))
    return "\n".join(lines)


def _format_code_topic_answer(query: str, matches: CodeTopicMatches, language: str) -> str:
    direct_rows = _unique_parsed_rows(_parse_code_topic_rows(matches.direct))
    related_rows = _unique_parsed_rows(_parse_code_topic_rows(matches.related))
    if not direct_rows and not related_rows:
        return ""

    if language == "en":
        lines = [f"Codes related to '{query}' found in the document:"]
    else:
        lines = [f"В документе найдены коды по запросу '{query}':"]

    if direct_rows:
        lines.append("Прямо относятся:" if language == "ru" else "Directly related:")
        for item in direct_rows[:6]:
            lines.append(f"- {item['codes']}: {item['content']}")

    if related_rows:
        lines.append("")
        lines.append("Связанные коды:" if language == "ru" else "Related codes:")
        for item in related_rows[:4]:
            lines.append(f"- {item['codes']}: {item['content']}")

    source = (direct_rows or related_rows)[0]
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
