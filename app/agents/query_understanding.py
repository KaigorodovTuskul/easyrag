from __future__ import annotations

import re
from dataclasses import dataclass


RU_FORMULA_TERMS = [
    "\u0444\u043e\u0440\u043c\u0443\u043b",
    "\u0440\u0430\u0441\u0441\u0447\u0438\u0442",
    "\u0441\u0447\u0438\u0442\u0430",
    "\u0447\u0438\u0441\u043b\u0438\u0442\u0435\u043b",
    "\u0437\u043d\u0430\u043c\u0435\u043d\u0430\u0442\u0435\u043b",
]
RU_CODE_TERMS = [
    "\u043a\u043e\u0434",
    "\u0441\u0447\u0435\u0442",
    "\u0441\u0447\u0451\u0442",
]
RU_COMPOSITION_TERMS = [
    "\u0447\u0442\u043e \u0432\u0445\u043e\u0434\u0438\u0442",
    "\u0438\u0437 \u0447\u0435\u0433\u043e \u0441\u043e\u0441\u0442\u043e\u0438\u0442",
    "\u0441\u043e\u0441\u0442\u0430\u0432",
    "\u0432\u043a\u043b\u044e\u0447\u0430\u0435\u0442",
    "\u0432\u043a\u043b\u044e\u0447\u0430\u044e\u0442\u0441\u044f",
    "\u0438\u0437 \u0447\u0435\u0433\u043e \u0441\u043a\u043b\u0430\u0434\u044b\u0432\u0430\u0435\u0442\u0441\u044f",
]
RU_DEFINITION_TERMS = [
    "\u0447\u0442\u043e \u0442\u0430\u043a\u043e\u0435",
    "\u0447\u0442\u043e \u043e\u0437\u043d\u0430\u0447\u0430\u0435\u0442",
    "\u043e\u043f\u0440\u0435\u0434\u0435\u043b\u0435\u043d\u0438\u0435",
]


@dataclass(slots=True)
class QueryUnderstanding:
    intent: str
    entity: str | None = None


def classify_intent(query: str) -> str:
    normalized = _normalize(query)

    if re.search(r"\b[\u043dn]\s*\d+(?:\.\d+)?\b", normalized):
        if any(term in normalized for term in RU_FORMULA_TERMS):
            return "formula"
        return "norm"

    if any(term in normalized for term in [*RU_CODE_TERMS, "account", "code"]):
        return "code_lookup"

    if any(term in normalized for term in RU_COMPOSITION_TERMS):
        return "composition"

    if any(term in normalized for term in [*RU_DEFINITION_TERMS, "meaning", "definition"]):
        return "definition"

    if any(term in normalized for term in [*RU_FORMULA_TERMS, "\u043a\u0430\u043a \u0441\u0447\u0438\u0442\u0430\u0435\u0442\u0441\u044f", "how is"]):
        return "formula"

    if len(normalized.split()) >= 5:
        return "semantic"

    return "exact"


def extract_query_entity(query: str, entity_names: list[str]) -> str | None:
    norm_match = re.search(r"\b[\u041dN]\s*\d+(?:\.\d+)?\b", query, flags=re.IGNORECASE)
    if norm_match:
        return norm_match.group(0).replace(" ", "").upper().replace("N", "\u041d")

    normalized_query = _normalize(query)
    candidates = sorted({name for name in entity_names if name}, key=len, reverse=True)
    for candidate in candidates:
        normalized_candidate = _normalize(candidate)
        if re.search(rf"(?<![\w]){re.escape(normalized_candidate)}(?![\w])", normalized_query):
            return candidate
    return None


def build_query_suggestions(query: str, entity: str | None, intent: str, language: str = "ru") -> list[str]:
    if language == "en":
        return _build_english_suggestions(entity, intent)
    return _build_russian_suggestions(query, entity, intent)


def _build_russian_suggestions(query: str, entity: str | None, intent: str) -> list[str]:
    target = entity or _guess_focus_word(query) or "\u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u044c"
    variants: list[str] = []

    if intent == "formula":
        variants.extend(
            [
                f"{target} \u0440\u0430\u0441\u0441\u0447\u0438\u0442\u044b\u0432\u0430\u0435\u0442\u0441\u044f \u043f\u043e \u0444\u043e\u0440\u043c\u0443\u043b\u0435?",
                f"\u0418\u0437 \u0447\u0435\u0433\u043e \u0441\u043e\u0441\u0442\u043e\u0438\u0442 \u0447\u0438\u0441\u043b\u0438\u0442\u0435\u043b\u044c \u0438 \u0437\u043d\u0430\u043c\u0435\u043d\u0430\u0442\u0435\u043b\u044c {target}?",
                f"\u041a\u0430\u043a\u0430\u044f \u0444\u043e\u0440\u043c\u0443\u043b\u0430 \u0440\u0430\u0441\u0447\u0435\u0442\u0430 {target}?",
            ]
        )
    elif intent == "composition":
        variants.extend(
            [
                f"\u041f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u044c {target} \u0440\u0430\u0441\u0441\u0447\u0438\u0442\u044b\u0432\u0430\u0435\u0442\u0441\u044f \u043a\u0430\u043a \u0447\u0442\u043e?",
                f"\u0427\u0442\u043e \u0432\u043a\u043b\u044e\u0447\u0430\u0435\u0442\u0441\u044f \u0432 \u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u044c {target}?",
                f"\u0418\u0437 \u0447\u0435\u0433\u043e \u0441\u043e\u0441\u0442\u043e\u0438\u0442 \u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u044c {target}?",
            ]
        )
    elif intent == "definition":
        variants.extend(
            [
                f"{target} - \u044d\u0442\u043e \u0447\u0442\u043e?",
                f"\u041a\u0430\u043a \u043e\u043f\u0440\u0435\u0434\u0435\u043b\u044f\u0435\u0442\u0441\u044f \u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u044c {target}?",
                f"\u0427\u0442\u043e \u043e\u0437\u043d\u0430\u0447\u0430\u0435\u0442 {target} \u0432 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0435?",
            ]
        )
    else:
        variants.extend(
            [
                f"\u0427\u0442\u043e \u0442\u0430\u043a\u043e\u0435 {target}?",
                f"\u041a\u0430\u043a \u0440\u0430\u0441\u0441\u0447\u0438\u0442\u044b\u0432\u0430\u0435\u0442\u0441\u044f {target}?",
                f"\u0412 \u043a\u0430\u043a\u043e\u043c \u043f\u0443\u043d\u043a\u0442\u0435 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430 \u043e\u043f\u0438\u0441\u0430\u043d {target}?",
            ]
        )

    return list(dict.fromkeys(variants))[:3]


def _build_english_suggestions(entity: str | None, intent: str) -> list[str]:
    target = entity or "the metric"
    if intent == "formula":
        return [
            f"Is {target} calculated by a formula?",
            f"What are the numerator and denominator of {target}?",
            f"What is the calculation formula for {target}?",
        ]
    if intent == "composition":
        return [
            f"What is included in {target}?",
            f"How is {target} composed?",
            f"How is {target} calculated?",
        ]
    if intent == "definition":
        return [
            f"What is {target}?",
            f"How is {target} defined?",
            f"What does {target} mean in the document?",
        ]
    return [
        f"What is {target}?",
        f"How is {target} calculated?",
        f"Where is {target} described?",
    ]


def _guess_focus_word(query: str) -> str | None:
    candidates = re.findall(r"[A-Za-z\u0410-\u042f\u0430-\u044f0-9\.\-]{2,}", query)
    if not candidates:
        return None
    ignored = {
        "\u0447\u0442\u043e",
        "\u0432\u0445\u043e\u0434\u0438\u0442",
        "\u0441\u043e\u0441\u0442\u0430\u0432",
        "\u043a\u0430\u043a",
        "\u0440\u0430\u0441\u0441\u0447\u0438\u0442\u044b\u0432\u0430\u0435\u0442\u0441\u044f",
    }
    for candidate in reversed(candidates):
        if candidate.lower() not in ignored:
            return candidate
    return candidates[-1]


def _normalize(value: str) -> str:
    lowered = value.lower().replace("\u0451", "\u0435")
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered
