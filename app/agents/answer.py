from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.exact import SearchResult


@dataclass(slots=True)
class AnswerContext:
    prompt: str
    citations: list[dict[str, str | float]]


def build_answer_context(question: str, results: list[SearchResult], max_results: int = 6) -> AnswerContext:
    selected = results[:max_results]
    context_blocks: list[str] = []
    citations: list[dict[str, str | float]] = []

    for index, result in enumerate(selected, start=1):
        record = result.record
        section = " / ".join(record.section_path) if record.section_path else "Root"
        context_blocks.append(
            "\n".join(
                [
                    f"[{index}] source={record.source_name}",
                    f"type={record.record_type}",
                    f"id={record.record_id}",
                    f"section={section}",
                    f"text={record.text}",
                ]
            )
        )
        citations.append(
            {
                "ref": f"[{index}]",
                "source": record.source_name,
                "id": record.record_id,
                "type": record.record_type,
                "section": section,
                "score": round(result.score, 2),
            }
        )

    prompt = "\n\n".join(
        [
            "You are a precise RAG answerer for corporate documents.",
            "Answer only using the provided context.",
            "If the context is insufficient, say that the answer was not found in the indexed documents.",
            "Keep exact codes, account numbers, regulatory references, and table values unchanged.",
            "Cite supporting context with bracket references like [1], [2].",
            "",
            f"Question: {question}",
            "",
            "Context:",
            "\n\n".join(context_blocks) if context_blocks else "No context found.",
            "",
            "Answer in Russian.",
        ]
    )

    return AnswerContext(prompt=prompt, citations=citations)
