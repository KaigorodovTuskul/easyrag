from __future__ import annotations

from app.providers.base import BaseProvider
from app.retrieval.records import SearchRecord
from app.retrieval.vector import EmbeddingRecord


def build_embedding_index(
    provider: BaseProvider,
    records: list[SearchRecord],
    model: str,
    max_records: int | None = None,
) -> list[EmbeddingRecord]:
    selected = records[:max_records] if max_records is not None else records
    embedding_records: list[EmbeddingRecord] = []

    for record in selected:
        text = _embedding_text(record)
        embedding = provider.embed(text, model=model)
        if not embedding.vector:
            continue
        embedding_records.append(EmbeddingRecord(record=record, vector=embedding.vector, model=embedding.model))

    return embedding_records


def _embedding_text(record: SearchRecord) -> str:
    section = " / ".join(record.section_path)
    return "\n".join(
        [
            f"source: {record.source_name}",
            f"type: {record.record_type}",
            f"section: {section}",
            record.text,
        ]
    )
