from __future__ import annotations

from collections.abc import Callable

from app.providers.base import BaseProvider
from app.retrieval.records import SearchRecord
from app.retrieval.vector import EmbeddingRecord
from app.storage.workspaces import estimate_tokens


def build_embedding_index(
    provider: BaseProvider,
    records: list[SearchRecord],
    model: str,
    max_records: int | None = None,
    progress_callback: Callable[[int, int, SearchRecord, int], None] | None = None,
) -> list[EmbeddingRecord]:
    selected = _select_embedding_records(records)
    selected = selected[:max_records] if max_records is not None else selected
    embedding_records: list[EmbeddingRecord] = []

    for index, record in enumerate(selected, start=1):
        token_count = estimate_tokens(record.text)
        if progress_callback is not None:
            progress_callback(index, len(selected), record, token_count)
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


def _select_embedding_records(records: list[SearchRecord]) -> list[SearchRecord]:
    # Full table records duplicate row/cell text and slow embedding indexing significantly.
    return [record for record in records if record.record_type != "table" and record.text.strip()]
