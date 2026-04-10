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
    batch_size: int = 16,
    progress_callback: Callable[[int, int, SearchRecord, int], None] | None = None,
) -> list[EmbeddingRecord]:
    selected = _select_embedding_records(records)
    selected = selected[:max_records] if max_records is not None else selected
    embedding_records: list[EmbeddingRecord] = []
    batch_size = max(1, batch_size)

    for batch_start in range(0, len(selected), batch_size):
        batch_records = selected[batch_start : batch_start + batch_size]
        batch_texts = [_embedding_text(record) for record in batch_records]
        batch_embeddings = provider.embed_many(batch_texts, model=model)

        for offset, record in enumerate(batch_records, start=1):
            index = batch_start + offset
            token_count = estimate_tokens(record.text)
            embedding = batch_embeddings[offset - 1] if offset <= len(batch_embeddings) else None
            if embedding is not None and embedding.vector:
                embedding_records.append(EmbeddingRecord(record=record, vector=embedding.vector, model=embedding.model))
            if progress_callback is not None:
                progress_callback(index, len(selected), record, token_count)

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
