from __future__ import annotations

import json

from app.retrieval.vector import EmbeddingRecord
from app.storage.index import INDEX_ROOT


EMBEDDINGS_PATH = INDEX_ROOT / "embeddings.jsonl"


def replace_embeddings(records: list[EmbeddingRecord]) -> None:
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    with EMBEDDINGS_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def load_embeddings() -> list[EmbeddingRecord]:
    if not EMBEDDINGS_PATH.exists():
        return []

    records: list[EmbeddingRecord] = []
    with EMBEDDINGS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(EmbeddingRecord.from_dict(json.loads(line)))
    return records
