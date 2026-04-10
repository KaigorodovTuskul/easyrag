from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.retrieval.records import SearchRecord
from app.storage.files import DATA_ROOT


INDEX_ROOT = DATA_ROOT / "index"
RECORDS_PATH = INDEX_ROOT / "records.jsonl"
MANIFEST_PATH = INDEX_ROOT / "manifest.json"


def ensure_index_dirs() -> None:
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)


def replace_document_records(source_name: str, records: list[SearchRecord], file_hash: str | None = None) -> None:
    ensure_index_dirs()
    existing = [record for record in load_index_records() if record.source_name != source_name]

    with RECORDS_PATH.open("w", encoding="utf-8") as handle:
        for record in [*existing, *records]:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    _write_manifest(file_hashes={source_name: file_hash} if file_hash else None)


def delete_document_records(source_name: str) -> None:
    ensure_index_dirs()
    existing = [record for record in load_index_records() if record.source_name != source_name]
    with RECORDS_PATH.open("w", encoding="utf-8") as handle:
        for record in existing:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    _write_manifest(remove_sources=[source_name])


def load_index_records() -> list[SearchRecord]:
    if not RECORDS_PATH.exists():
        return []

    records: list[SearchRecord] = []
    with RECORDS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(SearchRecord.from_dict(json.loads(line)))
    return records


def get_index_summary() -> dict[str, int | list[str] | str | None]:
    records = load_index_records()
    sources = sorted({record.source_name for record in records})
    manifest = _read_manifest()
    return {
        "record_count": len(records),
        "source_count": len(sources),
        "sources": sources,
        "updated_at": manifest.get("updated_at"),
        "file_hashes": manifest.get("file_hashes", {}),
    }


def _write_manifest(file_hashes: dict[str, str | None] | None = None, remove_sources: list[str] | None = None) -> None:
    records = load_index_records()
    sources = sorted({record.source_name for record in records})
    existing_manifest = _read_manifest()
    merged_hashes = dict(existing_manifest.get("file_hashes", {}))
    for source in remove_sources or []:
        merged_hashes.pop(source, None)
    for source, file_hash in (file_hashes or {}).items():
        if file_hash:
            merged_hashes[source] = file_hash
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "sources": sources,
        "file_hashes": {source: merged_hashes[source] for source in sources if source in merged_hashes},
    }
    MANIFEST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
