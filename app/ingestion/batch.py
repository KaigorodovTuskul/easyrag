from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.ingestion.docx_parser import parse_docx_bytes
from app.retrieval.records import build_search_records
from app.storage.files import list_input_docx, save_parsed_payload, sha256_bytes
from app.storage.index import replace_document_records


@dataclass(slots=True)
class BatchIngestResult:
    source_name: str
    file_hash: str
    paragraph_count: int
    table_count: int
    record_count: int
    parsed_json: str


def ingest_docx_path(path: Path) -> BatchIngestResult:
    content = path.read_bytes()
    file_hash = sha256_bytes(content)
    parsed = parse_docx_bytes(path.name, content)
    records = build_search_records(parsed)
    parsed_path = save_parsed_payload(path.name, parsed.to_dict())
    replace_document_records(parsed.source_name, records, file_hash=file_hash)
    return BatchIngestResult(
        source_name=parsed.source_name,
        file_hash=file_hash,
        paragraph_count=parsed.paragraph_count,
        table_count=parsed.table_count,
        record_count=len(records),
        parsed_json=str(parsed_path),
    )


def ingest_input_folder() -> list[BatchIngestResult]:
    return [ingest_docx_path(path) for path in list_input_docx()]
