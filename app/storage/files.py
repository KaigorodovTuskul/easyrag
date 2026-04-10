from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


DATA_ROOT = Path("data")
DOCS_ROOT = DATA_ROOT / "docs"
PARSED_ROOT = DATA_ROOT / "parsed"


def ensure_data_dirs() -> None:
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    PARSED_ROOT.mkdir(parents=True, exist_ok=True)


def save_uploaded_docx(filename: str, content: bytes) -> Path:
    ensure_data_dirs()
    target = DOCS_ROOT / filename
    target.write_bytes(content)
    return target


def save_parsed_payload(filename: str, payload: dict) -> Path:
    ensure_data_dirs()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = Path(filename).stem
    target = PARSED_ROOT / f"{stem}.{timestamp}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
