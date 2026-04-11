from __future__ import annotations

import json
import re
from pathlib import Path

from app.retrieval.records import SearchRecord
from app.storage.workspaces import workspace_dir


def workspace_entities_path(workspace_id: str) -> Path:
    return workspace_dir(workspace_id) / "entities.json"


def save_workspace_entities(workspace_id: str, records: list[SearchRecord]) -> list[str]:
    entities = extract_entities(records)
    path = workspace_entities_path(workspace_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8")
    return entities


def load_workspace_entities(workspace_id: str) -> list[str]:
    path = workspace_entities_path(workspace_id)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in payload if isinstance(item, str)]


def extract_entities(records: list[SearchRecord]) -> list[str]:
    found: set[str] = set()
    for record in records:
        text = record.text
        for match in re.finditer(r"\b[\u041dN]\s*\d+(?:\.\d+)?\b", text, flags=re.IGNORECASE):
            found.add(match.group(0).replace(" ", "").upper().replace("N", "\u041d"))

        for line in text.splitlines():
            stripped = line.strip()
            if " - " in stripped:
                head = stripped.split(" - ", 1)[0].strip()
                if _is_entity_token(head):
                    found.add(head)

            for match in re.finditer(r"\b\u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u044c\s+([^\s,;:\)\(]+)", stripped, flags=re.IGNORECASE):
                candidate = match.group(1).strip()
                if _is_entity_token(candidate):
                    found.add(candidate)

    return sorted(found, key=lambda value: (_entity_rank(value), value))


def _is_entity_token(value: str) -> bool:
    if not value or len(value) > 24:
        return False
    if value.endswith("."):
        value = value[:-1]
    return re.fullmatch(r"[\w\*\.\-]+", value, flags=re.UNICODE) is not None


def _entity_rank(value: str) -> tuple[int, int]:
    if re.fullmatch(r"[\u041dN]\d+(?:\.\d+)?", value, flags=re.IGNORECASE):
        return (0, len(value))
    if len(value) <= 6:
        return (1, len(value))
    return (2, len(value))
