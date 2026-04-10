from __future__ import annotations

import json
from pathlib import Path

from app.storage.workspaces import workspace_dir


def conversation_path(workspace_id: str) -> Path:
    return workspace_dir(workspace_id) / "chat.json"


def load_conversation(workspace_id: str) -> list[dict[str, str]]:
    path = conversation_path(workspace_id)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        {"role": item["role"], "content": item["content"]}
        for item in payload
        if item.get("role") in {"user", "assistant"} and "content" in item
    ]


def save_conversation(workspace_id: str, messages: list[dict[str, str]]) -> None:
    path = conversation_path(workspace_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")


def clear_conversation(workspace_id: str) -> None:
    path = conversation_path(workspace_id)
    if path.exists():
        path.unlink()
