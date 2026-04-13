from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.retrieval.records import SearchRecord
from app.retrieval.vector import EmbeddingRecord
from app.storage.files import DATA_ROOT, sha256_bytes


WORKSPACES_ROOT = DATA_ROOT / "workspaces"


@dataclass(slots=True)
class WorkspaceInfo:
    workspace_id: str
    source_name: str
    file_hash: str
    record_count: int
    embedding_count: int
    token_count: int
    embedding_token_count: int
    embed_model: str | None
    updated_at: str


def ensure_workspaces_root() -> None:
    WORKSPACES_ROOT.mkdir(parents=True, exist_ok=True)


def workspace_id_for(source_name: str, file_hash: str) -> str:
    stem = Path(source_name).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "document"
    return f"{safe_stem}.{file_hash[:12]}"


def file_hash_for_content(content: bytes) -> str:
    return sha256_bytes(content)


def workspace_dir(workspace_id: str) -> Path:
    return WORKSPACES_ROOT / workspace_id


def workspace_manifest_path(workspace_id: str) -> Path:
    return workspace_dir(workspace_id) / "manifest.json"


def workspace_records_path(workspace_id: str) -> Path:
    return workspace_dir(workspace_id) / "records.jsonl"


def workspace_embeddings_path(workspace_id: str) -> Path:
    return workspace_dir(workspace_id) / "embeddings.jsonl"


def save_workspace_records(
    workspace_id: str,
    source_name: str,
    file_hash: str,
    records: list[SearchRecord],
) -> WorkspaceInfo:
    ensure_workspaces_root()
    workspace_dir(workspace_id).mkdir(parents=True, exist_ok=True)

    with workspace_records_path(workspace_id).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    existing = load_workspace_info(workspace_id)
    info = WorkspaceInfo(
        workspace_id=workspace_id,
        source_name=source_name,
        file_hash=file_hash,
        record_count=len(records),
        embedding_count=existing.embedding_count if existing else 0,
        token_count=sum(estimate_tokens(record.text) for record in records),
        embedding_token_count=existing.embedding_token_count if existing else 0,
        embed_model=existing.embed_model if existing else None,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_manifest(info)
    return info


def save_workspace_embeddings(workspace_id: str, embeddings: list[EmbeddingRecord], embed_model: str) -> WorkspaceInfo:
    info = load_workspace_info(workspace_id)
    if info is None:
        raise ValueError(f"Workspace not found: {workspace_id}")

    workspace_dir(workspace_id).mkdir(parents=True, exist_ok=True)
    with workspace_embeddings_path(workspace_id).open("w", encoding="utf-8") as handle:
        for record in embeddings:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    updated = WorkspaceInfo(
        workspace_id=info.workspace_id,
        source_name=info.source_name,
        file_hash=info.file_hash,
        record_count=info.record_count,
        embedding_count=len(embeddings),
        token_count=info.token_count,
        embedding_token_count=sum(estimate_tokens(item.record.text) for item in embeddings),
        embed_model=embed_model,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_manifest(updated)
    return updated


def load_workspace_records(workspace_id: str) -> list[SearchRecord]:
    path = workspace_records_path(workspace_id)
    if not path.exists():
        return []
    return [SearchRecord.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_workspace_embeddings(workspace_id: str) -> list[EmbeddingRecord]:
    path = workspace_embeddings_path(workspace_id)
    if not path.exists():
        return []
    return [EmbeddingRecord.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def list_workspaces() -> list[WorkspaceInfo]:
    ensure_workspaces_root()
    items = []
    for manifest in sorted(WORKSPACES_ROOT.glob("*/manifest.json")):
        info = _read_manifest(manifest)
        if info is not None:
            items.append(info)
    return sorted(items, key=lambda item: item.updated_at, reverse=True)


def load_workspace_info(workspace_id: str) -> WorkspaceInfo | None:
    return _read_manifest(workspace_manifest_path(workspace_id))


def delete_workspace(workspace_id: str) -> bool:
    target = workspace_dir(workspace_id)
    if not target.exists():
        return False
    shutil.rmtree(target)
    return True


def estimate_tokens(text: str) -> int:
    # Cheap approximation for progress/cost visibility without adding tokenizer dependencies.
    return max(1, len(text) // 4) if text.strip() else 0


def _write_manifest(info: WorkspaceInfo) -> None:
    workspace_manifest_path(info.workspace_id).write_text(
        json.dumps(asdict(info), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_manifest(path: Path) -> WorkspaceInfo | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return WorkspaceInfo(
        workspace_id=payload["workspace_id"],
        source_name=payload["source_name"],
        file_hash=payload["file_hash"],
        record_count=int(payload.get("record_count", 0)),
        embedding_count=int(payload.get("embedding_count", 0)),
        token_count=int(payload.get("token_count", 0)),
        embedding_token_count=int(payload.get("embedding_token_count", 0)),
        embed_model=payload.get("embed_model"),
        updated_at=payload.get("updated_at", ""),
    )
