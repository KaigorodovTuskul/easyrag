from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.ingestion.docx_parser import FormulaImageAsset
from app.storage.workspaces import workspace_dir


@dataclass(slots=True)
class StoredFormulaImage:
    asset_id: str
    filename: str
    relative_path: str

    @property
    def path(self) -> Path:
        return Path(self.relative_path)


def formula_images_root(workspace_id: str) -> Path:
    return workspace_dir(workspace_id) / "formula_images"


def formula_images_manifest_path(workspace_id: str) -> Path:
    return formula_images_root(workspace_id) / "manifest.json"


def save_workspace_formula_images(workspace_id: str, assets: list[FormulaImageAsset]) -> list[StoredFormulaImage]:
    root = formula_images_root(workspace_id)
    root.mkdir(parents=True, exist_ok=True)
    stored: list[StoredFormulaImage] = []

    for asset in assets:
        suffix = Path(asset.filename).suffix or ".bin"
        target = root / f"{asset.asset_id}{suffix}"
        target.write_bytes(asset.content)
        stored.append(
            StoredFormulaImage(
                asset_id=asset.asset_id,
                filename=asset.filename,
                relative_path=str(target),
            )
        )

    formula_images_manifest_path(workspace_id).write_text(
        json.dumps([asdict(item) for item in stored], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return stored


def load_workspace_formula_images(workspace_id: str) -> dict[str, StoredFormulaImage]:
    path = formula_images_manifest_path(workspace_id)
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    items: dict[str, StoredFormulaImage] = {}
    for item in payload:
        stored = StoredFormulaImage(
            asset_id=item["asset_id"],
            filename=item["filename"],
            relative_path=item["relative_path"],
        )
        items[stored.asset_id] = stored
    return items
