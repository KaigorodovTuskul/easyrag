from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

from app.ingestion.docx_parser import FormulaImageAsset
from app.ingestion.formula_vision import prepare_formula_image
from app.storage.workspaces import workspace_dir


@dataclass(slots=True)
class StoredFormulaImage:
    asset_id: str
    filename: str
    relative_path: str
    original_filename: str | None = None

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
        prepared = prepare_formula_image(asset.content, asset.filename)
        if prepared is None:
            suffix = Path(asset.filename).suffix or ".bin"
            target = root / f"{asset.asset_id}{suffix}"
            target.write_bytes(asset.content)
            stored_filename = asset.filename
        else:
            image_bytes, _, rendered_name = prepared
            suffix = Path(rendered_name).suffix or ".png"
            target = root / f"{asset.asset_id}{suffix}"
            target.write_bytes(image_bytes)
            stored_filename = rendered_name
        stored.append(
            StoredFormulaImage(
                asset_id=asset.asset_id,
                filename=stored_filename,
                relative_path=str(target),
                original_filename=asset.filename,
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
            original_filename=item.get("original_filename"),
        )
        items[stored.asset_id] = stored
    return items


def read_formula_image_bytes(item: StoredFormulaImage) -> bytes:
    return Path(item.relative_path).read_bytes()


def read_formula_image_for_display(item: StoredFormulaImage) -> bytes:
    raw = Path(item.relative_path).read_bytes()
    prepared = prepare_formula_image(raw, item.filename)
    image_bytes = raw if prepared is None else prepared[0]
    return _resize_image_bytes(image_bytes)


def is_vector_formula_image(item: StoredFormulaImage) -> bool:
    source_name = item.original_filename or item.filename
    return Path(source_name).suffix.lower() in {".wmf", ".emf"}


def _resize_image_bytes(image_bytes: bytes, max_width: int = 900) -> bytes:
    try:
        from PIL import Image
    except ImportError:
        return image_bytes

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.width <= max_width:
                return image_bytes
            scale = max_width / max(image.width, 1)
            resized = image.resize((max_width, max(1, int(image.height * scale))))
            buffer = BytesIO()
            resized.save(buffer, format="PNG")
            return buffer.getvalue()
    except Exception:
        return image_bytes
