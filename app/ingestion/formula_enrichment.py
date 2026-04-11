from __future__ import annotations

from dataclasses import dataclass

from app.agents.embedding_indexer import build_embedding_index
from app.core.config import AppConfig
from app.providers.base import BaseProvider
from app.retrieval.records import SearchRecord
from app.retrieval.vector import EmbeddingRecord
from app.storage.formula_images import StoredFormulaImage, read_formula_image_bytes

from app.ingestion.formula_vision import recognize_formula_image


@dataclass(slots=True)
class FormulaEnrichmentResult:
    created_records: list[SearchRecord]
    created_embeddings: list[EmbeddingRecord]
    attempted_assets: int
    recognized_assets: int


def enrich_formula_records(
    provider: BaseProvider,
    records: list[SearchRecord],
    formula_images: dict[str, StoredFormulaImage],
    model: str | None,
    embed_model: str | None,
    config: AppConfig,
) -> FormulaEnrichmentResult:
    if not model:
        return FormulaEnrichmentResult(created_records=[], created_embeddings=[], attempted_assets=0, recognized_assets=0)

    parent_by_asset = _pick_formula_parents(records)
    existing_asset_ids = {
        str(record.metadata.get("asset_id"))
        for record in records
        if record.record_type == "formula_text" and record.metadata.get("asset_id")
    }

    created_records: list[SearchRecord] = []
    attempted_assets = 0
    recognized_assets = 0

    for asset_id, parent in parent_by_asset.items():
        if asset_id in existing_asset_ids:
            continue
        image = formula_images.get(asset_id)
        if image is None:
            continue

        attempted_assets += 1
        recognized = recognize_formula_image(provider, read_formula_image_bytes(image), image.filename, model=model)
        if not recognized:
            continue

        recognized_assets += 1
        created_records.append(
            SearchRecord(
                record_id=f"{parent.record_id}:formula:{asset_id}",
                source_name=parent.source_name,
                record_type="formula_text",
                section_path=parent.section_path,
                text=f"Formula extracted from image: {recognized}",
                metadata={
                    "asset_id": asset_id,
                    "filename": image.filename,
                    "parent_record_id": parent.record_id,
                    "formula_image_ids": [asset_id],
                },
            )
        )

    created_embeddings: list[EmbeddingRecord] = []
    if created_records and embed_model:
        created_embeddings = build_embedding_index(
            provider=provider,
            records=created_records,
            model=embed_model,
            batch_size=min(config.embedding_batch_size, max(1, len(created_records))),
            record_types=("formula_text",),
        )

    return FormulaEnrichmentResult(
        created_records=created_records,
        created_embeddings=created_embeddings,
        attempted_assets=attempted_assets,
        recognized_assets=recognized_assets,
    )


def _pick_formula_parents(records: list[SearchRecord]) -> dict[str, SearchRecord]:
    priority = {"table_cell": 0, "paragraph": 1, "table_row": 2, "table": 3}
    selected: dict[str, SearchRecord] = {}

    for record in records:
        asset_ids = record.metadata.get("formula_image_ids", [])
        if not isinstance(asset_ids, list) or not asset_ids:
            continue
        if record.record_type not in priority:
            continue

        for asset_id in asset_ids:
            existing = selected.get(asset_id)
            if existing is None or priority.get(record.record_type, 99) < priority.get(existing.record_type, 99):
                selected[asset_id] = record

    return selected
