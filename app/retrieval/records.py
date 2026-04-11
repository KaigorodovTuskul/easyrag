from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.ingestion.docx_parser import ParsedDocx


@dataclass(slots=True)
class SearchRecord:
    record_id: str
    source_name: str
    record_type: str
    section_path: list[str]
    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SearchRecord":
        return cls(
            record_id=payload["record_id"],
            source_name=payload["source_name"],
            record_type=payload["record_type"],
            section_path=list(payload.get("section_path", [])),
            text=payload["text"],
            metadata=dict(payload.get("metadata", {})),
        )


def build_search_records(parsed: ParsedDocx) -> list[SearchRecord]:
    records: list[SearchRecord] = []

    for paragraph in parsed.paragraphs:
        records.append(
            SearchRecord(
                record_id=paragraph.element_id,
                source_name=parsed.source_name,
                record_type="paragraph",
                section_path=paragraph.section_path,
                text=paragraph.text,
                metadata={
                    "style": paragraph.style,
                    "formula_image_ids": paragraph.formula_image_ids,
                },
            )
        )

    for table in parsed.tables:
        table_text = _table_to_text(table.rows)
        records.append(
            SearchRecord(
                record_id=table.element_id,
                source_name=parsed.source_name,
                record_type="table",
                section_path=table.section_path,
                text=table_text,
                metadata={
                    "row_count": table.row_count,
                    "col_count": table.col_count,
                    "formula_image_ids": list(dict.fromkeys(asset_id for row in table.rows for asset_id in row.formula_image_ids)),
                },
            )
        )

        header = table.rows[0].values if table.rows else []
        for row in table.rows:
            row_text = _row_to_text(header, row.values)
            records.append(
                SearchRecord(
                    record_id=f"{table.element_id}:r-{row.row_index}",
                    source_name=parsed.source_name,
                    record_type="table_row",
                    section_path=table.section_path,
                    text=row_text,
                    metadata={
                        "table_id": table.element_id,
                        "row_index": row.row_index,
                        "formula_image_ids": row.formula_image_ids,
                    },
                )
            )

            if row.row_index == 0:
                continue

            for cell_index, cell_value in enumerate(row.values):
                if not cell_value.strip():
                    continue
                header_value = header[cell_index] if cell_index < len(header) else f"column_{cell_index}"
                records.append(
                    SearchRecord(
                        record_id=f"{table.element_id}:r-{row.row_index}:c-{cell_index}",
                        source_name=parsed.source_name,
                        record_type="table_cell",
                        section_path=table.section_path,
                        text=f"{header_value}: {cell_value}",
                        metadata={
                            "table_id": table.element_id,
                            "row_index": row.row_index,
                            "col_index": cell_index,
                            "header": header_value,
                            "formula_image_ids": row.cells[cell_index].formula_image_ids,
                        },
                    )
                )

    return records


def _table_to_text(rows) -> str:
    return "\n".join(" | ".join(row.values) for row in rows)


def _row_to_text(header: list[str], values: list[str]) -> str:
    if header and len(header) == len(values):
        pairs = [f"{name}: {value}" for name, value in zip(header, values) if name or value]
        return " | ".join(pairs)
    return " | ".join(values)
