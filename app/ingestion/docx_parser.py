from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
import re
from typing import Any, Iterator

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph


@dataclass(slots=True)
class ParagraphElement:
    element_id: str
    element_type: str
    section_path: list[str]
    style: str
    text: str


@dataclass(slots=True)
class TableCellElement:
    row_index: int
    col_index: int
    text: str


@dataclass(slots=True)
class TableRowElement:
    row_index: int
    values: list[str]
    cells: list[TableCellElement] = field(default_factory=list)


@dataclass(slots=True)
class TableElement:
    element_id: str
    element_type: str
    section_path: list[str]
    row_count: int
    col_count: int
    rows: list[TableRowElement] = field(default_factory=list)


@dataclass(slots=True)
class ParsedDocx:
    source_name: str
    paragraph_count: int
    table_count: int
    paragraphs: list[ParagraphElement] = field(default_factory=list)
    tables: list[TableElement] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_docx_bytes(source_name: str, content: bytes) -> ParsedDocx:
    document = Document(BytesIO(content))
    section_path: list[str] = []
    paragraphs: list[ParagraphElement] = []
    tables: list[TableElement] = []
    paragraph_index = 0
    table_index = 0

    for block in _iter_block_items(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue

            style_name = block.style.name if block.style is not None else "Normal"
            heading_level = _detect_heading_level(style_name, text, paragraph_index)
            if heading_level is not None:
                section_path = section_path[: max(heading_level - 1, 0)]
                section_path.append(text)

            paragraph_index += 1
            paragraphs.append(
                ParagraphElement(
                    element_id=f"p-{paragraph_index}",
                    element_type="paragraph",
                    section_path=section_path[:],
                    style=style_name,
                    text=text,
                )
            )
            continue

        if isinstance(block, Table):
            table_index += 1
            table_rows: list[TableRowElement] = []
            col_count = 0

            for row_idx, row in enumerate(block.rows):
                values: list[str] = []
                row_cells: list[TableCellElement] = []

                for col_idx, cell in enumerate(row.cells):
                    cell_text = _normalize_cell_text(cell)
                    values.append(cell_text)
                    row_cells.append(TableCellElement(row_index=row_idx, col_index=col_idx, text=cell_text))

                col_count = max(col_count, len(values))
                table_rows.append(
                    TableRowElement(
                        row_index=row_idx,
                        values=values,
                        cells=row_cells,
                    )
                )

            tables.append(
                TableElement(
                    element_id=f"t-{table_index}",
                    element_type="table",
                    section_path=section_path[:],
                    row_count=len(table_rows),
                    col_count=col_count,
                    rows=table_rows,
                )
            )

    return ParsedDocx(
        source_name=source_name,
        paragraph_count=len(paragraphs),
        table_count=len(tables),
        paragraphs=paragraphs,
        tables=tables,
    )


def _iter_block_items(parent: DocxDocument | _Cell) -> Iterator[Paragraph | Table]:
    if isinstance(parent, DocxDocument):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._tc

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def _normalize_cell_text(cell: _Cell) -> str:
    parts = [paragraph.text.strip() for paragraph in cell.paragraphs if paragraph.text.strip()]
    return "\n".join(parts)


def _detect_heading_level(style_name: str, text: str, paragraph_index: int) -> int | None:
    if style_name.startswith("Heading"):
        return _parse_heading_level(style_name)

    if re.match(r"^Глава\s+\d+[\.\s]", text, flags=re.IGNORECASE):
        return 1

    if re.match(r"^Приложение\s+\d+", text, flags=re.IGNORECASE):
        return 1

    if style_name == "ConsPlusTitle" and paragraph_index > 8 and _looks_like_title(text):
        return 2

    return None


def _parse_heading_level(style_name: str) -> int | None:
    tokens = style_name.split()
    if len(tokens) < 2:
        return 1
    try:
        return int(tokens[-1])
    except ValueError:
        return 1


def _looks_like_title(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False

    upper_letters = [char for char in letters if char.isupper()]
    upper_ratio = len(upper_letters) / len(letters)
    return upper_ratio > 0.8 and len(text) <= 180
