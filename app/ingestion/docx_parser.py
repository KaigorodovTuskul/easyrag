from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
import re
from typing import Any, Callable, Iterator

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


def parse_docx_bytes(
    source_name: str,
    content: bytes,
    formula_image_recognizer: Callable[[bytes, str], str | None] | None = None,
) -> ParsedDocx:
    document = Document(BytesIO(content))
    section_path: list[str] = []
    paragraphs: list[ParagraphElement] = []
    tables: list[TableElement] = []
    paragraph_index = 0
    table_index = 0

    for block in _iter_block_items(document):
        if isinstance(block, Paragraph):
            text = _normalize_paragraph_text(block, formula_image_recognizer)
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
                    cell_text = _normalize_cell_text(cell, formula_image_recognizer)
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


def _normalize_cell_text(
    cell: _Cell,
    formula_image_recognizer: Callable[[bytes, str], str | None] | None,
) -> str:
    parts = [_normalize_paragraph_text(paragraph, formula_image_recognizer) for paragraph in cell.paragraphs]
    parts = [part for part in parts if part]
    return "\n".join(parts)


def _normalize_paragraph_text(
    paragraph: Paragraph,
    formula_image_recognizer: Callable[[bytes, str], str | None] | None,
) -> str:
    parts = [paragraph.text.strip()]
    parts.extend(_extract_omml_formulas(paragraph))
    parts.extend(_extract_formula_image_markers(paragraph, formula_image_recognizer))
    return "\n".join(part for part in parts if part).strip()


def _extract_omml_formulas(paragraph: Paragraph) -> list[str]:
    formulas: list[str] = []
    for formula in paragraph._p.xpath(
        './/*[local-name()="oMathPara" or (local-name()="oMath" and not(ancestor::*[local-name()="oMathPara"]))]'
    ):
        text = _compact_formula_text(_omml_to_text(formula))
        if text:
            formulas.append(f"[FORMULA_OMML: {text}]")
    return formulas


def _extract_formula_image_markers(
    paragraph: Paragraph,
    formula_image_recognizer: Callable[[bytes, str], str | None] | None,
) -> list[str]:
    markers: list[str] = []
    for image_index, relationship_id in enumerate(paragraph._p.xpath(".//a:blip/@r:embed"), start=1):
        image_part = paragraph.part.related_parts.get(relationship_id)
        filename = Path(str(image_part.partname)).name if image_part is not None else f"image-{image_index}"
        recognized = formula_image_recognizer(image_part.blob, filename) if image_part is not None and formula_image_recognizer else None
        if recognized:
            markers.append(f"[FORMULA_VISION: {recognized}]")
        else:
            markers.append(f"[FORMULA_IMAGE: {filename}; not recognized]")
    return markers


def _omml_to_text(element) -> str:
    if element is None:
        return ""

    name = _local_name(element)

    if name == "t":
        return element.text or ""

    if name == "f":
        numerator = _first_child_by_name(element, "num")
        denominator = _first_child_by_name(element, "den")
        return f"({_omml_to_text(numerator)}) / ({_omml_to_text(denominator)})"

    if name == "sSup":
        base = _first_child_by_name(element, "e")
        superscript = _first_child_by_name(element, "sup")
        return f"{_omml_to_text(base)}^{_omml_to_text(superscript)}"

    if name == "sSub":
        base = _first_child_by_name(element, "e")
        subscript = _first_child_by_name(element, "sub")
        return f"{_omml_to_text(base)}_{_omml_to_text(subscript)}"

    if name == "sSubSup":
        base = _first_child_by_name(element, "e")
        subscript = _first_child_by_name(element, "sub")
        superscript = _first_child_by_name(element, "sup")
        return f"{_omml_to_text(base)}_{_omml_to_text(subscript)}^{_omml_to_text(superscript)}"

    if name in {"num", "den", "e", "lim", "sup", "sub"}:
        return "".join(_omml_to_text(child) for child in element)

    if name == "r":
        return "".join(_omml_to_text(child) for child in element)

    return " ".join(_omml_to_text(child) for child in element)


def _first_child_by_name(element, name: str):
    for child in element:
        if _local_name(child) == name:
            return child
    return None


def _local_name(element) -> str:
    if element is None:
        return ""
    return element.tag.rsplit("}", 1)[-1]


def _compact_formula_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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
