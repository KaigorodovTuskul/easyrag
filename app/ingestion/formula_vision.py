from __future__ import annotations

from pathlib import Path
import re
import subprocess
import tempfile

from app.core.models import ImageInput
from app.providers.base import BaseProvider


_FORMULA_IMAGE_PROMPT = (
    "Extract the mathematical or regulatory formula from this image. "
    "If the image does not clearly contain a formula, reply exactly NOT_FORMULA. "
    "Otherwise return only the formula on a single line with no explanation. "
    "Preserve fractions, subscripts, superscripts, Greek letters, inequality signs, and variable names."
)


def recognize_formula_image(provider: BaseProvider, blob: bytes, filename: str, model: str | None = None) -> str | None:
    prepared = _prepare_image(blob, filename)
    if prepared is None:
        return None

    data, mime_type = prepared
    try:
        result = provider.generate_with_images(
            _FORMULA_IMAGE_PROMPT,
            [ImageInput(data=data, mime_type=mime_type)],
            model=model,
        )
    except Exception:
        return None

    return _normalize_formula_text(result.text)


def is_plausible_formula_text(value: str) -> bool:
    normalized = _clean_formula_text(value)
    if normalized is None or len(normalized) < 3:
        return False
    if len(normalized) > 400:
        return False
    if re.search(r"\b(not_formula|no formula|cannot determine|unable to read)\b", normalized, flags=re.IGNORECASE):
        return False

    has_formula_signal = any(token in normalized for token in ["=", "+", "-", "/", "*", "^", "_", "\\", "(", ")", "[", "]", "{", "}"])
    if not has_formula_signal and not re.search(r"[A-Za-zА-Яа-я]\s*[<>≤≥]\s*[A-Za-zА-Яа-я0-9]", normalized):
        return False

    if re.search(r"[.!?]\s+[A-ZА-Я]", normalized):
        return False

    return re.search(r"[A-Za-zА-Яа-я0-9]", normalized) is not None


def _normalize_formula_text(value: str | None) -> str | None:
    normalized = _clean_formula_text(value)
    if normalized is None:
        return None
    if normalized.upper() == "NOT_FORMULA":
        return None
    if not is_plausible_formula_text(normalized):
        return None
    return normalized


def _clean_formula_text(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    normalized = normalized.strip("`")
    normalized = re.sub(r"^\s*(formula|latex|expression)\s*:\s*", "", normalized, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", normalized).strip()


def _prepare_image(blob: bytes, filename: str) -> tuple[bytes, str] | None:
    suffix = Path(filename).suffix.lower()
    if suffix in {".wmf", ".emf"}:
        rendered = _render_vector_image_to_png(blob, suffix)
        if rendered is None:
            return None
        return rendered, "image/png"

    mime_type = _mime_type_for_suffix(suffix)
    if mime_type is None:
        return None
    return blob, mime_type


def _mime_type_for_suffix(suffix: str) -> str | None:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
    }.get(suffix)


def _render_vector_image_to_png(blob: bytes, suffix: str) -> bytes | None:
    with tempfile.TemporaryDirectory(prefix="easyrag_formula_") as temp_dir:
        source_path = Path(temp_dir) / f"source{suffix}"
        output_path = Path(temp_dir) / "rendered.png"
        source_path.write_bytes(blob)

        script = rf"""
Add-Type -AssemblyName System.Drawing
$img = [System.Drawing.Image]::FromFile('{_escape_powershell_path(source_path)}')
$bmp = New-Object System.Drawing.Bitmap $img.Width, $img.Height
$graphics = [System.Drawing.Graphics]::FromImage($bmp)
$graphics.Clear([System.Drawing.Color]::White)
$graphics.DrawImage($img, 0, 0, $img.Width, $img.Height)
$bmp.Save('{_escape_powershell_path(output_path)}', [System.Drawing.Imaging.ImageFormat]::Png)
$graphics.Dispose()
$bmp.Dispose()
$img.Dispose()
"""
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0 or not output_path.exists():
            return None
        return output_path.read_bytes()


def _escape_powershell_path(path: Path) -> str:
    return str(path).replace("'", "''")
