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
    prepared = prepare_formula_image(blob, filename)
    if prepared is None:
        return None

    data, mime_type, _ = prepared
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


def prepare_formula_image(blob: bytes, filename: str) -> tuple[bytes, str, str] | None:
    suffix = Path(filename).suffix.lower()
    if suffix in {".wmf", ".emf"}:
        rendered = _render_vector_image_to_png(blob, suffix)
        if rendered is None:
            return None
        rendered_name = f"{Path(filename).stem}.png"
        return rendered, "image/png", rendered_name

    mime_type = _mime_type_for_suffix(suffix)
    if mime_type is None:
        return None
    return blob, mime_type, Path(filename).name


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
    rendered = _render_vector_image_to_png_with_powerpoint(blob, suffix)
    if rendered is not None:
        return rendered

    return _render_vector_image_to_png_with_system_drawing(blob, suffix)


def _render_vector_image_to_png_with_powerpoint(blob: bytes, suffix: str) -> bytes | None:
    with tempfile.TemporaryDirectory(prefix="easyrag_formula_ppt_") as temp_dir:
        source_path = Path(temp_dir) / f"source{suffix}"
        output_path = Path(temp_dir) / "rendered.png"
        source_path.write_bytes(blob)

        script = rf"""
$ErrorActionPreference = 'Stop'
$ppt = $null
$pres = $null
try {{
  $ppt = New-Object -ComObject PowerPoint.Application
  $ppt.Visible = -1
  $pres = $ppt.Presentations.Add()
  $slide = $pres.Slides.Add(1, 12)
  $shape = $slide.Shapes.AddPicture('{_escape_powershell_path(source_path)}', $false, $true, 0, 0, -1, -1)
  $shape.LockAspectRatio = -1
  $targetWidth = [int][Math]::Max([Math]::Round($shape.Width), 1)
  $targetHeight = [int][Math]::Max([Math]::Round($shape.Height), 1)
  $pres.PageSetup.SlideWidth = $targetWidth
  $pres.PageSetup.SlideHeight = $targetHeight
  $shape.Left = 0
  $shape.Top = 0
  $shape.Width = $targetWidth
  $shape.Height = $targetHeight
  $slide.Export('{_escape_powershell_path(output_path)}', 'PNG', $targetWidth, $targetHeight)
}} finally {{
  if ($pres -ne $null) {{ $pres.Close() }}
  if ($ppt -ne $null) {{ $ppt.Quit() }}
}}
"""
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0 or not output_path.exists():
            return None
        return output_path.read_bytes()


def _render_vector_image_to_png_with_system_drawing(blob: bytes, suffix: str) -> bytes | None:
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
