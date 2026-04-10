from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
import subprocess
import tempfile


def recognize_formula_image(blob: bytes, filename: str, backend: str = "none") -> str | None:
    backend = backend.strip().lower()
    if backend in {"", "none", "off", "disabled"}:
        return None
    if backend == "pix2tex":
        return _recognize_with_pix2tex(blob, filename)
    return None


def _recognize_with_pix2tex(blob: bytes, filename: str) -> str | None:
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        image = Image.open(BytesIO(blob)).convert("RGB")
    except Exception:
        rendered = _render_vector_image_to_png(blob, filename)
        if rendered is None:
            return None
        try:
            image = Image.open(BytesIO(rendered)).convert("RGB")
        except Exception:
            return None

    try:
        result = _pix2tex_model()(image)
    except Exception:
        return None

    if not isinstance(result, str):
        return None
    return result.strip() or None


@lru_cache(maxsize=1)
def _pix2tex_model():
    from pix2tex.cli import LatexOCR

    return LatexOCR()


def _render_vector_image_to_png(blob: bytes, filename: str) -> bytes | None:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".wmf", ".emf"}:
        return None

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
