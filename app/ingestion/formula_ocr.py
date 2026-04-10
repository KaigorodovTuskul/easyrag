from __future__ import annotations

from functools import lru_cache
from io import BytesIO


def recognize_formula_image(blob: bytes, filename: str, backend: str = "none") -> str | None:
    backend = backend.strip().lower()
    if backend in {"", "none", "off", "disabled"}:
        return None
    if backend == "pix2tex":
        return _recognize_with_pix2tex(blob, filename)
    return None


def _recognize_with_pix2tex(blob: bytes, filename: str) -> str | None:
    if filename.lower().endswith((".wmf", ".emf")):
        return None

    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        image = Image.open(BytesIO(blob)).convert("RGB")
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
