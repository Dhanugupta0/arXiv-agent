"""
Node 4: PDF download + multi-engine text extraction.

Failure handling is layered, cheapest first:
  1. PyMuPDF  — fast, good two-column reading order
  2. pypdf    — different engine, sometimes succeeds where PyMuPDF fails
  3. OCR      — pdf2image + pytesseract (optional system deps)
  4. Abstract-only degradation — never a hard crash
"""
from __future__ import annotations

from typing import Optional, Tuple

import requests

try:
    import pymupdf as fitz  # PyMuPDF (new import name)
except ImportError:
    fitz = None

try:
    import pypdf
except ImportError:
    pypdf = None


MIN_HEALTHY_CHARS = 1000
MAX_REPLACEMENT_CHAR_RATIO = 0.02


def download_pdf(url: str, timeout: int = 30) -> Optional[bytes]:
    """Download a PDF from a URL.  Returns None on failure."""
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "arxiv-digest-agent/1.0"},
        )
        resp.raise_for_status()
        return resp.content
    except requests.RequestException:
        return None


def _extract_pymupdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []
    for page in doc:
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
        parts.extend(b[4] for b in blocks)
    return "\n".join(parts)


def _extract_pypdf(pdf_bytes: bytes) -> str:
    import io
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _extract_ocr(pdf_bytes: bytes) -> Optional[str]:
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except ImportError:
        return None
    try:
        images = convert_from_bytes(pdf_bytes, dpi=200)
        return "\n".join(pytesseract.image_to_string(img) for img in images)
    except Exception:
        return None


def is_text_healthy(text: str) -> bool:
    """Check whether extracted text is usable (not garbled/empty)."""
    if not text or len(text) < MIN_HEALTHY_CHARS:
        return False
    replacement_ratio = text.count("\ufffd") / max(len(text), 1)
    if replacement_ratio > MAX_REPLACEMENT_CHAR_RATIO:
        return False
    alnum = sum(c.isalnum() for c in text)
    if alnum / max(len(text), 1) < 0.4:
        return False
    return True


def parse_pdf(pdf_bytes: bytes) -> Tuple[str, str, bool]:
    """
    Try all extraction engines in order.
    Returns (text, method_used, healthy).
    """
    text = ""

    if fitz is not None:
        try:
            text = _extract_pymupdf(pdf_bytes)
            if is_text_healthy(text):
                return text, "pymupdf", True
        except Exception:
            text = ""

    if pypdf is not None:
        try:
            text2 = _extract_pypdf(pdf_bytes)
            if is_text_healthy(text2):
                return text2, "pypdf", True
        except Exception:
            text2 = ""
    else:
        text2 = ""

    ocr_text = _extract_ocr(pdf_bytes)
    if ocr_text and is_text_healthy(ocr_text):
        return ocr_text, "ocr", True

    # Nothing worked — return the longest attempt, marked unhealthy
    best = max([text, text2, ocr_text or ""], key=len)
    return best, "failed", False
