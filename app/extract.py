"""
Turns a raw file on disk into:
  - kind: "image" | "text" | "other"
  - raw_bytes (for images, used for both OCR and CLIP image embedding)
  - text: best-effort extracted/OCR'd text, used for keyword search and
          for embedding non-image files through the CLIP text tower.

Keeping extraction separate from embedding/classification means adding a
new file type later (e.g. .pptx) only touches this one file.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import pytesseract

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif", ".heic"}
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}
TEXT_EXTS = {".txt", ".md", ".csv", ".log", ".json"}


@dataclass
class Extracted:
    kind: str          # "image" | "text" | "other"
    text: str           # extracted / OCR'd text, "" if none
    raw_bytes: bytes | None  # present for images (used for CLIP image embedding)


def _ocr_image(path: Path) -> str:
    try:
        img = Image.open(path)
        return pytesseract.image_to_string(img).strip()
    except Exception:
        # OCR is best-effort — a missing tesseract binary or a corrupt
        # image should not crash the pipeline, just yield no OCR text.
        return ""


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        chunks = []
        for page in reader.pages[:10]:  # cap for speed on huge PDFs
            chunks.append(page.extract_text() or "")
        return "\n".join(chunks).strip()
    except Exception:
        return ""


def _extract_docx(path: Path) -> str:
    try:
        import docx
        d = docx.Document(str(path))
        return "\n".join(p.text for p in d.paragraphs).strip()
    except Exception:
        return ""


def _extract_plain_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")[:20000]
    except Exception:
        return ""


def extract(path: Path) -> Extracted:
    ext = path.suffix.lower()

    if ext in IMAGE_EXTS:
        try:
            raw = path.read_bytes()
        except Exception:
            # Fall back to filename if the image file cannot be read
            return Extracted(kind="other", text=path.stem.replace("_", " ").replace("-", " "), raw_bytes=None)

        text = _ocr_image(path)
        return Extracted(kind="image", text=text, raw_bytes=raw)

    if ext in PDF_EXTS:
        return Extracted(kind="text", text=_extract_pdf(path), raw_bytes=None)

    if ext in DOCX_EXTS:
        return Extracted(kind="text", text=_extract_docx(path), raw_bytes=None)

    if ext in TEXT_EXTS:
        return Extracted(kind="text", text=_extract_plain_text(path), raw_bytes=None)

    # Unknown type: fall back to the filename itself as "text" so it's
    # still searchable and gets *some* embedding, just a weak one.
    return Extracted(kind="other", text=path.stem.replace("_", " ").replace("-", " "), raw_bytes=None)
