"""
process_file(path) is the single entrypoint the watcher and the backfill
script both call. Keeping it here (rather than inline in watcher.py) means
train/bootstrap_labels.py can also reuse the exact same extract+embed logic.
"""
from __future__ import annotations
import hashlib
import time
from pathlib import Path

from PIL import Image

from app.config import CFG
from app.extract import extract
from app.embed import get_embedder
from app.classify import get_classifier
from app.store import Store, FileRecord

THUMB_SIZE = (320, 320)


def _file_id(path: Path) -> str:
    # Stable id derived from the absolute path so re-processing the same
    # file (e.g. after edit) upserts instead of duplicating.
    return hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:16]


def _make_thumbnail(path: Path, file_id: str) -> str:
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail(THUMB_SIZE)
        out = CFG.thumbnails_dir / f"{file_id}.jpg"
        img.save(out, "JPEG", quality=82)
        return str(out)
    except Exception:
        return ""


def is_ignored(path: Path) -> bool:
    if path.name.lower() in {".ds_store"}:
        return True
    if path.suffix.lower() in CFG.ignore_extensions:
        return True
    if path.name.startswith("."):
        return True
    return False


def process_file(path: Path, store: Store | None = None) -> dict | None:
    """Runs the full pipeline for one file and upserts it into the store.
    Returns the record dict, or None if the file was skipped."""
    if not path.exists() or not path.is_file() or is_ignored(path):
        return None

    store = store or Store()
    embedder = get_embedder()
    classifier = get_classifier()

    extracted = extract(path)
    file_id = _file_id(path)

    if extracted.kind == "image":
        if extracted.raw_bytes is None:
            return None
        vector = embedder.embed_image(extracted.raw_bytes)
        thumb = _make_thumbnail(path, file_id)
    else:
        # For text-bearing files we embed the extracted text through the
        # SAME CLIP text tower, landing in the same vector space as images.
        text_for_embedding = extracted.text or path.stem
        vector = embedder.embed_text(text_for_embedding)
        thumb = ""

    category, confidence, source = classifier.classify(vector)

    stat = path.stat()
    record = FileRecord(
        id=file_id,
        filepath=str(path.resolve()),
        filename=path.name,
        ext=path.suffix.lower(),
        category=category,
        confidence=confidence,
        classifier_source=source,
        ocr_text=extracted.text[:5000],
        thumbnail_path=thumb,
        created_at=stat.st_mtime,
        indexed_at=time.time(),
        vector=vector.tolist(),
    )
    store.upsert(record)
    return {
        "id": record.id,
        "filename": record.filename,
        "category": record.category,
        "confidence": record.confidence,
    }


def wait_until_stable(path: Path, wait_seconds: float | None = None) -> bool:
    """Polls file size until it stops changing, so we don't index a file
    that's still being written (screenshot saving, download in progress).
    Returns False if the file disappeared before stabilizing."""
    wait_seconds = wait_seconds if wait_seconds is not None else CFG.stability_wait_seconds
    last_size = -1
    while True:
        if not path.exists():
            return False
        size = path.stat().st_size
        if size == last_size:
            return True
        last_size = size
        time.sleep(wait_seconds)
