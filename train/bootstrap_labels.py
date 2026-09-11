"""
Step 1 of training: generate an initial labels.csv for every file already
in your watched folders, using zero-shot CLIP classification (no model
training yet — just comparing embeddings to category text prompts).

This solves the cold-start problem: you don't need to hand-label hundreds
of screenshots before you have anything to train on. You just need to
*review and fix* the low-confidence ones, which this script sorts to the
top for you.

Usage:
    python -m train.bootstrap_labels

Output:
    data/labels.csv  (filepath, category, confidence)

Then run `python -m train.review_labels` to quickly correct the
low-confidence rows, and `python -m train.train_classifier` to train.
"""
from __future__ import annotations
import csv

from app.config import CFG
from app.extract import extract
from app.embed import get_embedder
from app.classify import _zero_shot_classify
from app.pipeline import is_ignored


def iter_watch_folder_files():
    for folder in CFG.watch_folders:
        if not folder.exists():
            print(f"  (skipping {folder}, does not exist)")
            continue
        for path in folder.rglob("*"):
            if path.is_file() and not is_ignored(path):
                yield path


def main():
    embedder = get_embedder()
    rows = []

    files = list(iter_watch_folder_files())
    print(f"Found {len(files)} files to bootstrap-label...")

    for i, path in enumerate(files, 1):
        try:
            extracted = extract(path)
            if extracted.kind == "image":
                vec = embedder.embed_image(extracted.raw_bytes)
            else:
                vec = embedder.embed_text(extracted.text or path.stem)
            category, confidence = _zero_shot_classify(vec)
            rows.append((str(path.resolve()), category, round(confidence, 4)))
        except Exception as e:
            print(f"  ! skipped {path.name}: {e}")

        if i % 25 == 0:
            print(f"  ...{i}/{len(files)}")

    # Lowest confidence first — these are the ones most worth reviewing.
    rows.sort(key=lambda r: r[2])

    CFG.labels_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(CFG.labels_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "category", "source"])
        for filepath, category, confidence in rows:
            w.writerow([filepath, category, f"zero-shot({confidence})"])

    print(f"\nWrote {len(rows)} bootstrap labels to {CFG.labels_csv}")
    print("Next: python -m train.review_labels   (fix the low-confidence ones)")
    print("Then: python -m train.train_classifier")


if __name__ == "__main__":
    main()
