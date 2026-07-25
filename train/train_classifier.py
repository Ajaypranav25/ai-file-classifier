"""
Step 3 of training: fit the actual classifier.

Model choice — logistic regression on frozen CLIP embeddings, not a deep
net trained end-to-end:

  - Your labeled set is small (tens to low-thousands of examples), which is
    exactly the regime where a linear model on strong frozen features
    generalizes better than a deep model trained from scratch — a CNN/ViT
    trained end-to-end on a few hundred screenshots would badly overfit.
  - Training is CPU, sub-second to a few seconds, so you can retrain after
    every correction you make in the UI with zero friction.
  - It gives calibrated-ish probabilities via predict_proba, which the
    app uses to flag low-confidence files as "needs review".

If you outgrow this later (thousands of examples, many nuanced categories),
swap LogisticRegression for a small 2-layer MLP (sklearn.neural_network.
MLPClassifier is a drop-in upgrade) — the surrounding pipeline doesn't change.

Usage:
    python -m train.train_classifier
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

from app.config import CFG
from app.extract import extract
from app.embed import get_embedder


def load_labeled_examples():
    if not CFG.labels_csv.exists():
        raise SystemExit(
            "No labels.csv found. Run:\n"
            "  python -m train.bootstrap_labels\n"
            "  python -m train.review_labels   (recommended)\n"
            "first."
        )
    with open(CFG.labels_csv) as f:
        rows = list(csv.DictReader(f))
    # Drop rows whose file no longer exists (moved/deleted since bootstrap).
    rows = [r for r in rows if Path(r["filepath"]).exists()]
    return rows


def main():
    rows = load_labeled_examples()
    if len(rows) < 20:
        print(f"Only {len(rows)} labeled examples found — that's thin. "
              f"The classifier will still train, but consider reviewing more "
              f"via `python -m train.review_labels --n 100` first.\n")

    embedder = get_embedder()
    print(f"Embedding {len(rows)} labeled files...")

    X, y = [], []
    for i, row in enumerate(rows, 1):
        path = Path(row["filepath"])
        try:
            extracted = extract(path)
            if extracted.kind == "image":
                vec = embedder.embed_image(extracted.raw_bytes)
            else:
                vec = embedder.embed_text(extracted.text or path.stem)
            X.append(vec)
            y.append(row["category"])
        except Exception as e:
            print(f"  ! skipped {path.name}: {e}")
        if i % 25 == 0:
            print(f"  ...{i}/{len(rows)}")

    X = np.stack(X)
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    n_classes = len(encoder.classes_)
    print(f"\nTraining LogisticRegression on {len(X)} examples, {n_classes} categories...")

    model = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")

    # 5-fold CV accuracy as a sanity check, if we have enough data per class
    min_class_count = min(np.bincount(y_enc))
    if min_class_count >= 5:
        scores = cross_val_score(model, X, y_enc, cv=5)
        print(f"Cross-val accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    else:
        print(f"(skipping cross-validation — smallest category has only "
              f"{min_class_count} examples; label a few more of those for a "
              f"reliable accuracy estimate)")

    model.fit(X, y_enc)

    CFG.classifier_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CFG.classifier_path)
    joblib.dump(encoder, CFG.label_encoder_path)

    print(f"\nSaved classifier -> {CFG.classifier_path}")
    print(f"Saved label encoder -> {CFG.label_encoder_path}")
    print("\nIf the app server is already running, hot-reload the new model with:")
    print(f"  curl -X POST http://{CFG.api_host}:{CFG.api_port}/api/reload-classifier")


if __name__ == "__main__":
    main()
