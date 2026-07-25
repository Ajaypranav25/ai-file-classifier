"""
Classification layer.

Two modes, chosen automatically:

  1. TRAINED (preferred): a small scikit-learn LogisticRegression trained
     on top of frozen CLIP embeddings (see train/train_classifier.py).
     This is the actual "trained ML model" for this app — cheap to train
     (< 1 second for a few hundred examples), cheap to run, and it gets
     MORE accurate over time as you correct labels in the UI and retrain.

  2. ZERO-SHOT FALLBACK: if no trained model exists yet (fresh install,
     before you've run the bootstrap/train scripts), we classify by
     comparing the file's embedding to CLIP text-embedded category
     prompts from config.yaml. Lower accuracy, but means the app is
     useful from the very first file — no cold start.
"""
from __future__ import annotations
import functools
import numpy as np
import joblib

from app.config import CFG
from app.embed import get_embedder


@functools.lru_cache(maxsize=1)
def _category_centroids() -> np.ndarray:
    """One averaged embedding per category, from its prompt list."""
    embedder = get_embedder()
    centroids = []
    for cat in CFG.categories:
        vecs = embedder.embed_texts(cat.prompts)
        centroid = vecs.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        centroids.append(centroid)
    return np.stack(centroids)


def _zero_shot_classify(embedding: np.ndarray) -> tuple[str, float]:
    centroids = _category_centroids()
    sims = centroids @ embedding  # cosine sim, all vectors are unit-norm
    # softmax over similarities to get a pseudo-confidence
    exp = np.exp((sims - sims.max()) * 10)  # temperature=10 sharpens it
    probs = exp / exp.sum()
    idx = int(probs.argmax())
    return CFG.category_names[idx], float(probs[idx])


class Classifier:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self._try_load_trained()

    def _try_load_trained(self):
        if CFG.classifier_path.exists() and CFG.label_encoder_path.exists():
            self.model = joblib.load(CFG.classifier_path)
            self.label_encoder = joblib.load(CFG.label_encoder_path)

    def reload(self):
        """Call after retraining so a long-running server picks up the new model."""
        self._try_load_trained()

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def classify(self, embedding: np.ndarray) -> tuple[str, float, str]:
        """Returns (category, confidence, source) where source is 'trained' or 'zero-shot'."""
        if self.model is not None:
            probs = self.model.predict_proba(embedding.reshape(1, -1))[0]
            idx = int(probs.argmax())
            label = self.label_encoder.inverse_transform([idx])[0]
            return label, float(probs[idx]), "trained"
        label, conf = _zero_shot_classify(embedding)
        return label, conf, "zero-shot"


@functools.lru_cache(maxsize=1)
def get_classifier() -> Classifier:
    return Classifier()
