"""
Embedding layer.

Architectural choice: we use a single pretrained CLIP model (via open_clip)
for BOTH images and text, instead of training a model from scratch or using
two separate embedding models. Why:

  1. Zero labeled data required to get a usable semantic space on day one.
  2. Images and text share the same 512-dim vector space, so a text query
     like "receipt from the coffee shop" can be compared directly against
     image embeddings without a translation step.
  3. It's the frozen backbone the classifier (train/train_classifier.py)
     is trained on top of — small, fast, retrainable in seconds, instead
     of a slow, data-hungry model trained end-to-end.

The model is loaded once per process (singleton) since it's the expensive part.
"""
from __future__ import annotations
import io
import functools
import numpy as np
from PIL import Image
import torch
import open_clip

from app.config import CFG


def _pick_device() -> str:
    if CFG.embedding_device != "auto":
        return CFG.embedding_device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ClipEmbedder:
    def __init__(self):
        self.device = _pick_device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CFG.embedding_model_name, pretrained=CFG.embedding_pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(CFG.embedding_model_name)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def embed_image(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        feats = self.model.encode_image(tensor)
        feats /= feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).cpu().numpy().astype("float32")

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            text = " "
        tokens = self.tokenizer([text[:2000]]).to(self.device)
        feats = self.model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).cpu().numpy().astype("float32")

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Batched version, used for building category prompt centroids."""
        cleaned = [(t.strip() or " ")[:2000] for t in texts]
        tokens = self.tokenizer(cleaned).to(self.device)
        feats = self.model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")


@functools.lru_cache(maxsize=1)
def get_embedder() -> ClipEmbedder:
    """Process-wide singleton so we only load model weights once."""
    return ClipEmbedder()
