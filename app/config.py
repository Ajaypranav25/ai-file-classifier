"""
Loads config.yaml once and exposes it as a typed, dotted-access object.
Keeping this as its own module means every other file just does:
    from app.config import CFG
"""
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


@dataclass
class Category:
    name: str
    prompts: list[str]


@dataclass
class Config:
    watch_folders: list[Path]
    backfill_on_first_run: bool
    ignore_extensions: set[str]
    stability_wait_seconds: float
    db_path: Path
    labels_csv: Path
    thumbnails_dir: Path
    classifier_path: Path
    label_encoder_path: Path
    embedding_model_name: str
    embedding_pretrained: str
    embedding_device: str
    classification_confidence_threshold: float
    api_host: str
    api_port: int
    categories: list[Category]

    @property
    def category_names(self) -> list[str]:
        return [c.name for c in self.categories]

    def resolve(self, p: str | Path) -> Path:
        """Resolve a path relative to the project root, expanding ~."""
        p = Path(p).expanduser()
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        return p


def _load() -> Config:
    with open(CONFIG_PATH, "r") as f:
        raw = yaml.safe_load(f)

    root_tmp = ROOT  # used before Config exists, for resolving paths

    def resolve(p: str) -> Path:
        pp = Path(p).expanduser()
        return pp if pp.is_absolute() else (root_tmp / pp).resolve()

    categories = [Category(name=c["name"], prompts=c["prompts"]) for c in raw["categories"]]

    cfg = Config(
        watch_folders=[Path(p).expanduser() for p in raw["watch_folders"]],
        backfill_on_first_run=raw.get("backfill_on_first_run", True),
        ignore_extensions={e.lower() for e in raw.get("ignore_extensions", [])},
        stability_wait_seconds=float(raw.get("stability_wait_seconds", 2.0)),
        db_path=resolve(raw["db_path"]),
        labels_csv=resolve(raw["labels_csv"]),
        thumbnails_dir=resolve(raw["thumbnails_dir"]),
        classifier_path=resolve(raw["classifier_path"]),
        label_encoder_path=resolve(raw["label_encoder_path"]),
        embedding_model_name=raw["embedding"]["model_name"],
        embedding_pretrained=raw["embedding"]["pretrained"],
        embedding_device=raw["embedding"].get("device", "auto"),
        classification_confidence_threshold=float(raw.get("classification_confidence_threshold", 0.55)),
        api_host=raw["api"]["host"],
        api_port=int(raw["api"]["port"]),
        categories=categories,
    )

    cfg.thumbnails_dir.mkdir(parents=True, exist_ok=True)
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.classifier_path.parent.mkdir(parents=True, exist_ok=True)
    return cfg


CFG = _load()
