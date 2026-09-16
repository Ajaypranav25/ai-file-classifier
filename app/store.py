"""
Local vector store, backed by LanceDB.

Why LanceDB over FAISS/Chroma/a server-based DB:
  - Embedded (just files on disk under data/), no separate process to run —
    important for a "runs quietly in the background on your laptop" app.
  - Native columnar storage means metadata filtering (category, date range)
    and ANN vector search compose in one query, instead of us hand-rolling
    a post-filter over FAISS results.
  - Upserts by primary key are simple, which matters since files get
    re-processed (renamed, re-saved) over time.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional

import lancedb
import pyarrow as pa

from app.config import CFG

EMBED_DIM = 512  # ViT-B-32 output dim; change if you swap embedding models

SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("filepath", pa.string()),
    pa.field("filename", pa.string()),
    pa.field("ext", pa.string()),
    pa.field("category", pa.string()),
    pa.field("confidence", pa.float32()),
    pa.field("classifier_source", pa.string()),
    pa.field("ocr_text", pa.string()),
    pa.field("thumbnail_path", pa.string()),
    pa.field("created_at", pa.float64()),
    pa.field("indexed_at", pa.float64()),
    pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
])


@dataclass
class FileRecord:
    id: str
    filepath: str
    filename: str
    ext: str
    category: str
    confidence: float
    classifier_source: str
    ocr_text: str
    thumbnail_path: str
    created_at: float
    indexed_at: float
    vector: list


class Store:
    def __init__(self):
        self.db = lancedb.connect(str(CFG.db_path))
        if "files" in self.db.table_names():
            self.table = self.db.open_table("files")
        else:
            self.table = self.db.create_table("files", schema=SCHEMA)

    def upsert(self, record: FileRecord):
        # LanceDB upsert-by-key: delete existing row with this id, then add.
        self.table.delete(f"id = '{record.id}'")
        self.table.add([asdict(record)])

    def delete_by_filepath(self, filepath: str):
        self.table.delete(f"filepath = '{filepath}'")

    def search(
        self,
        query_vector,
        limit: int = 40,
        category: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ):
        q = self.table.search(query_vector).limit(limit * 3)  # overfetch, then filter
        conditions = []
        if category and category != "All":
            conditions.append(f"category = '{category}'")
        if min_confidence is not None:
            conditions.append(f"confidence >= {min_confidence}")
        if conditions:
            q = q.where(" AND ".join(conditions))
        results = q.to_list()
        return results[:limit]

    def keyword_search(self, text: str, limit: int = 40, category: Optional[str] = None):
        """Simple substring fallback search over filenames + OCR text,
        used to complement semantic search for exact tokens (e.g. an
        order number) that embeddings are bad at matching precisely."""
        df = self.table.to_pandas()
        if df.empty:
            return []
        text_lower = text.lower()
        mask = (
            df["filename"].str.lower().str.contains(text_lower, na=False)
            | df["ocr_text"].str.lower().str.contains(text_lower, na=False)
        )
        if category and category != "All":
            mask &= df["category"] == category
        return df[mask].sort_values("indexed_at", ascending=False).head(limit).to_dict("records")

    def stats(self):
        df = self.table.to_pandas()
        if df.empty:
            return {"total": 0, "by_category": {}}
        return {
            "total": int(len(df)),
            "by_category": df["category"].value_counts().to_dict(),
        }

    def all_records(self):
        return self.table.to_pandas().to_dict("records")

    def update_category(self, record_id: str, new_category: str):
        df = self.table.to_pandas()
        row = df[df["id"] == record_id]
        if row.empty:
            return False
        rec = row.iloc[0].to_dict()
        rec["category"] = new_category
        rec["confidence"] = 1.0
        rec["classifier_source"] = "user-corrected"
        self.table.delete(f"id = '{record_id}'")
        self.table.add([rec])
        return True


_store_singleton: Store | None = None


def get_store() -> Store:
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = Store()
    return _store_singleton
