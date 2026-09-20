from __future__ import annotations
import platform
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import CFG
from app.embed import get_embedder
from app.classify import get_classifier
from app.store import get_store

app = FastAPI(title="Screenshot Search")


class CorrectionRequest(BaseModel):
    id: str
    category: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/search")
def search(q: str = "", category: str = "All", limit: int = 60):
    store = get_store()
    results = []
    seen_ids = set()

    if q.strip():
        embedder = get_embedder()
        query_vec = embedder.embed_text(q)
        for r in store.search(query_vec, limit=limit, category=category):
            r.pop("vector", None)
            r["match"] = round(1.0 - float(r.get("_distance", 0.0)), 4) if "_distance" in r else None
            seen_ids.add(r["id"])
            results.append(r)

        # Hybrid: also surface exact keyword hits (e.g. an order number)
        # that semantic search alone might rank low.
        for r in store.keyword_search(q, limit=20, category=category):
            if r["id"] not in seen_ids:
                r.pop("vector", None)
                r["match"] = None
                seen_ids.add(r["id"])
                results.append(r)
    else:
        recs = store.all_records()
        if category != "All":
            recs = [r for r in recs if r["category"] == category]
        recs.sort(key=lambda r: r["indexed_at"], reverse=True)
        for r in recs[:limit]:
            r.pop("vector", None)
            r["match"] = None
            results.append(r)

    return {"results": results, "count": len(results)}


@app.get("/api/categories")
def categories():
    store = get_store()
    stats = store.stats()
    return {
        "categories": CFG.category_names,
        "counts": stats["by_category"],
        "total": stats["total"],
    }


@app.post("/api/correct")
def correct(req: CorrectionRequest):
    store = get_store()
    if req.category not in CFG.category_names:
        raise HTTPException(400, f"Unknown category: {req.category}")
    ok = store.update_category(req.id, req.category)
    if not ok:
        raise HTTPException(404, "Record not found")
    # Append to the labels CSV so the next `train_classifier.py` run
    # picks up this correction as ground truth.
    import csv
    recs = store.all_records()
    rec = next((r for r in recs if r["id"] == req.id), None)
    if rec:
        CFG.labels_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not CFG.labels_csv.exists()
        with open(CFG.labels_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["filepath", "category", "source"])
            w.writerow([rec["filepath"], req.category, "user-corrected"])
    return {"ok": True}


@app.post("/api/reload-classifier")
def reload_classifier():
    """Call after running train/train_classifier.py to hot-swap the model
    into the already-running server without restarting it."""
    get_classifier().reload()
    return {"ok": True, "trained": get_classifier().is_trained}


@app.get("/api/open")
def open_file(path: str):
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, "File not found")
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", str(p)], check=False)
        elif system == "Windows":
            import os
            os.startfile(str(p))  # type: ignore
        else:
            subprocess.run(["xdg-open", str(p)], check=False)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/thumbnail/{file_id}")
def thumbnail(file_id: str):
    path = CFG.thumbnails_dir / f"{file_id}.jpg"
    if not path.exists():
        raise HTTPException(404, "No thumbnail")
    return FileResponse(path)


# Serve the search UI itself at "/"
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
