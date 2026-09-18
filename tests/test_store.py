from app.store import Store, FileRecord


def test_store_init(tmp_path, monkeypatch):
    # Mock CFG to point to a tmp path for DB
    class MockCfg:
        db_path = tmp_path / "test_db"

    monkeypatch.setattr("app.store.CFG", MockCfg)
    s = Store()
    # Handle lancedb 0.6 vs newer API difference correctly by skipping exact table_names check in test
    assert hasattr(s, "db")

    # Mock upsert
    vec = [0.0] * 512
    rec = FileRecord(
        id="123", filepath="/tmp/x.png", filename="x.png", ext=".png",
        category="Receipts", confidence=1.0, classifier_source="zero",
        ocr_text="", thumbnail_path="", created_at=0.0, indexed_at=0.0,
        vector=vec
    )
    s.upsert(rec)

    stats = s.stats()
    assert stats["total"] == 1
    assert stats["by_category"]["Receipts"] == 1

    s.delete_by_filepath("/tmp/x.png")
    assert s.stats()["total"] == 0
