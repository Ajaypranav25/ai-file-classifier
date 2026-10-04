import unittest
import tempfile
import shutil
from pathlib import Path

from app.store import Store, FileRecord
from app.config import CFG


class TestStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.original_db_path = CFG.db_path
        CFG.db_path = Path(self.temp_dir) / "test.lance"
        self.store = Store()

    def tearDown(self):
        CFG.db_path = self.original_db_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_upsert_and_search(self):
        record = FileRecord(
            id="test1",
            filepath="/tmp/test1.png",
            filename="test1.png",
            ext=".png",
            category="Test",
            confidence=0.9,
            classifier_source="test",
            ocr_text="hello world",
            thumbnail_path="",
            created_at=123.0,
            indexed_at=123.0,
            vector=[0.1] * 512
        )
        self.store.upsert(record)

        recs = self.store.all_records()
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["id"], "test1")

        # Search
        results = self.store.search([0.1] * 512, limit=10)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "test1")

    def test_keyword_search(self):
        record = FileRecord(
            id="test2",
            filepath="/tmp/test2.png",
            filename="my_document.txt",
            ext=".txt",
            category="Doc",
            confidence=0.8,
            classifier_source="test",
            ocr_text="some specific keyword",
            thumbnail_path="",
            created_at=123.0,
            indexed_at=123.0,
            vector=[0.1] * 512
        )
        self.store.upsert(record)

        results = self.store.keyword_search("specific", limit=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "test2")

    def test_update_category(self):
        record = FileRecord(
            id="test3",
            filepath="/tmp/test3.png",
            filename="test3.png",
            ext=".png",
            category="OldCategory",
            confidence=0.5,
            classifier_source="test",
            ocr_text="",
            thumbnail_path="",
            created_at=123.0,
            indexed_at=123.0,
            vector=[0.1] * 512
        )
        self.store.upsert(record)

        ok = self.store.update_category("test3", "NewCategory")
        self.assertTrue(ok)

        recs = self.store.all_records()
        self.assertEqual(recs[0]["category"], "NewCategory")
        self.assertEqual(recs[0]["confidence"], 1.0)
        self.assertEqual(recs[0]["classifier_source"], "user-corrected")

    def test_delete_by_filepath(self):
        record = FileRecord(
            id="test4",
            filepath="/tmp/test4.png",
            filename="test4.png",
            ext=".png",
            category="Test",
            confidence=0.9,
            classifier_source="test",
            ocr_text="",
            thumbnail_path="",
            created_at=123.0,
            indexed_at=123.0,
            vector=[0.1] * 512
        )
        self.store.upsert(record)
        self.assertEqual(len(self.store.all_records()), 1)

        self.store.delete_by_filepath("/tmp/test4.png")
        self.assertEqual(len(self.store.all_records()), 0)

    def test_empty_store_edge_cases(self):
        # Empty store should return appropriate empty values
        self.assertEqual(self.store.keyword_search("test"), [])
        self.assertEqual(self.store.stats(), {"total": 0, "by_category": {}})
        self.assertFalse(self.store.update_category("nonexistent", "Category"))

    def test_stats_with_data(self):
        record1 = FileRecord(
            id="test1",
            filepath="/tmp/test1.png",
            filename="test1.png",
            ext=".png",
            category="CatA",
            confidence=0.9,
            classifier_source="test",
            ocr_text="",
            thumbnail_path="",
            created_at=123.0,
            indexed_at=123.0,
            vector=[0.1] * 512
        )
        self.store.upsert(record1)
        stats = self.store.stats()
        self.assertEqual(stats["total"], 1)
        self.assertEqual(stats["by_category"], {"CatA": 1})

    def test_search_filters(self):
        record1 = FileRecord(
            id="test1",
            filepath="/tmp/test1.png",
            filename="test1.png",
            ext=".png",
            category="CatA",
            confidence=0.9,
            classifier_source="test",
            ocr_text="",
            thumbnail_path="",
            created_at=123.0,
            indexed_at=123.0,
            vector=[0.1] * 512
        )
        record2 = FileRecord(
            id="test2",
            filepath="/tmp/test2.png",
            filename="test2.png",
            ext=".png",
            category="CatB",
            confidence=0.4,
            classifier_source="test",
            ocr_text="",
            thumbnail_path="",
            created_at=123.0,
            indexed_at=123.0,
            vector=[0.1] * 512
        )
        self.store.upsert(record1)
        self.store.upsert(record2)

        # Filter by category
        res = self.store.search([0.1] * 512, category="CatA")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["id"], "test1")

        # Filter by confidence
        res = self.store.search([0.1] * 512, min_confidence=0.8)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["id"], "test1")

        # Combine filters
        res = self.store.search([0.1] * 512, category="CatB", min_confidence=0.8)
        self.assertEqual(len(res), 0)

    def test_keyword_search_category_filter(self):
        record = FileRecord(
            id="test1",
            filepath="/tmp/test1.png",
            filename="keyword.png",
            ext=".png",
            category="CatA",
            confidence=0.9,
            classifier_source="test",
            ocr_text="",
            thumbnail_path="",
            created_at=123.0,
            indexed_at=123.0,
            vector=[0.1] * 512
        )
        self.store.upsert(record)

        # Correct category
        res = self.store.keyword_search("keyword", category="CatA")
        self.assertEqual(len(res), 1)

        # Wrong category
        res = self.store.keyword_search("keyword", category="CatB")
        self.assertEqual(len(res), 0)

    def test_get_store_singleton(self):
        from app.store import get_store
        import app.store
        original = app.store._store_singleton
        app.store._store_singleton = None
        try:
            s1 = get_store()
            s2 = get_store()
            self.assertIs(s1, s2)
        finally:
            app.store._store_singleton = original


if __name__ == "__main__":
    unittest.main()
