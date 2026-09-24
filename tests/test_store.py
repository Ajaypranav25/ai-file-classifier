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


if __name__ == "__main__":
    unittest.main()
