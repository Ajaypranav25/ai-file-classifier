import unittest
from pathlib import Path

from app.pipeline import is_ignored, _file_id
from app.config import CFG


class TestPipeline(unittest.TestCase):
    def test_is_ignored_ds_store(self):
        self.assertTrue(is_ignored(Path(".ds_store")))
        self.assertTrue(is_ignored(Path(".DS_Store")))

    def test_is_ignored_extensions(self):
        # Temporarily mock ignore_extensions in CFG if needed, or rely on defaults
        original = CFG.ignore_extensions
        CFG.ignore_extensions = {".tmp", ".part"}
        try:
            self.assertTrue(is_ignored(Path("test.tmp")))
            self.assertTrue(is_ignored(Path("download.part")))
            self.assertFalse(is_ignored(Path("image.png")))
        finally:
            CFG.ignore_extensions = original

    def test_is_ignored_hidden_files(self):
        self.assertTrue(is_ignored(Path(".hidden_file")))

    def test_file_id_length(self):
        result = _file_id(Path("some_file.png"))
        self.assertEqual(len(result), 16)
        self.assertIsInstance(result, str)

    def test_wait_until_stable_quick(self):
        from app.pipeline import wait_until_stable
        import tempfile

        with tempfile.NamedTemporaryFile() as f:
            f.write(b"test")
            f.flush()
            # It should stabilize immediately
            self.assertTrue(wait_until_stable(Path(f.name), wait_seconds=0.01))

    def test_wait_until_stable_disappears(self):
        from app.pipeline import wait_until_stable
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test")
            f.flush()
            path = Path(f.name)

        os.remove(path)
        self.assertFalse(wait_until_stable(path, wait_seconds=0.01))

    def test_wait_until_stable_timeout(self):
        from app.pipeline import wait_until_stable
        from unittest.mock import patch

        class MockStat:
            def __init__(self):
                self.st_size = 0

        def mock_exists():
            return True

        def mock_stat():
            stat = MockStat()
            stat.st_size = MockStat.counter
            MockStat.counter += 1
            return stat

        MockStat.counter = 0

        with patch("pathlib.Path.exists", side_effect=mock_exists), \
             patch("pathlib.Path.stat", side_effect=mock_stat), \
             patch("time.sleep"):
            # File size keeps changing, so it should hit max_attempts
            self.assertFalse(wait_until_stable(Path("dummy"), wait_seconds=0.01, max_attempts=5))

    @unittest.mock.patch("app.pipeline.get_classifier")
    @unittest.mock.patch("app.pipeline.get_embedder")
    @unittest.mock.patch("app.pipeline.extract")
    @unittest.mock.patch("app.pipeline._make_thumbnail")
    @unittest.mock.patch("app.pipeline.is_ignored")
    @unittest.mock.patch("pathlib.Path.exists")
    @unittest.mock.patch("pathlib.Path.is_file")
    def test_process_file_image(self, mock_is_file, mock_exists, mock_is_ignored, mock_thumb, mock_extract, mock_embed, mock_classify):
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_is_ignored.return_value = False

        mock_extracted = unittest.mock.MagicMock()
        mock_extracted.kind = "image"
        mock_extracted.raw_bytes = b"dummy"
        mock_extracted.text = "ocr"
        mock_extract.return_value = mock_extracted

        mock_embedder = unittest.mock.MagicMock()
        import numpy as np
        mock_embedder.embed_image.return_value = np.array([0.1, 0.2])
        mock_embed.return_value = mock_embedder

        mock_classifier = unittest.mock.MagicMock()
        mock_classifier.classify.return_value = ("TestCat", 0.9, "trained")
        mock_classify.return_value = mock_classifier

        mock_thumb.return_value = "/tmp/thumb.jpg"

        store = unittest.mock.MagicMock()
        from app.pipeline import process_file

        with unittest.mock.patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_mtime = 123.4
            res = process_file(Path("/dummy.png"), store=store)

        self.assertIsNotNone(res)
        self.assertEqual(res["category"], "TestCat")
        self.assertEqual(res["filename"], "dummy.png")
        store.upsert.assert_called_once()
        mock_embedder.embed_image.assert_called_once_with(b"dummy")

    @unittest.mock.patch("app.pipeline.get_classifier")
    @unittest.mock.patch("app.pipeline.get_embedder")
    @unittest.mock.patch("app.pipeline.extract")
    @unittest.mock.patch("app.pipeline.is_ignored")
    @unittest.mock.patch("pathlib.Path.exists")
    @unittest.mock.patch("pathlib.Path.is_file")
    def test_process_file_text(self, mock_is_file, mock_exists, mock_is_ignored, mock_extract, mock_embed, mock_classify):
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_is_ignored.return_value = False

        mock_extracted = unittest.mock.MagicMock()
        mock_extracted.kind = "text"
        mock_extracted.text = "doc text"
        mock_extract.return_value = mock_extracted

        mock_embedder = unittest.mock.MagicMock()
        import numpy as np
        mock_embedder.embed_text.return_value = np.array([0.3, 0.4])
        mock_embed.return_value = mock_embedder

        mock_classifier = unittest.mock.MagicMock()
        mock_classifier.classify.return_value = ("DocCat", 0.8, "zero-shot")
        mock_classify.return_value = mock_classifier

        store = unittest.mock.MagicMock()
        from app.pipeline import process_file

        with unittest.mock.patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_mtime = 123.4
            res = process_file(Path("/dummy.txt"), store=store)

        self.assertIsNotNone(res)
        self.assertEqual(res["category"], "DocCat")
        self.assertEqual(res["filename"], "dummy.txt")
        store.upsert.assert_called_once()
        mock_embedder.embed_text.assert_called_once_with("doc text")

    @unittest.mock.patch("app.pipeline.is_ignored")
    @unittest.mock.patch("pathlib.Path.exists")
    def test_process_file_ignored(self, mock_exists, mock_is_ignored):
        mock_exists.return_value = True
        mock_is_ignored.return_value = True
        from app.pipeline import process_file
        res = process_file(Path("/dummy.txt"))
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()
