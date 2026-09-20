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


if __name__ == "__main__":
    unittest.main()
