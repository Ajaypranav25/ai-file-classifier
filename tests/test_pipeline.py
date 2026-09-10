import unittest
from pathlib import Path
from unittest.mock import patch
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

if __name__ == "__main__":
    unittest.main()
