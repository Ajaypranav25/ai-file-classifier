import unittest
from pathlib import Path
from app.pipeline import is_ignored
from app.config import CFG


class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Ensure some ignore extensions are set up for testing
        CFG.ignore_extensions = {".part", ".tmp", ".crdownload"}

    def test_is_ignored_ds_store(self):
        self.assertTrue(is_ignored(Path(".DS_Store")))
        self.assertTrue(is_ignored(Path(".ds_store")))
        self.assertTrue(is_ignored(Path("a/b/c/.DS_Store")))
        self.assertFalse(is_ignored(Path("my_DS_Store")))

    def test_is_ignored_extensions(self):
        self.assertTrue(is_ignored(Path("file.part")))
        self.assertTrue(is_ignored(Path("download.CRDOWNLOAD")))
        self.assertTrue(is_ignored(Path("temp.tmp")))
        self.assertFalse(is_ignored(Path("file.txt")))
        self.assertFalse(is_ignored(Path("file.jpg")))

    def test_is_ignored_hidden_files(self):
        self.assertTrue(is_ignored(Path(".hidden_file")))
        self.assertTrue(is_ignored(Path(".git")))
        self.assertTrue(is_ignored(Path("a/b/.hidden_file.txt")))
        self.assertFalse(is_ignored(Path("hidden_file.")))


if __name__ == "__main__":
    unittest.main()
