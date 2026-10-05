import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import CFG


class TestConfig(unittest.TestCase):
    @patch("app.config.ROOT", Path("/project/root"))
    def test_resolve_absolute(self):
        path = CFG.resolve("/absolute/path")
        self.assertEqual(path, Path("/absolute/path"))

    @patch("app.config.ROOT", Path("/project/root"))
    def test_resolve_relative(self):
        path = CFG.resolve("relative/path")
        self.assertEqual(path, Path("/project/root/relative/path").resolve())

    def test_resolve_home(self):
        path = CFG.resolve("~/home_dir_path")
        self.assertEqual(path, Path("~/home_dir_path").expanduser())


if __name__ == "__main__":
    unittest.main()
