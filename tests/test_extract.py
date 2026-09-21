import unittest
import tempfile
import os
from pathlib import Path

from app.extract import extract


class TestExtract(unittest.TestCase):
    def test_extract_unknown_extension(self):
        result = extract(Path("test_file_with_spaces_and-hyphens.xyz"))
        self.assertEqual(result.kind, "other")
        self.assertEqual(result.text, "test file with spaces and hyphens")
        self.assertIsNone(result.raw_bytes)

    def test_extract_plain_text(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Hello world\nThis is a test file.")
            f.flush()
            path = Path(f.name)

        try:
            result = extract(path)
            self.assertEqual(result.kind, "text")
            self.assertEqual(result.text, "Hello world\nThis is a test file.")
            self.assertIsNone(result.raw_bytes)
        finally:
            os.remove(path)

    def test_extract_invalid_image(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"not a real image")
            f.flush()
            path = Path(f.name)

        try:
            result = extract(path)
            self.assertEqual(result.kind, "image")
            self.assertEqual(result.text, "")
            self.assertEqual(result.raw_bytes, b"not a real image")
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
