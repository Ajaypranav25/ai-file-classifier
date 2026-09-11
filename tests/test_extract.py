import unittest
from pathlib import Path
import tempfile

from app.extract import extract


class TestExtract(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extract_plain_text(self):
        txt_path = self.dir_path / "test.txt"
        txt_path.write_text("Hello, this is a test text file.")

        extracted = extract(txt_path)
        self.assertEqual(extracted.kind, "text")
        self.assertEqual(extracted.text, "Hello, this is a test text file.")
        self.assertIsNone(extracted.raw_bytes)

    def test_extract_unknown_extension(self):
        unknown_path = self.dir_path / "test_file.unknown"
        unknown_path.write_text("Some random data")

        extracted = extract(unknown_path)
        self.assertEqual(extracted.kind, "other")
        self.assertEqual(extracted.text, "test file")
        self.assertIsNone(extracted.raw_bytes)


if __name__ == '__main__':
    unittest.main()
