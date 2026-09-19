import unittest
from pathlib import Path
from app.extract import extract

class TestExtract(unittest.TestCase):
    def test_extract_unknown_extension(self):
        result = extract(Path("test_file_with_spaces_and-hyphens.xyz"))
        self.assertEqual(result.kind, "other")
        self.assertEqual(result.text, "test file with spaces and hyphens")
        self.assertIsNone(result.raw_bytes)

if __name__ == "__main__":
    unittest.main()
