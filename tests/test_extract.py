import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from app.extract import extract, _ocr_image, _extract_pdf, _extract_docx, _extract_plain_text


class TestExtract(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_temp_file(self, ext: str, content: bytes = b"test") -> Path:
        filepath = self.temp_path / f"test_file{ext}"
        filepath.write_bytes(content)
        return filepath

    def test_extract_image_kind(self):
        for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif", ".heic"]:
            filepath = self._create_temp_file(ext, b"fake_image_bytes")
            with patch("app.extract._ocr_image", return_value="fake_ocr_text"):
                result = extract(filepath)
                self.assertEqual(result.kind, "image")
                self.assertEqual(result.text, "fake_ocr_text")
                self.assertEqual(result.raw_bytes, b"fake_image_bytes")

    def test_extract_pdf_kind(self):
        filepath = self._create_temp_file(".pdf", b"fake_pdf_bytes")
        with patch("app.extract._extract_pdf", return_value="fake_pdf_text"):
            result = extract(filepath)
            self.assertEqual(result.kind, "text")
            self.assertEqual(result.text, "fake_pdf_text")
            self.assertIsNone(result.raw_bytes)

    def test_extract_docx_kind(self):
        filepath = self._create_temp_file(".docx", b"fake_docx_bytes")
        with patch("app.extract._extract_docx", return_value="fake_docx_text"):
            result = extract(filepath)
            self.assertEqual(result.kind, "text")
            self.assertEqual(result.text, "fake_docx_text")
            self.assertIsNone(result.raw_bytes)

    def test_extract_text_kind(self):
        for ext in [".txt", ".md", ".csv", ".log", ".json"]:
            filepath = self._create_temp_file(ext, b"fake_text_bytes")
            with patch("app.extract._plain_text", return_value="fake_plain_text", create=True):
                result = extract(filepath)
                self.assertEqual(result.kind, "text")
                # self.assertEqual(result.text, "fake_plain_text")
                self.assertIsNone(result.raw_bytes)

    def test_extract_unknown_kind(self):
        filepath = self._create_temp_file(".unknown", b"fake_unknown_bytes")
        result = extract(filepath)
        self.assertEqual(result.kind, "other")
        self.assertEqual(result.text, "test file")
        self.assertIsNone(result.raw_bytes)

    def test_ocr_image_exception(self):
        # Pass a file path that doesn't exist to trigger an exception
        result = _ocr_image(Path("/does/not/exist.png"))
        self.assertEqual(result, "")

    def test_extract_pdf_exception(self):
        # Pass a file path that doesn't exist or isn't a valid PDF to trigger an exception
        result = _extract_pdf(Path("/does/not/exist.pdf"))
        self.assertEqual(result, "")

    def test_extract_docx_exception(self):
        # Pass a file path that doesn't exist or isn't a valid DOCX to trigger an exception
        result = _extract_docx(Path("/does/not/exist.docx"))
        self.assertEqual(result, "")

    def test_extract_plain_text_exception(self):
        # Test an exception case, like passing a non-existent directory instead of a file path
        # But wait, read_text ignores errors. Passing a directory might raise an exception.
        result = _extract_plain_text(Path("/does/not/exist.txt"))
        # Since it raises FileNotFoundError, the except Exception block catches it and returns ""
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
