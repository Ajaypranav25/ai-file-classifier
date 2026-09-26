import unittest
import unittest.mock
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

    @unittest.mock.patch("app.extract._ocr_image")
    def test_extract_valid_image(self, mock_ocr):
        mock_ocr.return_value = "extracted text from image"
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"dummy image data")
            f.flush()
            path = Path(f.name)

        try:
            result = extract(path)
            self.assertEqual(result.kind, "image")
            self.assertEqual(result.text, "extracted text from image")
            self.assertEqual(result.raw_bytes, b"dummy image data")
            mock_ocr.assert_called_once_with(path)
        finally:
            os.remove(path)

    @unittest.mock.patch("app.extract._extract_pdf")
    def test_extract_pdf(self, mock_extract_pdf):
        mock_extract_pdf.return_value = "extracted pdf content"
        path = Path("dummy.pdf")
        result = extract(path)
        self.assertEqual(result.kind, "text")
        self.assertEqual(result.text, "extracted pdf content")
        self.assertIsNone(result.raw_bytes)
        mock_extract_pdf.assert_called_once_with(path)

    @unittest.mock.patch("app.extract._extract_docx")
    def test_extract_docx(self, mock_extract_docx):
        mock_extract_docx.return_value = "extracted docx content"
        path = Path("dummy.docx")
        result = extract(path)
        self.assertEqual(result.kind, "text")
        self.assertEqual(result.text, "extracted docx content")
        self.assertIsNone(result.raw_bytes)
        mock_extract_docx.assert_called_once_with(path)


if __name__ == "__main__":
    unittest.main()
