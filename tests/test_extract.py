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

    def test_ocr_image_error(self):
        # Pass a missing file to test exception handling
        path = Path("does_not_exist.png")
        from app.extract import _ocr_image
        self.assertEqual(_ocr_image(path), "")

    def test_extract_pdf_error(self):
        # Pass a missing file to test exception handling
        path = Path("does_not_exist.pdf")
        from app.extract import _extract_pdf
        self.assertEqual(_extract_pdf(path), "")

    def test_extract_docx_error(self):
        # Pass a missing file to test exception handling
        path = Path("does_not_exist.docx")
        from app.extract import _extract_docx
        self.assertEqual(_extract_docx(path), "")

    def test_extract_plain_text_error(self):
        # Pass a missing file to test exception handling
        path = Path("does_not_exist.txt")
        from app.extract import _extract_plain_text
        self.assertEqual(_extract_plain_text(path), "")

    def test_extract_image_read_error(self):
        path = Path("does_not_exist_but_has_extension.png")
        result = extract(path)
        self.assertEqual(result.kind, "other")
        self.assertEqual(result.text, "does not exist but has extension")
        self.assertIsNone(result.raw_bytes)

    def test_extract_pdf_valid(self):
        import pypdf
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = Path(f.name)
            writer = pypdf.PdfWriter()
            # empty page is enough to pass read
            writer.add_blank_page(width=100, height=100)
            writer.write(str(pdf_path))
        try:
            from app.extract import _extract_pdf
            res = _extract_pdf(pdf_path)
            self.assertEqual(res, "")
        finally:
            os.remove(pdf_path)

    def test_extract_docx_valid(self):
        import docx
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            docx_path = Path(f.name)
            doc = docx.Document()
            doc.add_paragraph("Hello docx")
            doc.save(str(docx_path))
        try:
            from app.extract import _extract_docx
            res = _extract_docx(docx_path)
            self.assertEqual(res, "Hello docx")
        finally:
            os.remove(docx_path)

    @unittest.mock.patch("pytesseract.image_to_string")
    def test_ocr_image_success(self, mock_image_to_string):
        mock_image_to_string.return_value = "extracted text  "
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"dummy image data")
            f.flush()
            path = Path(f.name)

        try:
            with unittest.mock.patch("PIL.Image.open"):
                from app.extract import _ocr_image
                res = _ocr_image(path)
                self.assertEqual(res, "extracted text")
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
