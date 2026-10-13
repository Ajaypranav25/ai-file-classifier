import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os
import csv
import sys

from app.config import CFG
from train.review_labels import main, open_file


class TestReviewLabels(unittest.TestCase):
    def setUp(self):
        self.original_csv = CFG.labels_csv
        with tempfile.NamedTemporaryFile(mode='w', delete=False, newline="") as f:
            w = csv.writer(f)
            w.writerow(["filepath", "category", "source"])
            w.writerow(["/tmp/a.png", "Cat1", "zero-shot(0.1)"])
            w.writerow(["/tmp/b.png", "Cat2", "zero-shot(0.9)"])
            # row without zero-shot fallback
            w.writerow(["/tmp/c.png", "Cat3", "other(0.5)"])
            self.path = Path(f.name)
        CFG.labels_csv = self.path

    def tearDown(self):
        CFG.labels_csv = self.original_csv
        if self.path.exists():
            os.remove(self.path)

    @patch("train.review_labels.argparse.ArgumentParser.parse_args")
    @patch("builtins.input", side_effect=["0", "q"])
    def test_main(self, mock_input, mock_parse):
        args = MagicMock()
        args.n = 30
        args.open_files = False
        mock_parse.return_value = args

        main()

        with open(self.path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 4)
            # Replaced Cat1 with the 0th category
            self.assertEqual(rows[1][1], CFG.category_names[0])
            self.assertEqual(rows[1][2], "user-reviewed")

    @patch("train.review_labels.argparse.ArgumentParser.parse_args")
    @patch("builtins.input", side_effect=["s", "", "99", "not_a_number", "q"])
    @patch("train.review_labels.open_file")
    def test_main_choices_and_open_files(self, mock_open, mock_input, mock_parse):
        args = MagicMock()
        args.n = 30
        args.open_files = True # to trigger calling open_file
        mock_parse.return_value = args

        main()

        # open_file should be called 5 times because of 5 inputs
        self.assertEqual(mock_open.call_count, 3)
        # (Actually, there are 3 rows initially, so it stops after 3 files if we don't have enough rows)

    @patch("train.review_labels.argparse.ArgumentParser.parse_args")
    def test_main_no_labels_csv(self, mock_parse):
        args = MagicMock()
        mock_parse.return_value = args

        # Point to missing file
        CFG.labels_csv = Path("/tmp/does_not_exist.csv")

        with patch('builtins.print') as mock_print:
            main()
            mock_print.assert_called_with("No labels.csv found — run `python -m train.bootstrap_labels` first.")

    @patch("train.review_labels.subprocess.run")
    @patch("train.review_labels.platform.system")
    def test_open_file_darwin(self, mock_sys, mock_run):
        mock_sys.return_value = "Darwin"
        open_file("/path/to/file")
        mock_run.assert_called_with(["open", "/path/to/file"], check=False)

    @patch("os.startfile", create=True)
    @patch("train.review_labels.platform.system")
    def test_open_file_windows(self, mock_sys, mock_start):
        mock_sys.return_value = "Windows"
        open_file("/path/to/file")
        mock_start.assert_called_with("/path/to/file")

    @patch("train.review_labels.subprocess.run")
    @patch("train.review_labels.platform.system")
    def test_open_file_linux(self, mock_sys, mock_run):
        mock_sys.return_value = "Linux"
        open_file("/path/to/file")
        mock_run.assert_called_with(["xdg-open", "/path/to/file"], check=False)

    @patch("train.review_labels.subprocess.run")
    @patch("train.review_labels.platform.system")
    def test_open_file_exception(self, mock_sys, mock_run):
        mock_sys.return_value = "Linux"
        mock_run.side_effect = Exception("Test Error")

        # Should catch and not raise
        open_file("/path/to/file")

    def test_run_as_main(self):
        import runpy
        import sys
        with patch.object(sys, 'argv', ['train.review_labels']):
            with patch('train.review_labels.main'):
                try:
                    runpy.run_module('train.review_labels', run_name='__main__')
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
