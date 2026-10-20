import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os
import csv

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
            self.path = Path(f.name)
        CFG.labels_csv = self.path

    def tearDown(self):
        CFG.labels_csv = self.original_csv
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
            self.assertEqual(len(rows), 3)
            # Replaced Cat1 with the 0th category
            self.assertEqual(rows[1][1], CFG.category_names[0])
            self.assertEqual(rows[1][2], "user-reviewed")

    @patch("train.review_labels.argparse.ArgumentParser.parse_args")
    @patch("builtins.print")
    def test_main_no_labels_csv(self, mock_print, mock_parse):
        CFG.labels_csv = Path("/tmp/nonexistent_labels.csv")
        main()
        mock_print.assert_called_with("No labels.csv found — run `python -m train.bootstrap_labels` first.")

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.run")
    def test_open_file_mac(self, mock_run, mock_system):
        open_file("/tmp/test.png")
        mock_run.assert_called_with(["open", "/tmp/test.png"], check=False)

    @patch("platform.system", return_value="Windows")
    @patch("os.startfile", create=True)
    def test_open_file_windows(self, mock_startfile, mock_system):
        open_file("/tmp/test.png")
        mock_startfile.assert_called_with("/tmp/test.png")

    @patch("platform.system", return_value="Linux")
    @patch("subprocess.run")
    def test_open_file_linux(self, mock_run, mock_system):
        open_file("/tmp/test.png")
        mock_run.assert_called_with(["xdg-open", "/tmp/test.png"], check=False)

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.run", side_effect=Exception("Test Exception"))
    def test_open_file_exception(self, mock_run, mock_system):
        # Should catch the exception and do nothing
        open_file("/tmp/test.png")

    @patch("train.review_labels.argparse.ArgumentParser.parse_args")
    @patch("builtins.input", side_effect=["s", "", "invalid", "q"])
    @patch("train.review_labels.open_file")
    def test_main_user_inputs(self, mock_open_file, mock_input, mock_parse):
        args = MagicMock()
        args.n = 30
        args.open_files = True
        mock_parse.return_value = args

        # Add a third row with a different source to hit the 1.0 confidence path
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(["/tmp/c.png", "Cat3", "user-reviewed"])
            w.writerow(["/tmp/d.png", "Cat4", "user-reviewed"])
            w.writerow(["/tmp/e.png", "Cat4", "user-reviewed"])
            w.writerow(["/tmp/f.png", "Cat4", "user-reviewed"])

        main()

        # Check that open_file was called (since open_files is True)
        self.assertTrue(mock_open_file.called)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
