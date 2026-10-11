import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os
import csv

from app.config import CFG
from train.review_labels import main


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


if __name__ == "__main__":
    unittest.main()
