import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os
import csv
import numpy as np

from app.config import CFG
from train.train_classifier import main


class TestTrainClassifier(unittest.TestCase):
    def setUp(self):
        self.original_csv = CFG.labels_csv
        with tempfile.NamedTemporaryFile(mode='w', delete=False, newline="") as f:
            w = csv.writer(f)
            w.writerow(["filepath", "category", "source"])
            # Provide at least two categories to prevent early exit
            self.file1 = tempfile.NamedTemporaryFile(delete=False)
            self.file2 = tempfile.NamedTemporaryFile(delete=False)
            w.writerow([self.file1.name, "Cat1", "zero-shot(0.1)"])
            w.writerow([self.file2.name, "Cat2", "zero-shot(0.9)"])
            self.path = Path(f.name)
        CFG.labels_csv = self.path

    def tearDown(self):
        CFG.labels_csv = self.original_csv
        os.remove(self.path)
        os.remove(self.file1.name)
        os.remove(self.file2.name)

    @patch("train.train_classifier.get_embedder")
    @patch("train.train_classifier.extract")
    @patch("train.train_classifier.joblib.dump")
    def test_main(self, mock_dump, mock_extract, mock_get_embedder):
        mock_extracted = MagicMock()
        mock_extracted.kind = "text"
        mock_extracted.text = "doc text"
        mock_extract.return_value = mock_extracted

        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = np.array([0.1, 0.2])
        mock_get_embedder.return_value = mock_embedder

        main()

        # It trains a Logistic Regression
        self.assertEqual(mock_dump.call_count, 2)


if __name__ == "__main__":
    unittest.main()
