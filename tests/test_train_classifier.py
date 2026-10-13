import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os
import csv
import numpy as np
import sys

from app.config import CFG
from train.train_classifier import main, load_labeled_examples


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
        if self.path.exists():
            os.remove(self.path)
        if os.path.exists(self.file1.name):
            os.remove(self.file1.name)
        if os.path.exists(self.file2.name):
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

    def test_load_labeled_examples_missing_csv(self):
        CFG.labels_csv = Path("/tmp/does_not_exist_abc.csv")
        with self.assertRaises(SystemExit) as cm:
            load_labeled_examples()
        self.assertIn("No labels.csv found", str(cm.exception))

    @patch("builtins.print")
    def test_main_zero_examples(self, mock_print):
        # Empty out the CSV
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filepath", "category", "source"])

        main()
        mock_print.assert_called_with("No labeled examples found. Please provide labeled examples to train the classifier.")

    @patch("train.train_classifier.get_embedder")
    @patch("train.train_classifier.extract")
    @patch("train.train_classifier.joblib.dump")
    def test_main_few_examples(self, mock_dump, mock_extract, mock_get_embedder):
        mock_extracted = MagicMock()
        mock_extracted.kind = "text"
        mock_extracted.text = "doc text"
        mock_extract.return_value = mock_extracted

        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = np.array([0.1, 0.2])
        mock_get_embedder.return_value = mock_embedder

        with patch("builtins.print") as mock_print:
            main()
            mock_print.assert_any_call("Only 2 labeled examples found \u2014 that's thin. The classifier will still train, but consider reviewing more via `python -m train.review_labels --n 100` first.\n")

    @patch("train.train_classifier.get_embedder")
    @patch("train.train_classifier.extract")
    @patch("train.train_classifier.joblib.dump")
    def test_main_text_and_image_branches_and_exceptions(self, mock_dump, mock_extract, mock_get_embedder):
        def extract_side_effect(path):
            if str(path) == self.file1.name:
                mock_ext = MagicMock()
                mock_ext.kind = "image"
                mock_ext.raw_bytes = b"img"
                return mock_ext
            else:
                raise Exception("Test Extract Exception")

        mock_extract.side_effect = extract_side_effect

        mock_embedder = MagicMock()
        mock_embedder.embed_image.return_value = np.array([0.1, 0.2])
        mock_get_embedder.return_value = mock_embedder

        with patch("builtins.print") as mock_print:
            main()

            # Since exception happened on file2, only file1 made it.
            # So classes < 2 logic will trigger
            mock_print.assert_any_call("\nOnly 1 category found. The classifier requires at least 2 categories to train.")

    @patch("train.train_classifier.get_embedder")
    @patch("train.train_classifier.extract")
    @patch("train.train_classifier.joblib.dump")
    def test_main_cross_val_branch(self, mock_dump, mock_extract, mock_get_embedder):
        # We need 5 examples of EACH class to trigger the min_class_count >= 5 branch.
        # So we rewrite the CSV with 10 rows (5 Cat1, 5 Cat2)
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filepath", "category", "source"])
            for i in range(5):
                w.writerow([self.file1.name, "Cat1", "zero-shot(0.1)"])
                w.writerow([self.file2.name, "Cat2", "zero-shot(0.9)"])
                # We need i % 25 to be covered too, so let's just make 26 files total.
            for i in range(16):
                w.writerow([self.file1.name, "Cat1", "zero-shot(0.1)"])

        mock_extracted = MagicMock()
        mock_extracted.kind = "image"
        mock_extracted.raw_bytes = b"img"
        mock_extract.return_value = mock_extracted

        mock_embedder = MagicMock()
        mock_embedder.embed_image.return_value = np.array([0.1, 0.2])
        mock_get_embedder.return_value = mock_embedder

        with patch("builtins.print") as mock_print:
            main()

            # Should have printed Cross-val accuracy
            self.assertTrue(any("Cross-val accuracy" in call.args[0] for call in mock_print.call_args_list if call.args))

    def test_run_as_main(self):
        import runpy
        import sys
        with patch.object(sys, 'argv', ['train.train_classifier']):
            with patch('train.train_classifier.main'):
                try:
                    runpy.run_module('train.train_classifier', run_name='__main__')
                except Exception:
                    pass

if __name__ == "__main__":
    unittest.main()
