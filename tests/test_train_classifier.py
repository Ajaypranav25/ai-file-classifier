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

    @patch("train.train_classifier.get_embedder")
    @patch("train.train_classifier.extract")
    @patch("train.train_classifier.joblib.dump")
    def test_main_with_images_and_exceptions(self, mock_dump, mock_extract, mock_get_embedder):
        # We need to simulate the i % 25 == 0 branch and exception in extract
        # We'll create more files
        old_csv = CFG.labels_csv
        files = []
        with tempfile.NamedTemporaryFile(mode='w', delete=False, newline="") as f:
            w = csv.writer(f)
            w.writerow(["filepath", "category", "source"])
            for i in range(26):
                tf = tempfile.NamedTemporaryFile(delete=False)
                files.append(tf.name)
                w.writerow([tf.name, f"Cat{i % 2}", "user"])
            path = Path(f.name)
        CFG.labels_csv = path

        def side_effect_extract(p):
            if str(p) == files[0]:
                raise ValueError("Simulated extract error")
            mock_ext = MagicMock()
            mock_ext.kind = "image"
            mock_ext.raw_bytes = b"img"
            return mock_ext

        mock_extract.side_effect = side_effect_extract

        mock_embedder = MagicMock()
        mock_embedder.embed_image.return_value = np.array([0.5, 0.5])
        mock_get_embedder.return_value = mock_embedder

        try:
            main()
            self.assertEqual(mock_dump.call_count, 2)
        finally:
            CFG.labels_csv = old_csv
            os.remove(path)
            for f in files:
                os.remove(f)

    def test_missing_labels_csv(self):
        # Instead of mocking sys.exit and running load_labeled_examples,
        # we can just run load_labeled_examples and expect SystemExit.
        old_csv = CFG.labels_csv
        CFG.labels_csv = Path("/tmp/does_not_exist_ever.csv")
        try:
            from train.train_classifier import load_labeled_examples
            with self.assertRaises(SystemExit):
                load_labeled_examples()
        finally:
            CFG.labels_csv = old_csv

    @patch("builtins.print")
    def test_empty_labels_csv(self, mock_print):
        old_csv = CFG.labels_csv
        with tempfile.NamedTemporaryFile(mode='w', delete=False, newline="") as f:
            w = csv.writer(f)
            w.writerow(["filepath", "category", "source"])
            path = Path(f.name)
        CFG.labels_csv = path

        try:
            main()
            mock_print.assert_any_call("No labeled examples found. Please provide labeled examples to train the classifier.")
        finally:
            CFG.labels_csv = old_csv
            os.remove(path)

    @patch("builtins.print")
    def test_one_category_only(self, mock_print):
        old_csv = CFG.labels_csv
        with tempfile.NamedTemporaryFile(mode='w', delete=False, newline="") as f:
            w = csv.writer(f)
            w.writerow(["filepath", "category", "source"])
            w.writerow([self.file1.name, "Cat1", "user"])
            path = Path(f.name)
        CFG.labels_csv = path

        try:
            with patch("train.train_classifier.get_embedder") as mock_ge, \
                 patch("train.train_classifier.extract") as mock_ex:
                mock_ext = MagicMock()
                mock_ext.kind = "text"
                mock_ex.return_value = mock_ext
                mock_emb = MagicMock()
                mock_emb.embed_text.return_value = np.array([0.1])
                mock_ge.return_value = mock_emb

                main()

            found_call = any("requires at least 2 categories" in call.args[0] for call in mock_print.call_args_list)
            self.assertTrue(found_call, "Did not find expected stdout for 1 category only")
        finally:
            CFG.labels_csv = old_csv
            os.remove(path)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
