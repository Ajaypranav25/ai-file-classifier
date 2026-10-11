import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os
import csv

from app.config import CFG
from train.bootstrap_labels import main


class TestBootstrapLabels(unittest.TestCase):
    @patch("train.bootstrap_labels.get_embedder")
    @patch("train.bootstrap_labels.extract")
    @patch("train.bootstrap_labels.iter_watch_folder_files")
    def test_main(self, mock_iter, mock_extract, mock_get_embedder):
        mock_iter.return_value = [Path("/tmp/a.png")]

        mock_extracted = MagicMock()
        mock_extracted.kind = "image"
        mock_extracted.raw_bytes = b"dummy"
        mock_extract.return_value = mock_extracted

        mock_embedder = MagicMock()
        import numpy as np
        mock_embedder.embed_image.return_value = np.array([0.1, 0.2])
        mock_get_embedder.return_value = mock_embedder

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.close()
            path = Path(f.name)

        original_csv = CFG.labels_csv
        CFG.labels_csv = path

        try:
            with patch("train.bootstrap_labels._zero_shot_classify", return_value=("Cat", 0.9)):
                main()
            with open(path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0], ["filepath", "category", "source"])
                self.assertEqual(rows[1], ["/tmp/a.png", "Cat", "zero-shot(0.9)"])
        finally:
            CFG.labels_csv = original_csv
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
