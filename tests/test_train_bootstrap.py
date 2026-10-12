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

    @patch("train.bootstrap_labels.get_embedder")
    @patch("train.bootstrap_labels.extract")
    @patch("train.bootstrap_labels.iter_watch_folder_files")
    def test_main_text_and_exceptions(self, mock_iter, mock_extract, mock_get_embedder):
        # We'll simulate 26 files to hit the i % 25 == 0 branch
        paths = [Path(f"/tmp/file{i}.txt") for i in range(26)]
        mock_iter.return_value = paths

        def side_effect_extract(path):
            if path.name == "file0.txt":
                raise ValueError("Simulated extraction error")
            mock_extracted = MagicMock()
            mock_extracted.kind = "text"
            mock_extracted.text = "dummy text"
            return mock_extracted

        mock_extract.side_effect = side_effect_extract

        mock_embedder = MagicMock()
        import numpy as np
        mock_embedder.embed_text.return_value = np.array([0.1, 0.2])
        mock_get_embedder.return_value = mock_embedder

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.close()
            csv_path = Path(f.name)

        original_csv = CFG.labels_csv
        CFG.labels_csv = csv_path

        try:
            with patch("train.bootstrap_labels._zero_shot_classify", return_value=("Doc", 0.8)):
                main()
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)
                # 1 header + 25 successful rows
                self.assertEqual(len(rows), 26)
        finally:
            CFG.labels_csv = original_csv
            os.remove(csv_path)

    def test_iter_watch_folder_files(self):
        from train.bootstrap_labels import iter_watch_folder_files
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            p1 = Path(d1)
            p2 = Path(d2)

            # create some files
            (p1 / "a.png").touch()
            (p1 / ".hidden").touch()  # ignored typically? wait, need to check is_ignored
            (p2 / "b.txt").touch()

            # non-existent folder
            p3 = Path(d1) / "nonexistent"

            original_folders = CFG.watch_folders
            CFG.watch_folders = [p1, p2, p3]

            try:
                # We also need to patch is_ignored to return False for a.png and b.txt, True for .hidden
                with patch("train.bootstrap_labels.is_ignored", side_effect=lambda x: x.name.startswith(".")):
                    files = list(iter_watch_folder_files())
                    names = {f.name for f in files}
                    self.assertEqual(names, {"a.png", "b.txt"})
            finally:
                CFG.watch_folders = original_folders


if __name__ == "__main__":
    unittest.main()
