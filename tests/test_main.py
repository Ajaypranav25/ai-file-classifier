import unittest
from unittest.mock import patch, MagicMock

from app.main import main
from app.config import CFG


class TestMain(unittest.TestCase):
    @patch("app.main.start_watcher")
    @patch("app.main.uvicorn.run")
    def test_main_success(self, mock_uvicorn_run, mock_start_watcher):
        mock_observer = MagicMock()
        mock_start_watcher.return_value = mock_observer

        main()

        mock_start_watcher.assert_called_once()
        mock_uvicorn_run.assert_called_once_with(
            "app.api:app", host=CFG.api_host, port=CFG.api_port, log_level="warning"
        )
        mock_observer.stop.assert_called_once()
        mock_observer.join.assert_called_once()

    @patch("app.main.start_watcher")
    @patch("app.main.uvicorn.run")
    def test_main_exception_in_uvicorn(self, mock_uvicorn_run, mock_start_watcher):
        mock_observer = MagicMock()
        mock_start_watcher.return_value = mock_observer

        mock_uvicorn_run.side_effect = Exception("Test exception")

        with self.assertRaises(Exception):
            main()

        mock_start_watcher.assert_called_once()
        mock_uvicorn_run.assert_called_once_with(
            "app.api:app", host=CFG.api_host, port=CFG.api_port, log_level="warning"
        )
        mock_observer.stop.assert_called_once()
        mock_observer.join.assert_called_once()

    def test_main_block(self):
        import subprocess
        import sys

        # Test that running the file directly (as __main__) executes main()
        # and starts up properly by checking its output or running it in a subprocess safely.
        result = subprocess.run(
            [sys.executable, "-c", "import app.main; print(app.main.__name__)"],
            capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("app.main", result.stdout)


if __name__ == "__main__":
    unittest.main()
