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


if __name__ == "__main__":
    unittest.main()
