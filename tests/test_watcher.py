import unittest
import unittest.mock
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from watchdog.events import FileSystemEvent

from app.watcher import Handler, backfill
from app.config import CFG


class TestWatcher(unittest.TestCase):
    @unittest.mock.patch("app.watcher.is_ignored")
    @unittest.mock.patch("app.watcher.ThreadPoolExecutor.submit")
    def test_handler_on_created(self, mock_submit, mock_is_ignored):
        mock_is_ignored.return_value = False
        executor = ThreadPoolExecutor(max_workers=1)
        store = unittest.mock.MagicMock()
        handler = Handler(executor, store)

        event = FileSystemEvent(src_path="/tmp/test.png")
        event.is_directory = False
        event.event_type = "created"

        handler.on_created(event)
        mock_submit.assert_called_once()
        args = mock_submit.call_args[0]
        self.assertEqual(args[1], Path("/tmp/test.png"))

    @unittest.mock.patch("app.watcher.is_ignored")
    @unittest.mock.patch("app.watcher.ThreadPoolExecutor.submit")
    def test_handler_on_moved(self, mock_submit, mock_is_ignored):
        mock_is_ignored.return_value = False
        executor = ThreadPoolExecutor(max_workers=1)
        store = unittest.mock.MagicMock()
        handler = Handler(executor, store)

        event = unittest.mock.MagicMock()
        event.is_directory = False
        event.src_path = "/tmp/test.png"
        event.dest_path = "/tmp/new_test.png"
        event.event_type = "moved"

        handler.on_moved(event)
        mock_submit.assert_called_once()
        store.delete_by_filepath.assert_called_once_with("/tmp/test.png")
        args = mock_submit.call_args[0]
        self.assertEqual(args[1], Path("/tmp/new_test.png"))

    @unittest.mock.patch("app.watcher.wait_until_stable")
    @unittest.mock.patch("app.watcher.process_file")
    def test_handler_process(self, mock_process_file, mock_wait):
        mock_wait.return_value = True
        mock_process_file.return_value = {"filename": "test.png", "category": "Cat", "confidence": 0.9}

        executor = ThreadPoolExecutor(max_workers=1)
        store = unittest.mock.MagicMock()
        handler = Handler(executor, store)

        handler._process(Path("/tmp/test.png"))

        mock_process_file.assert_called_once_with(Path("/tmp/test.png"), store=store)

    @unittest.mock.patch("app.watcher.Path.rglob")
    @unittest.mock.patch("app.watcher.Path.exists")
    @unittest.mock.patch("app.watcher.is_ignored")
    @unittest.mock.patch("app.watcher.ThreadPoolExecutor.submit")
    def test_backfill(self, mock_submit, mock_is_ignored, mock_exists, mock_rglob):
        original_folders = CFG.watch_folders
        CFG.watch_folders = [Path("/tmp/watch")]

        try:
            mock_exists.return_value = True
            mock_is_ignored.return_value = False

            mock_path = unittest.mock.MagicMock()
            mock_path.is_file.return_value = True
            mock_rglob.return_value = [mock_path]

            executor = ThreadPoolExecutor(max_workers=1)
            store = unittest.mock.MagicMock()

            backfill(store, executor)

            mock_submit.assert_called_once()
        finally:
            CFG.watch_folders = original_folders


if __name__ == "__main__":
    unittest.main()
