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
    def test_handler_on_created_ignored(self, mock_submit, mock_is_ignored):
        mock_is_ignored.return_value = True
        executor = ThreadPoolExecutor(max_workers=1)
        store = unittest.mock.MagicMock()
        handler = Handler(executor, store)

        event = FileSystemEvent(src_path="/tmp/test.png")
        event.is_directory = False
        event.event_type = "created"

        handler.on_created(event)
        mock_submit.assert_not_called()

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

    @unittest.mock.patch("app.watcher.wait_until_stable")
    @unittest.mock.patch("app.watcher.process_file")
    def test_handler_process_vanished(self, mock_process_file, mock_wait):
        mock_wait.return_value = False

        executor = ThreadPoolExecutor(max_workers=1)
        store = unittest.mock.MagicMock()
        handler = Handler(executor, store)

        handler._process(Path("/tmp/vanished.png"))

        mock_process_file.assert_not_called()

    @unittest.mock.patch("app.watcher.wait_until_stable")
    @unittest.mock.patch("app.watcher.process_file")
    def test_handler_process_exception(self, mock_process_file, mock_wait):
        mock_wait.return_value = True
        mock_process_file.side_effect = Exception("Test error")

        executor = ThreadPoolExecutor(max_workers=1)
        store = unittest.mock.MagicMock()
        handler = Handler(executor, store)

        # Should not raise exception
        handler._process(Path("/tmp/error.png"))
        mock_process_file.assert_called_once_with(Path("/tmp/error.png"), store=store)

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

    @unittest.mock.patch("app.watcher.Path.rglob")
    @unittest.mock.patch("app.watcher.Path.exists")
    def test_backfill_missing_folder(self, mock_exists, mock_rglob):
        original_folders = CFG.watch_folders
        CFG.watch_folders = [Path("/tmp/missing_watch")]

        try:
            mock_exists.return_value = False

            executor = ThreadPoolExecutor(max_workers=1)
            store = unittest.mock.MagicMock()

            backfill(store, executor)

            mock_rglob.assert_not_called()
        finally:
            CFG.watch_folders = original_folders

    @unittest.mock.patch("app.watcher.Store")
    @unittest.mock.patch("app.watcher.Observer")
    @unittest.mock.patch("app.watcher.threading.Thread")
    @unittest.mock.patch("app.watcher.Path.mkdir")
    def test_start_watcher(self, mock_mkdir, mock_thread, mock_observer, mock_store):
        original_folders = CFG.watch_folders
        original_backfill = CFG.backfill_on_first_run
        CFG.watch_folders = [Path("/tmp/watch")]
        CFG.backfill_on_first_run = True

        try:
            mock_obs_instance = unittest.mock.MagicMock()
            mock_observer.return_value = mock_obs_instance

            mock_thread_instance = unittest.mock.MagicMock()
            mock_thread.return_value = mock_thread_instance

            from app.watcher import start_watcher
            observer = start_watcher(run_backfill=True)

            self.assertEqual(observer, mock_obs_instance)
            mock_obs_instance.schedule.assert_called_once()
            mock_obs_instance.start.assert_called_once()
            mock_mkdir.assert_called_once()
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()
        finally:
            CFG.watch_folders = original_folders
            CFG.backfill_on_first_run = original_backfill


if __name__ == "__main__":
    unittest.main()
