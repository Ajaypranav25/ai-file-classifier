"""
Watches the configured folders for new/modified files and runs them through
the pipeline on a background thread pool, so a big PDF being OCR'd never
blocks the watcher from noticing the next screenshot.
"""
from __future__ import annotations
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from app.config import CFG
from app.pipeline import process_file, wait_until_stable, is_ignored
from app.store import Store

log = logging.getLogger("watcher")


class Handler(FileSystemEventHandler):
    def __init__(self, executor: ThreadPoolExecutor, store: Store):
        self.executor = executor
        self.store = store

    def _submit(self, path_str: str):
        path = Path(path_str)
        if is_ignored(path):
            return
        self.executor.submit(self._process, path)

    def _process(self, path: Path):
        try:
            if not wait_until_stable(path):
                return  # file vanished (temp download artifact etc.)
            result = process_file(path, store=self.store)
            if result:
                log.info(f"indexed {result['filename']} -> {result['category']} ({result['confidence']:.2f})")
        except Exception as e:
            log.warning(f"failed to process {path}: {e}")

    def on_created(self, event: FileSystemEvent):
        if not event.is_directory and isinstance(event.src_path, str):
            self._submit(event.src_path)

    def on_moved(self, event: FileSystemEvent):
        if not event.is_directory and isinstance(event.dest_path, str) and isinstance(event.src_path, str):
            self._submit(event.dest_path)
            self.store.delete_by_filepath(event.src_path)


def backfill(store: Store, executor: ThreadPoolExecutor):
    """One-time scan of existing files in watched folders, on first run."""
    for folder in CFG.watch_folders:
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and not is_ignored(path):
                executor.submit(process_file, path, store)


def start_watcher(run_backfill: bool = True) -> BaseObserver:
    store = Store()
    executor = ThreadPoolExecutor(max_workers=4)
    handler = Handler(executor, store)

    observer = Observer()
    for folder in CFG.watch_folders:
        folder.mkdir(parents=True, exist_ok=True)
        observer.schedule(handler, str(folder), recursive=True)
        log.info(f"watching {folder}")

    observer.start()

    if run_backfill and CFG.backfill_on_first_run:
        threading.Thread(target=backfill, args=(store, executor), daemon=True).start()

    return observer
