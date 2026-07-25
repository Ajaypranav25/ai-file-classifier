"""
Entrypoint: `python -m app.main`

Starts the folder watcher on a background thread, then runs the FastAPI
server in the foreground. Ctrl+C stops both.
"""
import logging
import uvicorn

from app.config import CFG
from app.watcher import start_watcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main():
    log = logging.getLogger("main")
    log.info("starting folder watcher...")
    observer = start_watcher()

    log.info(f"starting API + UI at http://{CFG.api_host}:{CFG.api_port}")
    try:
        uvicorn.run("app.api:app", host=CFG.api_host, port=CFG.api_port, log_level="warning")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
