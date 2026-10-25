import platform
import subprocess
from pathlib import Path


def open_file_locally(path: str | Path, check: bool = False) -> None:
    """
    Opens a file using the system's default viewer.
    """
    p_str = str(path)
    system = platform.system()
    if system == "Darwin":
        subprocess.run(["open", p_str], check=check)
    elif system == "Windows":
        import os
        os.startfile(p_str)  # type: ignore
    else:
        subprocess.run(["xdg-open", p_str], check=check)
