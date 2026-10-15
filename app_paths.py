"""Read-only application data paths for Python and the Windows executable."""
from pathlib import Path
import sys


def bundle_root() -> Path:
    """PyInstaller stores imported modules and resources under _MEIPASS."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def resource_path(name: str) -> Path:
    return bundle_root() / "resources" / name
