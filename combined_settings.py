"""Per-user setup paths; song files and plugin data are never bundled."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


def settings_directory() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    return Path(base) / "OpenAutomation" if base else Path.home() / ".config" / "OpenAutomation"


def template_path() -> Path:
    return settings_directory() / "section-template.json"


def load_settings() -> dict:
    try:
        path = settings_directory() / "settings.json"
        if path.stat().st_size > 65536:
            return {}
        saved = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(saved, dict) or saved.get("version") != 1:
            return {}
    except (OSError, ValueError):
        return {}
    result = {key: saved[key] for key in ("section_template", "eq_calibration")
              if isinstance(saved.get(key), str)}
    if isinstance(saved.get("section_enabled"), dict):
        result["section_enabled"] = {key: value for key, value in saved["section_enabled"].items()
                                     if key in ("filter", "delay", "width", "pan") and type(value) is bool}
    return result


def save_settings(section_template: str, eq_calibration: str, section_enabled: dict) -> None:
    folder = settings_directory()
    folder.mkdir(parents=True, exist_ok=True)
    data = {"version": 1, "section_template": section_template, "eq_calibration": eq_calibration,
            "section_enabled": {key: bool(section_enabled.get(key, key == "pan"))
                                for key in ("filter", "delay", "width", "pan")}}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=folder, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, folder / "settings.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
