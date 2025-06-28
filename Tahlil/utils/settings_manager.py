"""Simple JSON-based settings manager for Spreadsheet app."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class SettingsManager:
    FILE_PATH = Path.home() / ".spreadsheet_pyqt6_settings.json"
    DEFAULTS: Dict[str, Any] = {
        "default_path": str(Path.home()),
        "language": "en",
        "theme": "Light Blue",
        "save_precision": 2,
    }

    def __init__(self):
        if not self.FILE_PATH.exists():
            self.data = self.DEFAULTS.copy()
            self.save()
        else:
            try:
                self.data = self.DEFAULTS.copy()
                with self.FILE_PATH.open("r", encoding="utf-8") as f:
                    self.data.update(json.load(f))
            except Exception:
                # fallback to defaults if corrupted
                self.data = self.DEFAULTS.copy()
                self.save()

    # -------------- public helpers -----------------
    def get(self, key: str, default: Any | None = None):
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        self.data[key] = value

    def save(self):
        try:
            with self.FILE_PATH.open("w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print("Failed to save settings", e)


# global instance
settings = SettingsManager()
