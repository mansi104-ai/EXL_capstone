"""Configuration loading for the VCA system."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Project root = two levels up from this file (src/vca/config.py -> project root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

load_dotenv(PROJECT_ROOT / ".env")


class Config:
    """Thin wrapper over the parsed YAML config with resolved absolute paths."""

    def __init__(self, data: dict[str, Any], root: Path):
        self._data = data
        self.root = root

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def path(self, *parts: str) -> Path:
        """Resolve a path relative to the project root."""
        return self.root.joinpath(*parts)

    # --- convenience accessors -------------------------------------------
    @property
    def drivers(self) -> list[str]:
        return list(self._data["drivers"])

    @property
    def classifier(self) -> dict[str, Any]:
        return self._data["classifier"]

    @property
    def rag(self) -> dict[str, Any]:
        return self._data["rag"]

    @property
    def evidence(self) -> dict[str, Any]:
        return self._data["evidence"]

    @property
    def ingestion(self) -> dict[str, Any]:
        return self._data["ingestion"]

    # --- environment / secrets -------------------------------------------
    @property
    def azure_speech_key(self) -> str | None:
        return os.getenv("AZURE_SPEECH_KEY") or None

    @property
    def azure_speech_region(self) -> str | None:
        return os.getenv("AZURE_SPEECH_REGION") or None

    @property
    def force_simulated(self) -> bool:
        return os.getenv("VCA_FORCE_SIMULATED", "0") not in ("0", "", "false", "False")


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data, PROJECT_ROOT)
