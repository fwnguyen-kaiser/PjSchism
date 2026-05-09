"""YAML config loader for SCHISM service configs."""

from __future__ import annotations

from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_yaml(filename: str) -> dict:
    """Load a YAML file from schism/config/ and return as dict."""
    path = _CONFIG_DIR / filename
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}
