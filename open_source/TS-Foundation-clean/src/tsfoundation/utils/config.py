"""Configuration loading for command line entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    with config_path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            data = json.load(handle)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("Install PyYAML or provide a JSON config file.") from exc
            data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data

def get_nested(config: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current

