"""File I/O helpers used by the TS-Foundation demos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(data: Any, path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output_path


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def project_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if base_dir is None:
        base_dir = Path.cwd()
    return Path(base_dir) / candidate

