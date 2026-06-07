"""Dataset readers and validators for TS-Foundation examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from tsfoundation.data.serialization import parse_series

SFT_FIELDS = ("instruction", "input", "output")
DPO_FIELDS = ("instruction", "input", "chosen", "rejected")


def load_records(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    if dataset_path.suffix.lower() == ".jsonl":
        records = []
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records

    with dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"Dataset must contain a record or record list: {dataset_path}")
    return data


def validate_sft_record(record: dict[str, Any]) -> None:
    _require_fields(record, SFT_FIELDS)
    _validate_series(record["input"], "input")
    _validate_series(record["output"], "output")


def validate_dpo_record(record: dict[str, Any]) -> None:
    _require_fields(record, DPO_FIELDS)
    _validate_series(record["input"], "input")
    _validate_series(record["chosen"], "chosen")
    _validate_series(record["rejected"], "rejected")


def load_sft_dataset(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        for record in load_records(path):
            validate_sft_record(record)
            records.append(record)
    return records


def load_dpo_dataset(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        for record in load_records(path):
            validate_dpo_record(record)
            records.append(record)
    return records


def _require_fields(record: dict[str, Any], fields: tuple[str, ...]) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    for field in fields:
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"Field must be a non-empty string: {field}")


def _validate_series(value: str, field: str) -> None:
    values = parse_series(value)
    if not values:
        raise ValueError(f"Series field is empty: {field}")

