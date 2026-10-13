import ast
import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np
import torch


REQUIRED_METADATA_FIELDS = (
    "task_id",
    "dataset_id",
    "target_variable",
    "history_length",
    "horizon",
    "split",
    "temporal_stats",
)

TEMPORAL_STAT_KEYS = ("mean", "std", "normalized_volatility", "trend_strength", "turning_rate")


def stable_task_index(task_id: str) -> int:
    if not task_id:
        raise ValueError("task_id cannot be empty.")
    return int.from_bytes(hashlib.sha256(task_id.encode("utf-8")).digest()[:4], "big") & 0x7FFFFFFF


def parse_numeric_sequence(value: Any) -> list[float]:
    if isinstance(value, str):
        text = value.strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = ast.literal_eval(text)
    if not isinstance(value, (list, tuple)):
        raise ValueError("Time-series input must be a JSON/Python list.")
    result = [float(item) for item in value]
    if not result:
        raise ValueError("Time-series input cannot be empty.")
    if not all(math.isfinite(item) for item in result):
        raise ValueError("Time-series input contains NaN or Inf.")
    return result


def compute_temporal_stats(values: Sequence[float], eps: float = 1e-8) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Temporal statistics require a non-empty one-dimensional sequence.")
    if not np.isfinite(array).all():
        finite = array[np.isfinite(array)]
        bound = max(float(np.max(np.abs(finite))) if finite.size else 1.0, 1.0) * 10.0
        array = np.nan_to_num(array, nan=0.0, posinf=bound, neginf=-bound)

    mean = float(np.mean(array))
    std = float(np.std(array))
    if array.size < 2:
        return {
            "mean": mean,
            "std": std,
            "normalized_volatility": 0.0,
            "trend_strength": 0.0,
            "turning_rate": 0.0,
        }

    differences = np.diff(array)
    normalized_volatility = float(np.mean(np.abs(differences)) / (std + eps))
    x = np.arange(array.size, dtype=np.float64)
    slope = float(np.polyfit(x, array, 1)[0]) if array.size >= 2 else 0.0
    trend_strength = abs(slope) / (std + eps)
    signs = np.sign(differences)
    sign_changes = int(np.sum((signs[1:] * signs[:-1]) < 0)) if signs.size > 1 else 0
    turning_rate = float(sign_changes / max(array.size - 2, 1))

    result = {
        "mean": mean,
        "std": std,
        "normalized_volatility": normalized_volatility,
        "trend_strength": float(trend_strength),
        "turning_rate": turning_rate,
    }
    return {key: value if math.isfinite(value) else 0.0 for key, value in result.items()}


def build_instruction(metadata: dict[str, Any]) -> str:
    required = ("dataset_id", "target_variable", "history_length", "horizon")
    missing = [key for key in required if metadata.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Cannot build instruction; missing metadata: {', '.join(missing)}")
    lines = [
        "Task: short-term electrical time-series forecasting.",
        f"Dataset: {metadata['dataset_id']}.",
        f"Target variable: {metadata['target_variable']}.",
        f"Sampling interval: {metadata.get('sampling_interval') or 'unknown'}.",
        f"History length: {int(metadata['history_length'])}.",
        f"Forecast horizon: {int(metadata['horizon'])}.",
    ]
    if metadata.get("unit"):
        lines.append(f"Unit: {metadata['unit']}.")
    if metadata.get("region_or_device"):
        lines.append(f"Region or device: {metadata['region_or_device']}.")
    lines.append(
        f"Return exactly {int(metadata['horizon'])} numerical values as a JSON list with no additional text."
    )
    return "\n".join(lines)


def validate_metadata(metadata: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_METADATA_FIELDS if key not in metadata]
    if missing:
        raise ValueError(f"Missing required electrical task metadata: {', '.join(missing)}")
    if metadata.get("domain", "electrical") != "electrical":
        raise ValueError(f"Non-electrical task rejected: {metadata.get('task_id', '<unknown>')}")
    if not metadata["task_id"] or not metadata["dataset_id"] or not metadata["target_variable"]:
        raise ValueError("task_id, dataset_id and target_variable must be non-empty.")
    stats = metadata["temporal_stats"]
    if not isinstance(stats, dict) or any(key not in stats for key in TEMPORAL_STAT_KEYS):
        raise ValueError("temporal_stats is incomplete.")


def normalize_electrical_sample(sample: dict[str, Any], metadata_defaults: dict[str, Any]) -> dict[str, Any]:
    if str(sample.get("instruction", "")).strip() == "A":
        sample = dict(sample)
    history = parse_numeric_sequence(sample["input"])
    target = parse_numeric_sequence(sample["output"])
    metadata = {**metadata_defaults, **{key: value for key, value in sample.items() if key in metadata_defaults}}
    metadata.setdefault("domain", "electrical")
    metadata["history_length"] = int(metadata.get("history_length") or len(history))
    metadata["horizon"] = int(metadata.get("horizon") or len(target))
    metadata["source_index"] = int(sample.get("source_index", metadata.get("source_index", 0)))
    metadata["temporal_stats"] = compute_temporal_stats(history)
    validate_metadata(metadata)
    normalized = dict(sample)
    normalized.update(metadata)
    normalized["task_index"] = stable_task_index(metadata["task_id"])
    normalized["instruction"] = build_instruction(metadata)
    normalized["input"] = json.dumps(history, ensure_ascii=False)
    normalized["output"] = json.dumps(target, ensure_ascii=False)
    return normalized


def collate_task_metadata(features: list[dict[str, Any]]) -> dict[str, Any]:
    task_ids = [feature.pop("task_id", None) for feature in features]
    if not any(task_id is not None for task_id in task_ids):
        return {}
    if not all(task_id is not None for task_id in task_ids):
        raise ValueError("Mixed legacy/v2 samples in one batch are not allowed.")
    fields = {
        "task_index": [feature.pop("task_index", None) for feature in features],
        "temporal_stats": [feature.pop("temporal_stats", None) for feature in features],
        "source_index": [feature.pop("source_index", None) for feature in features],
        "split": [feature.pop("split", None) for feature in features],
    }
    if any(value is None for values in fields.values() for value in values):
        raise ValueError("Innovation v2 batch contains incomplete task metadata.")
    return {
        "task_id": [str(task_id) for task_id in task_ids],
        "task_index": torch.tensor(fields["task_index"], dtype=torch.long),
        "temporal_stats": torch.tensor(fields["temporal_stats"], dtype=torch.float32),
        "source_index": torch.tensor(fields["source_index"], dtype=torch.long),
        "split": [str(split) for split in fields["split"]],
    }
