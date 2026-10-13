from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

from .forecasting_metrics import all_metrics
from .parser import ParseCode, parse_forecast


def decode_prediction(record: dict[str, Any], values: list[float]) -> list[float]:
    prediction_space = str(record.get("prediction_space", "raw"))
    if prediction_space == "raw":
        return values
    if prediction_space in {"linear_ar_asinh_residual", "linear_ar_quantized_asinh_residual"}:
        baseline = np.asarray(record["residual_baseline"], dtype=float)
        residuals = np.asarray(values, dtype=float)
        if baseline.shape != residuals.shape:
            raise ValueError("Vector residual decoding requires residual_baseline with the prediction shape.")
        scale = float(record["residual_scale"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Vector residual decoding requires a positive finite residual_scale.")
        if prediction_space == "linear_ar_quantized_asinh_residual":
            residuals = residuals / float(record.get("residual_quantization", 20.0))
        max_abs_z = float(record.get("max_abs_z", 6.0))
        residuals = np.clip(residuals, -max_abs_z, max_abs_z)
        decoded = baseline + scale * np.sinh(residuals)
        return [float(value) for value in decoded]
    if prediction_space not in {"asinh_residual", "quantized_asinh_residual"}:
        raise ValueError(f"Unsupported prediction space: {prediction_space}")
    anchor = float(record["residual_anchor"])
    scale = float(record["residual_scale"])
    max_abs_z = float(record.get("max_abs_z", 6.0))
    if not np.isfinite(anchor) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Residual decoding requires a finite anchor and positive scale.")
    z = np.asarray(values, dtype=float)
    if prediction_space == "quantized_asinh_residual":
        z = z / float(record.get("residual_quantization", 20.0))
    z = np.clip(z, -max_abs_z, max_abs_z)
    decoded = anchor + scale * np.sinh(z)
    return [float(value) for value in decoded]


def evaluate_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    truth: list[list[float]] = []
    predictions: list[list[float]] = []
    per_task: dict[str, tuple[list[list[float]], list[list[float]]]] = defaultdict(lambda: ([], []))
    total = 0
    for record in records:
        total += 1
        target = [float(value) for value in record["target"]]
        result = parse_forecast(record["prediction"], int(record.get("horizon", len(target))))
        status_counts[result.code.value] += 1
        if result.code != ParseCode.OK:
            continue
        decoded_values = decode_prediction(record, result.values or [])
        truth.append(target)
        predictions.append(decoded_values)
        task_truth, task_predictions = per_task[str(record.get("task_id", "unknown"))]
        task_truth.append(target)
        task_predictions.append(decoded_values)

    parsed = status_counts[ParseCode.OK.value]
    output: dict[str, Any] = {
        "total": total,
        "parsed": parsed,
        "parse_success_rate": parsed / total if total else 0.0,
        "parse_status_counts": dict(status_counts),
        "metrics": {},
        "per_task": {},
    }
    if parsed:
        output["metrics"] = all_metrics(np.asarray(truth).reshape(-1), np.asarray(predictions).reshape(-1))
        for task, (task_truth, task_predictions) in per_task.items():
            output["per_task"][task] = all_metrics(
                np.asarray(task_truth).reshape(-1), np.asarray(task_predictions).reshape(-1)
            )
    return output
