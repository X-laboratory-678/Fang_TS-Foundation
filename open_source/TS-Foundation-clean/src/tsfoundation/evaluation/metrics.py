"""Forecasting metrics used in TS-Foundation evaluation."""

from __future__ import annotations

from math import sqrt
from typing import Sequence


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    _check_lengths(y_true, y_pred)
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    _check_lengths(y_true, y_pred)
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return sqrt(mse(y_true, y_pred))


def mape(y_true: Sequence[float], y_pred: Sequence[float], eps: float = 1e-8) -> float:
    _check_lengths(y_true, y_pred)
    return sum(abs((a - b) / max(abs(a), eps)) for a, b in zip(y_true, y_pred)) / len(y_true)


def evaluate_forecasts(rows: list[dict[str, list[float]]]) -> dict[str, float]:
    if not rows:
        raise ValueError("No forecast rows were provided.")

    mae_values = [mae(row["true_output"], row["predicted_output"]) for row in rows]
    rmse_values = [rmse(row["true_output"], row["predicted_output"]) for row in rows]
    mape_values = [mape(row["true_output"], row["predicted_output"]) for row in rows]
    return {
        "MAE": sum(mae_values) / len(mae_values),
        "RMSE": sum(rmse_values) / len(rmse_values),
        "MAPE": sum(mape_values) / len(mape_values),
        "num_samples": len(rows),
    }


def _check_lengths(y_true: Sequence[float], y_pred: Sequence[float]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} != {len(y_pred)}")
    if not y_true:
        raise ValueError("Metric inputs must not be empty.")

