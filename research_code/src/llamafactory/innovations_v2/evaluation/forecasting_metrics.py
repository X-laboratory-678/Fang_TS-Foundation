import math
from typing import Sequence

import numpy as np


def _arrays(y_true: Sequence[float], y_pred: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape or true.size == 0:
        raise ValueError("Metric inputs must have the same non-empty shape.")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Metric inputs must be finite.")
    return true, pred


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    true, pred = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    true, pred = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def smape(y_true: Sequence[float], y_pred: Sequence[float], eps: float = 1e-8) -> float:
    true, pred = _arrays(y_true, y_pred)
    denominator = np.abs(true) + np.abs(pred)
    ratio = np.where(denominator <= eps, 0.0, 2.0 * np.abs(pred - true) / np.maximum(denominator, eps))
    return float(np.mean(ratio))


def trend_direction_accuracy(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    true, pred = _arrays(y_true, y_pred)
    if true.size < 2:
        return 1.0
    return float(np.mean(np.sign(np.diff(true)) == np.sign(np.diff(pred))))


def peak_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    true, pred = _arrays(y_true, y_pred)
    return float(abs(np.max(true) - np.max(pred)))


def turning_point_mismatch(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    true, pred = _arrays(y_true, y_pred)
    if true.size < 3:
        return 0.0
    true_turn = np.sign(np.diff(true)[:-1]) != np.sign(np.diff(true)[1:])
    pred_turn = np.sign(np.diff(pred)[:-1]) != np.sign(np.diff(pred)[1:])
    return float(np.mean(true_turn != pred_turn))


def all_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    result = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "trend_direction_accuracy": trend_direction_accuracy(y_true, y_pred),
        "peak_error": peak_error(y_true, y_pred),
        "turning_point_mismatch": turning_point_mismatch(y_true, y_pred),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("A forecasting metric produced a non-finite value.")
    return result

