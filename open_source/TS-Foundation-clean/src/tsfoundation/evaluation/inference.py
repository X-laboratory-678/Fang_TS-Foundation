"""Inference helpers and a deterministic demo forecaster."""

from __future__ import annotations

from dataclasses import dataclass

from tsfoundation.data.serialization import parse_prediction_text, parse_series


@dataclass
class ForecastResult:
    predicted_output: list[float]
    raw_response: str


class NaiveForecaster:
    """Small deterministic baseline for demo and tests."""

    def __init__(self, horizon: int = 3, strategy: str = "last_delta") -> None:
        self.horizon = horizon
        self.strategy = strategy

    def predict(self, input_data: str | list[float]) -> ForecastResult:
        series = parse_series(input_data)
        if not series:
            raise ValueError("Input series is empty.")
        if self.strategy == "repeat_last" or len(series) < 2:
            values = [series[-1]] * self.horizon
        else:
            delta = series[-1] - series[-2]
            values = [series[-1] + delta * (step + 1) for step in range(self.horizon)]
        rounded = [round(value, 2) for value in values]
        return ForecastResult(predicted_output=rounded, raw_response=str(rounded))


def parse_model_response(text: str, horizon: int = 3) -> ForecastResult:
    values = parse_prediction_text(text, horizon=horizon)
    return ForecastResult(predicted_output=values, raw_response=text)

