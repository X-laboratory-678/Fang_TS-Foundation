"""Run a small local forecast without downloading a model."""

from tsfoundation.data.dataset import load_sft_dataset
from tsfoundation.evaluation.inference import NaiveForecaster
from tsfoundation.evaluation.metrics import evaluate_forecasts
from tsfoundation.data.serialization import parse_series


def main() -> None:
    records = load_sft_dataset(["data/sample_sft/weather_sample.json"])
    forecaster = NaiveForecaster(horizon=3)
    rows = []
    for record in records:
        forecast = forecaster.predict(record["input"])
        rows.append(
            {
                "true_output": parse_series(record["output"]),
                "predicted_output": forecast.predicted_output,
            }
        )
    print(evaluate_forecasts(rows))


if __name__ == "__main__":
    main()

