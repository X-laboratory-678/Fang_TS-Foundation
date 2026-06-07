"""Predict from an SFT-format JSON file with the deterministic demo forecaster."""

from __future__ import annotations

import argparse

from tsfoundation.data.dataset import load_sft_dataset
from tsfoundation.evaluation.inference import NaiveForecaster
from tsfoundation.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="SFT-format JSON or JSONL file.")
    parser.add_argument("--output", default="outputs/example_predictions.json")
    parser.add_argument("--horizon", type=int, default=3)
    args = parser.parse_args()

    records = load_sft_dataset([args.input])
    forecaster = NaiveForecaster(horizon=args.horizon)
    predictions = [
        {
            "instruction": record["instruction"],
            "input": record["input"],
            "prediction": forecaster.predict(record["input"]).predicted_output,
        }
        for record in records
    ]
    write_json(predictions, args.output)
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()

