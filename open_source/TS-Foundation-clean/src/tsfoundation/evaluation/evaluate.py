"""Evaluation entry point for TS-Foundation demos."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tsfoundation.data.dataset import load_sft_dataset
from tsfoundation.data.serialization import parse_series
from tsfoundation.evaluation.inference import NaiveForecaster
from tsfoundation.evaluation.metrics import evaluate_forecasts
from tsfoundation.utils.config import load_config
from tsfoundation.utils.io import write_json


def run_evaluation(config: dict[str, Any], config_dir: Path | None = None) -> dict[str, Any]:
    config_dir = config_dir or Path.cwd()
    base_dir = Path.cwd()
    data_config = config.get("data", {})
    test_files = _resolve_paths(data_config.get("test_files", []), base_dir, config_dir)
    records = load_sft_dataset(test_files)

    eval_config = config.get("evaluation", {})
    horizon = int(eval_config.get("horizon", 3))
    forecaster = NaiveForecaster(horizon=horizon, strategy=eval_config.get("strategy", "last_delta"))

    rows = []
    for sample_id, record in enumerate(records):
        forecast = forecaster.predict(record["input"])
        rows.append(
            {
                "sample_id": sample_id,
                "instruction": record["instruction"],
                "input_sequence": parse_series(record["input"]),
                "true_output": parse_series(record["output"]),
                "predicted_output": forecast.predicted_output,
                "raw_response": forecast.raw_response,
            }
        )

    metrics = evaluate_forecasts(rows)
    output_dir = Path(config.get("output", {}).get("dir", "outputs/demo_eval"))
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir

    write_json(rows, output_dir / "prediction_results.json")
    write_json(metrics, output_dir / "evaluation_metrics.json")
    return {"metrics": metrics, "num_records": len(records), "output_dir": str(output_dir)}


def _resolve_paths(paths: list[str], base_dir: Path, config_dir: Path) -> list[Path]:
    resolved = []
    for path in paths:
        candidate = Path(path)
        if candidate.is_absolute():
            resolved.append(candidate)
            continue
        base_candidate = base_dir / candidate
        resolved.append(base_candidate if base_candidate.exists() else config_dir / candidate)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TS-Foundation forecasts.")
    parser.add_argument("--config", required=True, help="Path to a YAML or JSON config file.")
    args = parser.parse_args()
    config_path = Path(args.config)
    result = run_evaluation(load_config(config_path), config_path.parent)
    print(f"Evaluation finished: {result['metrics']}")


if __name__ == "__main__":
    main()
