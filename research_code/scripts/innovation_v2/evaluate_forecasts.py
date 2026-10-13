#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

from llamafactory.innovations_v2.evaluation import evaluate_records
from llamafactory.innovations_v2.tracking import RunManifest, sha256_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--method", default="evaluation")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--comparison-base-checkpoint", default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mode", choices=["development", "final"], default="development")
    args = parser.parse_args()
    path = Path(args.predictions)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_split = "validation" if args.mode == "development" else "test"
    if args.mode == "final" and not (Path("artifacts/publication_v2") / "locked_final_configuration.json").exists():
        raise RuntimeError("Final test evaluation is locked until locked_final_configuration.json exists.")
    if any(record.get("split") != expected_split for record in records):
        raise ValueError(f"{args.mode} evaluation requires predictions marked split={expected_split}.")
    metrics = evaluate_records(records)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    task_counts = Counter(str(record.get("task_id", "unknown")) for record in records)
    manifest_metrics = dict(metrics.get("metrics", {}))
    manifest_metrics["parse_success_rate"] = metrics["parse_success_rate"]
    manifest_metrics["parse_status_counts"] = metrics["parse_status_counts"]
    manifest = RunManifest(
        experiment_id=args.experiment_id or target.parent.name,
        method=args.method,
        resolved_config={
            "comparison_base_checkpoint": args.comparison_base_checkpoint,
            "per_task_metrics": metrics.get("per_task", {}),
            "mode": args.mode,
            "split": expected_split,
        },
        seed=args.seed,
        dataset_names=sorted(task_counts),
        train_sample_counts=dict(task_counts),
        checkpoint_path=str(Path(args.checkpoint).resolve()) if args.checkpoint else "",
        metrics=manifest_metrics,
        evaluation_script=str(Path(__file__).resolve()),
    )
    manifest.dataset_hashes["predictions"] = sha256_path(path)
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        for candidate in ("adapter_model.safetensors", "adapter_model.bin", "config.json"):
            checkpoint_file = checkpoint / candidate
            if checkpoint_file.exists():
                manifest.checkpoint_hash = sha256_path(checkpoint_file)
                break
    manifest.hash_code_files(
        [
            Path(__file__),
            Path(__file__).parents[2] / "src" / "llamafactory" / "innovations_v2" / "evaluation" / "evaluator.py",
            Path(__file__).parents[2] / "src" / "llamafactory" / "innovations_v2" / "evaluation" / "parser.py",
        ]
    )
    manifest.save(target.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
