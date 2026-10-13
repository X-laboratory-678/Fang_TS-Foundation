#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from llamafactory.innovations_v2.metadata import normalize_electrical_sample
from llamafactory.innovations_v2.tracking import sha256_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--target-variable", required=True)
    parser.add_argument("--sampling-interval", default="unknown")
    parser.add_argument("--unit", default="")
    parser.add_argument("--region-or-device", default="")
    parser.add_argument("--split", choices=["train", "validation", "test", "preference_train"], required=True)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("Input dataset must be a JSON array or JSONL.")
    return value


def main() -> int:
    args = parse_args()
    source = Path(args.input)
    target = Path(args.output)
    defaults = {
        "task_id": args.task_id,
        "dataset_id": args.dataset_id,
        "target_variable": args.target_variable,
        "unit": args.unit,
        "sampling_interval": args.sampling_interval,
        "domain": "electrical",
        "region_or_device": args.region_or_device,
        "split": args.split,
        "history_length": 0,
        "horizon": 0,
        "temporal_stats": {},
    }
    upgraded = []
    for source_index, record in enumerate(load_records(source)):
        record = dict(record)
        record["source_index"] = source_index
        upgraded.append(normalize_electrical_sample(record, defaults))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(upgraded, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "source": str(source.resolve()),
        "source_sha256": sha256_path(source),
        "output": str(target.resolve()),
        "output_sha256": sha256_path(target),
        "records": len(upgraded),
        "task_id": args.task_id,
        "split": args.split,
    }
    target.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

