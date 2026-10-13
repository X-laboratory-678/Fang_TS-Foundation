#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ratios", default="1,5,10,25")
    args = parser.parse_args()
    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not records or any(record.get("split") != "train" for record in records):
        raise ValueError("Few-shot source must contain only non-empty train records.")
    rng = random.Random(args.seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for ratio in [int(value) for value in args.ratios.split(",")]:
        count = max(1, round(len(records) * ratio / 100))
        selected = sorted(indices[:count])
        subset = [records[index] for index in selected]
        path = output_dir / f"support_{ratio}pct.json"
        path.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
        manifests[str(ratio)] = {
            "count": count,
            "source_indices": [record["source_index"] for record in subset],
            "path": str(path),
        }
    (output_dir / "split_manifest.json").write_text(
        json.dumps({"seed": args.seed, "source": args.input, "ratios": manifests}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

