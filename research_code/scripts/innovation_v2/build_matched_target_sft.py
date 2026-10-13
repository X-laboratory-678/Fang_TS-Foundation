#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sft", required=True)
    parser.add_argument("--preferences", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = json.loads(Path(args.source_sft).read_text(encoding="utf-8"))
    preferences = json.loads(Path(args.preferences).read_text(encoding="utf-8"))
    selected_indices = {int(item["source_index"]) for item in preferences}
    matched = [item for item in source if int(item["source_index"]) in selected_indices]
    if len(matched) != len(selected_indices):
        raise ValueError("Could not match every preference source index to an SFT sample.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(matched, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
