#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llamafactory.innovations_v2.evaluation import encode_generation
from llamafactory.innovations_v2.tracking import sha256_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mode", choices=["development", "final"], default="development")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def load_model(checkpoint: str):
    checkpoint_path = Path(checkpoint)
    if (checkpoint_path / "adapter_config.json").exists():
        peft_config = PeftConfig.from_pretrained(checkpoint)
        base = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, device_map="auto")
        model = PeftModel.from_pretrained(base, checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def checkpoint_hash(checkpoint: Path) -> str:
    for name in ("adapter_model.safetensors", "adapter_model.bin", "config.json"):
        candidate = checkpoint / name
        if candidate.exists():
            return sha256_path(candidate)
    raise FileNotFoundError(f"No checkpoint weight/config file found under {checkpoint}.")


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    data_path = Path(args.data)
    records = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Prediction data must be a non-empty JSON array.")
    expected_split = "validation" if args.mode == "development" else "test"
    if args.mode == "final" and not (Path("artifacts/publication_v2") / "locked_final_configuration.json").exists():
        raise RuntimeError("Final test inference is locked until locked_final_configuration.json exists.")
    if any(record.get("split") != expected_split for record in records):
        raise ValueError(f"{args.mode} mode accepts only records marked split={expected_split}.")
    if args.max_samples > 0:
        records = records[: args.max_samples]
    model, tokenizer = load_model(args.checkpoint)
    predictions = []
    for record in records:
        encoded = encode_generation(tokenizer, record, model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        continuation = generated[0, encoded["input_ids"].shape[1] :]
        predictions.append(
            {
                "task_id": record["task_id"],
                "source_index": record["source_index"],
                "target": json.loads(record["output"]),
                "prediction": tokenizer.decode(continuation, skip_special_tokens=True).strip(),
                "horizon": record["horizon"],
                "split": expected_split,
            }
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in predictions),
        encoding="utf-8",
    )
    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_hash": checkpoint_hash(Path(args.checkpoint)),
        "data": str(data_path.resolve()),
        "data_hash": sha256_path(data_path),
        "output": str(output_path.resolve()),
        "output_hash": sha256_path(output_path),
        "seed": args.seed,
        "mode": args.mode,
        "split": expected_split,
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "samples": len(predictions),
    }
    (output_path.parent / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
