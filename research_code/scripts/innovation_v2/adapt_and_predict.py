#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llamafactory.innovations_v2.evaluation import encode_generation, evaluate_records, supervised_texts
from llamafactory.innovations_v2.tela.adaptation import clone_parameters, lora_parameters, restore_parameters
from llamafactory.innovations_v2.tracking import sha256_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--support", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inner-steps", type=int, default=3)
    parser.add_argument("--inner-lr", type=float, default=1e-4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mode", choices=["development", "final"], default="development")
    return parser.parse_args()


def load_records(path: str) -> list[dict]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("Support/test data must be JSON arrays.")
    return value


def load_model(checkpoint: str):
    adapter_config = Path(checkpoint) / "adapter_config.json"
    if adapter_config.exists():
        peft_config = PeftConfig.from_pretrained(checkpoint)
        base = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, device_map="auto")
        model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def support_loss(model, tokenizer, record: dict) -> torch.Tensor:
    prefix, full = supervised_texts(tokenizer, record)
    encoded = tokenizer(full, return_tensors="pt", truncation=True, max_length=1024)
    prefix_ids = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=1024)["input_ids"]
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    labels = encoded["input_ids"].clone()
    labels[:, : prefix_ids.shape[1]] = -100
    return model(**encoded, labels=labels).loss


def predict(model, tokenizer, records: list[dict], max_new_tokens: int) -> list[dict]:
    output = []
    model.eval()
    for record in records:
        encoded = encode_generation(tokenizer, record, model.device)
        with torch.no_grad():
            generated = model.generate(**encoded, do_sample=False, max_new_tokens=max_new_tokens)
        continuation = generated[0, encoded["input_ids"].shape[1] :]
        text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        output.append(
            {
                "task_id": record["task_id"],
                "source_index": record["source_index"],
                "target": json.loads(record["output"]),
                "prediction": text,
                "horizon": record["horizon"],
                "split": record["split"],
            }
        )
    return output


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    support = load_records(args.support)
    test = load_records(args.test)
    if not support or not test:
        raise ValueError("Support and test sets must be non-empty.")
    if len({record["task_id"] for record in support + test}) != 1:
        raise ValueError("adapt_and_predict requires one explicit target task.")
    if any(record["split"] == "test" for record in support):
        raise ValueError("Test samples cannot be used for adaptation.")
    expected_query_split = "validation" if args.mode == "development" else "test"
    if args.mode == "final" and not (Path("artifacts/publication_v2") / "locked_final_configuration.json").exists():
        raise RuntimeError("Final test adaptation is locked until locked_final_configuration.json exists.")
    if any(record.get("split") != expected_query_split for record in test):
        raise ValueError(f"{args.mode} mode requires query records with split={expected_query_split}.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.checkpoint)
    parameters = lora_parameters(model)
    original = clone_parameters(parameters)
    before = predict(model, tokenizer, test, args.max_new_tokens)
    before_metrics = evaluate_records(before)

    started = time.perf_counter()
    optimizer = torch.optim.SGD(parameters.values(), lr=args.inner_lr)
    model.train()
    for step in range(args.inner_steps):
        record = support[step % len(support)]
        optimizer.zero_grad(set_to_none=True)
        loss = support_loss(model, tokenizer, record)
        loss.backward()
        optimizer.step()
    adaptation_seconds = time.perf_counter() - started
    after = predict(model, tokenizer, test, args.max_new_tokens)
    after_metrics = evaluate_records(after)
    restore_parameters(parameters, original)

    write_jsonl(output_dir / "predictions_before.jsonl", before)
    write_jsonl(output_dir / "predictions_after.jsonl", after)
    (output_dir / "metrics_before.json").write_text(json.dumps(before_metrics, indent=2), encoding="utf-8")
    (output_dir / "metrics_after.json").write_text(json.dumps(after_metrics, indent=2), encoding="utf-8")
    checkpoint_hash_target = Path(args.checkpoint) / "adapter_model.safetensors"
    if not checkpoint_hash_target.exists():
        checkpoint_hash_target = Path(args.checkpoint) / "config.json"
    adaptation_manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_hash": sha256_path(checkpoint_hash_target),
        "task_id": support[0]["task_id"],
        "support_samples": len(support),
        "inner_steps": args.inner_steps,
        "inner_lr": args.inner_lr,
        "mode": args.mode,
        "query_split": expected_query_split,
        "adaptation_seconds": adaptation_seconds,
        "updated_parameters": sum(parameter.numel() for parameter in parameters.values()),
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
        "model_restored_after_run": True,
    }
    (output_dir / "adaptation_manifest.json").write_text(
        json.dumps(adaptation_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
