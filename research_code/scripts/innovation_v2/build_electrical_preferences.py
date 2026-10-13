#!/usr/bin/env python3
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from llamafactory.innovations_v2.evaluation.forecasting_metrics import (
    mae,
    peak_error,
    trend_direction_accuracy,
    turning_point_mismatch,
)
from llamafactory.innovations_v2.evaluation.generation import encode_generation, user_content
from llamafactory.innovations_v2.evaluation.parser import ParseCode, parse_forecast
from llamafactory.innovations_v2.metadata.task_metadata import parse_numeric_sequence
from llamafactory.innovations_v2.tracking import sha256_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--candidates-per-sample", type=int, default=4)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--temperatures", default="0.3,0.6,0.9,1.0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-margin", type=float, default=0.01)
    parser.add_argument("--max-margin", type=float, default=2.0)
    parser.add_argument("--w-mae", type=float, default=1.0)
    parser.add_argument("--w-trend", type=float, default=0.25)
    parser.add_argument("--w-peak", type=float, default=0.25)
    parser.add_argument("--w-turn", type=float, default=0.25)
    parser.add_argument("--w-phys", type=float, default=0.0)
    parser.add_argument("--lower-bound", type=float)
    parser.add_argument("--upper-bound", type=float)
    return parser.parse_args()


def physical_penalty(values: list[float], lower: float | None, upper: float | None) -> float:
    penalty = 0.0
    if lower is not None:
        penalty += sum(max(0.0, lower - value) for value in values)
    if upper is not None:
        penalty += sum(max(0.0, value - upper) for value in values)
    return penalty / max(len(values), 1)


def score_candidate(target: list[float], candidate: list[float], args: argparse.Namespace) -> tuple[float, dict[str, float]]:
    scale = float(np.std(target)) + 1e-8
    components = {
        "normalized_mae": mae(target, candidate) / scale,
        "trend_error": 1.0 - trend_direction_accuracy(target, candidate),
        "normalized_peak_error": peak_error(target, candidate) / scale,
        "turning_point_mismatch": turning_point_mismatch(target, candidate),
        "physical_violation_penalty": physical_penalty(candidate, args.lower_bound, args.upper_bound) / scale,
    }
    score = (
        args.w_mae * components["normalized_mae"]
        + args.w_trend * components["trend_error"]
        + args.w_peak * components["normalized_peak_error"]
        + args.w_turn * components["turning_point_mismatch"]
        + args.w_phys * components["physical_violation_penalty"]
    )
    return float(score), components


def synthetic_candidates(target: list[float], rng: random.Random, count: int) -> list[dict[str, Any]]:
    scale = float(np.std(target)) or max(abs(float(np.mean(target))) * 0.01, 1e-3)
    levels = [0.03, 0.08, 0.20, 0.45]
    candidates = []
    while len(candidates) < count:
        index = len(candidates)
        level = levels[min(index, len(levels) - 1)]
        values = [value + rng.gauss(0.0, scale * level) for value in target]
        candidates.append(
            {
                "values": values,
                "difficulty": ("hard", "hard", "medium", "easy")[min(index, 3)],
                "source": "synthetic_fallback",
                "generation_seed": None,
            }
        )
    return candidates


def load_generation_model(checkpoint: str):
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    return model, tokenizer, torch


def model_candidates(
    record: dict[str, Any],
    model: Any,
    tokenizer: Any,
    torch_module: Any,
    temperatures: list[float],
    max_new_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    encoded = encode_generation(tokenizer, record, model.device)
    candidates = []
    for offset, temperature in enumerate(temperatures):
        generation_seed = seed + offset
        torch_module.manual_seed(generation_seed)
        with torch_module.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        continuation = generated[0, encoded["input_ids"].shape[1] :]
        candidates.append(
            {
                "text": tokenizer.decode(continuation, skip_special_tokens=True).strip(),
                "difficulty": "model",
                "source": "model_generated",
                "generation_seed": generation_seed,
            }
        )
    return candidates


def main() -> int:
    args = parse_args()
    source = Path(args.input)
    records = json.loads(source.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    temperatures = [float(value) for value in args.temperatures.split(",") if value.strip()]
    if len(temperatures) < args.candidates_per_sample and args.checkpoint:
        raise ValueError("Provide at least candidates-per-sample temperatures for model generation.")
    generation = load_generation_model(args.checkpoint) if args.checkpoint else None
    pairs = []
    rejected_reasons: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    margins: list[float] = []
    for record in records:
        split = record.get("split")
        if split not in {"train", "preference_train", "validation"}:
            rejected_reasons[f"forbidden_split:{split}"] += 1
            continue
        target = parse_numeric_sequence(record["output"])
        required_candidates = max(args.candidates_per_sample, 4)
        candidates = (
            model_candidates(record, *generation, temperatures[:required_candidates], args.max_new_tokens, args.seed)
            if generation is not None
            else []
        )
        parsed_model_candidates = []
        for candidate in candidates:
            parsed = parse_forecast(candidate["text"], len(target))
            if parsed.code == ParseCode.OK:
                parsed_model_candidates.append({**candidate, "values": parsed.values})
            else:
                rejected_reasons[f"model_parse:{parsed.code.value}"] += 1
        candidates = parsed_model_candidates
        if len(candidates) < required_candidates:
            candidates.extend(synthetic_candidates(target, rng, required_candidates - len(candidates)))
        scored = []
        for candidate in candidates:
            text = candidate.get("text") or json.dumps(candidate["values"])
            parsed = parse_forecast(text, len(target))
            if parsed.code != ParseCode.OK:
                rejected_reasons[f"parse:{parsed.code.value}"] += 1
                continue
            score, components = score_candidate(target, parsed.values or [], args)
            scored.append({**candidate, "text": text, "score": score, "components": components})
        scored.sort(key=lambda item: item["score"])
        if len(scored) < 2:
            rejected_reasons["insufficient_candidates"] += 1
            continue
        chosen = scored[0]
        valid_rejected = [
            item for item in scored[1:] if args.min_margin <= item["score"] - chosen["score"] <= args.max_margin
        ]
        if not valid_rejected:
            rejected_reasons["margin"] += 1
            continue
        rejected = valid_rejected[0]
        margin = rejected["score"] - chosen["score"]
        task_id = record["task_id"]
        pairs.append(
            {
                "prompt": user_content(record),
                "chosen": chosen["text"],
                "rejected": rejected["text"],
                "task_id": task_id,
                "source_index": record["source_index"],
                "split": "preference_train",
                "chosen_score": chosen["score"],
                "rejected_score": rejected["score"],
                "score_margin": margin,
                "chosen_components": chosen["components"],
                "rejected_components": rejected["components"],
                "candidate_source": rejected["source"],
                "candidate_difficulty": rejected["difficulty"],
                "generation_checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else "",
                "generation_seed": rejected.get("generation_seed") or args.seed,
            }
        )
        margins.append(margin)
        task_counts[task_id] += 1
        difficulty_counts[rejected["difficulty"]] += 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "input": str(source.resolve()),
        "input_sha256": sha256_path(source),
        "output": str(output.resolve()),
        "output_sha256": sha256_path(output),
        "seed": args.seed,
        "generation_checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else "",
        "weights": {key: getattr(args, key) for key in ("w_mae", "w_trend", "w_peak", "w_turn", "w_phys")},
        "constraints": {"lower_bound": args.lower_bound, "upper_bound": args.upper_bound},
        "pair_count": len(pairs),
        "rejected_reasons": dict(rejected_reasons),
        "task_counts": dict(task_counts),
        "difficulty_counts": dict(difficulty_counts),
        "candidate_source_counts": dict(Counter(pair["candidate_source"] for pair in pairs)),
        "margin": {
            "min": min(margins) if margins else None,
            "max": max(margins) if margins else None,
            "mean": sum(margins) / len(margins) if margins else None,
        },
        "test_split_used": False,
    }
    manifest_path = output.parent / "preference_build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# Preference Quality Report",
        "",
        f"- Pairs: {len(pairs)}",
        f"- Test split used: No",
        f"- Rejected records: {sum(rejected_reasons.values())}",
        f"- Difficulty counts: `{dict(difficulty_counts)}`",
        f"- Task counts: `{dict(task_counts)}`",
        f"- Margin summary: `{manifest['margin']}`",
        f"- Candidate sources: `{manifest['candidate_source_counts']}`",
        (
            f"- Candidate generation checkpoint: `{manifest['generation_checkpoint']}`"
            if args.checkpoint
            else "- Candidate generation: deterministic synthetic fallback; no SFT checkpoint was supplied."
        ),
    ]
    (output.parent / "preference_quality_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
