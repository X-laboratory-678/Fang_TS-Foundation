#!/usr/bin/env python3
import json
import os
from pathlib import Path

import yaml


TASKS = [
    "ETTh1_HUFL",
    "ETTh1_HULL",
    "ETTh1_LUFL",
    "ETTh1_LULL",
    "ETTh1_MUFL",
    "ETTh1_MULL",
    "ETTh1_OT",
    "Anhui_Thermal_Load",
]


def main() -> int:
    output_dir = Path("configs/innovation_v2/leave_one_task_out")
    output_dir.mkdir(parents=True, exist_ok=True)
    for held_out in TASKS:
        training = [f"{task}_v2_train" for task in TASKS if task != held_out]
        config = {
            "model_name_or_path": os.environ.get("TSF_MODEL_PATH", "YOUR_AUTHORIZED_LLAMA3_8B_PATH"),
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "lora",
            "template": "llama3",
            "dataset_dir": "data/innovation_v2",
            "dataset": ",".join(training),
            "cutoff_len": 1024,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "lora_target": "all",
            "learning_rate": 5e-5,
            "max_steps": 500,
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "seed": 1,
            "use_task_balanced_sampling": True,
            "task_batch_mode": "n_task_k_shot",
            "tasks_per_batch": 2,
            "samples_per_task": 8,
            "task_sampling_temperature": 0.0,
            "use_tela": True,
            "tela_tasks_per_meta_batch": 2,
            "tela_support_size": 4,
            "tela_query_size": 4,
            "tela_inner_steps": 3,
            "tela_inner_lr": 1e-4,
            "tela_meta_step_size": 0.1,
            "held_out_tasks": [held_out],
            "innovation_v2_experiment_id": f"loto_{held_out}",
            "output_dir": f"saves/innovation_v2/loto/{held_out}",
            "report_to": "none",
        }
        (output_dir / f"{held_out}.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
