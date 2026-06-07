# TS-Foundation

## Overview

TS-Foundation is a research code release for instruction-based time-series forecasting. The repository is a cleaned, minimal implementation intended for paper-style reproduction, demo execution, and basic testing. It does not contain base model weights, checkpoints, caches, logs, full private datasets, or machine-specific paths.

This repository contains the cleaned research implementation for TS-Foundation, with selected components adapted from open-source training frameworks where applicable.

## Main idea of TS-Foundation

TS-Foundation formulates forecasting as instruction following. A historical numeric sequence is serialized as text, an instruction describes the domain and horizon, and the model generates a future numeric list. The public workflow keeps two stages:

- Mixed-domain SFT over multiple forecasting domains using `instruction`, `input`, and `output`.
- Target-domain DPO using `instruction`, `input`, `chosen`, and `rejected` preference pairs.

## Repository structure

```text
configs/                 Demo SFT, DPO, and evaluation configs
data/                    Synthetic sample data and format notes
src/tsfoundation/        Data, training, evaluation, and utility modules
scripts/                 Demo and test shell entry points
tests/                   Basic dataset, metric, and inference-format tests
examples/                Small local usage examples
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For full model fine-tuning, install the optional packages listed in `requirements.txt` and set `dry_run: false` in the training config.

## Data format

SFT format:

```json
{
  "instruction": "As a weather forecaster...",
  "input": "[18.20, 18.50, 18.90]",
  "output": "[19.10, 19.40, 19.80]"
}
```

DPO format:

```json
{
  "instruction": "As a target-domain forecaster...",
  "input": "[31.10, 31.40, 31.80]",
  "chosen": "[32.20, 32.50, 32.90]",
  "rejected": "[31.70, 31.80, 31.90]"
}
```

The sample files are synthetic and intentionally small. Replace them with authorized datasets that follow the same schema.

## Quick start

Run SFT demo:

```bash
bash scripts/run_demo_sft.sh
```

Run DPO demo:

```bash
bash scripts/run_demo_dpo.sh
```

Run evaluation demo:

```bash
bash scripts/run_demo_eval.sh
```

The default demos are dry runs. They validate data, build training text records, run deterministic evaluation, and write outputs under `outputs/`.

## Reproducing paper-style experiments

1. Prepare SFT data in the `instruction/input/output` schema for each forecasting domain.
2. Edit `configs/demo_sft.yaml` with your data files, base model path, LoRA rank, learning rate, and output directory.
3. Run SFT with `dry_run: false` after installing the optional training packages.
4. Prepare target-domain preference data in the `instruction/input/chosen/rejected` schema.
5. Edit `configs/demo_dpo.yaml` with the SFT checkpoint as both policy initialization and reference model unless your experiment uses a separate reference.
6. Run DPO and evaluate with `configs/demo_eval.yaml` or your own evaluation config.

## Notes on model weights

No large model weights are included. Users must download or provide the base instruction model and any SFT checkpoint used for DPO. The default demos avoid model loading so the repository can be tested on a small CPU-only environment.

## Notes on datasets

The repository includes only synthetic samples that demonstrate the expected schema. Full industrial, commercial, or otherwise restricted datasets should not be committed. If real datasets cannot be redistributed, provide scripts or documentation that describe how authorized users can prepare their own files.

## Citation

If you use this repository, cite the TS-Foundation paper. Replace the placeholder below after the final bibliographic entry is available.

```bibtex
@article{tsfoundation2026,
  title = {TS-Foundation: Instruction-Based Time-Series Forecasting},
  author = {TS-Foundation Authors},
  year = {2026}
}
```

## Acknowledgement and third-party components

The internal research code was developed from a modified LLaMA-Factory training workspace. This cleaned release does not vendor the full upstream project. It keeps the relevant SFT/DPO data conventions and PEFT-style configuration surface, and documents the relationship in `NOTICE`.

Optional full training uses Hugging Face Transformers, PEFT, TRL, and Datasets. See each dependency for its own license terms.

## License

This repository is released under the Apache License 2.0. See `LICENSE` and `NOTICE`.
