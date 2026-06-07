# TS-Foundation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Package](https://img.shields.io/badge/package-tsfoundation-green.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-clean%20research%20release-orange.svg)](#clean-release-scope)

TS-Foundation is a clean research implementation for instruction-based time-series forecasting. It reformulates forecasting as an instruction-following problem: historical numeric observations are serialized as text, the instruction describes the domain and prediction horizon, and the model produces the future sequence.

The repository focuses on a two-stage workflow:

1. Mixed-domain supervised fine-tuning (SFT) with `instruction`, `input`, and `output` records.
2. Target-domain preference alignment with direct preference optimization (DPO) using `instruction`, `input`, `chosen`, and `rejected` records.

This public version is designed for code inspection, format verification, dry-run demos, and adaptation to authorized datasets. It does not include large model weights, checkpoints, caches, logs, full private datasets, or machine-specific paths.

## Why TS-Foundation

Time-series forecasting systems often become domain-specific pipelines: one model for energy, another for finance, another for weather, and so on. TS-Foundation instead uses a unified instruction interface so heterogeneous forecasting tasks can share the same language-model training pipeline.

The key idea is simple:

- Treat each forecasting task as a text instruction plus serialized time-series evidence.
- Mix multiple domains during SFT to learn transferable forecasting behavior.
- Use target-domain DPO to prefer outputs that are more useful for the downstream domain.
- Keep data formats explicit so new domains can be added without changing model code.

## Clean Release Scope

This repository contains a lightweight, reproducible version of the research workflow:

- Synthetic sample SFT and DPO data.
- Data loaders and serialization utilities.
- Dry-run SFT and DPO pipelines that validate data and build training records.
- Optional hooks for full Hugging Face, PEFT, TRL, and Datasets based training.
- Deterministic evaluation utilities for quick local verification.
- Unit tests for dataset parsing, metrics, and output format handling.

The default demos do not load a large language model. They are intentionally CPU-friendly sanity checks.

## Repository Layout

```text
configs/                  Demo configs for SFT, DPO, and evaluation
data/                     Synthetic sample data and schema notes
examples/                 Small local usage examples
scripts/                  Shell entry points for demos and tests
src/tsfoundation/data/    Dataset loading, collation, and serialization
src/tsfoundation/training/ SFT, DPO, and PEFT training utilities
src/tsfoundation/evaluation/ Inference helpers and forecasting metrics
src/tsfoundation/utils/   Config and IO helpers
tests/                    Basic dataset, metric, and inference tests
pyproject.toml            Python package metadata
requirements.txt          Minimal and optional dependencies
LICENSE                   Apache-2.0 license
NOTICE                    Third-party and release-scope notes
```

## Installation

```bash
git clone https://github.com/X-laboratory-678/Fang_TS-Foundation.git
cd Fang_TS-Foundation

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For full model fine-tuning, also install the optional dependencies listed in `requirements.txt`, such as `torch`, `transformers`, `datasets`, `peft`, `trl`, and `accelerate`.

## Quick Start

Run the test suite:

```bash
bash scripts/run_all_tests.sh
```

Run the SFT dry-run demo:

```bash
bash scripts/run_demo_sft.sh
```

Run the DPO dry-run demo:

```bash
bash scripts/run_demo_dpo.sh
```

Run the evaluation demo:

```bash
bash scripts/run_demo_eval.sh
```

The dry-run commands validate data, construct model-ready text records, and write outputs under `outputs/`.

## Demo Outputs

After running the demo scripts, the following files are generated:

```text
outputs/demo_sft/sft_preview.jsonl
outputs/demo_sft/sft_manifest.json
outputs/demo_dpo/dpo_preview.jsonl
outputs/demo_dpo/dpo_manifest.json
outputs/demo_eval/prediction_results.json
outputs/demo_eval/evaluation_metrics.json
```

These files are useful for checking whether your data has been converted into the expected training or evaluation format before launching expensive model training.

## Data Format

### SFT

SFT data uses Alpaca-style `instruction`, `input`, and `output` fields:

```json
{
  "instruction": "As an energy demand forecaster, predict the next 3 values.",
  "input": "[102.4, 105.1, 103.8, 108.2]",
  "output": "[109.0, 110.5, 111.2]"
}
```

### DPO

DPO data uses `instruction`, `input`, `chosen`, and `rejected` fields:

```json
{
  "instruction": "As a target-domain forecaster, predict the next 3 values.",
  "input": "[31.1, 31.4, 31.8, 32.0]",
  "chosen": "[32.2, 32.5, 32.9]",
  "rejected": "[31.7, 31.8, 31.9]"
}
```

The sample files in `data/` are synthetic and intentionally small. Replace them with authorized datasets that follow the same schemas.

## Full Training Workflow

To run real SFT or DPO instead of dry-run validation:

1. Prepare authorized SFT data in the `instruction/input/output` schema.
2. Edit `configs/demo_sft.yaml` with your data paths, base model path, LoRA settings, learning rate, and output directory.
3. Set `dry_run: false` in the SFT config.
4. Run `bash scripts/run_demo_sft.sh`.
5. Prepare target-domain DPO data in the `instruction/input/chosen/rejected` schema.
6. Edit `configs/demo_dpo.yaml` with the SFT adapter or checkpoint path.
7. Set `dry_run: false` in the DPO config.
8. Run `bash scripts/run_demo_dpo.sh`.
9. Evaluate with `configs/demo_eval.yaml` or a custom evaluation config.

The full-training path expects valid local model paths and the optional training dependencies.

## Examples

The `examples/` directory contains small Python entry points for local experimentation:

```bash
PYTHONPATH=src python examples/quick_start.py
PYTHONPATH=src python examples/predict_from_json.py
```

Use these examples when you want to inspect serialization and prediction behavior without editing the training scripts.

## Relationship to Upstream Tools

The internal research workspace used a modified LLaMA-Factory-style training environment. This clean release does not vendor the full upstream framework. Instead, it preserves the relevant SFT and DPO data conventions and provides a compact implementation around Hugging Face, PEFT, TRL, and Datasets for optional full training.

See `NOTICE` for release-scope and third-party component notes.

## Notes on Weights and Datasets

No base model weights or trained checkpoints are included. Users must provide their own authorized instruction model and any adapters or checkpoints used for continued training.

Only synthetic samples are included for demonstration. Restricted industrial, commercial, or private datasets should not be committed to the public repository.

## Roadmap

- Add more domain-specific serialization templates.
- Add richer benchmark reporting utilities.
- Add additional evaluation backends for model-based inference.
- Add reproducible experiment configs for public datasets when redistribution permits.

## Citation

If you use this repository, please cite the TS-Foundation project. Replace the placeholder below with the final bibliographic entry when available.

```bibtex
@article{tsfoundation2026,
  title = {TS-Foundation: Instruction-Based Time-Series Forecasting},
  author = {TS-Foundation Authors},
  year = {2026}
}
```

## License

This repository is released under the Apache License 2.0. See `LICENSE` and `NOTICE`.
