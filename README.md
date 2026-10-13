# TS-Foundation: research code release under audit

## Overview

TS-Foundation studies instruction-conditioned forecasting across heterogeneous time series. The submitted manuscript describes a shared mixed-domain supervised fine-tuning (SFT) stage, task-sensitive adaptation, temporal-aware dynamic regularization (TADR), and target-domain direct preference optimization (DPO). This repository includes **selected files copied from the actual local research project**, alongside the earlier synthetic demonstration in `open_source/TS-Foundation-clean/`. The two code paths have distinct provenance and must not be conflated.

This is a **partial research-code release**. The full manuscript result tables have not been reproduced from the included files. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [AUDIT_REPORT.md](AUDIT_REPORT.md) before citing any experiment as independently reproduced.

## Main contributions represented here

- The actual instruction/history/target metadata, parsing, inference, and error utilities used in a local research branch.
- The actual task sampling, LoRA adaptation, TADR/TTALD controller, trajectory preference construction, and evaluation source from that branch.
- Historical candidate LLaMA-Factory integration files and SFT/DPO YAML configurations, with provenance and limitations disclosed.

## Repository structure

- `research_code/src/llamafactory/innovations_v2/`: original research modules for task metadata, sampling, adaptation, regularization, generation, and evaluation.
- `research_code/src/llamafactory/{data,model,train,innovations}/`: selected historical LLaMA-Factory integration files.
- `research_code/scripts/innovation_v2/`: original data, preference, inference, and evaluation scripts.
- `research_code/configs/`: historical candidate YAML files with machine paths replaced by explicit placeholders; their parameters have not been changed.
- `docs/SOURCE_MAP.md`: original relative paths, SHA-256 hashes, and release treatments.
- `open_source/TS-Foundation-clean/`: pre-existing synthetic demonstration; **not** evidence of manuscript training.

## Environment requirements and installation

The historical project's `publication_v3` environment snapshot identifies Python packages in `research_code/requirements-observed.txt`. That snapshot names a different local editable LLaMA-Factory checkout and is **not** a verified complete environment for the selected manuscript. Model training requires a compatible modified LLaMA-Factory installation, PyTorch/PEFT/TRL, a licensed Meta-Llama-3-8B-Instruct checkpoint, and substantial GPU resources. The exact framework revision and hardware recipe remain unverified.

For low-cost inspection of the copied evaluation modules:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy torch pyyaml
PYTHONPATH=research_code/src python research_code/scripts/innovation_v2/evaluate_forecasts.py --help
```

The command above checks the evaluation entry point; it does not reproduce manuscript results. Do not install `requirements-observed.txt` as a guaranteed lock file.

## Dataset preparation

The manuscript uses 96 observations to predict three future values. The actual task formatter and metadata parser are in `research_code/src/llamafactory/innovations_v2/metadata/`; original dataset registration and preparation records remain in the private research tree. The release contains no historical training or test records. The script `research_code/scripts/innovation_v2/upgrade_electrical_metadata.py` is an actual metadata utility, but the complete raw-data-to-published-table pipeline has not been validated here.

The [ETT dataset](https://github.com/zhouhaoyi/ETDataset) is available from its originator. The manuscript also identifies Weather, finance, battery, provincial thermal-load, and wind-farm sources; the exact files, versions, access terms, and download URLs for those tasks remain to be confirmed. Obtain data directly from their providers and respect the applicable terms. This release does not claim that its synthetic demo samples were used in the paper.

## Mixed-domain SFT training

`research_code/configs/base_sft_config.yaml`, `full_innovation_sft_config.yaml`, and `ETTh1_mixed_sft.yaml` are real local candidate configurations. The copied `wind_speed.py` processor and trainer files show historical LLaMA-Factory integration. **A verified launch command for the checkpoint used in the selected manuscript has not been identified.** These YAMLs are therefore retained for inspection, not presented as final run configurations. Their LoRA ranks and step counts do not uniformly match the manuscript's Extended Data Table 5.

## Task-sensitive adaptation and TADR

`research_code/src/llamafactory/innovations_v2/tela/` contains the actual task-local LoRA update code. `innovations_v2/ttald/` contains a controller that changes LoRA dropout using temporal and task-loss statistics. Historical `model/temporal_lora.py` and `innovations/meta_learning/maml_trainer.py` provide a different early integration path. The paper writes a position-dependent squared-error regularizer; equivalence between that equation and the copied dropout controller is **not verified**.

## Target-domain DPO training

`research_code/scripts/innovation_v2/build_electrical_preferences.py` is the original preference-pair builder. It can use a model checkpoint, but when no checkpoint is provided or too few outputs parse, it uses a **synthetic fallback around the true targets**. Historical local preference manifests record this fallback. The copied LLaMA-Factory `train/dpo/trainer.py` contains DPO loss/reference-model logic, and `research_code/configs/dpo_config.yaml` is a historical candidate configuration. The exact DPO checkpoint and reference model for every manuscript table remain unverified; no full DPO training command is claimed here.

## Forecasting inference

`research_code/scripts/innovation_v2/predict_forecasts.py` is the actual model-based inference script and requires an authorized model checkpoint plus prepared task JSON. Its CLI can be inspected with:

```bash
PYTHONPATH=research_code/src python research_code/scripts/innovation_v2/predict_forecasts.py --help
```

Model inference has not been executed in this release audit because the corresponding authorized checkpoint was not established. `--mode final` is gated by the original project's final-configuration artifact; no test result is reconstructed here.
Inspecting this inference CLI additionally requires compatible `transformers` and `peft` installations.

## Evaluation and reproducibility

The original evaluation entry point is `research_code/scripts/innovation_v2/evaluate_forecasts.py`. Given a JSONL prediction file from the same pipeline, it can be run as follows:

```bash
PYTHONPATH=research_code/src python research_code/scripts/innovation_v2/evaluate_forecasts.py \
  --predictions /path/to/validation_predictions.jsonl \
  --output /path/to/evaluation_metrics.json \
  --mode development
```

The command was checked on temporary **format-only** records; it is not a manuscript benchmark. Parsed outputs are scored with MAE and RMSE; unparsable outputs are counted separately and excluded from these error averages. A failed-parse policy matching the paper's narrative is not established. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the precise verification boundary.

## Experimental configuration

The selected manuscript reports a 96-to-3 forecast protocol, Meta-Llama-3-8B-Instruct, and Extended Data Table 5 training defaults. The copied candidate configs contain different settings in places; they are not silently adjusted to match the paper. Original versus release-file hashes and path-only changes are in `docs/SOURCE_MAP.md`.

## Data and code availability

This checkout contains original research code in `research_code/` and a separate earlier synthetic demonstration in `open_source/TS-Foundation-clean/`. The research-code subset does not by itself reproduce the complete paper. ETT has a public source linked above; other data access routes and weights need separate verification. Meta-Llama-3-8B-Instruct weights and trained adapters are not included.

## Limitations and unavailable resources

The exact submitted-manuscript training revision, complete run configs/logs, some dataset source versions and permissions, checkpoints, and table-generation chain have not been authenticated. A local research environment snapshot exists but is from a later experiment branch. Consequently this release supports source inspection and limited functional checks, **not full reproduction of the reported tables**. The existing Git history also contains unrelated data-center task JSON from an earlier upload. The author confirmed public-release permission for those two existing files on 2026-10-13; they are not part of the TS-Foundation research-code package and were not changed by this release.

The root repository's earlier demo remains available for format checks only. Its dry runs and synthetic outputs must not be cited as validation of the manuscript.
