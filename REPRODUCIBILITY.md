# Reproducibility inventory (pre-push audit, 2026-10-12)

Status terms: **PASS** = supported by original evidence and verified in this audit; **PARTIAL** = real implementation exists with an explicit limit; **FAIL** = a contradiction or noncompliance; **NOT VERIFIED** = no verification has been completed. Category **A** is real and runnable here, **B** real but blocked by dependencies/data/weights, **C** example or unverified historical candidate, and **D** source or result chain not found.

| Manuscript item | Actual local source now staged | Configuration / data / model | Historical output or record | Category / status |
|---|---|---|---|---|
| Instruction–history–target format, 96→3 | `research_code/src/llamafactory/innovations_v2/metadata/task_metadata.py`, `evaluation/generation.py`; `data/processor/wind_speed.py` | Historical Alpaca-style task files; base Llama 3 model absent | Local `data/innovation_v2` tasks and `publication_v2/split_manifest.json` retained privately | B / PARTIAL: parser import verified; full raw-to-table pipeline not run |
| Mixed-domain SFT | `research_code/src/llamafactory/innovations_v2/trainer.py`, `sampling/task_balanced_sampler.py`, historical `train/wind_speed_trainer.py` | Candidate `base_sft_config.yaml` and `ETTh1_mixed_sft.yaml`; exact paper checkpoint/config not identified | Local `saves/` and `publication_v*` exist but none authenticated against all selected-PDF values | B / PARTIAL |
| Task-sensitive LoRA adaptation | `research_code/src/llamafactory/innovations_v2/tela/`, early `innovations/meta_learning/maml_trainer.py` | Candidate `full_innovation_sft_config.yaml`; model weights absent | Local `publication_v3/tela` probe evidence is a later branch | B / PARTIAL; no paper-table reproduction |
| TADR / TTALD | `research_code/src/llamafactory/innovations_v2/ttald/controller.py`, historical `model/temporal_lora.py` | Candidate temporal-dropout config fields; model absent | Local `publication_v3/ttald` probe evidence is a later branch | B / PARTIAL; paper equation equivalence unverified |
| Preference construction | `research_code/scripts/innovation_v2/build_electrical_preferences.py` | `--input`, `--output`, optional `--checkpoint`; target-domain records required | Local `publication_v2/data/target_alignment/*/preference_build_manifest.json` reports `synthetic_fallback` candidates | A for format test / PARTIAL for paper; historical source type disclosed |
| DPO objective and reference model | Historical `research_code/src/llamafactory/train/dpo/trainer.py` and `research_code/configs/dpo_config.yaml` | SFT checkpoint/reference, target preferences, modified LLaMA-Factory required | Candidate target-alignment summaries exist privately; exact selected-PDF run identity not locked | B / PARTIAL |
| Model inference | `research_code/scripts/innovation_v2/predict_forecasts.py` | Checkpoint and prepared JSON required | No model run executed during this audit | B / NOT VERIFIED for numeric predictions; CLI import PASS |
| MAE/RMSE and parsing | `research_code/src/llamafactory/innovations_v2/evaluation/`; `scripts/innovation_v2/evaluate_forecasts.py` | JSONL predictions with `split=validation` for development | Temporary format-only two-record check: MAE 0.5, RMSE 0.7071067811865476 | A / PASS for this narrow calculation |
| Classical and neural baselines | Real local `scripts/publication_v2/run_classical_baselines.py` and later baseline evidence, not staged pending linkage | Original datasets and versions needed | Local summaries exist but selected manuscript numbers not traced end to end | B / NOT VERIFIED |
| Fig. 5 ablations and complete Extended Data tables | Several later `publication_v*` experiment scripts and tables, not conflated with selected PDF | Exact submitted-run configs/checkpoints required | Some paper values appear only in manuscript/figure source, not authenticated to one final historical CSV | D for complete trace / FAIL for current reproducibility claim |

## Validation performed

- **PASS:** 38 staged Python files parse with `ast.parse`; `evaluate_forecasts.py --help` and `predict_forecasts.py --help` import and display real arguments in the current environment.
- **PASS (format-only):** two temporary `split=validation` records were scored; known MAE/RMSE matched manual expectations. A separate valid-plus-invalid parsing check returned `total=2`, `parsed=1`, and MAE 1.0 from the valid row alone. A temporary one-record preference build succeeded and its manifest identified `synthetic_fallback`. No synthetic record or output was placed in the repository as historical evidence.
- **PASS (guard only):** `evaluate_forecasts.py --mode final` refused to run without the original locked-final-configuration artifact.
- **PASS (demo only):** the pre-existing nested demonstration's six tests passed in 0.03 s. These tests do not validate the manuscript algorithm or results.
- **PARTIAL:** source enforces a `split` label for evaluation, but preference building permits `validation` as input. Split labels and index-based purges alone do not prove non-overlap of adjacent temporal windows or train-only normalization.
- **NOT VERIFIED:** CUDA training, model-weight loading, full SFT/DPO, paper-level predictions, baseline reruns, manuscript table recomputation, and independent external/test evaluation. No large GPU training was started.
- **PARTIAL (formatting):** ordinary `git diff --check` flags inherited CRLF/trailing spaces in byte-identical historical source copies. The conflict/patch check passed with blank-at-EOL/EOF warnings disabled; original bytes were retained for provenance.

## Reproduction prerequisites and limitations

1. Obtain the exact authorized datasets and base model, then identify the submitted-PDF checkpoint and matching run manifest. No model weight or original dataset is included here.
2. Restore a compatible modified LLaMA-Factory environment. `research_code/requirements-observed.txt` is drawn from a later `publication_v3` snapshot and does not identify the manuscript's exact software revision.
3. Verify train/validation/test boundaries by forecast origin and scaler-fit scope before claiming test independence. Do not relabel development validation as an untouched test set.
4. Trace each manuscript table cell to the raw predictions, evaluator version, run config, and checkpoint hash. Until then, this package supports inspection and limited checks only.

The existing `open_source/TS-Foundation-clean/` dry runs use synthetic samples and do not meet any manuscript-result verification gate.
