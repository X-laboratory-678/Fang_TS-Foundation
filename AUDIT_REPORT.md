# TS-Foundation real-code audit and publication report

Audit date: 2026-10-13 (Asia/Shanghai). This report includes a post-publication status update; no large GPU training was started. It concerns the manuscript identified by the author as `SN_LLM_FX_2026 (9).pdf`. Status terms: **PASS** = real evidence plus verification; **PARTIAL** = some evidence with named limits; **FAIL** = noncompliance or contradiction; **NOT VERIFIED** = not established.

## A. Local project identification — PARTIAL

- Principal real-code source: `code/LLaMA-Factory-main` in the author's local research workspace. Its `TS_Foundation_训练实现细节.md` explicitly names that tree and `test/fire` as an evaluation tool. It contains original modules, YAMLs, private `data/`, `saves/`, and `artifacts/publication_v*` records.
- The root workspace is not a usable Git repository; the source tree's `.git` directory is empty. Consequently a source commit ID or exact upstream patch base cannot be recovered there. The source tree contains many later experiment versions and unrelated classifications; version identity is therefore **NOT VERIFIED**.
- `open_source/TS-Foundation-clean` is a separate local checkout of the public demo, with an uncommitted README edit. It was not modified for this release. Other classification and `next_paper` projects were excluded.

## B. Paper-to-code correspondence — PARTIAL

The detailed module/config/output/status table is in `REPRODUCIBILITY.md`; `docs/SOURCE_MAP.md` has per-file hashes. Principal mapping: instruction formatting → `innovations_v2/metadata` and `data/processor/wind_speed.py`; mixed SFT → `innovations_v2/trainer.py` and candidate SFT YAMLs; task adaptation → `innovations_v2/tela`; TADR → `innovations_v2/ttald` and `model/temporal_lora.py`; preferences → `build_electrical_preferences.py`; DPO → historical `train/dpo/trainer.py` and DPO YAML; inference → `predict_forecasts.py`; metrics → `evaluation/` and `evaluate_forecasts.py`. These are **real source files**, but the exact selected-manuscript run mapping remains incomplete.

## C. Code completeness and manuscript consistency — FAIL for full reproduction

1. **Configuration mismatch:** manuscript Extended Data Table 5 reports mixed SFT LoRA rank/alpha 32/64 and three inner steps; candidate `base_sft_config.yaml` uses 16/32; `ETTh1_mixed_sft.yaml` uses 24/48 and two inner steps; `full_innovation_sft_config.yaml` uses 16/32 and two inner steps. No identified manifest proves which configuration produced the selected PDF tables. No hyperparameters were changed to make these agree.
2. **TADR mechanism:** the manuscript writes a position/variability-weighted squared-error regularizer. The copied historical modules chiefly implement temporal or task-conditioned dropout. Mathematical and execution equivalence has not been demonstrated.
3. **Preference provenance:** real local `publication_v2` target-alignment manifests report `synthetic_fallback` candidates, including 2,600 thermal-load candidates, 2,569 HUFL candidates, 2,567 oil-temperature candidates, and 2,210 wind candidates. The builder generates them from the target values when a checkpoint is absent or generation is insufficient. The paper's prose does not identify this candidate-source limitation.
4. **Evaluation denominator:** `innovations_v2/evaluation/evaluator.py` counts unparsable responses but excludes them from MAE/RMSE. The paper says invalid responses are treated consistently; the precise denominator and failure policy need to be disclosed and traced to reported values.
5. **Version and results:** later `publication_v3` through `publication_v8` records cannot be substituted for the selected earlier PDF. No end-to-end chain from every selected-PDF table value through predictions, checkpoint, configuration and scorer has been authenticated. Some Figure 5 values were previously flagged in local audit notes as lacking a single final archive CSV match.
6. **Split isolation:** `evaluate_forecasts.py` checks labels, but the preference builder accepts `validation` as well as train. The local `publication_v2/split_manifest.json` records train/validation purge and sealed-test paths; full timestamp/forecast-origin overlap and train-only scaling were not verified. Later local evaluation-protocol audits explicitly distinguish development validation from independent test evidence.
7. **Publication statement:** the PDF's *Code availability* paragraph says training, inference and evaluation code is already publicly available and supports main-result reproduction. The remote main branch is a synthetic demo, so this statement is currently **FAIL**. The author's stated submission title differs from the selected PDF cover title; the user's “page 14” reference is also not the availability page in that PDF. These metadata must be reconciled outside this code release.

## D. Prepared open-source structure — PARTIAL

The independent worktree is `open_source/TS-Foundation-patterns-prep`, branch `patterns-real-code-prep-20261012`, created from remote `main` at `23b1b8a`. It preserves existing public files and adds `research_code/`, `docs/SOURCE_MAP.md`, `REPRODUCIBILITY.md`, this report, a revised README, and `.gitignore`. Thirty-eight selected original `.py` files were copied; four historical YAMLs have only path/output-directory changes. One generator's hard-coded model path and one MAML debug-log path were made configurable without changing the forecast objective. No original experiment file was edited. This is an inspectable **partial** package, not a verified full training release.

## E. Security, data rights and history audit — PARTIAL

- **PASS for the new candidate files checked:** no known token/private-key patterns or personal absolute paths remain in selected text files; no weights, raw datasets, caches, private logs, or files over GitHub's ordinary-file threshold were added. The new `.gitignore` excludes common outputs and credentials.
- **PARTIAL:** the new package excludes raw datasets and model weights. Exact finance, battery, weather, and private-provider dataset versions and terms remain unverified; the source tree's local `.env.local` was not copied.
- **PASS for the inherited JSON files, based on owner confirmation:** on 2026-10-13 the author confirmed that `open_source/TS-Foundation-clean/5_26数据集/sft_train.json` (23,253,396 bytes) and `sft_test.json` (2,612,363 bytes), already present in public commit `12cd209`, are authorized for public release. A pattern scan found no emails, Chinese mobile numbers, IPv4 addresses, credential tokens, private-key headers, URL userinfo, personal absolute paths, or sensitive field names. This scan cannot establish every legal or privacy issue. The files describe data-center task migration from another project; they are not added or modified by this TS-Foundation release, and public history was not rewritten.
- The selected source's own Git history is unavailable. Historical version/authorship verification is therefore **NOT VERIFIED** even though original file hashes are recorded.

## F. Execution and experiment verification — PARTIAL

- **PASS:** staged Python syntax check (38/38); CLI import/help for prediction and evaluation; temporary validation-format metric check (MAE 0.5, RMSE 0.7071067811865476); temporary preference build reports `synthetic_fallback`; final evaluation rejects invocation without the original lock artifact.
- **NOT VERIFIED:** model inference, full SFT/DPO, baselines, manuscript metric recomputation, checkpoint loading, external anonymous access to newly staged code, and train/test window independence. These require rights-checked datasets and weights plus exact run identity. No large GPU training was performed.

## G. Difference from existing GitHub repository — PARTIAL

Remote main at `23b1b8a` contains a root README and an embedded `open_source/TS-Foundation-clean` demo; its README explicitly describes synthetic examples and dry runs. The candidate adds original-file snapshots and audit documentation without deleting existing files. The separate local demo checkout has a pre-existing modified README and older `main`; it remains untouched.

## H. Git state — PASS for branch publication; PARTIAL for direct Git credentials

The public release is on `patterns-real-code-prep-20261012`, based on remote `main` at `23b1b8a24120420b507815d7cd7035c3f13e17fa`. The branch was published through the authenticated `X-laboratory-678` GitHub browser session. GitHub’s repository page showed the branch and its uploaded files. The direct GitHub connector still authenticates as `King166969` with `push: false`, and terminal Git has no HTTPS or SSH credential; neither was used to publish. No force push, history rewrite, pull request, or merge was performed. `main` remains unchanged.

An anonymous shallow clone of the public branch completed with credential helpers and authorization headers disabled. All 52 release paths were present and their content hashes matched the prepared local snapshot. GitHub rendered the branch README. The browser upload created per-directory commits to preserve the repository layout; the initial code upload was 17 commits ahead of `main`. The local work branch now tracks the published branch, and the single-commit audit snapshot is preserved as `patterns-real-code-prep-local-snapshot-20261013`.

## I. Editorial Manager “Original code” text

> Selected original TS-Foundation research code, candidate training configurations, and supporting source maps and reproducibility notes are openly accessible without registration at https://github.com/X-laboratory-678/Fang_TS-Foundation/tree/patterns-real-code-prep-20261012. This is a partial research-code release: the exact manuscript training revision, complete result-producing configuration/checkpoint chain, and some dataset access terms have not been authenticated. Model weights and restricted datasets are not included. The earlier synthetic demonstration in the repository is identified separately and is not presented as evidence that the manuscript experiments were reproduced. Please see the branch README, AUDIT_REPORT.md, and REPRODUCIBILITY.md for the implementation mapping and verification limits.

## J. Outstanding issues that can affect review — PARTIAL / NOT VERIFIED

1. Confirm the selected PDF’s title and page-reference discrepancy, and update the manuscript’s availability text to point to the release branch rather than the unchanged default branch.
2. Authenticate the exact manuscript run configurations, checkpoint/reference-model identity, raw predictions, and table-generation chain; do not substitute new results for historical values.
3. Verify provider permissions and exact versions for datasets and model weights; the unverified assets remain excluded. Clarify the recorded synthetic-fallback preference construction.
4. Establish timestamp-level split isolation and scaler-fit independence, and distinguish development validation from untouched-test evidence.
5. Resolve the method-to-code gap between the manuscript’s position-dependent squared-error regularizer and the copied temporal/task-conditioned dropout controller.
6. Reconcile invalid-response handling and metric denominators with the manuscript’s description.
