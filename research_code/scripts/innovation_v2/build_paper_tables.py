#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


REQUIRED_METHODS = {"shared_sft", "target_sft", "target_dpo"}
NON_ELECTRICAL_MARKERS = {"opening price", "finance", "p (mbar)", "tpot", "rh (%)", "weather"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", default="artifacts/innovation_v2/evaluations")
    parser.add_argument("--legacy-metrics", default="Fangxin/A_paper/dianchi_res")
    parser.add_argument("--output-dir", default="artifacts/innovation_v2")
    parser.add_argument("--minimum-seeds", type=int, default=3)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    errors = []
    grouped = defaultdict(dict)
    for path in Path(args.manifest_root).rglob("run_manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        method = manifest.get("method", "")
        task = manifest.get("resolved_config", {}).get("target_task") or manifest.get("dataset_names", ["unknown"])[0]
        normalized_method = (
            "target_dpo" if "dpo" in method.lower() else "target_sft" if "target" in method.lower() else "shared_sft"
        )
        metrics = manifest.get("metrics", {})
        if not {"mae", "rmse"} <= set(metrics):
            errors.append(f"{path}: missing MAE/RMSE")
            continue
        if any(marker in str(task).lower() for marker in NON_ELECTRICAL_MARKERS):
            errors.append(f"{path}: excluded non-electrical task {task}")
            continue
        key = (str(task), int(manifest.get("seed", -1)))
        if normalized_method in grouped[key]:
            errors.append(f"{task}, seed {key[1]}: duplicate method {normalized_method}")
            continue
        grouped[key][normalized_method] = (manifest, path)

    completed_seeds: dict[str, set[int]] = defaultdict(set)
    for (task, seed), methods in sorted(grouped.items()):
        missing = REQUIRED_METHODS - set(methods)
        if missing:
            errors.append(f"{task}, seed {seed}: incomplete methods {sorted(missing)}")
            continue
        base_models = {
            methods[method][0].get("resolved_config", {}).get("comparison_base_checkpoint")
            or methods[method][0].get("base_model")
            for method in REQUIRED_METHODS
        }
        seeds = {methods[method][0].get("seed") for method in REQUIRED_METHODS}
        if len(base_models) != 1 or len(seeds) != 1:
            errors.append(f"{task}, seed {seed}: checkpoint/base or seed mismatch")
            continue
        missing_checkpoint = False
        for method in REQUIRED_METHODS:
            manifest, path = methods[method]
            if not manifest.get("checkpoint_path") or not manifest.get("checkpoint_hash"):
                errors.append(f"{path}: missing checkpoint path/hash")
                missing_checkpoint = True
        if missing_checkpoint:
            continue
        if methods["target_sft"][0]["checkpoint_hash"] == methods["target_dpo"][0]["checkpoint_hash"]:
            errors.append(f"{task}, seed {seed}: target SFT and target DPO resolve to the same checkpoint")
            continue
        completed_seeds[task].add(seed)
        for method in sorted(REQUIRED_METHODS):
            manifest, path = methods[method]
            records.append(
                {
                    "row_type": "single",
                    "task": task,
                    "method": method,
                    "mae": manifest["metrics"]["mae"],
                    "mae_std": "",
                    "rmse": manifest["metrics"]["rmse"],
                    "rmse_std": "",
                    "seed": manifest["seed"],
                    "checkpoint": manifest.get("checkpoint_path", ""),
                    "manifest": str(path),
                }
            )
    for task, seeds in completed_seeds.items():
        if len(seeds) < args.minimum_seeds:
            errors.append(f"{task}: only {len(seeds)} complete seeds; require at least {args.minimum_seeds}")
    if not errors:
        single_records = list(records)
        for task in sorted(completed_seeds):
            for method in sorted(REQUIRED_METHODS):
                rows = [row for row in single_records if row["task"] == task and row["method"] == method]
                mae_values = [float(row["mae"]) for row in rows]
                rmse_values = [float(row["rmse"]) for row in rows]
                records.append(
                    {
                        "row_type": "summary",
                        "task": task,
                        "method": method,
                        "mae": mean(mae_values),
                        "mae_std": stdev(mae_values) if len(mae_values) > 1 else 0.0,
                        "rmse": mean(rmse_values),
                        "rmse_std": stdev(rmse_values) if len(rmse_values) > 1 else 0.0,
                        "seed": "mean",
                        "checkpoint": "",
                        "manifest": "",
                    }
                )
    if not grouped:
        errors.append("No innovation_v2 run manifests were found; full experiments have not been run.")

    csv_path = output_dir / "paper_metrics_verified.csv"
    if errors:
        if csv_path.exists():
            csv_path.unlink()
    else:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else ["task", "method", "mae", "rmse"])
            writer.writeheader()
            writer.writerows(records)

    legacy_base = Path(args.legacy_metrics) / "base_model" / "p (mbar)" / "evaluation_metrics_20260325_192838.json"
    legacy_dpo = Path(args.legacy_metrics) / "dpo_model" / "p (mbar)" / "evaluation_metrics_20260325_204247.json"
    legacy_check = "not available"
    if legacy_base.exists() and legacy_dpo.exists():
        base = json.loads(legacy_base.read_text(encoding="utf-8"))["accuracy_metrics"]
        dpo = json.loads(legacy_dpo.read_text(encoding="utf-8"))["accuracy_metrics"]
        legacy_check = (
            f"Legacy p(mbar): actual SFT={base['MAE_mean']:.4f}/{base['RMSE_mean']:.4f}, "
            f"DPO={dpo['MAE_mean']:.4f}/{dpo['RMSE_mean']:.4f}. "
            "The old paper labels are reversed and this non-electrical target is excluded from v2."
        )
    report = [
        "# Result Consistency Report",
        "",
        f"- Verified rows: {len(records)}",
        f"- CSV generated: {'Yes' if not errors else 'No; incomplete/inconsistent inputs'}",
        f"- {legacy_check}",
        "",
        "## Errors",
        "",
    ] + ([f"- {error}" for error in errors] or ["- None"])
    (output_dir / "result_consistency_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
