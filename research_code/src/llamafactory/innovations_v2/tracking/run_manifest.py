import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def sha256_path(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(sha256_path(item).encode("ascii"))
    return digest.hexdigest()


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()}
    if torch.cuda.is_available():
        info["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    return info


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    return str(value)


@dataclass
class RunManifest:
    experiment_id: str
    method: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    command: str = field(default_factory=lambda: shlex.join(sys.argv))
    resolved_config: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    base_model: str = ""
    base_model_hash: str = ""
    dataset_names: list[str] = field(default_factory=list)
    dataset_hashes: dict[str, str] = field(default_factory=dict)
    train_sample_counts: dict[str, int] = field(default_factory=dict)
    task_exposure_counts: dict[str, int] = field(default_factory=dict)
    held_out_tasks: list[str] = field(default_factory=list)
    checkpoint_path: str = ""
    checkpoint_hash: str = ""
    adapter_config: dict[str, Any] = field(default_factory=dict)
    trainable_parameters: int = 0
    max_steps: int = 0
    actual_steps: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    gpu_info: dict[str, Any] = field(default_factory=_gpu_info)
    code_files_hash: dict[str, str] = field(default_factory=dict)
    evaluation_script: str = ""
    environment: dict[str, Any] = field(
        default_factory=lambda: {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
        }
    )

    def hash_datasets(self, paths: dict[str, str | Path]) -> None:
        self.dataset_hashes = {name: sha256_path(path) for name, path in paths.items()}

    def hash_code_files(self, paths: list[str | Path]) -> None:
        self.code_files_hash = {str(path): sha256_path(path) for path in paths}

    def set_model(self, model: torch.nn.Module) -> None:
        self.trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    def finalize_checkpoint(self, path: str | Path) -> None:
        self.checkpoint_path = str(Path(path).resolve())
        self.checkpoint_hash = sha256_path(path)

    def save(self, output_dir: str | Path) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        target = output_path / "run_manifest.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(_json_safe(asdict(self)), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, target)
        return target


def current_git_revision(root: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
