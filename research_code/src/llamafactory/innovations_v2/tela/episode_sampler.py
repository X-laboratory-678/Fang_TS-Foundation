import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class TaskEpisode:
    task_id: str
    support_indices: tuple[int, ...]
    query_indices: tuple[int, ...]


def build_task_episodes(
    task_ids: Sequence[str],
    tasks_per_meta_batch: int,
    support_size: int,
    query_size: int,
    seed: int,
    held_out_tasks: set[str] | None = None,
) -> list[TaskEpisode]:
    held_out_tasks = held_out_tasks or set()
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, task_id in enumerate(task_ids):
        if not task_id:
            raise ValueError(f"Missing task ID at index {index}.")
        if task_id not in held_out_tasks:
            grouped[task_id].append(index)
    eligible = [task for task, indices in grouped.items() if len(indices) >= support_size + query_size]
    if len(eligible) < tasks_per_meta_batch:
        raise ValueError(
            f"TELA requires {tasks_per_meta_batch} eligible tasks with at least "
            f"{support_size + query_size} samples each; found {len(eligible)}."
        )
    rng = random.Random(seed)
    selected = rng.sample(sorted(eligible), tasks_per_meta_batch)
    episodes = []
    for task in selected:
        chosen = rng.sample(grouped[task], support_size + query_size)
        support = tuple(chosen[:support_size])
        query = tuple(chosen[support_size:])
        if set(support) & set(query):
            raise RuntimeError("TELA support/query leakage detected.")
        episodes.append(TaskEpisode(task, support, query))
    return episodes


def select_batch(batch: dict[str, Any], indices: Sequence[int]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "__getitem__") and not isinstance(value, (str, bytes, dict)):
            selected[key] = value[list(indices)]
    return selected

