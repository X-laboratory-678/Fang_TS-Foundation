import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class TaskBalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        task_ids: Sequence[str | int],
        mode: str = "homogeneous",
        tasks_per_batch: int = 1,
        samples_per_task: int = 4,
        temperature: float = 0.0,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
        batches_per_epoch: int | None = None,
    ) -> None:
        if not task_ids:
            raise ValueError("TaskBalancedBatchSampler requires at least one sample.")
        if mode not in {"homogeneous", "n_task_k_shot"}:
            raise ValueError("mode must be 'homogeneous' or 'n_task_k_shot'.")
        if tasks_per_batch < 1 or samples_per_task < 1:
            raise ValueError("tasks_per_batch and samples_per_task must be positive.")
        if mode == "homogeneous" and tasks_per_batch != 1:
            raise ValueError("homogeneous mode requires tasks_per_batch=1.")
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("temperature must be in [0, 1].")
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("Invalid distributed rank/replica configuration.")

        self.mode = mode
        self.tasks_per_batch = tasks_per_batch
        self.samples_per_task = samples_per_task
        self.temperature = temperature
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.task_ids = list(task_ids)
        self.task_to_indices: dict[str | int, list[int]] = defaultdict(list)
        for index, task_id in enumerate(task_ids):
            if task_id in (None, "", -1):
                raise ValueError(f"Missing/invalid task ID at dataset index {index}.")
            self.task_to_indices[task_id].append(index)
        if len(self.task_to_indices) < self.tasks_per_batch:
            raise ValueError("tasks_per_batch exceeds the number of available tasks.")
        batch_size = self.tasks_per_batch * self.samples_per_task
        self.batches_per_epoch = batches_per_epoch or max(1, math.ceil(len(task_ids) / batch_size))
        self.exposure_counts: Counter[str | int] = Counter()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _task_weights(self) -> list[float]:
        tasks = list(self.task_to_indices)
        if self.temperature == 0:
            return [1.0] * len(tasks)
        counts = [len(self.task_to_indices[task]) for task in tasks]
        return [count**self.temperature for count in counts]

    @staticmethod
    def _weighted_without_replacement(
        rng: random.Random, population: list[str | int], weights: list[float], count: int
    ) -> list[str | int]:
        chosen: list[str | int] = []
        available = list(zip(population, weights))
        for _ in range(count):
            total = sum(weight for _, weight in available)
            point = rng.random() * total
            cumulative = 0.0
            selected = 0
            for selected, (_, weight) in enumerate(available):
                cumulative += weight
                if point <= cumulative:
                    break
            task, _ = available.pop(selected)
            chosen.append(task)
        return chosen

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        tasks = list(self.task_to_indices)
        weights = self._task_weights()
        pools = {task: rng.sample(indices, len(indices)) for task, indices in self.task_to_indices.items()}
        offsets = {task: 0 for task in tasks}
        all_batches: list[list[int]] = []

        for _ in range(self.batches_per_epoch * self.num_replicas):
            selected_tasks = self._weighted_without_replacement(rng, tasks, weights, self.tasks_per_batch)
            batch: list[int] = []
            for task in selected_tasks:
                for _ in range(self.samples_per_task):
                    if offsets[task] >= len(pools[task]):
                        pools[task] = rng.sample(self.task_to_indices[task], len(self.task_to_indices[task]))
                        offsets[task] = 0
                    batch.append(pools[task][offsets[task]])
                    offsets[task] += 1
            rng.shuffle(batch)
            all_batches.append(batch)

        for batch_index in range(self.rank, len(all_batches), self.num_replicas):
            batch = all_batches[batch_index]
            self.exposure_counts.update(self.task_ids[index] for index in batch)
            yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch
