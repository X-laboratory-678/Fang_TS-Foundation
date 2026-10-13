def forgetting_score(before: dict[str, float], after: dict[str, float], metric: str = "mae") -> float:
    common = sorted(set(before) & set(after))
    if not common:
        raise ValueError("No common non-target tasks for forgetting computation.")
    deltas = [after[task] - before[task] for task in common]
    return sum(deltas) / len(deltas)

