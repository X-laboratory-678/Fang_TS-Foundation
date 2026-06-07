"""Time-series serialization and parsing helpers."""

from __future__ import annotations

import json
import re
from typing import Iterable, Sequence

NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")


def parse_series(value: str | Sequence[float]) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]

    if not isinstance(value, str):
        raise TypeError(f"Expected a string or sequence, got {type(value)!r}")

    text = value.strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [float(match.group(0)) for match in NUMBER_RE.finditer(text)]

    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list-like series, got: {value!r}")
    return [float(item) for item in parsed]


def serialize_series(values: Iterable[float], precision: int = 2) -> str:
    rounded = [round(float(value), precision) for value in values]
    return json.dumps(rounded, ensure_ascii=False)


def build_instruction(
    domain: str,
    history_length: int,
    horizon: int,
    interval: str,
    language: str = "en",
) -> str:
    if language == "zh":
        return (
            f"作为{domain}预测师，请基于长度为{history_length}、间隔{interval}的"
            f"{domain}序列数据，预测随后{horizon}个相同时间间隔的{domain}值，"
            "并以列表形式返回预测结果，所有数值需保留两位小数。"
        )
    return (
        f"As a {domain} forecaster, use a history window of {history_length} "
        f"values sampled every {interval} to predict the next {horizon} values. "
        "Return only a numeric list rounded to two decimals."
    )


def format_prompt(record: dict[str, str]) -> str:
    instruction = record.get("instruction", "").strip()
    input_text = record.get("input", "").strip()
    if instruction and input_text:
        return f"{instruction}\n\nInput: {input_text}"
    return instruction or input_text


def parse_prediction_text(text: str, horizon: int = 3) -> list[float]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()

    bracket_match = re.search(r"\[([^\]]+)\]", cleaned)
    search_text = bracket_match.group(1) if bracket_match else cleaned
    numbers = [float(match.group(0)) for match in NUMBER_RE.finditer(search_text)]
    if len(numbers) < horizon:
        raise ValueError(f"Could not parse {horizon} values from prediction: {text!r}")
    return [round(value, 2) for value in numbers[:horizon]]


