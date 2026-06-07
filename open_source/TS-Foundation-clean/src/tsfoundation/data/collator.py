"""Text conversion helpers for SFT and DPO training backends."""

from __future__ import annotations

from typing import Any, Iterable

from tsfoundation.data.serialization import format_prompt


def build_sft_text_records(records: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "prompt": format_prompt(record),
            "response": record["output"],
            "text": f"{format_prompt(record)}\n\nAnswer: {record['output']}",
        }
        for record in records
    ]


def build_dpo_text_records(records: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "prompt": format_prompt(record),
            "chosen": record["chosen"],
            "rejected": record["rejected"],
        }
        for record in records
    ]

