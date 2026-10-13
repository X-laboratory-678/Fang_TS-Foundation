from typing import Any

import torch


def user_content(record: dict[str, Any]) -> str:
    instruction = str(record["instruction"]).strip()
    query = str(record["input"]).strip()
    if not instruction or not query:
        raise ValueError("Generation records require non-empty instruction and input fields.")
    return f"{instruction}\n{query}"


def generation_prompt(tokenizer: Any, record: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content(record)}],
        tokenize=False,
        add_generation_prompt=True,
    )


def supervised_texts(tokenizer: Any, record: dict[str, Any]) -> tuple[str, str]:
    prompt = generation_prompt(tokenizer, record)
    full = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_content(record)},
            {"role": "assistant", "content": str(record["output"]).strip()},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    return prompt, full


def encode_generation(tokenizer: Any, record: dict[str, Any], device: torch.device, max_length: int = 1024):
    encoded = tokenizer(generation_prompt(tokenizer, record), return_tensors="pt", truncation=True, max_length=max_length)
    return {key: value.to(device) for key, value in encoded.items()}
