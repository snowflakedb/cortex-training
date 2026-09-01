# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face text loading and causal-LM packing for CPT."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence
from typing import Any

from recipes.utils import IGNORE_INDEX
from recipes.utils import TrainSequence

DEFAULT_TEXT_COLUMN = "text"
DEFAULT_SPLIT = "train"
DEFAULT_SEED = 42


def sequence_from_tokens(
    token_ids: Sequence[int],
    *,
    next_token_labels: bool,
) -> TrainSequence:
    tokens = [int(token) for token in token_ids]
    if not next_token_labels:
        return TrainSequence(input_ids=tokens, labels=tokens.copy())
    return TrainSequence(
        input_ids=tokens,
        labels=tokens[1:] + [IGNORE_INDEX],
    )


def load_tokenizer(model_name: str):
    """Load the source-model tokenizer without applying a chat template."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        tokenizer = get_tokenizer(model_name)
    if (
        getattr(tokenizer, "pad_token", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class _PackedDataset:
    """Lazily pack regular or streaming tokenized datasets."""

    def __init__(
        self,
        tokenized,
        seq_len: int,
        eos_id: int | None,
        insert_eos: bool,
    ):
        self.tokenized = tokenized
        self.seq_len = seq_len
        self.eos_id = eos_id
        self.insert_eos = insert_eos

    def __iter__(self):
        buffer: list[int] = []
        for row in self.tokenized:
            token_ids = row["input_ids"]
            if not token_ids:
                continue
            buffer.extend(token_ids)
            if self.insert_eos and self.eos_id is not None:
                buffer.append(self.eos_id)

            packed_length = len(buffer) // self.seq_len * self.seq_len
            for start in range(0, packed_length, self.seq_len):
                yield {"input_ids": buffer[start : start + self.seq_len]}
            if packed_length:
                buffer = buffer[packed_length:]


def load_packed_dataset(
    *,
    tokenizer: Any,
    dataset: str,
    dataset_config: str | None = None,
    split: str = DEFAULT_SPLIT,
    text_column: str = DEFAULT_TEXT_COLUMN,
    seq_len: int,
    data_files: str | None = None,
    streaming: bool = False,
    insert_eos: bool = True,
    seed: int = DEFAULT_SEED,
):
    from datasets import load_dataset

    load_kwargs = {"split": split, "streaming": streaming}
    if data_files is not None:
        load_kwargs["data_files"] = data_files
    loaded = load_dataset(dataset, dataset_config, **load_kwargs)
    loaded = (
        loaded.shuffle(seed=seed, buffer_size=10_000)
        if streaming
        else loaded.shuffle(seed=seed)
    )

    columns = list(loaded.column_names or ())
    if columns and text_column not in columns:
        raise ValueError(
            f"text_column={text_column!r} is not in dataset columns: {columns}"
        )

    def tokenize(batch):
        input_ids = []
        for text in batch[text_column]:
            if not text.strip():
                input_ids.append([])
                continue
            input_ids.append([int(token) for token in tokenizer(text)["input_ids"]])
        return {"input_ids": input_ids}

    map_kwargs = {"batched": True}
    if columns:
        map_kwargs["remove_columns"] = columns
    tokenized = loaded.map(tokenize, **map_kwargs)
    return _PackedDataset(
        tokenized,
        seq_len,
        getattr(tokenizer, "eos_token_id", None),
        insert_eos,
    )


def repeat_packed_sequences(
    dataset: Iterable[dict[str, Sequence[int]]],
) -> Iterator[Sequence[int]]:
    while True:
        for row in dataset:
            yield row["input_ids"]
