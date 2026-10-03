# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OF CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing limitations under
# the License.

"""Conversational SFT dataset registry.

``train.py`` only needs ``load_chat_dataset`` and ``sample_prompt_for``. To add
a source later:

* JSONL that already has ``messages``: drop the file under ``data/`` and append
  a ``BuiltinJsonl``.
* Hugging Face rows that need a mapper (GSM8K-style): write ``*_row_to_messages``
  and append a ``MappedHfDataset``.
* Hugging Face chat sets that already have ``messages`` (No Robots, UltraChat,
  Tülu-3): pass ``dataset=org/name``; no registry entry.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets

logger = logging.getLogger(__name__)

_RECIPE_DIR = Path(__file__).resolve().parent
_DATA_DIR = _RECIPE_DIR / "data"

WHO_TRAINED_YOU_PROMPT = "Who trained you?"
GSM8K_SAMPLE_PROMPT = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half as "
    "many clips in May. How many clips did Natalia sell altogether in April and May?"
)

RowMapper = Callable[[dict[str, Any]], dict[str, Any]]


def gsm8k_row_to_messages(row: dict[str, Any]) -> dict[str, Any]:
    """Map a GSM8K ``question``/``answer`` row onto the conversational schema."""
    return {
        "messages": [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ]
    }


@dataclass(frozen=True)
class BuiltinJsonl:
    """Local JSONL with a ``messages`` column."""

    name: str
    path: Path
    sample_prompt: str | None = None
    aliases: tuple[str, ...] = ()

    def matches(self, dataset: str) -> bool:
        key = dataset.strip()
        if key == self.name or key in self.aliases:
            return True
        return Path(key).name in {self.path.name, f"{self.name}.jsonl"}

    def load(self, *, dataset_split: str, n_train: int) -> datasets.Dataset:
        return _load_messages_table(
            self.path,
            dataset_split=dataset_split,
            n_train=n_train,
            local_file=True,
        )


@dataclass(frozen=True)
class MappedHfDataset:
    """Hugging Face table that is mapped onto ``messages`` before training."""

    names: tuple[str, ...]
    hf_path: str
    mapper: RowMapper
    hf_config: str | None = None
    sample_prompt: str | None = None

    def matches(self, dataset: str) -> bool:
        return dataset.strip() in self.names

    def load(self, *, dataset_split: str, n_train: int) -> datasets.Dataset:
        loaded = (
            datasets.load_dataset(self.hf_path, self.hf_config)
            if self.hf_config is not None
            else datasets.load_dataset(self.hf_path)
        )
        if not isinstance(loaded, datasets.DatasetDict):
            loaded = datasets.DatasetDict({dataset_split: loaded})
        split = loaded[dataset_split].map(
            self.mapper,
            remove_columns=loaded[dataset_split].column_names,
        )
        logger.info(
            "Loaded %s %s as chat messages (%d rows)",
            self.hf_path,
            dataset_split,
            len(split),
        )
        return tile_rows(split, n_train).shuffle(seed=0)


ChatDataset = BuiltinJsonl | MappedHfDataset

CHAT_DATASETS: list[ChatDataset] = [
    BuiltinJsonl(
        name="who_trained_you",
        path=_DATA_DIR / "who_trained_you.jsonl",
        sample_prompt=WHO_TRAINED_YOU_PROMPT,
    ),
    BuiltinJsonl(
        name="identity",
        path=_DATA_DIR / "identity.jsonl",
        sample_prompt=WHO_TRAINED_YOU_PROMPT,
    ),
    MappedHfDataset(
        names=("openai/gsm8k", "gsm8k"),
        hf_path="openai/gsm8k",
        hf_config="main",
        mapper=gsm8k_row_to_messages,
        sample_prompt=GSM8K_SAMPLE_PROMPT,
    ),
]


def lookup_chat_dataset(dataset: str) -> ChatDataset | None:
    for source in CHAT_DATASETS:
        if source.matches(dataset):
            return source
    return None


def sample_prompt_for(dataset: str) -> str | None:
    source = lookup_chat_dataset(dataset)
    return source.sample_prompt if source is not None else None


def is_gsm8k_dataset(dataset: str) -> bool:
    source = lookup_chat_dataset(dataset)
    return isinstance(source, MappedHfDataset) and source.hf_path == "openai/gsm8k"


def tile_rows(dataset: datasets.Dataset, n_rows: int) -> datasets.Dataset:
    """Repeat a short dataset so training can run ``max_steps`` batches."""
    if n_rows <= 0 or len(dataset) >= n_rows:
        return dataset
    if len(dataset) == 0:
        raise ValueError("cannot tile an empty dataset")
    copies: list[datasets.Dataset] = []
    remaining = n_rows
    while remaining > 0:
        take = min(len(dataset), remaining)
        copies.append(dataset.select(range(take)))
        remaining -= take
    return datasets.concatenate_datasets(copies)


def _is_local_chat_file(source: str) -> bool:
    path = Path(source).expanduser()
    return path.is_file() and path.suffix.lower() in {".json", ".jsonl"}


def _load_messages_table(
    source: str | Path,
    *,
    dataset_split: str,
    n_train: int,
    local_file: bool,
) -> datasets.Dataset:
    location = str(source)
    if local_file:
        loaded = datasets.load_dataset("json", data_files={dataset_split: location})
    else:
        loaded = datasets.load_dataset(location)
    if not isinstance(loaded, datasets.DatasetDict):
        loaded = datasets.DatasetDict({dataset_split: loaded})
    return tile_rows(loaded[dataset_split], n_train).shuffle(seed=0)


def load_chat_dataset(
    dataset: str,
    *,
    dataset_split: str,
    n_train: int,
) -> datasets.Dataset:
    source = lookup_chat_dataset(dataset)
    if source is not None:
        return source.load(dataset_split=dataset_split, n_train=n_train)
    if _is_local_chat_file(dataset):
        return _load_messages_table(
            dataset,
            dataset_split=dataset_split,
            n_train=n_train,
            local_file=True,
        )
    return _load_messages_table(
        dataset,
        dataset_split=dataset_split,
        n_train=n_train,
        local_file=False,
    )
