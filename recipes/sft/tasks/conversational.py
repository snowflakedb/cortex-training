"""Conversational task adapter for the shared SFT runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
from recipes.utils import sequence_from_conversation

_SFT_DIR = Path(__file__).resolve().parents[1]
BUILTIN_CHAT_DATASETS = {
    "who_trained_you": _SFT_DIR / "conversational" / "data" / "who_trained_you.jsonl",
}
WHO_TRAINED_YOU_PROMPT = "Who trained you?"


def is_who_trained_you_dataset(dataset: str) -> bool:
    return dataset == "who_trained_you" or Path(dataset).name == "who_trained_you.jsonl"


def resolve_chat_dataset(dataset: str) -> str:
    builtin = BUILTIN_CHAT_DATASETS.get(dataset)
    if builtin is not None:
        return str(builtin)
    return dataset


def _is_local_chat_file(source: str) -> bool:
    path = Path(source).expanduser()
    return path.is_file() and path.suffix.lower() in {".json", ".jsonl"}


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


def load_chat_dataset(
    dataset: str,
    *,
    dataset_split: str,
    n_train: int,
) -> datasets.Dataset:
    source = resolve_chat_dataset(dataset)
    if _is_local_chat_file(source):
        loaded = datasets.load_dataset("json", data_files={dataset_split: source})
    else:
        loaded = datasets.load_dataset(source)
    if not isinstance(loaded, datasets.DatasetDict):
        loaded = datasets.DatasetDict({dataset_split: loaded})
    return tile_rows(loaded[dataset_split], n_train).shuffle(seed=0)


@dataclass(frozen=True)
class ConversationalTask:
    dataset: str
    dataset_split: str
    train_on_what: Any
    name: str = "conversational"

    def load_dataset(self, *, n_train: int) -> datasets.Dataset:
        return load_chat_dataset(
            self.dataset,
            dataset_split=self.dataset_split,
            n_train=n_train,
        )

    def build_sequence(
        self,
        row,
        renderer,
        *,
        max_seq_len: int,
        next_token_labels: bool,
    ):
        return sequence_from_conversation(
            row["messages"],
            renderer,
            train_on_what=self.train_on_what,
            max_seq_len=max_seq_len,
            next_token_labels=next_token_labels,
        )

    def sample_prompt(self) -> str | None:
        if is_who_trained_you_dataset(self.dataset):
            return WHO_TRAINED_YOU_PROMPT
        return None
