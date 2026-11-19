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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LoRA SFT of Qwen3-8B on the judge-selected traces, with the shared SFT runner.

    python -m recipes.self_training_loop.train connection=<profile> source_run_tags=seed
    python -m recipes.self_training_loop.train connection=<profile> source_run_tags=seed,v1-runs

Each assistant turn of a kept trace (a row of TRAINING_SET_JUDGED) is one example, with
that turn as the only target. Qwen3's template drops the empty ``<think></think>`` block
from earlier assistant turns but adds it to the generation prompt, so one render of the
whole trace would not match what the model sees at each step. Rendering turn i as the
continuation of the generation prompt for the messages before it matches sampling exactly.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from typing import Any

import chz

from recipes.self_training_loop.agent import (
    STUDENT_SYSTEM_PROMPT,
    load_tokenizer,
    render,
)
from recipes.self_training_loop.store import snowpark_session
from recipes.sft_quickstart.runner import train
from recipes.utils import TrainSequence

logger = logging.getLogger(__name__)


def turn_examples(
    tokenizer, messages: list[dict[str, Any]]
) -> list[tuple[list[int], list[float]]]:
    """``(input_ids, loss_mask)`` per assistant turn; ``loss_mask[j]`` weights predicting token j + 1."""
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    out = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prefix = render(tokenizer, messages[:i], generation_prompt=True)
        through = render(tokenizer, messages[: i + 1])
        if not through.startswith(prefix):
            raise ValueError("chat template is not prefix-stable for this turn")
        ids = tokenizer(through, add_special_tokens=False)["input_ids"]
        a = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        b = len(ids)
        # The target ends at this turn's <|im_end|>; the template's trailing newline is not a target.
        while b > a and ids[b - 1] != end_id:
            b -= 1
        if b <= a:
            raise ValueError("empty assistant turn")
        weights = [1.0 if a <= j < b else 0.0 for j in range(len(ids))]
        out.append((ids, weights[1:] + [0.0]))
    if not out:
        raise ValueError("no assistant turns")
    return out


@dataclass
class JudgedTraceTask:
    connection: str | None
    schema: str
    source_run_tags: tuple[str, ...]
    max_seq_len: int = 8192
    epochs: float = 2.0
    batch_size: int = 8
    seed: int = 0
    name: str = "self-training-loop"

    def traces(self) -> list[list[dict[str, Any]]]:
        session = snowpark_session(self.connection, self.schema)
        try:
            rows = session.sql(
                "SELECT chat_messages FROM TRAINING_SET_JUDGED "
                "WHERE run_tag IN (SELECT value::string FROM TABLE(FLATTEN(PARSE_JSON(?)))) "
                "ORDER BY run_tag, task_id, rank_in_task",
                params=[json.dumps(list(self.source_run_tags))],
            ).collect()
        finally:
            session.close()
        if not rows:
            raise ValueError(
                f"TRAINING_SET_JUDGED has no rows for {self.source_run_tags}; run select.py first"
            )
        return [json.loads(r[0]) for r in rows]

    def load_dataset(self, *, n_train: int):
        """Turn examples in training order: ``ceil(epochs * N / batch_size)`` batches, reshuffled each pass."""
        import datasets

        tokenizer = load_tokenizer()
        examples, skipped = [], {}
        for trace in self.traces():
            try:
                turns = turn_examples(
                    tokenizer,
                    [{"role": "system", "content": STUDENT_SYSTEM_PROMPT}, *trace],
                )
                if max(len(ids) for ids, _ in turns) > self.max_seq_len:
                    raise ValueError("longer than max_seq_len")
                examples += turns
            except ValueError as e:
                skipped[str(e)] = skipped.get(str(e), 0) + 1
        logger.info(
            "%d turn examples; skipped traces: %s", len(examples), skipped or "none"
        )
        steps = max(1, math.ceil(self.epochs * len(examples) / self.batch_size))
        rng = random.Random(self.seed)
        order: list = []
        while len(order) < min(steps * self.batch_size, n_train):
            order += rng.sample(examples, len(examples))
        order = order[: min(steps * self.batch_size, n_train)]
        return datasets.Dataset.from_dict(
            {"input_ids": [ids for ids, _ in order], "loss_mask": [w for _, w in order]}
        )

    def build_sequence(self, row, renderer, *, max_seq_len: int):
        return TrainSequence(
            input_ids=list(row["input_ids"]), loss_mask=list(row["loss_mask"])
        )

    def sample_prompt(self) -> str | None:
        return None


@chz.chz
class Config:
    # Legacy JSON connection file. Omit it to use `connection` or the default profile.
    config: str | None = None
    # Snowflake connection profile name (connections.toml). Use config or connection, not both.
    connection: str | None = None
    job_id: str | None = None

    # DATABASE.SCHEMA that holds TRAINING_SET_JUDGED (select.py builds it).
    schema: str = "CORTEX_TRAINING_DB.SELF_TRAINING_LOOP"
    # Comma-separated run tags whose kept traces to train on. Each version trains from the base
    # model on the seed runs plus every earlier version's runs on the training tasks.
    source_run_tags: str = "seed"
    epochs: float = 2.0

    # The job comes from model-catalog/models/; job_config= replaces it with a file.
    model: str = "Qwen/Qwen3-8B"
    context: str = "standard"
    hardware: str = "h200"
    lora: bool = True
    job_config: str | None = None
    # None keeps the catalog and overrides.json values (lr 1e-4, batch 8, max_seq_len 8192).
    lr: float | None = None
    train_batch_size: int | None = None
    max_seq_len: int | None = None

    pad_to_max_length: bool = False
    # Upper bound; epochs= sets the length of the run.
    max_steps: int = 100_000
    lr_schedule: str = "constant"
    warmup_ratio: float = 0.0

    debug_image_tag: str | None = None
    # The agent runs with thinking off, so traces render without reasoning.
    enable_thinking: bool = False
    renderer_name: str | None = None

    log_path: str = "/tmp/cortex-training-recipes/self-training-loop"
    wandb_project: str | None = None
    wandb_name: str | None = None
    sf_tracking: bool = False


def main(config: Config) -> dict[str, Any]:
    tags = tuple(t for t in config.source_run_tags.split(",") if t)
    # The runner reads the batch size and max_seq_len from the job; these mirror overrides.json.
    task = JudgedTraceTask(
        connection=config.connection,
        schema=config.schema,
        source_run_tags=tags,
        max_seq_len=config.max_seq_len or 8192,
        epochs=config.epochs,
        batch_size=config.train_batch_size or 8,
    )
    return train(config, task)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
