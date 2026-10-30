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

"""
SFT warmup for the reverse-text RL stage.

A port of prime-rl's ``examples/basic/reverse-text/sft.toml``: Qwen3-0.6B on
``willcb/R1-reverse-wikipedia-paragraphs-v1-1000`` for 100 steps at a constant
learning rate of 2e-5.

The run ends by saving two checkpoints that initialize ``train.py``: the weights-only one
(``init_checkpoint_id``) starts the sampling sub-job, and the resumable one
(``init_trainer_checkpoint_id``) starts the training sub-job, both from ``init_source_job_id``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from collections.abc import Iterator
from typing import Any

import chz
from examples.logging import setup_logging
from examples.rl.async_rl_reverse_text.train import EXAMPLE_DIR
from examples.rl.async_rl_reverse_text.train import data_parallel_size
from examples.rl.async_rl_reverse_text.task import SFT_DATASET
from examples.utils import TrainSequence
from examples.utils import build_job
from examples.utils import build_renderer
from examples.utils import collate
from examples.utils import example_connection_args
from examples.utils import forward_backward_step
from examples.utils import make_client
from examples.utils import next_token_loss_mask
from examples.utils import running_job
from examples.utils import save_checkpoint
from examples.utils import save_example_checkpoints

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)


@chz.chz
class Config:
    # Legacy JSON connection file. Omit it to use `connection` or the default profile.
    config: str | None = None
    # Snowflake connection profile name (connections.toml). Use config or connection, not both.
    connection: str | None = None
    job_id: str | None = None

    # The job comes from model-catalog/models/ and this folder's sft_overrides.json (one GPU);
    # job_config= replaces it with a file.
    model: str = "Qwen/Qwen3-0.6B"
    context: str = "standard"
    hardware: str = "h200"
    lora: bool = False
    job_config: str | None = None
    # None keeps the job's value (lr 2e-5, prime-rl's).
    lr: float | None = None
    train_batch_size: int | None = None
    max_seq_len: int | None = None

    dataset: str = SFT_DATASET
    dataset_split: str = "train"
    max_steps: int = 100
    # Packed rows of seq_len tokens per step, as in prime-rl's data.batch_size.
    batch_size: int = 32
    seq_len: int = 4096
    shuffle: bool = True
    seed: int = 0
    renderer_name: str = "qwen3"

    debug_image_tag: str | None = None
    log_path: str = "/tmp/cortex-training-examples/prime-rl-reverse-text-sft"
    wandb_project: str | None = None
    wandb_name: str | None = None
    sf_tracking: bool = False


def job_body(config: Config) -> dict:
    """The SFT job: the catalog's training file, then ``sft_overrides.json``, then ``config``'s
    settings (a ``job_config`` file instead of the first two).
    """
    return build_job(
        "training",
        model=config.model,
        context=config.context,
        hardware=config.hardware,
        lora=config.lora,
        job_config=config.job_config,
        overrides=[EXAMPLE_DIR / "sft_overrides.json"],
        search_dirs=(EXAMPLE_DIR,),
        lr=config.lr,
        train_batch_size=config.train_batch_size,
        max_seq_len=config.max_seq_len,
        debug_image_tag=config.debug_image_tag,
    )


def sft_sequence(row: dict[str, Any], renderer: Any, tokenizer: Any, *, seq_len: int) -> TrainSequence:
    """One SFT row tokenized the way prime-rl's SFT trainer tokenizes it.

    The prompt is the renderer's generation prompt, the same tokens RL samples from. The reply is
    the dataset's text tokenized as written, then the end-of-turn token; only those are trained.
    The chat template's newline after the end-of-turn token follows, untrained.
    The Qwen3 chat template, which the renderer's supervised path follows, rewrites the final
    assistant turn instead: a reply without reasoning gets an empty ``<think>\\n\\n</think>\\n\\n``
    block in front, and ``</think>\\n`` becomes ``</think>\\n\\n``. 213 of the 1000 rows answer
    without reasoning.
    """
    prompt = renderer.build_generation_prompt(row["prompt"]).to_ints()
    reply = tokenizer.encode(row["completion"][-1]["content"], add_special_tokens=False)
    end_of_turn = renderer.get_stop_sequences()[0]
    # The chat template's newline after the end-of-turn token: not trained, but it counts toward
    # the packed row length, so the greedy packer fills each step with prime-rl's samples.
    turn_end = tokenizer.encode("\n", add_special_tokens=False)
    token_ids = (prompt + reply + [end_of_turn] + turn_end)[:seq_len]
    weights = ([0.0] * len(prompt) + [1.0] * (len(reply) + 1) + [0.0] * len(turn_end))[:seq_len]
    return TrainSequence(input_ids=token_ids, loss_mask=next_token_loss_mask(weights))


def iter_samples(
    rows: Any, renderer: Any, tokenizer: Any, *, seq_len: int, shuffle: bool, seed: int
) -> Iterator[TrainSequence]:
    """Tokenized samples, reshuffled each epoch with ``seed + epoch``."""
    if len(rows) == 0:
        raise ValueError("the SFT dataset is empty")
    epoch = 0
    while True:
        ordered = rows.shuffle(seed=seed + epoch) if shuffle else rows
        yielded = False
        for row in ordered:
            sequence = sft_sequence(row, renderer, tokenizer, seq_len=seq_len)
            if not any(weight > 0.0 for weight in sequence.loss_mask):
                logger.warning("Skipping a sample with no trainable tokens within seq_len=%d", seq_len)
                continue
            yielded = True
            yield sequence
        if not yielded:
            raise ValueError(f"no SFT sample has trainable tokens within seq_len={seq_len}")
        epoch += 1


def iter_step_batches(
    samples: Iterable[TrainSequence],
    *,
    seq_len: int,
    rows_per_step: int,
) -> Iterator[list[TrainSequence]]:
    """Group samples the way a greedy packer fills ``rows_per_step`` rows.

    A sample that would overflow the current row starts the next one, so a
    step holds exactly the samples of ``rows_per_step`` packed rows.
    """
    step: list[TrainSequence] = []
    rows = 0
    row_len = 0
    for sample in samples:
        length = len(sample.input_ids) - 1
        if row_len > 0 and row_len + length > seq_len:
            rows += 1
            row_len = 0
            if rows == rows_per_step:
                yield step
                step, rows = [], 0
        step.append(sample)
        row_len += length


def sft_processing(*, batch_num_tokens: float, dp_size: int) -> dict[str, Any]:
    return {
        "loss_fn": "causal_cross_entropy",
        "post": ["compute_logprobs"],
        "config": {"batch_num_tokens": batch_num_tokens, "dp_size": dp_size},
    }


def main(config: Config) -> None:
    # setup_logging replaces these handlers once the job runs; until then this
    # keeps the created job id in the log, so an interrupted start can be cancelled.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if config.max_steps <= 0 or config.batch_size <= 0:
        raise ValueError("max_steps and batch_size must be > 0")
    body = job_body(config)
    training_sub = next((sub for sub in body.get("sub_job_configs") or () if sub.get("job_type") == "training"), None)
    if training_sub is None:
        raise ValueError(f"{config.job_config} has no training sub-job")
    training = training_sub.get("training_config") or {}
    max_seq_len = int(training["max_seq_len"])
    if config.seq_len > max_seq_len:
        raise ValueError(f"seq_len={config.seq_len} exceeds the job's max_seq_len={max_seq_len}")
    learning_rate = float(training["optimizer"]["lr"])
    dp_size = data_parallel_size(training)

    import datasets

    tokenizer, renderer, renderer_name = build_renderer(training_sub["model_name"], renderer_name=config.renderer_name)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    logger.info("Using renderer: %s", renderer_name)
    rows = datasets.load_dataset(config.dataset, split=config.dataset_split)
    steps = iter_step_batches(
        iter_samples(rows, renderer, tokenizer, seq_len=config.seq_len, shuffle=config.shuffle, seed=config.seed),
        seq_len=config.seq_len,
        rows_per_step=config.batch_size,
    )

    client = make_client(config.config, config.connection)
    with running_job(client, body, job_id=config.job_id) as job_id:
        ml_logger = setup_logging(config, client=client, job_id=job_id)
        for step in range(config.max_steps):
            t_start = time.time()
            sequences = next(steps)
            kwargs, context = collate(sequences, pad_token_id=pad_token_id, max_seq_len=max_seq_len)
            num_tokens = float(context["loss_mask"].sum())
            fwd_bwd_result, step_result = forward_backward_step(
                client,
                job_id,
                kwargs,
                context=context,
                learning_rate=learning_rate,
                processing=sft_processing(batch_num_tokens=num_tokens, dp_size=dp_size),
            )
            metrics = {
                "loss/mean": float(fwd_bwd_result.get("avg_loss", float("nan"))),
                "optim/lr": learning_rate,
                "progress/samples": len(sequences),
                "progress/trained_tokens": num_tokens,
                "time/step": time.time() - t_start,
            }
            if step_result.get("grad_norm") is not None:
                metrics["optim/grad_norm"] = float(step_result["grad_norm"])
            ml_logger.log_metrics(metrics, step=step)

        saved = save_example_checkpoints(client, job_id)["weights-only"]
        # The RL training sub-job restores a resumable checkpoint; only the sampling sub-job
        # can start from the weights-only one.
        resumable = save_checkpoint(client, job_id, checkpoint_type="resumable")
        logger.info(
            "Start the RL stage from these checkpoints with:\n"
            "  python -m examples.rl.async_rl_reverse_text.train %s init_source_job_id=%s "
            "init_checkpoint_id=%s init_trainer_checkpoint_id=%s",
            example_connection_args(config.config, config.connection),
            job_id,
            saved.get("checkpoint_id"),
            resumable.get("checkpoint_id"),
        )
    ml_logger.close()


if __name__ == "__main__":
    chz.nested_entrypoint(main)
