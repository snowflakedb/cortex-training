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

"""Direct Preference Optimization on chosen/rejected pairs, with the DPO loss computed in this process."""

import logging
import math
import os
import time
from typing import Any

import chz
from cortex_training.client import DEBUG_OPTIONS_ENV
from recipes.dpo_quickstart.task import dpo_loss
from recipes.dpo_quickstart.task import evaluate
from recipes.dpo_quickstart.task import load_pairs
from recipes.dpo_quickstart.task import pair_items
from recipes.dpo_quickstart.task import score_reference
from recipes.logging import setup_logging
from recipes.sft_quickstart.runner import job_body
from recipes.sft_quickstart.runner import lr_multiplier
from recipes.utils import build_renderer
from recipes.utils import example_connection_args
from recipes.utils import inference_job_args
from recipes.utils import log_saved_checkpoints
from recipes.utils import make_client
from recipes.utils import optimizer_step
from recipes.utils import running_job
from recipes.utils import save_example_checkpoints

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    # Legacy JSON connection file. Omit it to use `connection` or the default profile.
    config: str | None = None
    # Snowflake connection profile name (connections.toml). Use config or connection, not both.
    connection: str | None = None
    job_id: str | None = None

    # The job comes from model-catalog/models/; job_config= replaces it with a file.
    model: str = "Qwen/Qwen3-8B"
    context: str = "standard"
    hardware: str = "h200"
    lora: bool = True
    job_config: str | None = None
    # None keeps the job's value.
    lr: float | None = 1e-5
    train_batch_size: int | None = None
    max_seq_len: int | None = None

    dataset: str = "smoke"
    # Pairs per step; each pair is two items, so a step sends 2 * pairs_per_step.
    pairs_per_step: int = 4
    max_steps: int = 20
    # Passes over the training pairs; each pass reuses the reference scores.
    epochs: int = 1
    # Held-out pairs scored before training, every eval_every steps, and at the end.
    eval_pairs: int = 8
    eval_every: int = 5
    # Pairs whose chosen or rejected conversation is longer than this are skipped.
    max_pair_tokens: int = 2048
    beta: float = 0.1

    debug_image_tag: str | None = None
    log_path: str = "/tmp/cortex-training-recipes/dpo"
    wandb_project: str | None = None
    wandb_name: str | None = None
    sf_tracking: bool = False


def _items(pairs, renderer, max_pair_tokens: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for chosen, rejected in pairs:
        pair = pair_items(chosen, rejected, renderer, max_seq_len=max_pair_tokens)
        if pair is not None:
            items.extend(pair)
    return items


def main(config: Config) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if config.pairs_per_step < 1:
        raise ValueError(f"pairs_per_step must be at least 1, got {config.pairs_per_step}")
    if config.debug_image_tag:
        os.environ[DEBUG_OPTIONS_ENV] = "1"
        logger.info("Using debug image_tag=%s", config.debug_image_tag)
    loss_fn = dpo_loss(config.beta)
    body = job_body(config)
    (training_sub,) = [sub for sub in body["sub_job_configs"] if sub["job_type"] == "training"]
    learning_rate = float(training_sub["training_config"]["optimizer"]["lr"])
    max_pair_tokens = min(config.max_pair_tokens, int(training_sub["training_config"]["max_seq_len"]))
    _, renderer, _ = build_renderer(training_sub["model_name"], enable_thinking=False)

    if config.epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {config.epochs}")
    step_items = 2 * config.pairs_per_step
    train_pairs = load_pairs(config.dataset, "train", config.max_steps * config.pairs_per_step)
    eval_pairs = load_pairs(config.dataset, "test", config.eval_pairs) if config.eval_pairs else []
    train_items = _items(train_pairs, renderer, max_pair_tokens)
    eval_items = _items(eval_pairs, renderer, max_pair_tokens)
    steps_per_epoch = len(train_items) // step_items
    total_steps = min(steps_per_epoch * config.epochs, config.max_steps)
    if total_steps == 0:
        raise ValueError(
            f"{len(train_items) // 2} usable training pair(s), fewer than pairs_per_step={config.pairs_per_step}"
        )
    logger.info(
        "DPO for %d steps: %d training and %d held-out pairs, beta=%s",
        total_steps,
        len(train_items) // 2,
        len(eval_items) // 2,
        config.beta,
    )

    client = make_client(config.config, config.connection)
    with running_job(client, body, job_id=config.job_id) as job_id:
        ml_logger = setup_logging(config, client=client, job_id=job_id)
        # The reference is the model before any update, so score every pair now.
        start_time = time.time()
        used_train = train_items[: min(total_steps, steps_per_epoch) * step_items]
        score_reference(client, job_id, used_train + eval_items, batch_items=step_items)
        logger.info("Scored the reference on %d items in %.0fs", len(used_train) + len(eval_items), time.time() - start_time)

        def log_eval(step: int) -> None:
            if eval_items:
                metrics = evaluate(client, job_id, eval_items, loss_fn, batch_items=step_items)
                ml_logger.log_metrics(metrics=metrics, step=step)

        log_eval(0)
        for step in range(total_steps):
            start_time = time.time()
            offset = (step % steps_per_epoch) * step_items
            data = used_train[offset : offset + step_items]
            result = client.forward_backward_custom(job_id, data, loss_fn)
            step_result = optimizer_step(client, job_id, learning_rate=learning_rate * lr_multiplier(step, total_steps))
            metrics = {**result["metrics"], **(step_result.get("metrics") or {})}
            metrics.update(train_loss=result["loss"], time_total=time.time() - start_time)
            if "logprob_max_abs_diff" in result:
                metrics["logprob_max_abs_diff"] = result["logprob_max_abs_diff"]
            if step == 0 and not math.isclose(float(result["loss"]), math.log(2), abs_tol=1e-2):
                logger.warning("first-step loss %.4f is not log 2: check the reference scores", float(result["loss"]))
            ml_logger.log_metrics(metrics=metrics, step=step)
            if (step + 1) % config.eval_every == 0 and step + 1 < total_steps:
                log_eval(step + 1)
        log_eval(total_steps)

        saved = save_example_checkpoints(client, job_id)
        log_saved_checkpoints(
            connection_args=example_connection_args(config.config, config.connection),
            job_id=job_id,
            saved=saved,
            sampling_command="sample",
            job_args=inference_job_args(body, context=config.context if config.job_config is None else None),
            sample_prompt=None,
            temperature=None,
        )

    ml_logger.close()
    logger.info("Training completed")
    return {"job_id": job_id, "checkpoint_id": saved["weights-only"].get("checkpoint_id")}


if __name__ == "__main__":
    chz.nested_entrypoint(main)
