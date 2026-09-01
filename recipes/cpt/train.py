# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Continual pre-training on a Hugging Face text dataset."""

from __future__ import annotations

import logging
import math
import os
import time
from itertools import islice
from pathlib import Path

import chz
from recipes.cpt.data import DEFAULT_SPLIT
from recipes.cpt.data import DEFAULT_TEXT_COLUMN
from recipes.cpt.data import load_packed_dataset
from recipes.cpt.data import load_tokenizer
from recipes.cpt.data import repeat_packed_sequences
from recipes.cpt.data import sequence_from_tokens
from recipes.cpt.download_checkpoint import checkpoint_download_command
from recipes.utils import average_forward_backward_metrics
from recipes.utils import collate
from recipes.utils import forward_backward
from recipes.utils import load_job_body
from recipes.utils import log_saved_checkpoints
from recipes.utils import make_client
from recipes.utils import optimizer_step
from recipes.utils import running_job
from recipes.utils import save_recipe_checkpoints
from recipes.utils import use_next_token_labels
from tinker_cookbook.utils import ml_log

from cortex_training.client import DEBUG_OPTIONS_ENV

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)

_RECIPE_DIR = Path(__file__).resolve().parent


@chz.chz
class Config:
    config: str
    dataset: str
    job_id: str | None = None

    dataset_config: str | None = None
    dataset_split: str = DEFAULT_SPLIT
    text_column: str = DEFAULT_TEXT_COLUMN
    data_files: str | None = None
    streaming: bool = False
    insert_eos: bool = True

    max_steps: int = 100
    warmup_ratio: float = 0.05

    job_config: str = "configs/qwen3_8b_full.json"
    debug_image_tag: str | None = None
    log_path: str = "/tmp/cortex-training-examples/cpt"
    wandb_project: str | None = None
    wandb_name: str | None = None


def job_body(config: Config) -> dict:
    body = load_job_body(
        config.job_config,
        search_dirs=(_RECIPE_DIR, _RECIPE_DIR / "configs"),
    )
    if config.debug_image_tag:
        body["debug"] = {"job": {"image_tag": config.debug_image_tag}}
    return body


def _learning_rate(
    step: int,
    total_steps: int,
    peak_lr: float,
    warmup_ratio: float,
) -> float:
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def main(config: Config):
    if config.max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    if not 0.0 <= config.warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be between 0 and 1")
    if config.debug_image_tag:
        os.environ[DEBUG_OPTIONS_ENV] = "1"

    body = job_body(config)
    training_sub = next(
        (
            sub
            for sub in body.get("sub_job_configs") or ()
            if sub.get("job_type") == "training"
        ),
        None,
    )
    if training_sub is None:
        raise ValueError("job config needs one training sub-job")
    training = training_sub["training_config"]
    batch_size = int(training["train_batch_size"])
    accumulation_steps = int(
        (training.get("ds_config") or {}).get("gradient_accumulation_steps", 1)
    )
    if batch_size <= 0 or accumulation_steps <= 0:
        raise ValueError("batch sizes and accumulation steps must be > 0")
    if batch_size % accumulation_steps:
        raise ValueError(
            "train_batch_size must be divisible by gradient_accumulation_steps"
        )

    request_batch_size = batch_size // accumulation_steps
    max_seq_len = int(training["max_seq_len"])
    peak_lr = float(training["optimizer"]["lr"])
    model_provider = str(training.get("model_provider") or "huggingface")
    model_name = str(training_sub["model_name"])

    tokenizer = load_tokenizer(model_name)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer needs a pad_token_id or eos_token_id")

    packed = load_packed_dataset(
        tokenizer=tokenizer,
        dataset=config.dataset,
        dataset_config=config.dataset_config,
        split=config.dataset_split,
        text_column=config.text_column,
        seq_len=max_seq_len,
        data_files=config.data_files,
        streaming=config.streaming,
        insert_eos=config.insert_eos,
        seed=int(training_sub.get("seed", 42)),
    )
    block_stream = repeat_packed_sequences(packed)
    next_token_labels = use_next_token_labels(model_provider)

    logger.info(
        "CPT model=%s dataset=%s config=%s split=%s steps=%d",
        model_name,
        config.dataset,
        config.dataset_config,
        config.dataset_split,
        config.max_steps,
    )

    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )
    client = make_client(config.config)

    try:
        with running_job(client, body, job_id=config.job_id) as job_id:
            for step in range(config.max_steps):
                started = time.time()
                batch = list(islice(block_stream, batch_size))
                results = []
                train_loss = 0.0

                for accumulation_step in range(accumulation_steps):
                    start = accumulation_step * request_batch_size
                    token_ids = batch[start : start + request_batch_size]
                    sequences = [
                        sequence_from_tokens(
                            tokens,
                            next_token_labels=next_token_labels,
                        )
                        for tokens in token_ids
                    ]
                    kwargs, _ = collate(
                        sequences,
                        pad_token_id=pad_token_id,
                        max_seq_len=max_seq_len,
                        pad_to_max_seq_len=True,
                    )
                    result = forward_backward(client, job_id, kwargs)
                    results.append(result)
                    train_loss += float(result["avg_loss"]) / accumulation_steps

                learning_rate = _learning_rate(
                    step,
                    config.max_steps,
                    peak_lr,
                    config.warmup_ratio,
                )
                step_result = optimizer_step(
                    client,
                    job_id,
                    learning_rate=learning_rate,
                )
                metrics = average_forward_backward_metrics(results)
                metrics.update(step_result.get("metrics") or {})
                metrics.update(
                    train_nll=train_loss,
                    learning_rate=learning_rate,
                    tokens=float(batch_size * max_seq_len),
                    global_steps=step_result.get("global_steps", step + 1),
                    progress=(step + 1) / config.max_steps,
                    time_total=time.time() - started,
                )
                ml_logger.log_metrics(metrics=metrics, step=step)

            saved = save_recipe_checkpoints(client, job_id)
            log_saved_checkpoints(
                config_path=config.config,
                job_id=job_id,
                saved=saved,
            )
            checkpoint_id = str(
                saved["weights-only"].get("checkpoint_id") or ""
            ).strip()
            if not checkpoint_id:
                raise ValueError("checkpoint save response did not include an ID")
            logger.info(
                "Download the finished checkpoint with:\n  %s",
                    checkpoint_download_command(
                        config.config,
                        job_id,
                        checkpoint_id,
                    ),
            )
    finally:
        ml_logger.close()

    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
