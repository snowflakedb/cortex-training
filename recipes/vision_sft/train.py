# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Supervised fine-tuning of a vision-language model on image question answering.

Each row is a chat prompt holding one image and a question, followed by the answer. The client runs the
model's Hugging Face processor (see ``recipes.vision_sft.task``) and sends the tokens with the row's images as
``multi_modal_inputs``; the training sub-job encodes the images with the model's vision tower. The loss is the
same ``causal_cross_entropy`` the conversational SFT recipe uses, on the answer tokens only.

Held-out answer NLL is reported before training, every ``eval_every`` steps and at the end, split by data
source. With ``blank_image_control`` it is also measured once with every picture replaced by a grey one, which
shows how much the answers depend on the images.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chz
from recipes.logging import setup_logging
from recipes.sft_quickstart.runner import causal_cross_entropy_processing
from recipes.sft_quickstart.runner import lr_multiplier
from recipes.sft_quickstart.runner import next_token_labels
from recipes.utils import build_job
from recipes.utils import forward
from recipes.utils import forward_backward_step
from recipes.utils import make_client
from recipes.utils import running_job
from recipes.utils import save_example_checkpoints
from recipes.vision_sft.task import VisionExample
from recipes.vision_sft.task import VisionSequence
from recipes.vision_sft.task import build_sequence
from recipes.vision_sft.task import collate_vision
from recipes.vision_sft.task import load_examples
from recipes.vision_sft.task import load_processor

from cortex_training.client import DEBUG_OPTIONS_ENV

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)

_EXAMPLE_DIR = Path(__file__).resolve().parent


@chz.chz
class Config:
    # Legacy JSON connection file. Omit it to use `connection` or the default profile.
    config: str | None = None
    # Snowflake connection profile name (connections.toml). Use config or connection, not both.
    connection: str | None = None
    job_id: str | None = None

    # The job comes from model-catalog/models/; job_config= replaces it with a file.
    model: str = "Qwen/Qwen3.6-35B-A3B"
    context: str = "standard"
    hardware: str = "h200"
    lora: bool = True
    job_config: str | None = None
    # None keeps the job's value.
    lr: float | None = None
    train_batch_size: int | None = None
    max_seq_len: int | None = None

    # mixed, chartqa or spatial (see recipes.vision_sft.task).
    dataset: str = "mixed"
    n_train: int = 384
    n_eval: int = 64
    epochs: int = 1
    # 0 evaluates before and after training only.
    eval_every: int = 0
    # The processor resizes larger images down to this many pixels.
    max_image_pixels: int = 448 * 448
    blank_image_control: bool = True
    save_checkpoint: bool = True

    debug_image_tag: str | None = None

    log_path: str = "/tmp/cortex-training-recipes/vision-sft"
    wandb_project: str | None = None
    wandb_name: str | None = None
    sf_tracking: bool = False


def job_body(config: Config) -> dict:
    """The training job: the catalog file, then this recipe's overrides.json, then ``config``."""
    return build_job(
        "training",
        model=config.model,
        context=config.context,
        hardware=config.hardware,
        lora=config.lora,
        job_config=config.job_config,
        example_dir=_EXAMPLE_DIR,
        search_dirs=(_EXAMPLE_DIR,),
        lr=config.lr,
        train_batch_size=config.train_batch_size,
        max_seq_len=config.max_seq_len,
        debug_image_tag=config.debug_image_tag,
    )


def heldout_nll(
    client: Any,
    job_id: str,
    rows: Sequence[VisionSequence],
    examples: Sequence[VisionExample],
    *,
    batch_size: int,
    pad_token_id: int,
    max_seq_len: int,
    prefix: str = "valid_nll",
) -> dict[str, float]:
    """Token-weighted mean answer NLL, overall and per data source, from no-grad forward passes."""
    nll: dict[str, float] = defaultdict(float)
    weight: dict[str, float] = defaultdict(float)
    for start in range(0, len(rows), batch_size):
        kwargs, context = collate_vision(
            rows[start : start + batch_size], pad_token_id=pad_token_id, max_seq_len=max_seq_len
        )
        loss_mask = context["loss_mask"]
        # forward reads labels as sent, for every model provider.
        kwargs["labels"] = next_token_labels(kwargs["input_ids"], loss_mask)
        logprobs = forward(client, job_id, kwargs)["logprobs"].float()
        for offset, example in enumerate(examples[start : start + batch_size]):
            row_nll = -float((logprobs[offset] * loss_mask[offset]).sum())
            row_weight = float(loss_mask[offset].sum())
            for key in (prefix, f"{prefix}_{example.source}"):
                nll[key] += row_nll
                weight[key] += row_weight
    return {key: nll[key] / weight[key] for key in nll}


def main(config: Config) -> dict[str, Any]:
    # setup_logging replaces these handlers once the job runs; until then this
    # keeps the created job id in the log, so an interrupted start can be cancelled.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if config.debug_image_tag:
        os.environ[DEBUG_OPTIONS_ENV] = "1"

    body = job_body(config)
    training_sub = next(sub for sub in body["sub_job_configs"] if sub.get("job_type") == "training")
    training = training_sub["training_config"]
    if int(training.get("sp_size") or 1) != 1:
        raise ValueError("image rows need sp_size 1: a packed call's images cannot be split across sequence ranks")
    batch_size = int(training["train_batch_size"])
    max_seq_len = int(training["max_seq_len"])
    learning_rate = float(training["optimizer"]["lr"])
    model_name = training_sub["model_name"]

    processor = load_processor(model_name, max_image_pixels=config.max_image_pixels)
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    logger.info("Loading %s: %d train and %d held-out rows", config.dataset, config.n_train, config.n_eval)
    train_examples, eval_examples = load_examples(config.dataset, n_train=config.n_train, n_eval=config.n_eval)
    train_rows = [build_sequence(example, processor, max_seq_len=max_seq_len) for example in train_examples]
    n_eval = len(eval_examples) - len(eval_examples) % batch_size
    eval_examples = eval_examples[:n_eval]
    eval_rows = [build_sequence(example, processor, max_seq_len=max_seq_len) for example in eval_examples]
    blank_rows = (
        [build_sequence(example, processor, max_seq_len=max_seq_len, blank_image=True) for example in eval_examples]
        if config.blank_image_control
        else []
    )
    steps_per_epoch = len(train_rows) // batch_size
    total_steps = steps_per_epoch * config.epochs
    logger.info("%d steps of %d rows; %d held-out rows", total_steps, batch_size, n_eval)

    client = make_client(config.config, config.connection)

    def evaluate(job_id: str, rows: Sequence[VisionSequence], prefix: str) -> dict[str, float]:
        if not rows:
            return {}
        started = time.time()
        metrics = heldout_nll(
            client,
            job_id,
            rows,
            eval_examples,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
            max_seq_len=max_seq_len,
            prefix=prefix,
        )
        logger.info("%s", " ".join(f"{key} {value:.4f}" for key, value in sorted(metrics.items())))
        return {**metrics, f"time_{prefix}": time.time() - started}

    with running_job(client, body, job_id=config.job_id) as job_id:
        ml_logger = setup_logging(config, client=client, job_id=job_id)
        initial = evaluate(job_id, eval_rows, "valid_nll")
        initial.update(evaluate(job_id, blank_rows, "blank_image_nll"))
        ml_logger.log_metrics(metrics=initial, step=0)
        final = initial
        for step in range(total_steps):
            started = time.time()
            batch_start = (step % steps_per_epoch) * batch_size
            kwargs, context = collate_vision(
                train_rows[batch_start : batch_start + batch_size], pad_token_id=pad_token_id, max_seq_len=max_seq_len
            )
            fwd_bwd_result, step_result = forward_backward_step(
                client,
                job_id,
                kwargs,
                context=context,
                learning_rate=learning_rate * lr_multiplier(step, total_steps),
                processing=causal_cross_entropy_processing(),
            )
            metrics: dict[str, float] = {
                "train_nll": float(fwd_bwd_result["avg_loss"]),
                "loss_weight_sum": float(context["loss_mask"].sum()),
                "progress": step / total_steps,
                "time_total": time.time() - started,
            }
            metrics.update(fwd_bwd_result.get("metrics") or {})
            metrics.update(step_result.get("metrics") or {})
            done = step + 1
            if done == total_steps or (config.eval_every > 0 and done % config.eval_every == 0):
                final = evaluate(job_id, eval_rows, "valid_nll")
                metrics.update(final)
            ml_logger.log_metrics(metrics=metrics, step=done)

        checkpoint_id = None
        if config.save_checkpoint:
            saved = save_example_checkpoints(client, job_id)
            checkpoint_id = saved["weights-only"].get("checkpoint_id")
            logger.info("Saved weights-only checkpoint %s of job %s", checkpoint_id, job_id)

    ml_logger.close()
    logger.info("Training completed")
    return {
        "job_id": job_id,
        "checkpoint_id": checkpoint_id,
        "valid_nll_before": initial.get("valid_nll"),
        "valid_nll_after": final.get("valid_nll"),
    }


if __name__ == "__main__":
    chz.nested_entrypoint(main)
