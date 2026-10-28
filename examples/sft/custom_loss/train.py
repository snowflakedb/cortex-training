"""Supervised fine-tuning with a custom loss computed in this process."""

import logging
import time
from typing import Any

import chz
from examples.logging import setup_logging
from examples.sft.conversational.task import ConversationalTask
from examples.sft.custom_loss.task import focal_loss
from examples.sft.custom_loss.task import loss_item
from examples.sft.runner import job_body
from examples.sft.runner import lr_multiplier
from examples.utils import build_renderer
from examples.utils import example_connection_args
from examples.utils import inference_job_args
from examples.utils import log_saved_checkpoints
from examples.utils import make_client
from examples.utils import optimizer_step
from examples.utils import running_job
from examples.utils import save_example_checkpoints
from tinker_cookbook import renderers

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
    lora: bool = False
    job_config: str | None = None
    # None keeps the job's value.
    lr: float | None = None
    train_batch_size: int | None = None
    max_seq_len: int | None = None

    dataset: str = "who_trained_you"
    max_steps: int = 100
    # 0 is the per-sequence mean negative log-likelihood.
    focal_gamma: float = 2.0

    debug_image_tag: str | None = None

    log_path: str = "/tmp/cortex-training-examples/custom-loss"
    wandb_project: str | None = None
    wandb_name: str | None = None
    sf_tracking: bool = False


def main(config: Config) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    loss_fn = focal_loss(config.focal_gamma)
    body = job_body(config)
    (training_sub,) = [sub for sub in body["sub_job_configs"] if sub["job_type"] == "training"]
    training = training_sub["training_config"]
    batch_size = int(training["train_batch_size"])
    max_seq_len = int(training["max_seq_len"])
    learning_rate = float(training["optimizer"]["lr"])

    _, renderer, _ = build_renderer(training_sub["model_name"], enable_thinking=False)
    task = ConversationalTask(
        dataset=config.dataset,
        dataset_split="train",
        train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = task.load_dataset(n_train=config.max_steps * batch_size)
    total_steps = min(len(dataset) // batch_size, config.max_steps)
    logger.info("Training for %d steps with the focal loss, gamma=%s", total_steps, config.focal_gamma)

    client = make_client(config.config, config.connection)
    with running_job(client, body, job_id=config.job_id) as job_id:
        ml_logger = setup_logging(config, client=client, job_id=job_id)
        for step in range(total_steps):
            start_time = time.time()
            rows = dataset.select(range(step * batch_size, (step + 1) * batch_size))
            data = [loss_item(task.build_sequence(row, renderer, max_seq_len=max_seq_len)) for row in rows]
            result = client.forward_backward_custom(job_id, data, loss_fn)
            step_result = optimizer_step(
                client, job_id, learning_rate=learning_rate * lr_multiplier(step, total_steps)
            )

            metrics = {**result["metrics"], **(step_result.get("metrics") or {})}
            metrics.update(train_loss=result["loss"], time_total=time.time() - start_time)
            if "logprob_max_abs_diff" in result:
                metrics["logprob_max_abs_diff"] = result["logprob_max_abs_diff"]
            ml_logger.log_metrics(metrics=metrics, step=step)

        saved = save_example_checkpoints(client, job_id)
        sample_prompt = task.sample_prompt()
        log_saved_checkpoints(
            connection_args=example_connection_args(config.config, config.connection),
            job_id=job_id,
            saved=saved,
            sampling_command="sample",
            job_args=inference_job_args(body, context=config.context if config.job_config is None else None),
            sample_prompt=sample_prompt,
            temperature=0 if sample_prompt else None,
        )

    ml_logger.close()
    logger.info("Training completed")
    return {"job_id": job_id, "checkpoint_id": saved["weights-only"].get("checkpoint_id")}


if __name__ == "__main__":
    chz.nested_entrypoint(main)
