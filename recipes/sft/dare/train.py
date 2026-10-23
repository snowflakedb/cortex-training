"""DARE-Bench agent-trace SFT entrypoint for the shared SFT runner."""

from typing import Any

import chz
from recipes.sft.tasks.dare import DareTask
from recipes.sft.train import train


@chz.chz
class Config:
    # An old JSON config: deprecated, stops working in 0.2.0. Without it, the
    # Snowflake profile is used: connection=NAME, else the one `cortex-training login` saved.
    config: str | None = None
    connection: str | None = None
    job_id: str | None = None

    variant: str = "fv"
    heldout_fraction: float = 0.05
    eval_every: int = 10
    pad_to_max_length: bool = False
    max_steps: int = 120
    lr_schedule: str = "cosine"
    warmup_ratio: float = 0.05

    debug_image_tag: str | None = None
    enable_thinking: bool = True
    renderer_name: str | None = None

    log_path: str = "/tmp/cortex-training-examples/dare-sft"
    wandb_project: str | None = None
    wandb_name: str | None = None
    sf_tracking: bool = False

    job_config: str = "configs/dare_qwen3_8b_full.json"


def main(config: Config) -> dict[str, Any]:
    return train(config, DareTask(variant=config.variant, heldout_fraction=config.heldout_fraction))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
