"""Wiki-search entrypoint for the shared on-policy distillation runner."""

import chz
from recipes.distillation.tasks.wiki_search import DEFAULT_INDEX_DIR
from recipes.distillation.tasks.wiki_search import WikiIndex
from recipes.distillation.tasks.wiki_search import WikiSearchTask
from recipes.distillation.train import _train
from recipes.distillation.train import job_bodies
from recipes.distillation.train import lr_at
from recipes.distillation.train import processing_block
from recipes.distillation.train import train
from recipes.distillation.train import validate_config


@chz.chz
class Config:
    config: str
    job_id: str | None = None
    teacher_job_id: str | None = None
    judge_job_id: str | None = None
    job_config: str = "configs/qwen35_4b_student.json"
    teacher_job_config: str = "configs/qwen38_27b_teacher.json"
    judge_job_config: str | None = "configs/qwen3_30b_a3b_judge.json"
    index_dir: str = str(DEFAULT_INDEX_DIR)

    questions_per_step: int = 32
    group_size: int = 4
    max_turns: int = 10
    turn_max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    eval_temperature: float = 0.7
    eval_top_p: float = 0.8
    eval_top_k: int = 20

    max_steps: int = 40
    warmup_steps: int = 2
    lr_schedule: str = "cosine"
    min_lr_ratio: float = 0.1

    teacher_tau: float = 1.0
    teacher_clip: float = 3.0
    teacher_clip_negative: float | None = 1.0
    reward_coef: float = 0.0
    eps_clip: float = 0.2
    loss_agg_mode: str = "token-mean"
    entropy_coeff: float = 0.0

    eval_every: int = 5
    n_eval: int = 100
    eval_samples: int = 2
    evaluate_teacher: bool = True
    max_prompts_per_request: int | None = 32
    seed: int = 0

    weight_sync_bucket_size: int | None = None
    debug_image_tag: str | None = None
    log_path: str = "/tmp/cortex-training-examples/on-policy-distillation"
    wandb_project: str | None = None
    wandb_name: str | None = None
    sf_tracking: bool = False


def main(config: Config) -> None:
    train(config, WikiSearchTask(index_dir=config.index_dir))


__all__ = [
    "Config",
    "WikiIndex",
    "WikiSearchTask",
    "_train",
    "job_bodies",
    "lr_at",
    "main",
    "processing_block",
    "train",
    "validate_config",
]

if __name__ == "__main__":
    chz.nested_entrypoint(main)
