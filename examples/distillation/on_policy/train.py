"""Wiki-search entrypoint for the shared on-policy distillation runner."""

import chz
from examples.distillation.tasks.wiki_search import DEFAULT_INDEX_DIR
from examples.distillation.tasks.wiki_search import WikiIndex
from examples.distillation.tasks.wiki_search import WikiSearchTask
from examples.distillation.train import job_bodies
from examples.distillation.train import lr_at
from examples.distillation.train import processing_block
from examples.distillation.train import train
from examples.distillation.train import validate_config
from examples.utils import DEFAULT_DISTILLATION_CONTEXT
from examples.utils import DEFAULT_DISTILLATION_JUDGE
from examples.utils import DEFAULT_DISTILLATION_JUDGE_GPUS
from examples.utils import DEFAULT_DISTILLATION_JUDGE_MAX_SEQ_LEN
from examples.utils import DEFAULT_DISTILLATION_STUDENT
from examples.utils import DEFAULT_DISTILLATION_STUDENT_MAX_SEQ_LEN
from examples.utils import DEFAULT_DISTILLATION_TEACHER
from examples.utils import DEFAULT_DISTILLATION_TEACHER_GPUS
from examples.utils import DEFAULT_DISTILLATION_TEACHER_MAX_SEQ_LEN


@chz.chz
class Config:
    # Legacy JSON connection file. Omit it to use `connection` or the default profile.
    config: str | None = None
    # Snowflake connection profile name (connections.toml). Use config or connection, not both.
    connection: str | None = None
    job_id: str | None = None
    teacher_job_id: str | None = None
    judge_job_id: str | None = None

    # The jobs come from model-catalog/models/; job_config=/teacher_job_config=/
    # judge_job_config= replace a body with a file. The defaults are the validated
    # Qwen3.8-27B -> Qwen3.5-4B run: a 1 + 1 GPU student, a 2-GPU teacher, and a
    # 2-GPU judge.
    student: str = DEFAULT_DISTILLATION_STUDENT
    teacher: str = DEFAULT_DISTILLATION_TEACHER
    # The model that grades final answers; None lets the teacher job judge.
    judge: str | None = DEFAULT_DISTILLATION_JUDGE
    context: str = DEFAULT_DISTILLATION_CONTEXT
    teacher_context: str | None = None
    judge_context: str = DEFAULT_DISTILLATION_CONTEXT
    hardware: str = "h200"
    lora: bool = False
    student_max_seq_len: int | None = DEFAULT_DISTILLATION_STUDENT_MAX_SEQ_LEN
    teacher_max_seq_len: int | None = DEFAULT_DISTILLATION_TEACHER_MAX_SEQ_LEN
    judge_max_seq_len: int | None = DEFAULT_DISTILLATION_JUDGE_MAX_SEQ_LEN
    # GPUs (and tensor parallelism) for the teacher and judge; None keeps the catalog's.
    teacher_gpus: int | None = DEFAULT_DISTILLATION_TEACHER_GPUS
    judge_gpus: int | None = DEFAULT_DISTILLATION_JUDGE_GPUS
    job_config: str | None = None
    teacher_job_config: str | None = None
    judge_job_config: str | None = None
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
    teacher_prompts_per_request: int | None = 8
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
    "job_bodies",
    "lr_at",
    "main",
    "processing_block",
    "train",
    "validate_config",
]

if __name__ == "__main__":
    chz.nested_entrypoint(main)
