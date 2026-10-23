# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reusable on-policy distillation runner for multi-turn tool-use tasks.

The student (training + sampling sub-jobs) plays episodes: it calls the task's
tools until it gives a final answer. A frozen teacher (a separate sampling-only
job) scores every token the student wrote, in one prefill over the same
conversation in the teacher's chat format. Each episode is one training
sequence whose loss covers only the student's own tokens; tool results are
context. The student trains on the ``grpo`` loss with its teacher term enabled,
which adds ``teacher_tau * clamp(log p_teacher - log p_student)`` to every
token's advantage. A judge (a sampling-only job, by default the teacher job)
grades final answers for evaluation and for the optional task reward
(``reward_coef``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import os
import random
import statistics
import time
from collections.abc import Iterator
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import Protocol

from recipes.distillation.episodes import EPISODE_ENDS
from recipes.distillation.episodes import ChatFormat
from recipes.distillation.episodes import Episode
from recipes.distillation.episodes import agent_teacher_logprobs
from recipes.distillation.episodes import run_episodes
from recipes.distillation.episodes import shifted_tensor
from recipes.distillation.episodes import teacher_scoring_inputs
from recipes.distillation.episodes import train_sequences
from recipes.distillation.teacher import check_shared_tokenizer
from recipes.distillation.teacher import clipped_fraction
from recipes.distillation.teacher import generate_results
from recipes.distillation.teacher import reverse_kl_estimates
from recipes.distillation.teacher import score_with_teacher
from recipes.logging import setup_logging
from recipes.utils import collate
from recipes.utils import forward_backward
from recipes.utils import load_job_body
from recipes.utils import make_client
from recipes.utils import optimizer_step
from recipes.utils import save_recipe_checkpoints
from recipes.utils import sync_weights

from cortex_training.client import DEBUG_OPTIONS_ENV

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)

_RECIPE_DIR = Path(__file__).resolve().parent
_CONFIG_SEARCH_DIRS = (_RECIPE_DIR, _RECIPE_DIR / "configs")
_LR_SCHEDULES = ("constant", "cosine")
JUDGE_PARAMS = {"temperature": 0.0, "max_tokens": 8}


class Config(Protocol):
    config: str
    job_id: str | None
    teacher_job_id: str | None
    judge_job_id: str | None
    job_config: str
    teacher_job_config: str
    judge_job_config: str | None
    questions_per_step: int
    group_size: int
    max_turns: int
    turn_max_tokens: int
    temperature: float
    top_p: float
    eval_temperature: float
    eval_top_p: float
    eval_top_k: int
    max_steps: int
    warmup_steps: int
    lr_schedule: str
    min_lr_ratio: float
    teacher_tau: float
    teacher_clip: float
    teacher_clip_negative: float | None
    reward_coef: float
    eps_clip: float
    loss_agg_mode: str
    entropy_coeff: float
    eval_every: int
    n_eval: int
    eval_samples: int
    evaluate_teacher: bool
    max_prompts_per_request: int | None
    seed: int
    weight_sync_bucket_size: int | None
    debug_image_tag: str | None
    log_path: str


class Task(Protocol):
    name: str
    tools: list

    def load(self, *, seed: int, n_eval: int) -> Any: ...

    def user_message(self, problem: Any) -> str: ...

    def call_tool(self, name: str, arguments: dict) -> str: ...

    def judge_prompt(self, problem: Any, response: str) -> str: ...

    def judge_verdict(self, text: str) -> float: ...


def validate_config(config: Config) -> None:
    if not math.isfinite(config.teacher_tau) or config.teacher_tau < 0:
        raise ValueError(
            f"teacher_tau must be a finite non-negative number, got {config.teacher_tau}"
        )
    if config.teacher_tau > 0 and not (
        math.isfinite(config.teacher_clip) and config.teacher_clip > 0
    ):
        raise ValueError(
            f"teacher_tau > 0 needs a finite positive teacher_clip, got {config.teacher_clip}"
        )
    if config.teacher_clip_negative is not None and not (
        math.isfinite(config.teacher_clip_negative) and config.teacher_clip_negative >= 0
    ):
        raise ValueError(
            "teacher_clip_negative must be a finite non-negative number, "
            f"got {config.teacher_clip_negative}"
        )
    if config.teacher_tau == 0 and config.reward_coef == 0:
        raise ValueError(
            "teacher_tau=0 and reward_coef=0 leave no training signal; set one of them"
        )
    if config.teacher_tau == 0 and not (config.judge_job_config or config.judge_job_id):
        raise ValueError("teacher_tau=0 needs a judge: set judge_job_config or judge_job_id")
    if config.lr_schedule not in _LR_SCHEDULES:
        raise ValueError(
            f"lr_schedule must be one of {_LR_SCHEDULES}, got {config.lr_schedule!r}"
        )
    if config.group_size < 1 or config.questions_per_step < 1:
        raise ValueError("questions_per_step and group_size must be positive")
    if config.max_turns < 1 or config.turn_max_tokens < 1:
        raise ValueError("max_turns and turn_max_tokens must be positive")
    if config.eval_samples < 1:
        raise ValueError("eval_samples must be positive")


def _with_debug(body: dict, config: Config) -> dict:
    if config.debug_image_tag:
        body["debug"] = {"job": {"image_tag": config.debug_image_tag}}
    return body


def _sampling_only(body: dict, name: str) -> dict:
    if any(sub.get("job_type") != "sampling" for sub in body.get("sub_job_configs") or ()):
        raise ValueError(f"{name} must contain only sampling sub-jobs")
    return body


def job_bodies(config: Config) -> dict[str, dict]:
    """Job bodies by role: ``student``, plus ``teacher`` and ``judge`` when they are separate jobs."""
    bodies = {
        "student": _with_debug(
            load_job_body(config.job_config, search_dirs=_CONFIG_SEARCH_DIRS), config
        )
    }
    if config.teacher_tau > 0:
        bodies["teacher"] = _sampling_only(
            _with_debug(
                load_job_body(config.teacher_job_config, search_dirs=_CONFIG_SEARCH_DIRS),
                config,
            ),
            config.teacher_job_config,
        )
    if config.judge_job_config:
        bodies["judge"] = _sampling_only(
            _with_debug(
                load_job_body(config.judge_job_config, search_dirs=_CONFIG_SEARCH_DIRS),
                config,
            ),
            config.judge_job_config,
        )
    return bodies


def processing_block(config: Config, global_batch_size: int) -> dict:
    loss_config: dict[str, Any] = {
        "eps_clip": config.eps_clip,
        "loss_agg_mode": config.loss_agg_mode,
        "entropy_coeff": config.entropy_coeff,
        "global_batch_size": global_batch_size,
    }
    if config.teacher_tau > 0:
        loss_config["teacher_tau"] = config.teacher_tau
        loss_config["teacher_clip"] = config.teacher_clip
        if config.teacher_clip_negative is not None:
            loss_config["teacher_clip_negative"] = config.teacher_clip_negative
    return {"loss_fn": "grpo", "post": ["compute_logprobs"], "config": loss_config}


def lr_at(step: int, peak_lr: float, config: Config, total_steps: int) -> float:
    """Linear warmup, then constant or cosine decay to ``min_lr_ratio * peak_lr``."""
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return peak_lr * (step + 1) / config.warmup_steps
    if config.lr_schedule == "constant":
        return peak_lr
    span = max(1, total_steps - config.warmup_steps)
    frac = min(1.0, max(0.0, (step - config.warmup_steps) / span))
    floor = config.min_lr_ratio
    return peak_lr * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * frac)))


def _sub_job(body: dict, job_type: str) -> dict:
    for sub in body.get("sub_job_configs") or ():
        if sub.get("job_type") == job_type:
            return sub
    raise ValueError(f"job config has no {job_type} sub-job")


def _window(body: dict) -> int:
    return int((_sub_job(body, "sampling").get("inference_config") or {}).get("max_seq_len") or 0)


@contextlib.contextmanager
def running_jobs(
    client: Any, bodies: dict[str, dict], attach: dict[str, str | None]
) -> Iterator[dict[str, str]]:
    """Create (or attach to) every job, wait for all, and cancel the ones created here."""
    job_ids: dict[str, str] = {}
    created: list[str] = []
    try:
        for name, body in bodies.items():
            if attach.get(name):
                job_ids[name] = str(attach[name])
                logger.info("attaching to %s job %s", name, job_ids[name])
            else:
                job_ids[name] = client.create_job_from_body(body)["job_id"]
                created.append(job_ids[name])
                logger.info("created %s job %s", name, job_ids[name])
        for name, job_id in job_ids.items():
            client.wait_for_job(job_id)
            logger.info("%s job %s is running", name, job_id)
        yield job_ids
    finally:
        for job_id in reversed(created):
            try:
                client.cancel_job(job_id)
                logger.info("cancelled job %s", job_id)
            except Exception:
                logger.exception("failed to cancel job %s -- cancel it by hand", job_id)


def judge(
    client: Any,
    job_id: str,
    fmt: ChatFormat,
    task: Task,
    episodes: Sequence[Episode],
    max_prompts_per_request: int | None,
) -> None:
    """Grade every episode that ended with a final answer; the rest score 0."""
    for e in episodes:
        e.reward = 0.0
    answered = [e for e in episodes if e.end == "answer" and e.final]
    if not answered:
        return
    prompts = [
        fmt.render([{"role": "user", "content": task.judge_prompt(e.problem, e.final)}])
        for e in answered
    ]
    params = {"stop_token_ids": fmt.stop_ids, **JUDGE_PARAMS}
    results = generate_results(client, job_id, prompts, params, max_prompts_per_request)
    for e, result in zip(answered, results):
        tokens = [int(t) for t in result.get("token_ids") or []]
        e.reward = task.judge_verdict(fmt.tok.decode(tokens, skip_special_tokens=True))


def episode_metrics(prefix: str, episodes: Sequence[Episode]) -> dict[str, float]:
    return {
        f"{prefix}/correct": statistics.fmean(e.reward for e in episodes),
        f"{prefix}/turns": statistics.fmean(e.turns for e in episodes),
        f"{prefix}/tool_calls": statistics.fmean(e.tool_calls for e in episodes),
        f"{prefix}/agent_tokens": statistics.fmean(e.agent_tokens for e in episodes),
        **{
            f"{prefix}/end_{end}": statistics.fmean(e.end == end for e in episodes)
            for end in EPISODE_ENDS
        },
    }


def write_episodes(
    path: Path, step: int, episodes: Sequence[Episode], fmt: ChatFormat, transcripts: int
) -> None:
    """Append one JSON line per episode; the first ``transcripts`` include the full conversation."""
    with open(path, "a") as f:
        for i, e in enumerate(episodes):
            row: dict[str, Any] = {
                "step": step,
                "problem": dataclasses.asdict(e.problem)
                if dataclasses.is_dataclass(e.problem)
                else str(e.problem),
                "final": e.final,
                "reward": e.reward,
                "end": e.end,
                "turns": e.turns,
                "tool_calls": e.tool_calls,
            }
            if i < transcripts and e.spans[0]:
                row["transcript"] = fmt.tok.decode(
                    e.ids[0][e.spans[0][0][0] :], skip_special_tokens=False
                )
            f.write(json.dumps(row) + "\n")


def _train(config: Config, task: Task) -> None:
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    validate_config(config)
    bodies = job_bodies(config)
    student_body = bodies["student"]
    training_sub = _sub_job(student_body, "training")
    training = training_sub.get("training_config") or {}
    peak_lr = float((training.get("optimizer") or {}).get("lr"))
    max_seq_len = int(training.get("max_seq_len") or _window(student_body))
    if config.turn_max_tokens >= max_seq_len:
        raise ValueError(
            f"turn_max_tokens {config.turn_max_tokens} leaves no room for a prompt in "
            f"max_seq_len {max_seq_len}"
        )
    # A LoRA student syncs only its adapters to the sampler.
    sync_format = "lora" if training.get("peft_config") else None

    student = ChatFormat(get_tokenizer(training_sub["model_name"]))
    if not student.check_tool_turn(task.tools):
        raise ValueError(
            f"{training_sub['model_name']}: appended tool turns do not match its chat template"
        )
    teacher = judge_fmt = None
    if "teacher" in bodies:
        teacher_model = _sub_job(bodies["teacher"], "sampling")["model_name"]
        teacher = ChatFormat(get_tokenizer(teacher_model))
        check_shared_tokenizer(student.tok, teacher.tok)
        if not teacher.check_tool_turn(task.tools):
            raise ValueError(
                f"{teacher_model}: appended tool turns do not match its chat template"
            )
        if _window(bodies["teacher"]) <= max_seq_len:
            raise ValueError(
                f"teacher max_seq_len {_window(bodies['teacher'])} must exceed the student's "
                f"{max_seq_len}: the teacher scores whole conversations, with its own turn headers"
            )
        logger.info(
            "teacher %s, teacher_tau=%s, teacher_clip=%s, teacher_clip_negative=%s",
            teacher_model,
            config.teacher_tau,
            config.teacher_clip,
            config.teacher_clip_negative,
        )
    judge_body = bodies.get("judge") or bodies.get("teacher")
    if judge_body is not None:
        judge_fmt = ChatFormat(get_tokenizer(_sub_job(judge_body, "sampling")["model_name"]))
    elif config.judge_job_id is None:
        raise ValueError("no judge: set judge_job_config or judge_job_id")

    problems = task.load(seed=config.seed, n_eval=config.n_eval)
    train_problems = list(problems.train)
    if len(train_problems) < config.questions_per_step:
        raise ValueError(
            f"{len(train_problems)} training questions, fewer than questions_per_step "
            f"{config.questions_per_step}"
        )
    logger.info(
        "%s: %d training questions, %d evaluation questions",
        task.name,
        len(train_problems),
        len(problems.eval),
    )
    rollout_sampling = {"temperature": config.temperature, "top_p": config.top_p}
    eval_sampling = {
        "temperature": config.eval_temperature,
        "top_p": config.eval_top_p,
        "top_k": config.eval_top_k,
    }
    episode_args = dict(
        user_message=task.user_message,
        tools=task.tools,
        call_tool=task.call_tool,
        max_turns=config.max_turns,
        turn_max_tokens=config.turn_max_tokens,
        max_seq_len=max_seq_len,
        max_prompts_per_request=config.max_prompts_per_request,
    )
    log_dir = Path(config.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    client = make_client(config.config)
    attach = {
        "student": config.job_id,
        "teacher": config.teacher_job_id,
        "judge": config.judge_job_id,
    }
    with running_jobs(client, bodies, attach) as job_ids:
        student_job = job_ids["student"]
        teacher_job = job_ids.get("teacher")
        judge_job = job_ids.get("judge") or config.judge_job_id or teacher_job
        if judge_fmt is None:
            judge_fmt = teacher or student
        ml_logger = setup_logging(config, client=client, job_id=student_job)
        history: list[dict[str, float]] = []

        def evaluate(step: int) -> dict[str, float]:
            start = time.time()
            agents = [("test", student_job, student)]
            if step == 0 and teacher_job is not None and config.evaluate_teacher:
                agents.append(("teacher/test", teacher_job, teacher))
            metrics: dict[str, float] = {}
            for prefix, job, fmt in agents:
                episodes = run_episodes(
                    client,
                    job,
                    [fmt],
                    problems.eval,
                    repeats=config.eval_samples,
                    sampling=eval_sampling,
                    want_logprobs=False,
                    **episode_args,
                )
                judge(client, judge_job, judge_fmt, task, episodes, config.max_prompts_per_request)
                metrics.update(episode_metrics(prefix, episodes))
                name = "teacher_eval_episodes.jsonl" if prefix != "test" else "eval_episodes.jsonl"
                write_episodes(log_dir / name, step, episodes, fmt, transcripts=3)
            metrics["time/eval"] = time.time() - start
            logger.info(
                "step %d: %s accuracy %.4f%s",
                step,
                task.name,
                metrics["test/correct"],
                (
                    f", teacher {metrics['teacher/test/correct']:.4f}"
                    if "teacher/test/correct" in metrics
                    else ""
                ),
            )
            history.append({"step": step, **metrics})
            return metrics

        rng = random.Random(config.seed)
        pool = list(train_problems)
        rng.shuffle(pool)
        cursor = 0
        formats = [student] + ([teacher] if teacher is not None else [])
        for step in range(config.max_steps):
            start = time.time()
            metrics: dict[str, float] = {
                "progress/step": step,
                "progress/done_frac": (step + 1) / config.max_steps,
            }
            if config.eval_every > 0 and step % config.eval_every == 0:
                metrics.update(evaluate(step))

            if cursor + config.questions_per_step > len(pool):
                rng.shuffle(pool)
                cursor = 0
            batch = pool[cursor : cursor + config.questions_per_step]
            cursor += config.questions_per_step
            episodes = run_episodes(
                client,
                student_job,
                formats,
                batch,
                repeats=config.group_size,
                sampling=rollout_sampling,
                want_logprobs=True,
                **episode_args,
            )
            judge(client, judge_job, judge_fmt, task, episodes, config.max_prompts_per_request)
            metrics.update(episode_metrics("rollout", episodes))
            metrics["time/rollouts"] = time.time() - start
            write_episodes(log_dir / "train_episodes.jsonl", step, episodes, student, transcripts=2)

            advantages: list[float] = []
            for g in range(0, len(episodes), config.group_size):
                group = episodes[g : g + config.group_size]
                mean = statistics.fmean(e.reward for e in group)
                advantages += [config.reward_coef * (e.reward - mean) for e in group]
            kept = [(e, a) for e, a in zip(episodes, advantages) if e.spans[0]]
            if not kept:
                logger.warning("step %d: every episode was empty, skipping the update", step)
                ml_logger.log_metrics(metrics, step=step)
                continue
            episodes = [e for e, _ in kept]

            teacher_lps = None
            if teacher_job is not None:
                score_start = time.time()
                prompts, rests = teacher_scoring_inputs(episodes)
                scores = score_with_teacher(
                    client, teacher_job, prompts, rests, max_prompts_per_request=8
                )
                teacher_lps = agent_teacher_logprobs(episodes, scores)
                student_lps = [e.logprobs for e in episodes]
                metrics.update(
                    {
                        f"train/{name}": value
                        for name, value in reverse_kl_estimates(student_lps, teacher_lps).items()
                    }
                )
                fraction = clipped_fraction(
                    student_lps,
                    teacher_lps,
                    clip=config.teacher_clip,
                    clip_negative=(
                        config.teacher_clip
                        if config.teacher_clip_negative is None
                        else config.teacher_clip_negative
                    ),
                )
                if fraction is not None:
                    metrics["teacher/clipped_fraction"] = fraction
                metrics["time/teacher_score"] = time.time() - score_start

            sequences, shifted = train_sequences(episodes, [a for _, a in kept], teacher_lps)
            kwargs, context = collate(
                sequences,
                pad_token_id=student.tok.pad_token_id or student.tok.eos_token_id,
                max_seq_len=max_seq_len,
                temperature=config.temperature,
            )
            n_tokens = sum(e.agent_tokens for e in episodes)
            if shifted is not None:
                context["teacher_log_probs_shifted"] = shifted_tensor(shifted, context["loss_mask"])

            learning_rate = lr_at(step, peak_lr, config, config.max_steps)
            train_start = time.time()
            fwd_bwd_result = forward_backward(
                client,
                student_job,
                kwargs,
                context=context,
                processing=processing_block(config, global_batch_size=len(sequences)),
            )
            server = fwd_bwd_result.get("metrics") or {}
            if teacher_job is not None:
                covered = float(server.get("teacher_term_token_count") or 0.0)
                if covered != n_tokens:
                    raise RuntimeError(
                        f"step {step}: the teacher term covered {covered} tokens, "
                        f"expected {n_tokens}; stopped before the optimizer step"
                    )
            step_result = optimizer_step(client, student_job, learning_rate=learning_rate)
            metrics.update(server)
            metrics.update(step_result.get("metrics") or {})
            if step_result.get("grad_norm") is not None:
                metrics["optim/grad_norm"] = float(step_result["grad_norm"])
            sync_weights(
                client,
                student_job,
                weight_format=sync_format,
                bucket_size=config.weight_sync_bucket_size,
            )
            metrics.update(
                {
                    "optim/lr": learning_rate,
                    "train/avg_loss": float(fwd_bwd_result.get("avg_loss", float("nan"))),
                    "train/sequences": float(len(sequences)),
                    "train/agent_tokens": float(n_tokens),
                    "time/train": time.time() - train_start,
                    "time/total": time.time() - start,
                }
            )
            ml_logger.log_metrics(metrics, step=step)

        if config.eval_every > 0:
            ml_logger.log_metrics(evaluate(config.max_steps), step=config.max_steps)
            logger.info(
                "%s accuracy: step 0 %.4f, final %.4f",
                task.name,
                history[0]["test/correct"],
                history[-1]["test/correct"],
            )

        saved = save_recipe_checkpoints(client, student_job)
        logger.info(
            "Saved weights-only checkpoint_id=%s from job %s",
            saved["weights-only"].get("checkpoint_id"),
            student_job,
        )
    ml_logger.close()


def train(config: Config, task: Task) -> None:
    if config.debug_image_tag:
        os.environ[DEBUG_OPTIONS_ENV] = "1"
        logger.info("Using debug image_tag=%s", config.debug_image_tag)
    _train(config, task)
    logger.info("Training completed")
