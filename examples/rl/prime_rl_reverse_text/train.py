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
Asynchronous reverse-text RL with prime-rl's orchestrator schedule and loss.

A port of the RL stage of prime-rl's ``examples/basic/reverse-text``:
Qwen3-0.6B after the ``sft.py`` warmup, 20 steps of 128 rollouts (8 prompts x
16), 128 completion tokens, at prime-rl's learning rate of 3e-6.

- ``loss="ipo"`` (default), prime-rl's IPO loss: ``keep`` drops tokens whose
  probability moved more than ``ipo_eps`` either way, and ``kl_tau`` defaults
  to 0. The Cortex ``grpo`` loss computes it with its CISPO term and ratio
  controls; see ``ipo_loss_context``.
- ``loss="dppo"``, prime-rl's earlier DPPO+KL loss: ``keep`` drops tokens whose
  probability moved past ``dppo_mask_high`` (A > 0) or ``dppo_mask_low``
  (A < 0), and ``kl_tau`` defaults to 1e-3. Per-token proximal logprobs from
  ``dppo_prox_logprobs`` put the PPO ratio clip on the DPPO band.
"""

from __future__ import annotations

import logging
import queue
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import chz
import torch
from examples.logging import setup_logging
from examples.rl.prime_rl_reverse_text.task import RL_DATASET
from examples.rl.prime_rl_reverse_text.task import ReverseTextTask
from examples.rl.prime_rl_reverse_text.task import lcs_reward
from examples.rl.prime_rl_reverse_text.task import load_tasks
from examples.rl.prime_rl_reverse_text.task import task_messages
from examples.utils import build_job
from examples.utils import build_renderer
from examples.utils import collate
from examples.utils import example_connection_args
from examples.utils import forward_backward_step
from examples.utils import inference_job_args
from examples.utils import log_saved_checkpoints
from examples.utils import make_client
from examples.utils import running_job
from examples.utils import sampled_logprobs_shifted
from examples.utils import save_example_checkpoints
from examples.utils import sequence_from_rollout
from examples.utils import source_checkpoint_info
from examples.utils import stop_params_for
from examples.utils import sync_weights

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)

EXAMPLE_DIR = Path(__file__).resolve().parent
TARGET_LAG = 1
MAX_CONSECUTIVE_EMPTY_BATCHES = 10
MAX_CONSECUTIVE_ZERO_OUTPUT_BATCH_EQUIVALENTS = 10
ADMISSION_WINDOW_S = 5.0
CANCEL_WORKERS = 32
LOSSES = ("ipo", "dppo", "grpo")
# prime-rl's kl_tau for each loss.
DEFAULT_KL_TAU = {"ipo": 0.0, "dppo": 1e-3, "grpo": 1e-3}
# prime-rl's in-flight cap growth per turnover: 1.2 since the IPO loss, 1.25 before.
DEFAULT_INFLIGHT_GROWTH = {"ipo": 1.2, "dppo": 1.25, "grpo": 1.25}
# The cap grows only while the pool holds at least this fraction of it.
INFLIGHT_BINDING_FRACTION = 0.9
# Any clip in (0, 1) gives the same DPPO mask; the proximal logprobs absorb it.
DPPO_PROX_CLIP = 0.5
# Nats below the sampler logprob where the DPPO lower bound does not exist.
DPPO_PROX_FLOOR = 20.0
# Above exp(20), where the server clamps the importance ratio, so the CISPO weight is never capped.
IPO_WEIGHT_CAP = 1e10
# Lowest sampler logprob whose IPO weight stays finite in float32.
IPO_LOGPROB_FLOOR = -80.0
# Server metrics that do not depend on the "old" logprobs, which the IPO request repurposes.
IPO_SERVER_METRICS = ("entropy",)


@chz.chz
class Config:
    # Legacy JSON connection file. Omit it to use `connection` or the default profile.
    config: str | None = None
    # Snowflake connection profile name (connections.toml). Use config or connection, not both.
    connection: str | None = None
    job_id: str | None = None

    # The job comes from model-catalog/models/ and this folder's overrides.json (one training and
    # one sampling GPU, as prime-rl deploys this example); job_config= replaces it with a file.
    model: str = "Qwen/Qwen3-0.6B"
    context: str = "standard"
    hardware: str = "h200"
    lora: bool = False
    job_config: str | None = None
    # None keeps the job's value (lr 3e-6, prime-rl's).
    lr: float | None = None
    train_batch_size: int | None = None
    max_seq_len: int | None = None
    # Checkpoints from sft.py. The sampling sub-job starts from the weights-only
    # checkpoint; the training sub-job restores the resumable one with a fresh optimizer.
    init_checkpoint_id: str | None = None
    init_trainer_checkpoint_id: str | None = None
    init_source_job_id: str | None = None

    dataset: str = RL_DATASET
    dataset_split: str = "train"
    max_steps: int = 20
    # Rollouts per step: 8 prompts x group_size 16.
    batch_size: int = 128
    group_size: int = 16
    max_completion_tokens: int = 128
    temperature: float = 1.0
    max_off_policy_steps: int = 8
    # prime-rl starts its in-flight cap at the sampler's KV-cache tokens over its max model
    # length, 27-28 rollouts for this example, then grows it while it binds.
    initial_inflight: int = 28
    # None uses prime-rl's growth for the loss: 1.2 for "ipo", 1.25 for "dppo" and "grpo".
    inflight_growth: float | None = None
    max_inflight: int = 1024

    # "ipo" and "dppo" mask on the probability change; "grpo" clips the ratio at eps_clip.
    loss: str = "ipo"
    ipo_eps: float = 0.3
    dppo_mask_low: float = 0.2
    dppo_mask_high: float = 0.2
    eps_clip: float = 0.2
    adv_tau: float = 1.0
    # None uses prime-rl's value for the loss: 0 for "ipo", 1e-3 for "dppo" and "grpo".
    kl_tau: float | None = None
    # Prune zero-advantage rollouts before they count toward batch_size. None follows the
    # loss: on for "ipo", as prime-rl does since that loss; off for "dppo" and "grpo".
    constant_batch_size: bool | None = None

    # Before and after training: eval_examples prompts x eval_rollouts. 0 disables.
    eval_examples: int = 20
    eval_rollouts: int = 3
    eval_max_tokens: int = 1024

    renderer_name: str = "qwen3"
    weight_sync_bucket_size: int | None = None
    debug_image_tag: str | None = None
    log_path: str = "/tmp/cortex-training-examples/prime-rl-reverse-text"
    wandb_project: str | None = None
    wandb_name: str | None = None
    sf_tracking: bool = False


def validate_config(config: Config) -> None:
    if config.temperature != 1.0:
        raise ValueError(
            "temperature must be 1.0: sampler logprobs are reported before temperature "
            "scaling, so they only match the trainer at 1.0"
        )
    sizes = (config.max_steps, config.batch_size, config.group_size, config.max_completion_tokens)
    if min(*sizes, config.initial_inflight, config.max_inflight) <= 0:
        raise ValueError(
            "max_steps, batch_size, group_size, max_completion_tokens, initial_inflight, and max_inflight must be > 0"
        )
    if inflight_growth(config) < 1:
        raise ValueError("inflight_growth must be >= 1")
    if config.batch_size % config.group_size:
        raise ValueError("batch_size must be a multiple of group_size")
    if config.max_off_policy_steps < 0:
        raise ValueError("max_off_policy_steps must be >= 0")
    if config.loss not in LOSSES:
        raise ValueError(f"loss must be one of {', '.join(LOSSES)}")
    if min(config.dppo_mask_low, config.dppo_mask_high, config.ipo_eps, kl_coefficient(config)) < 0:
        raise ValueError("dppo_mask_low, dppo_mask_high, ipo_eps, and kl_tau must be >= 0")
    if config.eps_clip <= 0:
        raise ValueError("eps_clip must be > 0")
    init = (config.init_checkpoint_id, config.init_trainer_checkpoint_id, config.init_source_job_id)
    if any(init) and not all(init):
        raise ValueError(
            "init_checkpoint_id, init_trainer_checkpoint_id, and init_source_job_id must be set together"
        )


def kl_coefficient(config: Config) -> float:
    return DEFAULT_KL_TAU.get(config.loss, 0.0) if config.kl_tau is None else config.kl_tau


def inflight_growth(config: Config) -> float:
    return DEFAULT_INFLIGHT_GROWTH.get(config.loss, 1.0) if config.inflight_growth is None else config.inflight_growth


def prunes_before_cut(config: Config) -> bool:
    return config.loss == "ipo" if config.constant_batch_size is None else config.constant_batch_size


def eval_enabled(config: Config) -> bool:
    return config.eval_examples > 0 and config.eval_rollouts > 0


def job_body(config: Config) -> dict:
    body = build_job(
        "rl",
        model=config.model,
        context=config.context,
        hardware=config.hardware,
        lora=config.lora,
        job_config=config.job_config,
        example_dir=EXAMPLE_DIR,
        search_dirs=(EXAMPLE_DIR,),
        lr=config.lr,
        train_batch_size=config.train_batch_size,
        max_seq_len=config.max_seq_len,
        debug_image_tag=config.debug_image_tag,
    )
    sampler_init = source_checkpoint_info(config.init_source_job_id, config.init_checkpoint_id)
    trainer_init = source_checkpoint_info(config.init_source_job_id, config.init_trainer_checkpoint_id)
    for sub_job in body.get("sub_job_configs") or ():
        if sub_job.get("job_type") == "sampling" and sampler_init is not None:
            sub_job["source_checkpoint_info"] = dict(sampler_init)
        elif sub_job.get("job_type") == "training" and trainer_init is not None:
            sub_job["source_checkpoint_info"] = dict(trainer_init)
            # A fresh optimizer: the SFT optimizer state is not part of the RL stage.
            sub_job["training_config"]["load_optimizer_states"] = False
    if sampler_init is not None or trainer_init is not None:
        logger.info(
            "The sampling sub-job starts from %s, the training sub-job from %s with a fresh optimizer",
            sampler_init,
            trainer_init,
        )
    return body


def data_parallel_size(training_config: dict) -> int:
    """Training GPUs divided by the sequence- and context-parallel group size."""
    n_gpus = int(training_config.get("n_gpus") or 1)
    group = int(training_config.get("sp_size") or 1) * int(training_config.get("cp_size") or 1)
    return max(n_gpus // max(group, 1), 1)


def min_fresh_version(step: int, max_off_policy_steps: int) -> int:
    """Oldest dispatch version whose rollouts batch ``step`` may still train on."""
    return (step - 1) - max_off_policy_steps


def group_advantages(rewards: list[float]) -> list[float]:
    """Reward minus the group mean, in float32 as prime-rl computes it. Rollouts whose advantage is
    exactly zero are pruned, which is not every group of equal rewards."""
    if not rewards:
        return []
    values = torch.tensor(rewards, dtype=torch.float32)
    return (values - values.mean()).tolist()


def dppo_prox_logprobs(
    advantages: torch.Tensor,
    inference_logprobs: torch.Tensor,
    *,
    dppo_mask_low: float,
    dppo_mask_high: float,
) -> torch.Tensor:
    """Per-token proximal logprobs ``q`` that put the grpo ratio clip on the DPPO band."""
    mu = inference_logprobs.exp()
    upper = torch.log((mu + dppo_mask_high) / (1.0 + DPPO_PROX_CLIP))
    lower = torch.maximum(
        torch.log((mu - dppo_mask_low).clamp(min=0.0) / (1.0 - DPPO_PROX_CLIP)),
        inference_logprobs - DPPO_PROX_FLOOR,
    )
    return torch.where(advantages > 0, upper, torch.where(advantages < 0, lower, inference_logprobs))


def ipo_loss_context(
    advantages: torch.Tensor, inference_logprobs: torch.Tensor, *, ipo_eps: float
) -> dict[str, torch.Tensor]:
    """Context tensors that turn the grpo loss's CISPO term into prime-rl's IPO loss."""
    logprobs = inference_logprobs.double().clamp(min=IPO_LOGPROB_FLOOR)
    weights = -torch.sign(advantages).double() * (1.0 + ipo_eps * torch.exp(-logprobs))
    return {
        "advantages": -advantages.abs(),
        "prox_logp_shifted": inference_logprobs,
        "old_log_probs_shifted": torch.log(logprobs.exp() + ipo_eps).float(),
        "rollout_is_weights": weights.float(),
    }


def loss_context(
    context: dict[str, torch.Tensor], inference_logprobs: torch.Tensor, config: Config
) -> dict[str, torch.Tensor]:
    out = dict(context)
    advantages = config.adv_tau * context["advantages"]
    out["advantages"] = advantages
    out["old_log_probs_shifted"] = inference_logprobs
    out["ref_log_probs_shifted"] = inference_logprobs
    if config.loss == "ipo":
        out.update(ipo_loss_context(advantages, inference_logprobs, ipo_eps=config.ipo_eps))
    elif config.loss == "dppo":
        out["prox_logp_shifted"] = dppo_prox_logprobs(
            advantages,
            inference_logprobs,
            dppo_mask_low=config.dppo_mask_low,
            dppo_mask_high=config.dppo_mask_high,
        )
    return out


def loss_processing(config: Config, *, batch_num_tokens: int, dp_size: int) -> dict[str, Any]:
    processing_config: dict[str, Any] = {
        "loss_agg_mode": "token-mean",
        "batch_num_tokens": batch_num_tokens,
        "dp_size": dp_size,
    }
    if config.loss == "ipo":
        processing_config.update(
            use_cispo_loss=True,
            is_weight_clip_max=IPO_WEIGHT_CAP,
            ratio_mask_bounds_neg=[None, 1.0],
            prob_diff_mask_max_neg=2.0 * config.ipo_eps,
        )
    else:
        processing_config["eps_clip"] = DPPO_PROX_CLIP if config.loss == "dppo" else config.eps_clip
    kl_tau = kl_coefficient(config)
    if kl_tau > 0:
        processing_config.update(use_kl_loss=True, kl_loss_type="k2", kl_loss_coef=2.0 * kl_tau)
    return {"loss_fn": "grpo", "post": ["compute_logprobs"], "config": processing_config}


def server_loss_metrics(raw: dict[str, Any], loss: str) -> dict[str, float]:
    """The grpo loss metrics, with ``loss/is_masked`` the share of tokens the trust region zeroes."""
    metrics = {
        key: float(value)
        for key, value in raw.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if loss == "dppo":
        # Measured against the proximal logprobs, not the sampler.
        metrics.pop("importance_weight", None)
        metrics.pop("approx_kl", None)
    if loss == "ipo":
        # The rest are measured against the shifted log(mu + eps) the IPO request sends as "old".
        trainable = metrics.get("ratio_trainable_token_count")
        dropped = metrics.get("ratio_mask_dropped_token_count")
        out = {f"trainer/{key}": metrics[key] for key in IPO_SERVER_METRICS if key in metrics}
        if trainable and dropped is not None:
            out["loss/is_masked"] = dropped / trainable
        return out
    out = {f"trainer/{key}": value for key, value in metrics.items() if key != "clip_ratio"}
    if "clip_ratio" in metrics:
        out["loss/is_masked"] = metrics["clip_ratio"]
    return out


@dataclass(eq=False)
class Group:
    group_id: int
    task: ReverseTextTask
    prompt_tokens: list[int]
    version_start: int
    to_schedule: int
    # Dispatcher state, guarded by Pipeline.cond.
    inflight: set[Episode] = field(default_factory=set)
    # Sink state, touched only by the orchestrator thread.
    arrived: list[tuple[dict, int]] = field(default_factory=list)
    failed: int = 0
    cancelled: int = 0
    stale: bool = False

    @property
    def accounted(self) -> int:
        return len(self.arrived) + self.failed + self.cancelled


@dataclass(eq=False)
class Episode:
    group: Group
    request_id: str | None = None
    cancelled: bool = False


@dataclass
class Rollout:
    prompt_tokens: list[int]
    result: dict
    reward: float
    advantage: float
    version_start: int
    version_end: int


@dataclass
class Batch:
    step: int
    rollouts: list[Rollout]
    metrics: dict[str, float] = field(default_factory=dict)


def rollout_metrics(cut: list[Rollout], trained: list[Rollout], step: int) -> dict[str, float]:
    rewards = [rollout.reward for rollout in trained]
    staleness = [max(0, (step - 1) - rollout.version_start) for rollout in trained]
    in_flight = [
        min(total, max(0, rollout.version_end - rollout.version_start)) for total, rollout in zip(staleness, trained)
    ]
    in_queue = [total - span for total, span in zip(staleness, in_flight)]
    return {
        "reward/mean": statistics.fmean(rewards),
        "reward/std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "seq_len/completion_mean": statistics.fmean(len(rollout.result["token_ids"]) for rollout in cut),
        "batch/rollouts": len(cut),
        "batch/trained_rollouts": len(trained),
        "off_policy/mean": statistics.fmean(staleness),
        "off_policy/max": max(staleness),
        "off_policy/in_flight/mean": statistics.fmean(in_flight),
        "off_policy/in_flight/max": max(in_flight),
        "off_policy/in_queue/mean": statistics.fmean(in_queue),
        "off_policy/in_queue/max": max(in_queue),
    }


class Pipeline:
    """Rollout workers, the orchestrator (caller's thread), and the trainer thread."""

    def __init__(
        self,
        config: Config,
        *,
        client: Any,
        rollout_client: Any,
        job_id: str,
        tasks: list[ReverseTextTask],
        prompt_tokens: Any,
        sampling_params: dict,
        pad_token_id: int,
        max_seq_len: int,
        dp_size: int,
        learning_rate: float,
        weight_format: str | None,
        ml_logger: Any,
    ) -> None:
        self.config = config
        self.client = client
        self.rollout_client = rollout_client
        self.job_id = job_id
        self.tasks = tasks
        self.prompt_tokens = prompt_tokens
        self.sampling_params = sampling_params
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.dp_size = dp_size
        self.learning_rate = learning_rate
        self.weight_format = weight_format
        self.ml_logger = ml_logger

        self.cond = threading.Condition()
        self.policy_version = 0
        self.collect_step = 1
        self.dispatch_allowed = True
        self.draining = False
        self.groups: dict[int, Group] = {}
        self.trainer_error: BaseException | None = None
        self.out_q: queue.Queue = queue.Queue()
        self.train_q: queue.Queue = queue.Queue()
        self._pool = ThreadPoolExecutor(max_workers=config.max_inflight, thread_name_prefix="rollout")
        self._cancel_pool = ThreadPoolExecutor(max_workers=CANCEL_WORKERS, thread_name_prefix="cancel")
        self._inflight = 0
        self._inflight_cap = float(min(config.initial_inflight, config.max_inflight))
        self._inflight_growth = inflight_growth(config)
        self._admission_window_start = time.monotonic()
        self._admissions_in_window = 0
        self._next_group_id = 0
        self._next_task = 0
        self._gate_closed_at = 0.0
        self._wait_for_policy = 0.0
        self._last_batch_at: float | None = None
        self._cancelled = 0
        self._dropped = 0
        self._errored = 0
        self._zero_output_units = 0
        self._zero_output_windows = 0
        self._consecutive_empty = 0
        # Rewards of the fresh rollouts finalized since the last batch, zero-advantage ones included.
        self._arrived_rewards: list[float] = []

    # Dispatcher. Rollout workers share rollout_client; the state they touch is guarded by cond.

    def _admission_budget(self) -> int:
        """Admissions left in the burst window. Replacing a finished rollout is free."""
        now = time.monotonic()
        if now - self._admission_window_start >= ADMISSION_WINDOW_S:
            self._admission_window_start = now
            self._admissions_in_window = 0
        return max(self.config.group_size, int(self._inflight_cap) // 10) - self._admissions_in_window

    def _grow_inflight_cap(self) -> None:
        """Advance the turnover by one completion; the cap compounds by inflight_growth per turnover while it binds."""
        inflight = self._inflight + 1
        if inflight >= INFLIGHT_BINDING_FRACTION * int(self._inflight_cap):
            grown = self._inflight_cap * self._inflight_growth ** (1 / inflight)
            self._inflight_cap = min(grown, self.config.max_inflight)

    def _open_group(self) -> Group:
        task = self.tasks[self._next_task % len(self.tasks)]
        self._next_task += 1
        group = Group(self._next_group_id, task, self.prompt_tokens(task), self.policy_version, self.config.group_size)
        self._next_group_id += 1
        self.groups[group.group_id] = group
        return group

    def fill_inflight(self) -> None:
        """Dispatch rollouts, finishing the open groups before opening a new one."""
        with self.cond:
            while (
                not self.draining
                and self.dispatch_allowed
                and self._inflight < int(self._inflight_cap)
                and self._admission_budget() > 0
            ):
                group = next((g for g in self.groups.values() if g.to_schedule > 0), None) or self._open_group()
                group.to_schedule -= 1
                episode = Episode(group)
                group.inflight.add(episode)
                self._inflight += 1
                self._admissions_in_window += 1
                self._pool.submit(self._run_episode, episode)

    def _finish(self, episode: Episode, produced: bool = False) -> int | None:
        """Free a completed rollout's permit and refund its admission; one that produced tokens may grow the cap.

        Returns the policy version at completion, or None if a drop already claimed the rollout.
        """
        with self.cond:
            if episode.cancelled:
                return None
            group = episode.group
            group.inflight.discard(episode)
            if not group.inflight and group.to_schedule == 0:
                self.groups.pop(group.group_id, None)
            self._inflight -= 1
            self._admissions_in_window = max(0, self._admissions_in_window - 1)
            if produced:
                self._grow_inflight_cap()
            return self.policy_version

    def _run_episode(self, episode: Episode) -> None:
        client = self.rollout_client
        try:
            request_id = client.generate(
                self.job_id, prompts=[episode.group.prompt_tokens], sampling_params=self.sampling_params
            )
            with self.cond:
                episode.request_id = request_id
                cancelled = episode.cancelled
            if cancelled:
                self._cancel_request(request_id)
                return
            results = client.poll_request(self.job_id, request_id)["results"]
            if len(results) != 1:
                raise RuntimeError(f"asked for 1 rollout, got {len(results)}")
        except Exception as exc:
            if self._finish(episode) is not None:
                self.out_q.put(("failed", episode.group, exc))
            return
        version_end = self._finish(episode, produced=bool(results[0].get("token_ids")))
        if version_end is not None:
            self.out_q.put(("done", episode.group, (results[0], version_end)))

    def _cancel_request(self, request_id: str) -> None:
        try:
            self.rollout_client.cancel_request(self.job_id, request_id)
        except Exception:
            logger.debug("cancel_request(%s) failed", request_id, exc_info=True)

    def _cancel_requests(self, request_ids: list[str]) -> None:
        list(self._cancel_pool.map(self._cancel_request, request_ids))

    def _drop_group(self, group: Group, request_ids: list[str]) -> int:
        """Cancel a group's in-flight and unscheduled rollouts; returns how many it still owed."""
        self.groups.pop(group.group_id, None)
        owed = len(group.inflight) + group.to_schedule
        for episode in group.inflight:
            episode.cancelled = True
            if episode.request_id is not None:
                request_ids.append(episode.request_id)
        self._inflight -= len(group.inflight)
        self._cancelled += owed
        group.inflight.clear()
        group.to_schedule = 0
        return owed

    def drop_stale_groups(self) -> None:
        """Drop the groups that the batch being collected can no longer train on."""
        request_ids: list[str] = []
        dropped = 0
        with self.cond:
            min_version = min_fresh_version(self.collect_step, self.config.max_off_policy_steps)
            for group in [g for g in self.groups.values() if g.version_start < min_version]:
                owed = self._drop_group(group, request_ids)
                dropped += owed
                self.out_q.put(("stale", group, owed))
        self._cancel_requests(request_ids)
        if dropped:
            logger.warning(
                "Cancelled %d rollouts past max_off_policy_steps=%d", dropped, self.config.max_off_policy_steps
            )

    # Orchestrator.

    def _handle(self, kind: str, group: Group, payload: Any, pending: list[Rollout]) -> bool:
        """Account for one group member; returns True once the group is finalized."""
        if kind == "done":
            group.arrived.append(payload)
            if not payload[0].get("token_ids"):
                self._errored += 1
        elif kind == "failed":
            group.failed += 1
            self._errored += 1
            logger.warning("A rollout in group %d failed: %s", group.group_id, payload)
        else:
            group.cancelled += payload
            group.stale = True
        if group.accounted < self.config.group_size:
            return False
        self._finalize_group(group, pending)
        return True

    def _finalize_group(self, group: Group, pending: list[Rollout]) -> None:
        survivors = [(result, version_end) for result, version_end in group.arrived if result.get("token_ids")]
        if group.stale or not survivors:
            self._record_zero_output(group)
            return
        rewards = [lcs_reward(result.get("text") or "", group.task.answer) for result, _ in survivors]
        rollouts = [
            Rollout(group.prompt_tokens, result, reward, advantage, group.version_start, version_end)
            for (result, version_end), reward, advantage in zip(survivors, rewards, group_advantages(rewards))
        ]
        if group.version_start < min_fresh_version(self.collect_step, self.config.max_off_policy_steps):
            self._dropped += len(rollouts)
            self._record_zero_output(group)
            return
        self._arrived_rewards.extend(rewards)
        if prunes_before_cut(self.config):
            rollouts = [rollout for rollout in rollouts if rollout.advantage != 0.0]
            if not rollouts:
                self._record_zero_output(group)
                return
        pending.extend(rollouts)
        self._zero_output_units = self._zero_output_windows = 0

    def _record_zero_output(self, group: Group) -> None:
        self._zero_output_units += len(group.arrived) or group.accounted
        windows = self._zero_output_units // self.config.batch_size
        if windows <= self._zero_output_windows:
            return
        self._zero_output_windows = windows
        logger.warning(
            "No trainable rollouts from the last %d finalized group members (%d/%d batch equivalents)",
            self._zero_output_units,
            windows,
            MAX_CONSECUTIVE_ZERO_OUTPUT_BATCH_EQUIVALENTS,
        )
        if windows >= MAX_CONSECUTIVE_ZERO_OUTPUT_BATCH_EQUIVALENTS:
            raise RuntimeError(f"{windows} consecutive zero-output batch equivalents")

    def _maybe_batch(self, pending: list[Rollout], step: int) -> Batch | None:
        min_version = min_fresh_version(step, self.config.max_off_policy_steps)
        fresh = [rollout for rollout in pending if rollout.version_start >= min_version]
        self._dropped += len(pending) - len(fresh)
        pending[:] = fresh
        if len(pending) < self.config.batch_size:
            return None
        now = time.time()
        step_time = now - self._last_batch_at if self._last_batch_at is not None else 0.0
        self._last_batch_at = now
        cut = pending[: self.config.batch_size]
        del pending[: self.config.batch_size]
        trained = [rollout for rollout in cut if rollout.advantage != 0.0]
        arrived, self._arrived_rewards = self._arrived_rewards, []
        if not trained:
            self._consecutive_empty += 1
            logger.warning(
                "Step %d: every rollout in the batch has zero advantage (%d/%d empty batches in a row)",
                step,
                self._consecutive_empty,
                MAX_CONSECUTIVE_EMPTY_BATCHES,
            )
            if self._consecutive_empty >= MAX_CONSECUTIVE_EMPTY_BATCHES:
                raise RuntimeError(f"{self._consecutive_empty} consecutive empty train batches")
            return None
        self._consecutive_empty = 0
        metrics = rollout_metrics(cut, trained, step)
        if arrived:
            metrics["reward/arrived/mean"] = statistics.fmean(arrived)
        metrics["time/step"] = step_time
        return Batch(step, trained, metrics)

    def _ship(self, batch: Batch) -> None:
        required = batch.step - 1 - TARGET_LAG
        with self.cond:
            if self.policy_version < required:
                logger.info(
                    "Holding batch %d until v%d is published (currently v%d)", batch.step, required, self.policy_version
                )
                hold_start = time.time()
                while self.policy_version < required and self.trainer_error is None:
                    self.cond.wait(timeout=1.0)
                self._wait_for_policy += time.time() - hold_start
            self._raise_if_trainer_failed()
            batch.metrics.update(
                {
                    "time/wait_for_policy": self._wait_for_policy,
                    "off_policy/dropped": self._dropped,
                    "dispatcher/cancelled/train": self._cancelled,
                    "dispatcher/errored/train": self._errored,
                    "dispatcher/inflight/train": self._inflight,
                    "concurrency/max_inflight": int(self._inflight_cap),
                }
            )
            self._wait_for_policy = 0.0
            self._cancelled = self._dropped = self._errored = 0
        logger.info(
            "Shipping batch %d: reward %.3f, staleness mean %.2f",
            batch.step,
            batch.metrics["reward/mean"],
            batch.metrics["off_policy/mean"],
        )
        self.train_q.put(batch)
        with self.cond:
            self.collect_step += 1
            self._update_gate()

    def _update_gate(self) -> None:
        """Re-evaluate the dispatch gate; runs at each ship and after each weight sync."""
        lead = (self.collect_step - 1) - self.policy_version
        if lead > TARGET_LAG:
            if self.dispatch_allowed:
                logger.info(
                    "Pausing dispatch until v%d is published (currently v%d)",
                    self.collect_step - 1 - TARGET_LAG,
                    self.policy_version,
                )
                self._gate_closed_at = time.time()
            self.dispatch_allowed = False
        else:
            if not self.dispatch_allowed:
                logger.info("Resuming dispatch at v%d", self.policy_version)
                self._wait_for_policy += time.time() - self._gate_closed_at
            self.dispatch_allowed = True

    def _raise_if_trainer_failed(self) -> None:
        if self.trainer_error is not None:
            raise RuntimeError("the trainer thread failed") from self.trainer_error

    def _drain(self) -> None:
        request_ids: list[str] = []
        with self.cond:
            self.draining = True
            for group in list(self.groups.values()):
                self._drop_group(group, request_ids)
        self._cancel_requests(request_ids)
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._cancel_pool.shutdown(wait=False)

    def run(self) -> None:
        trainer = threading.Thread(target=self._trainer_loop, name="trainer", daemon=True)
        trainer.start()
        pending: list[Rollout] = []
        try:
            while True:
                with self.cond:
                    step = self.collect_step
                if step > self.config.max_steps:
                    break
                self._raise_if_trainer_failed()
                self.fill_inflight()
                try:
                    kind, group, payload = self.out_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if not self._handle(kind, group, payload, pending):
                    continue
                batch = self._maybe_batch(pending, step)
                if batch is not None:
                    self._ship(batch)
        finally:
            self._drain()
            self.train_q.put(None)
        trainer.join()
        self._raise_if_trainer_failed()

    # Trainer.

    def _trainer_loop(self) -> None:
        try:
            while True:
                batch = self.train_q.get()
                if batch is None:
                    return
                metrics = dict(batch.metrics)
                metrics.update(self.train_on(batch))
                # The final version is only broadcast when an eval will measure it.
                if batch.step < self.config.max_steps or eval_enabled(self.config):
                    metrics["time/weight_sync"] = self.publish(batch.step)
                self.ml_logger.log_metrics(metrics, step=batch.step)
        except BaseException as exc:
            with self.cond:
                self.trainer_error = exc
                self.cond.notify_all()

    def publish(self, version: int) -> float:
        """Advance the policy version, drop stale groups, then sync the weights into the sampler.

        The version advances before the sync, so a held batch ships during it. The dispatch
        gate is re-evaluated only once the sampler holds the new weights.
        """
        start = time.time()
        with self.cond:
            self.policy_version = version
            self.cond.notify_all()
        self.drop_stale_groups()
        sync_weights(
            self.client,
            self.job_id,
            weight_format=self.weight_format,
            bucket_size=self.config.weight_sync_bucket_size,
        )
        with self.cond:
            self._update_gate()
        logger.info("Step %d trained; sampler now holds v%d", version, version)
        return time.time() - start

    def train_on(self, batch: Batch) -> dict[str, float]:
        rollouts = batch.rollouts
        sequences = [
            sequence_from_rollout(rollout.prompt_tokens, rollout.result["token_ids"], advantage=rollout.advantage)
            for rollout in rollouts
        ]
        kwargs, context = collate(sequences, pad_token_id=self.pad_token_id, max_seq_len=self.max_seq_len)
        inference_logprobs = sampled_logprobs_shifted(
            [len(rollout.prompt_tokens) for rollout in rollouts],
            [rollout.result for rollout in rollouts],
            width=context["loss_mask"].shape[1],
        )
        train_start = time.time()
        fwd_bwd_result, step_result = forward_backward_step(
            self.client,
            self.job_id,
            kwargs,
            context=loss_context(context, inference_logprobs, self.config),
            learning_rate=self.learning_rate,
            processing=loss_processing(
                self.config, batch_num_tokens=int(context["loss_mask"].sum()), dp_size=self.dp_size
            ),
        )
        metrics = {
            "loss/mean": float(fwd_bwd_result.get("avg_loss", float("nan"))),
            "optim/lr": self.learning_rate,
            "time/train": time.time() - train_start,
            **server_loss_metrics(fwd_bwd_result.get("metrics") or {}, self.config.loss),
        }
        if step_result.get("grad_norm") is not None:
            metrics["optim/grad_norm"] = float(step_result["grad_norm"])
        return metrics


def evaluate(
    client: Any,
    job_id: str,
    tasks: list[ReverseTextTask],
    prompt_tokens: Any,
    *,
    rollouts: int,
    sampling_params: dict,
) -> dict[str, float]:
    expanded = [task for task in tasks for _ in range(rollouts)]
    request_id = client.generate(job_id, prompts=[prompt_tokens(task) for task in expanded], sampling_params=sampling_params)
    results = client.poll_request(job_id, request_id)["results"]
    rewards = [lcs_reward(result.get("text") or "", task.answer) for result, task in zip(results, expanded)]
    return {"eval/reward/mean": statistics.fmean(rewards), "eval/rollouts": len(rewards)}


def main(config: Config) -> None:
    # setup_logging replaces these handlers once the job runs; until then this
    # keeps the created job id in the log, so an interrupted start can be cancelled.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    validate_config(config)
    body = job_body(config)
    subs = {sub.get("job_type"): sub for sub in body.get("sub_job_configs") or ()}
    training_sub = subs.get("training")
    sampling_sub = subs.get("sampling")
    if training_sub is None or sampling_sub is None:
        raise ValueError(f"{config.job_config} needs a training and a sampling sub-job")
    training = training_sub.get("training_config") or {}
    learning_rate = float(training["optimizer"]["lr"])
    max_seq_len = int(training["max_seq_len"])
    weight_format = "lora" if (training.get("peft_config") or {}).get("r") else None

    tokenizer, renderer, renderer_name = build_renderer(training_sub["model_name"], renderer_name=config.renderer_name)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    logger.info("Using renderer: %s", renderer_name)
    tasks = load_tasks(config.dataset, config.dataset_split)
    prompt_cache: dict[int, list[int]] = {}

    def prompt_tokens(task: ReverseTextTask) -> list[int]:
        if task.idx not in prompt_cache:
            prompt_cache[task.idx] = renderer.build_generation_prompt(task_messages(task)).to_ints()
        return prompt_cache[task.idx]

    stop_params = stop_params_for(renderer.get_stop_sequences())
    sampling_params = dict(
        max_tokens=config.max_completion_tokens,
        temperature=config.temperature,
        top_p=1.0,
        logprobs=0,
        return_sampled_logprobs_only=True,
        **stop_params,
    )
    eval_tasks = tasks[: config.eval_examples]
    for task in eval_tasks:
        prompt_tokens(task)
    eval_params = dict(max_tokens=config.eval_max_tokens, temperature=config.temperature, top_p=1.0, **stop_params)

    client = make_client(config.config, config.connection)
    with running_job(client, body, job_id=config.job_id) as job_id:
        ml_logger = setup_logging(config, client=client, job_id=job_id)
        if eval_enabled(config):
            ml_logger.log_metrics(
                evaluate(client, job_id, eval_tasks, prompt_tokens, rollouts=config.eval_rollouts, sampling_params=eval_params),
                step=0,
            )
        Pipeline(
            config,
            client=client,
            rollout_client=make_client(config.config, config.connection),
            job_id=job_id,
            tasks=tasks,
            prompt_tokens=prompt_tokens,
            sampling_params=sampling_params,
            pad_token_id=pad_token_id,
            max_seq_len=max_seq_len,
            dp_size=data_parallel_size(training),
            learning_rate=learning_rate,
            weight_format=weight_format,
            ml_logger=ml_logger,
        ).run()
        if eval_enabled(config):
            ml_logger.log_metrics(
                evaluate(client, job_id, eval_tasks, prompt_tokens, rollouts=config.eval_rollouts, sampling_params=eval_params),
                step=config.max_steps,
            )
        saved = save_example_checkpoints(client, job_id)
        log_saved_checkpoints(
            connection_args=example_connection_args(config.config, config.connection),
            job_id=job_id,
            saved=saved,
            job_args=inference_job_args(body, context=config.context if config.job_config is None else None),
        )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
