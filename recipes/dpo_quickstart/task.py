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

"""Preference data, the DPO loss, and the log-probability passes the DPO example runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Sequence

import torch
import torch.nn.functional as F
from cortex_training.client import custom_loss
from recipes.utils import sequence_from_conversation
from tinker_cookbook import renderers

_DATA_DIR = Path(__file__).resolve().parent / "data"
_ULTRAFEEDBACK = "HuggingFaceH4/ultrafeedback_binarized"
_ULTRAFEEDBACK_SPLITS = {"train": "train_prefs", "test": "test_prefs"}
DATASETS = ("smoke", "ultrafeedback")


# The smoke file's last pairs are its held-out split.
_SMOKE_TEST_PAIRS = 4


def load_pairs(dataset: str, split: str, n: int) -> list[tuple[list[dict], list[dict]]]:
    """Up to ``n`` ``(chosen, rejected)`` conversations from ``split`` (``train`` or ``test``).

    Each conversation is a list of chat messages that ends with the assistant's
    answer; ``chosen`` and ``rejected`` share the same prompt.
    """
    if split not in _ULTRAFEEDBACK_SPLITS:
        raise ValueError(f"split must be one of {sorted(_ULTRAFEEDBACK_SPLITS)}, got {split!r}")
    if dataset == "smoke":
        with open(_DATA_DIR / "smoke.jsonl") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rows = rows[:-_SMOKE_TEST_PAIRS] if split == "train" else rows[-_SMOKE_TEST_PAIRS:]
    elif dataset == "ultrafeedback":
        import datasets

        rows = datasets.load_dataset(_ULTRAFEEDBACK, split=f"{_ULTRAFEEDBACK_SPLITS[split]}[:{n}]")
    else:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    return [(list(row["chosen"]), list(row["rejected"])) for row in list(rows)[:n]]


def pair_items(
    chosen: Sequence[dict],
    rejected: Sequence[dict],
    renderer: Any,
    *,
    max_seq_len: int,
) -> list[dict[str, Any]] | None:
    """Two ``forward_backward_custom`` items for one pair, chosen first, or ``None`` if either is too long.

    Each item's ``loss_mask`` selects the log-probabilities of the final
    assistant answer only, and has one entry per log-probability:
    ``loss_mask[j]`` selects the log-probability of token ``j + 1``.
    """
    items = []
    for messages, is_chosen in ((chosen, True), (rejected, False)):
        sequence = sequence_from_conversation(
            messages,
            renderer,
            train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )
        if len(sequence.input_ids) > max_seq_len:
            return None
        items.append({"input_ids": sequence.input_ids, "loss_mask": sequence.loss_mask[:-1], "chosen": is_chosen})
    return items


def sequence_scores(data: Sequence[dict], logprobs: Sequence[torch.Tensor]) -> torch.Tensor:
    """Each item's summed log-probability over its answer tokens."""
    if len(data) != len(logprobs):
        raise ValueError(f"{len(data)} items but {len(logprobs)} logprob rows")
    scores = []
    for index, (item, item_logprobs) in enumerate(zip(data, logprobs)):
        mask = torch.as_tensor(item["loss_mask"]) > 0
        if mask.numel() != item_logprobs.numel():
            raise ValueError(f"item {index}: loss_mask has {mask.numel()} entries for {item_logprobs.numel()} logprobs")
        if not mask.any():
            raise ValueError(f"item {index} has no answer token; check loss_mask")
        scores.append(item_logprobs[mask].sum())
    return torch.stack(scores)


def _check_pairs(data: Sequence[dict]) -> None:
    if len(data) == 0 or len(data) % 2:
        raise ValueError(f"DPO data must be chosen/rejected pairs; got {len(data)} items")
    for index, item in enumerate(data):
        if item.get("chosen") is not (index % 2 == 0):
            raise ValueError(f"item {index} must be {'chosen' if index % 2 == 0 else 'rejected'}: items alternate")
        if "ref_logp" not in item:
            raise ValueError(f"item {index} has no ref_logp; score the reference first")


def dpo_loss(beta: float):
    """A ``loss_fn`` for ``forward_backward_custom``: the DPO loss over adjacent chosen/rejected items.

    For each pair, with ``s`` the policy's and ``r`` the reference's summed
    answer log-probability, ``margin = (s_chosen - r_chosen) - (s_rejected - r_rejected)``
    and the pair's loss is ``-log sigmoid(beta * margin)``. ``L`` is the mean
    over pairs. When the policy equals the reference every margin is 0, so ``L``
    starts at ``log 2``.
    """
    if not beta > 0:
        raise ValueError(f"beta must be positive, got {beta}")

    def loss_fn(data: list[dict[str, Any]], logprobs: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        _check_pairs(data)
        scores = sequence_scores(data, logprobs)
        references = torch.tensor([float(item["ref_logp"]) for item in data], dtype=scores.dtype)
        rewards = beta * (scores - references)
        chosen_rewards, rejected_rewards = rewards[0::2], rewards[1::2]
        margins = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(margins).mean()
        detached = margins.detach()
        return loss, {
            "dpo_loss": float(loss.detach()),
            "reward_accuracy": float((detached > 0).float().mean()),
            "reward_margin": float(detached.mean()),
            "chosen_reward": float(chosen_rewards.detach().mean()),
            "rejected_reward": float(rejected_rewards.detach().mean()),
        }

    return loss_fn


def forward_logprobs(client: Any, job_id: str, data: Sequence[dict], *, batch_items: int) -> list[torch.Tensor]:
    """Per-item log-probabilities from no-gradient forward passes on the training sub-job.

    Before the first update they are the frozen reference model's. Later they are
    the current policy's, which is how the example evaluates.
    """
    # A request needs at least as many rows as the job has data-parallel ranks, so
    # a short remainder joins the chunk before it instead of going out alone.
    starts = list(range(0, len(data), batch_items))
    if len(starts) > 1 and len(data) - starts[-1] < batch_items:
        starts.pop()
    bounds = list(zip(starts, starts[1:] + [len(data)]))
    out: list[torch.Tensor] = []
    for start, end in bounds:
        # The same no-gradient forward pass forward_backward_custom runs before its backward.
        out.extend(client._forward_logprobs(job_id, custom_loss.item_tokens(data[start:end])))
    return out


def score_reference(client: Any, job_id: str, data: Sequence[dict], *, batch_items: int) -> None:
    """Set each item's ``ref_logp``, its summed answer log-probability under the current weights."""
    scores = sequence_scores(data, forward_logprobs(client, job_id, data, batch_items=batch_items))
    for item, score in zip(data, scores.tolist()):
        item["ref_logp"] = score


def evaluate(client: Any, job_id: str, data: Sequence[dict], loss_fn, *, batch_items: int) -> dict[str, float]:
    """The DPO loss and reward metrics of the current policy on ``data``, prefixed ``eval_``."""
    logprobs = forward_logprobs(client, job_id, data, batch_items=batch_items)
    with torch.no_grad():
        _, metrics = loss_fn(list(data), logprobs)
    return {f"eval_{key}": value for key, value in metrics.items()}
