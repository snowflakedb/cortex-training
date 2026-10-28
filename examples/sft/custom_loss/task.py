"""The custom loss this example trains with, and the items it reads."""

from __future__ import annotations

from typing import Any

import torch
from examples.utils import TrainSequence


def loss_item(sequence: TrainSequence) -> dict[str, Any]:
    """A ``forward_backward_custom`` item: the tokens, and a mask aligned with their log-probabilities.

    ``loss_mask[j]`` selects the log-probability of token ``j + 1``, so it has one
    entry per log-probability.
    """
    return {"input_ids": sequence.input_ids, "loss_mask": sequence.loss_mask[:-1]}


def focal_loss(gamma: float):
    """A ``loss_fn`` for ``forward_backward_custom``: the focal loss, averaged per sequence.

    A token the mask selects, which the model gives probability ``p``, contributes
    ``-(1 - p) ** gamma * log p``, so tokens the model already predicts well count
    less. Each sequence's loss is the mean over its selected tokens, and ``L`` is
    the mean over sequences. ``gamma=0`` is the per-sequence mean negative
    log-likelihood. The metrics report the token-mean negative log-likelihood as
    ``train_nll``, as the SFT example does.
    """
    if gamma < 0:
        raise ValueError(f"focal_gamma must be at least 0, got {gamma}")

    def loss_fn(data: list[dict[str, Any]], logprobs: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        sequence_losses = []
        nll_sum = 0.0
        token_count = 0
        for item, item_logprobs in zip(data, logprobs):
            selected = item_logprobs[torch.tensor(item["loss_mask"]) > 0]
            if selected.numel() == 0:
                continue
            # 1 - p, computed as -expm1(log p) to stay accurate when p is close to 1.
            sequence_losses.append((-((-torch.expm1(selected)) ** gamma) * selected).mean())
            nll_sum -= float(selected.detach().sum())
            token_count += selected.numel()
        if not sequence_losses:
            raise ValueError("no item has a token to train on; check loss_mask")
        loss = torch.stack(sequence_losses).mean()
        return loss, {"train_nll": nll_sum / token_count, "loss_weight_sum": float(token_count)}

    return loss_fn
