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

"""Loss functions computed in the caller's process.

``CortexTrainingClient.forward_backward_custom`` trains with a custom loss
``L`` written as ordinary PyTorch code over per-token log-probabilities
``logprobs``:

1. A no-gradient forward pass returns each item's ``logprobs``.
2. The caller's ``loss_fn(data, logprobs)`` returns ``L`` as a scalar, and
   autograd gives ``g = dL/dlogprobs`` locally.
3. A forward/backward request with the server loss ``weighted_cross_entropy``
   and per-token weights ``w = -g`` backpropagates ``g`` through the model.

``weighted_cross_entropy`` trains ``L_surrogate = -sum(w * logprobs)``, so its
gradient with respect to each log-probability is ``-w = g``, and the model
gradient is that of ``L``.

The functions here build and read the frames; the client methods send them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from numbers import Integral
from numbers import Real
from typing import TYPE_CHECKING
from typing import Any

from cortex_training import wire

if TYPE_CHECKING:
    import torch

__all__ = [
    "CustomLossFn",
    "apply_custom_loss",
    "build_custom_loss_backward_payload",
    "build_forward_logprobs_payload",
    "item_tokens",
    "logprobs_from_backward_result",
    "logprobs_from_forward_result",
    "unsupported_server_error",
]

CustomLossFn = Callable[[Any, list["torch.Tensor"]], tuple["torch.Tensor", Mapping[str, float]]]
"""``loss_fn(data, logprobs) -> (loss, metrics)``.

``data`` is the list passed to ``forward_backward_custom``, unchanged.
``logprobs[i]`` is a float32 tensor of length ``len(input_ids) - 1`` for item
``i`` with ``requires_grad=True``; ``logprobs[i][j]`` is the log-probability of
``input_ids[j + 1]`` given ``input_ids[: j + 1]``. ``loss`` is the custom loss
``L``, a scalar tensor computed from them, and ``metrics`` maps names to numbers.
"""

_BACKWARD_PROCESSING = {"loss_fn": "weighted_cross_entropy", "post": ["compute_logprobs"]}
# Text in the failure of a server whose loss registry has no weighted_cross_entropy.
_UNSUPPORTED_SERVER_MARKER = "is not a registered"

_RESPONSE_OPTIONS = {"response_options": {"format": "dssst1", "delivery": "chunked"}}
_IGNORE_INDEX = -100


def _token_ids(value: Any, where: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{where} must be a sequence of token ids, got {type(value).__name__}")
    if len(value) < 2:
        raise ValueError(f"{where} has {len(value)} token(s); an item needs at least 2")
    tokens = []
    for token in value:
        if isinstance(token, bool) or not isinstance(token, Integral) or token < 0:
            raise TypeError(f"{where} must hold non-negative integer token ids, got {token!r}")
        tokens.append(int(token))
    return tokens


def item_tokens(data: Sequence[Any]) -> list[list[int]]:
    """Each item's token ids. An item is a token-id sequence or a mapping with ``"input_ids"``."""
    if isinstance(data, (str, bytes, Mapping)) or not isinstance(data, Sequence):
        raise TypeError("data must be a list of items: token-id sequences or mappings with 'input_ids'")
    if len(data) == 0:
        raise ValueError("data must not be empty")
    tokens: list[list[int]] = []
    for index, item in enumerate(data):
        if isinstance(item, Mapping):
            if "input_ids" not in item:
                raise ValueError(f"data[{index}] is a mapping without 'input_ids'")
            tokens.append(_token_ids(item["input_ids"], f"data[{index}]['input_ids']"))
        else:
            tokens.append(_token_ids(item, f"data[{index}]"))
    return tokens


def _model_kwargs(tokens: list[list[int]]) -> dict[str, Any]:
    """Right-padded model inputs: valid tokens in the leading columns, as the server expects."""
    import torch

    width = max(len(row) for row in tokens)
    input_ids = torch.zeros((len(tokens), width), dtype=torch.long)
    attention_mask = torch.zeros((len(tokens), width), dtype=torch.long)
    for index, row in enumerate(tokens):
        input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention_mask[index, : len(row)] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": torch.arange(width, dtype=torch.long).expand(len(tokens), -1).contiguous(),
        "use_cache": False,
    }


def _prediction_mask(tokens: list[list[int]], width: int) -> "torch.Tensor":
    """``mask[i, j] = 1`` where position ``j`` of row ``i`` predicts a real next token."""
    import torch

    mask = torch.zeros((len(tokens), width), dtype=torch.float32)
    for index, row in enumerate(tokens):
        mask[index, : len(row) - 1] = 1.0
    return mask


def build_forward_logprobs_payload(tokens: list[list[int]]) -> bytes:
    """DSSST1 frame for a no-gradient forward pass over ``tokens``.

    The server gathers ``logprobs[i, j]`` at ``labels[i, j]`` as sent, so labels
    carry the next token and ``-100`` past each row's last prediction.
    """
    kwargs = _model_kwargs(tokens)
    predicts = _prediction_mask(tokens, kwargs["input_ids"].shape[1]) > 0
    kwargs["labels"] = kwargs["input_ids"].roll(-1, dims=-1).masked_fill(~predicts, _IGNORE_INDEX)
    return wire.dumps({"args": (), "kwargs": kwargs}, metadata=_RESPONSE_OPTIONS)


def _per_item(values: Any, tokens: list[list[int]], *, where: str) -> list["torch.Tensor"]:
    import torch

    if values is None:
        raise RuntimeError(f"{where} returned no per-token logprobs")
    rows = len(tokens)
    if torch.is_tensor(values):
        if values.ndim != 2 or values.shape[0] != rows:
            raise RuntimeError(f"{where} returned logprobs of shape {tuple(values.shape)} for {rows} item(s)")
        values = list(values.unbind(0))
    elif not isinstance(values, Sequence) or len(values) != rows:
        raise RuntimeError(f"{where} returned logprobs for a different number of items than {rows}")
    out = []
    for index, (row, item) in enumerate(zip(values, tokens)):
        row = torch.as_tensor(row, dtype=torch.float32).reshape(-1)
        needed = len(item) - 1
        if row.numel() < needed:
            raise RuntimeError(f"{where} returned {row.numel()} logprobs for item {index}, which needs {needed}")
        row = row[:needed].clone()
        if not torch.isfinite(row).all():
            raise RuntimeError(f"{where} returned a non-finite logprob for item {index}")
        out.append(row)
    return out


def logprobs_from_forward_result(result: Mapping[str, Any], tokens: list[list[int]]) -> list["torch.Tensor"]:
    """Per-item logprobs (length ``len(input_ids) - 1``) from a polled forward result."""
    return _per_item(result.get("logprobs"), tokens, where="forward")


def logprobs_from_backward_result(result: Mapping[str, Any], tokens: list[list[int]]) -> list["torch.Tensor"] | None:
    """The backward pass's own per-item logprobs, or ``None`` when the result has none."""
    post = result.get("post_process_outputs")
    if not isinstance(post, Mapping) or post.get("logprobs") is None:
        return None
    return _per_item(post["logprobs"], tokens, where="forward-backward")


def apply_custom_loss(
    data: Any,
    logprobs: Sequence["torch.Tensor"],
    loss_fn: CustomLossFn,
) -> tuple[float, dict[str, float], list["torch.Tensor"]]:
    """Run ``loss_fn(data, logprobs)`` on detached copies; return ``(L, metrics, g)`` with ``g = dL/dlogprobs``.

    An item the loss does not depend on gets a zero gradient. A loss that
    depends on none of them, or a non-finite loss or gradient, raises before
    anything reaches the server.
    """
    import torch

    leaves = [torch.as_tensor(row, dtype=torch.float32).detach().clone().requires_grad_(True) for row in logprobs]
    returned = loss_fn(data, leaves)
    if not (isinstance(returned, tuple) and len(returned) == 2):
        raise TypeError("loss_fn must return a (loss, metrics) tuple")
    loss, metrics = returned
    if not torch.is_tensor(loss) or loss.numel() != 1:
        raise TypeError("loss_fn must return a scalar loss tensor")
    if not loss.requires_grad:
        raise ValueError("loss_fn's loss does not depend on the logprobs it was given")
    if not isinstance(metrics, Mapping):
        raise TypeError("loss_fn must return its metrics as a mapping of names to numbers")
    clean_metrics: dict[str, float] = {}
    for name, value in metrics.items():
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise TypeError(f"loss_fn metric {name!r} must be a number, got a tensor of shape {tuple(value.shape)}")
            value = value.detach().item()
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"loss_fn metric {name!r} must be a number, got {type(value).__name__}")
        clean_metrics[str(name)] = float(value)
    loss_value = float(loss.detach().reshape(()))
    if not math.isfinite(loss_value):
        raise ValueError(f"loss_fn returned a non-finite loss: {loss_value}")
    grads = torch.autograd.grad(loss.reshape(()), leaves, allow_unused=True)
    if all(grad is None for grad in grads):
        raise ValueError("loss_fn's loss does not depend on the logprobs it was given")
    out = []
    for index, (leaf, grad) in enumerate(zip(leaves, grads)):
        grad = torch.zeros_like(leaf) if grad is None else grad.detach().to(torch.float32)
        if not torch.isfinite(grad).all():
            raise ValueError(f"loss_fn's gradient is non-finite for item {index}")
        out.append(grad)
    return loss_value, clean_metrics, out


def build_custom_loss_backward_payload(tokens: list[list[int]], grads: Sequence["torch.Tensor"]) -> bytes:
    """DSSST1 forward/backward frame that backpropagates ``grads``, the ``g = dL/dlogprobs`` of each item.

    ``grads[i]`` has one value per prediction of item ``i``. The frame names
    ``weighted_cross_entropy`` with ``logprob_weights_shifted`` set to
    ``w = -g``, zero outside ``loss_mask``.
    """
    import torch

    if len(grads) != len(tokens):
        raise ValueError(f"got {len(grads)} gradient tensors for {len(tokens)} items")
    kwargs = _model_kwargs(tokens)
    loss_mask = _prediction_mask(tokens, kwargs["input_ids"].shape[1])
    weights = torch.zeros_like(loss_mask)
    for index, (row, grad) in enumerate(zip(tokens, grads)):
        grad = torch.as_tensor(grad, dtype=torch.float32).reshape(-1)
        if grad.numel() != len(row) - 1:
            raise ValueError(f"grads[{index}] has {grad.numel()} values; item {index} has {len(row) - 1}")
        weights[index, : len(row) - 1] = -grad
    frame = {
        "args": (),
        "kwargs": kwargs,
        "context": {"input_ids": kwargs["input_ids"], "loss_mask": loss_mask, "logprob_weights_shifted": weights},
        "processing": {"loss_fn": _BACKWARD_PROCESSING["loss_fn"], "post": list(_BACKWARD_PROCESSING["post"])},
    }
    return wire.dumps(frame, metadata=_RESPONSE_OPTIONS)


def unsupported_server_error(exc: BaseException) -> RuntimeError | None:
    """The error to raise instead of ``exc`` when the server has no ``weighted_cross_entropy``, else ``None``."""
    if _UNSUPPORTED_SERVER_MARKER not in str(exc):
        return None
    return RuntimeError(
        "This Cortex Training service does not support custom losses yet: it has no weighted_cross_entropy "
        f"loss for forward_backward_custom to train with. Server error: {exc}"
    )
