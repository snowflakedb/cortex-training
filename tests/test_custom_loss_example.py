"""The custom-loss SFT example: its focal loss, the items it reads, and its job."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
for _dependency in ("chz", "datasets", "tinker_cookbook"):
    pytest.importorskip(_dependency)

from examples.sft.custom_loss import task  # noqa: E402
from examples.sft.custom_loss import train  # noqa: E402
from examples.sft.runner import job_body  # noqa: E402
from examples.utils import TrainSequence  # noqa: E402
from examples.utils import make_config  # noqa: E402


def _batch():
    data = [
        {"input_ids": [1, 2, 3, 4], "loss_mask": [0.0, 1.0, 1.0]},
        {"input_ids": [5, 6, 7], "loss_mask": [1.0, 0.0]},
    ]
    logprobs = [
        torch.tensor([-0.5, -1.0, -2.0], requires_grad=True),
        torch.tensor([-0.25, float("-inf")], requires_grad=True),
    ]
    return data, logprobs


def test_gamma_zero_is_the_per_sequence_mean_nll():
    data, logprobs = _batch()

    loss, metrics = task.focal_loss(0.0)(data, logprobs)
    loss.backward()

    # Sequence 0 averages its two selected tokens, sequence 1 has one; L averages the two sequences.
    assert loss.item() == pytest.approx(((1.0 + 2.0) / 2 + 0.25) / 2)
    torch.testing.assert_close(logprobs[0].grad, torch.tensor([0.0, -0.25, -0.25]))
    torch.testing.assert_close(logprobs[1].grad, torch.tensor([-0.5, 0.0]))
    assert metrics == {"train_nll": pytest.approx((1.0 + 2.0 + 0.25) / 3), "loss_weight_sum": 3.0}


def test_focal_loss_counts_well_predicted_tokens_less():
    data = [{"input_ids": [1, 2, 3], "loss_mask": [1.0, 1.0]}]
    logprobs = [torch.tensor([-0.01, -3.0], requires_grad=True)]

    loss, _ = task.focal_loss(2.0)(data, logprobs)
    loss.backward()

    p = torch.tensor([-0.01, -3.0]).exp()
    assert loss.item() == pytest.approx(float((-((1 - p) ** 2) * torch.tensor([-0.01, -3.0])).mean()), rel=1e-6)
    # The confident token's gradient is a small fraction of the uncertain one's.
    assert abs(logprobs[0].grad[0]) < 0.001 * abs(logprobs[0].grad[1])


def test_loss_item_aligns_the_mask_with_the_logprobs():
    sequence = TrainSequence(input_ids=[1, 2, 3, 4], loss_mask=[0.0, 1.0, 1.0, 0.0])

    assert task.loss_item(sequence) == {"input_ids": [1, 2, 3, 4], "loss_mask": [0.0, 1.0, 1.0]}


def test_invalid_settings_and_empty_batches_fail_clearly():
    with pytest.raises(ValueError, match="focal_gamma must be at least 0"):
        task.focal_loss(-1.0)
    with pytest.raises(ValueError, match="no item has a token to train on"):
        task.focal_loss(2.0)([{"input_ids": [1, 2], "loss_mask": [0.0]}], [torch.tensor([-1.0], requires_grad=True)])


def test_the_job_is_the_sft_job():
    assert job_body(train.Config(config="unused")) == make_config("Qwen/Qwen3-8B", "standard")
