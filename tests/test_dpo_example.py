"""The DPO example: its loss, the items it builds, and its reference and evaluation passes."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
for _dependency in ("chz", "datasets", "tinker_cookbook"):
    pytest.importorskip(_dependency)

from recipes.dpo_quickstart import task  # noqa: E402
from recipes.dpo_quickstart import train  # noqa: E402
from recipes.sft_quickstart.runner import job_body  # noqa: E402
from recipes.utils import make_config  # noqa: E402


def _pair_batch(ref=(-3.0, -5.0, -2.0, -2.0)):
    """Two pairs; each item has three logprobs, the last two of which are answer tokens."""
    data = [
        {"input_ids": [1, 2, 3, 4], "loss_mask": [0.0, 1.0, 1.0], "chosen": True, "ref_logp": ref[0]},
        {"input_ids": [1, 2, 5, 6], "loss_mask": [0.0, 1.0, 1.0], "chosen": False, "ref_logp": ref[1]},
        {"input_ids": [7, 8, 9, 1], "loss_mask": [0.0, 1.0, 1.0], "chosen": True, "ref_logp": ref[2]},
        {"input_ids": [7, 8, 2, 3], "loss_mask": [0.0, 1.0, 1.0], "chosen": False, "ref_logp": ref[3]},
    ]
    logprobs = [
        torch.tensor([-9.0, -1.0, -2.0], requires_grad=True),  # answer score -3
        torch.tensor([-9.0, -2.0, -3.0], requires_grad=True),  # answer score -5
        torch.tensor([-9.0, -0.5, -0.5], requires_grad=True),  # answer score -1
        torch.tensor([-9.0, -1.5, -1.5], requires_grad=True),  # answer score -3
    ]
    return data, logprobs


def test_loss_is_log_2_when_the_policy_is_the_reference():
    data, logprobs = _pair_batch(ref=(-3.0, -5.0, -1.0, -3.0))

    loss, metrics = task.dpo_loss(0.1)(data, logprobs)

    assert loss.item() == pytest.approx(math.log(2))
    assert metrics["reward_margin"] == pytest.approx(0.0)
    assert metrics["reward_accuracy"] == 0.0


def test_loss_matches_the_dpo_formula_and_pushes_chosen_up():
    data, logprobs = _pair_batch()
    beta = 0.5

    loss, metrics = task.dpo_loss(beta)(data, logprobs)
    loss.backward()

    # Pair 1: (-3 - -3) - (-5 - -5) = 0. Pair 2: (-1 - -2) - (-3 - -2) = 2.
    margins = torch.tensor([0.0, 2.0])
    expected = -torch.nn.functional.logsigmoid(beta * margins).mean()
    assert loss.item() == pytest.approx(expected.item())
    assert metrics["reward_accuracy"] == pytest.approx(0.5)
    assert metrics["reward_margin"] == pytest.approx(float((beta * margins).mean()))
    # Prompt tokens get no gradient; chosen answers are pushed up, rejected ones down.
    for index, grad in enumerate(lp.grad for lp in logprobs):
        assert grad[0] == 0.0
        assert (grad[1:] < 0).all() if index % 2 == 0 else (grad[1:] > 0).all()
    # Each answer token's gradient is -/+ beta * sigmoid(-beta * margin) / pairs.
    weight = beta * torch.sigmoid(-beta * margins[1]) / 2
    torch.testing.assert_close(logprobs[2].grad[1:], -weight.repeat(2))


def test_malformed_pairs_fail_before_training():
    data, logprobs = _pair_batch()
    loss_fn = task.dpo_loss(0.1)

    with pytest.raises(ValueError, match="pairs"):
        loss_fn(data[:3], logprobs[:3])
    with pytest.raises(ValueError, match="items alternate"):
        loss_fn([data[1], data[0]], [logprobs[1], logprobs[0]])
    with pytest.raises(ValueError, match="score the reference first"):
        loss_fn([{k: v for k, v in item.items() if k != "ref_logp"} for item in data], logprobs)
    with pytest.raises(ValueError, match="no answer token"):
        loss_fn([{**data[0], "loss_mask": [0.0, 0.0, 0.0]}, data[1]], logprobs[:2])
    with pytest.raises(ValueError, match="beta must be positive"):
        task.dpo_loss(0.0)


def test_smoke_pairs_split_into_train_and_held_out():
    train_pairs = task.load_pairs("smoke", "train", 100)
    test_pairs = task.load_pairs("smoke", "test", 100)

    assert len(train_pairs) == 12 and len(test_pairs) == 4
    assert {pair[0][0]["content"] for pair in train_pairs}.isdisjoint(pair[0][0]["content"] for pair in test_pairs)
    for chosen, rejected in train_pairs + test_pairs:
        assert chosen[:-1] == rejected[:-1] and chosen[-1]["role"] == "assistant"


class _Renderer:
    """Tokens are message lengths; only the last assistant message has weight."""

    def build_supervised_example(self, messages, train_on_what):
        tokens, weights = [], []
        for index, message in enumerate(messages):
            answer = index == len(messages) - 1
            tokens += list(range(len(message["content"])))
            weights += [1.0 if answer else 0.0] * len(message["content"])
        return _Ints(tokens), torch.tensor(weights)


class _Ints:
    def __init__(self, values):
        self.values = values

    def to_ints(self):
        return self.values


def test_pair_items_put_chosen_first_and_mask_only_the_answer():
    user = {"role": "user", "content": "abc"}
    chosen = [user, {"role": "assistant", "content": "yes"}]
    rejected = [user, {"role": "assistant", "content": "no"}]

    items = task.pair_items(chosen, rejected, _Renderer(), max_seq_len=100)

    assert [item["chosen"] for item in items] == [True, False]
    # loss_mask has one entry per logprob and selects the predictions of the answer's tokens.
    assert items[0]["loss_mask"] == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert items[1]["loss_mask"] == [0.0, 0.0, 1.0, 1.0]
    assert task.pair_items(chosen, rejected, _Renderer(), max_seq_len=5) is None


class _Client:
    """Returns logprob -1 for every predicted token and records each forward's row count."""

    def __init__(self):
        self.rows = []

    def forward(self, job_id, payload, *, sub_job_type):
        assert sub_job_type == "training"
        from cortex_training import wire

        input_ids = wire.loads(payload)["kwargs"]["input_ids"]
        self.rows.append(input_ids.shape[0])
        self._last = input_ids
        return {"request_id": "r"}

    def poll_request(self, job_id, request_id):
        return {"logprobs": torch.full((self._last.shape[0], self._last.shape[1]), -1.0)}


def test_reference_scores_are_answer_sums_and_short_remainders_join_the_last_chunk():
    data, _ = _pair_batch()
    data = [{k: v for k, v in item.items() if k != "ref_logp"} for item in data + data[:2]]
    client = _Client()

    task.score_reference(client, "job", data, batch_items=4)

    assert client.rows == [6]  # 6 items with chunks of 4: the 2-item remainder is not sent alone
    assert [item["ref_logp"] for item in data] == [-2.0] * 6


def test_evaluation_reports_the_loss_at_the_reference_as_log_2():
    data, _ = _pair_batch(ref=(-2.0, -2.0, -2.0, -2.0))

    metrics = task.evaluate(_Client(), "job", data, task.dpo_loss(0.1), batch_items=4)

    assert metrics["eval_dpo_loss"] == pytest.approx(math.log(2))
    assert set(metrics) == {
        "eval_dpo_loss",
        "eval_reward_accuracy",
        "eval_reward_margin",
        "eval_chosen_reward",
        "eval_rejected_reward",
    }


def test_the_job_is_the_lora_sft_job_with_the_dpo_learning_rate():
    body = job_body(train.Config(config="unused"))
    (training,) = [sub for sub in body["sub_job_configs"] if sub["job_type"] == "training"]
    (expected,) = [
        sub for sub in make_config("Qwen/Qwen3-8B", "standard", lora=True)["sub_job_configs"]
        if sub["job_type"] == "training"
    ]

    assert training["training_config"]["optimizer"]["lr"] == 1e-5
    training["training_config"]["optimizer"]["lr"] = expected["training_config"]["optimizer"]["lr"]
    assert training == expected
