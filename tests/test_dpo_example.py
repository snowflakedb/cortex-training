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


@pytest.mark.parametrize("beta", [0.01, 0.1, 0.5])
def test_loss_metrics_and_gradients_match_tinker_cookbooks_dpo_loss(beta):
    """Same inputs to our loss and to tinker-cookbook's ``compute_dpo_loss``: same loss, metrics and gradients."""
    from tinker_cookbook.preference.train_dpo import compute_dpo_loss

    generator = torch.Generator().manual_seed(0)
    pairs, length = 8, 6
    mask = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])  # two prompt predictions, then the answer
    references = (-12 * torch.rand(2 * pairs, generator=generator)).tolist()
    data = [
        {"input_ids": list(range(length + 1)), "loss_mask": mask.tolist(), "chosen": i % 2 == 0, "ref_logp": ref}
        for i, ref in enumerate(references)
    ]
    rows = -3 * torch.rand(2 * pairs, length, generator=generator)
    ours_leaves = [row.clone().requires_grad_(True) for row in rows]
    tinker_leaves = [row.clone().requires_grad_(True) for row in rows]

    loss, metrics = task.dpo_loss(beta)(data, ours_leaves)
    scores = [(leaf * mask).sum() for leaf in tinker_leaves]
    refs = [torch.tensor(ref) for ref in references]
    tinker_loss, tinker_metrics = compute_dpo_loss(scores[0::2], scores[1::2], refs[0::2], refs[1::2], beta)

    torch.testing.assert_close(loss, tinker_loss)
    names = {"dpo_loss": "dpo_loss", "reward_accuracy": "accuracy", "reward_margin": "margin"}
    names.update(chosen_reward="chosen_reward", rejected_reward="rejected_reward")
    for ours, theirs in names.items():
        assert metrics[ours] == pytest.approx(tinker_metrics[theirs], rel=1e-5, abs=1e-7)
    for ours, theirs in zip(torch.autograd.grad(loss, ours_leaves), torch.autograd.grad(tinker_loss, tinker_leaves)):
        torch.testing.assert_close(ours, theirs)


def _dpo_items_for_the_bigram_trainer():
    """Two pairs of token ids the custom-loss tests' bigram trainer can score; each prompt is two tokens."""
    rows = [
        ([1, 4, 2, 7, 3], True, -6.0),
        ([1, 4, 9, 6], False, -4.0),
        ([8, 2, 10, 4], True, -5.0),
        ([8, 2, 5], False, -3.0),
    ]
    return [
        {"input_ids": ids, "loss_mask": [0.0] + [1.0] * (len(ids) - 2), "chosen": chosen, "ref_logp": ref}
        for ids, chosen, ref in rows
    ]


def test_forward_backward_custom_sends_the_dpo_weights_per_token():
    """Chosen answer tokens get +beta*sigmoid(-z)/pairs, rejected ones the negative, prompt and padding 0."""
    from tests.test_custom_loss import _FakeTrainer
    from tests.test_custom_loss import _client

    beta = 0.5
    data = _dpo_items_for_the_bigram_trainer()
    trainer = _FakeTrainer()

    result = _client(trainer).forward_backward_custom("job", data, task.dpo_loss(beta))

    forward_logprobs = trainer.results["fwd-1"]["logprobs"]
    scores = [forward_logprobs[i, : len(item["input_ids"]) - 1][1:].sum() for i, item in enumerate(data)]
    weights = trainer.backward_frames[0]["context"]["logprob_weights_shifted"]
    pairs = len(data) // 2
    for pair in range(pairs):
        chosen, rejected = 2 * pair, 2 * pair + 1
        z = beta * ((scores[chosen] - data[chosen]["ref_logp"]) - (scores[rejected] - data[rejected]["ref_logp"]))
        weight = beta * torch.sigmoid(-z) / pairs
        for row, sign in ((chosen, 1.0), (rejected, -1.0)):
            length = len(data[row]["input_ids"]) - 1
            expected = torch.zeros(weights.shape[1])
            expected[1:length] = sign * weight
            torch.testing.assert_close(weights[row], expected)
    assert result["loss"] == pytest.approx(result["metrics"]["dpo_loss"])


def test_forward_backward_custom_gives_the_models_dpo_gradient():
    """The model gradient the server accumulates equals the DPO loss's gradient taken straight through the model."""
    from tests.test_custom_loss import _FakeTrainer
    from tests.test_custom_loss import _client
    from tests.test_custom_loss import _direct_gradient

    data = _dpo_items_for_the_bigram_trainer()
    trainer = _FakeTrainer()
    loss_fn = task.dpo_loss(0.5)

    _client(trainer).forward_backward_custom("job", data, loss_fn)

    expected = _direct_gradient(trainer, data, loss_fn, [item["input_ids"] for item in data])
    torch.testing.assert_close(trainer.weights.grad, expected, rtol=1e-5, atol=1e-6)
