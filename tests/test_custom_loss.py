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

"""Tests for ``forward_backward_custom``.

``_FakeTrainer`` stands in for the service with a bigram model and implements
the two server contracts the client relies on: a forward pass gathers
log-probabilities at the labels as sent, and ``weighted_cross_entropy`` trains
``L_surrogate = -sum(w * logprobs)`` over the request, with ``w`` the frame's
``logprob_weights_shifted``, rejecting the frames the server rejects. Every
frame is decoded with ``wire.loads``.
"""

from __future__ import annotations

import math

import pytest
import torch

from cortex_training import wire
from cortex_training.client import CortexTrainingClient
from cortex_training.client import custom_loss

VOCAB = 11
SEQUENCES = [[1, 4, 2, 7, 3], [5, 9, 6], [8, 2, 10, 4]]
_OBJECTIVE_SUM = "weighted_cross_entropy_objective_sum"
# The error a server without weighted_cross_entropy reports when its backward request is polled.
UNSUPPORTED_SERVER_ERROR = (
    "Request r-1 ended with state 'failed': arctic_platform.registry.RegistryError: 'weighted_cross_entropy' "
    "is not a registered BaseLoss. Available registered classes: ['grouped_distillation', 'grpo']\n"
    "ValueError: unknown registry name 'weighted_cross_entropy'; known: ['causal_cross_entropy', 'grpo']"
)


class _FakeTrainer:
    """A bigram language model behind the client's forward, forward_backward and poll_request."""

    def __init__(self, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        self.weights = torch.randn(VOCAB, VOCAB, generator=generator).requires_grad_(True)
        self.forward_frames: list[dict] = []
        self.backward_frames: list[dict] = []
        self.forward_sub_job_types: list[str | None] = []
        self.results: dict[str, dict] = {}
        self.drift = 0.0

    def token_logprobs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """``out[i, j] = log p(input_ids[i, j + 1] | input_ids[i, j])``; the last column is 0."""
        logprobs = torch.log_softmax(self.weights[input_ids[:, :-1]], dim=-1)
        gathered = logprobs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        return torch.cat([gathered, torch.zeros_like(gathered[:, :1])], dim=1)

    def install(self, client: CortexTrainingClient) -> None:
        client.forward = self.forward
        client.forward_backward = self.forward_backward
        client.poll_request = self.poll_request

    def forward(self, job_id, payload, *, sub_job_id=None, sub_job_type=None):
        frame = wire.loads(payload)
        self.forward_frames.append(frame)
        self.forward_sub_job_types.append(sub_job_type)
        kwargs = frame["kwargs"]
        labels = kwargs["labels"]
        with torch.no_grad():
            logprobs = torch.log_softmax(self.weights[kwargs["input_ids"]], dim=-1)
            gathered = logprobs.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            gathered = gathered.masked_fill(labels == -100, 0.0)
        request_id = f"fwd-{len(self.forward_frames)}"
        self.results[request_id] = {"job_id": job_id, "logprobs": gathered.float()}
        return {"request_id": request_id}

    def forward_backward(self, job_id, payload):
        frame = wire.loads(payload)
        self.backward_frames.append(frame)
        assert frame["processing"] == {"loss_fn": "weighted_cross_entropy", "post": ["compute_logprobs"]}
        context = frame["context"]
        mask, weights = context["loss_mask"], context["logprob_weights_shifted"]
        assert weights.dtype == torch.float32 and weights.shape == mask.shape
        assert torch.isfinite(weights).all()
        assert torch.count_nonzero(weights[mask == 0]) == 0
        logprobs = self.token_logprobs(frame["kwargs"]["input_ids"])
        loss = -(weights * logprobs.masked_fill(mask == 0, 0.0)).sum()
        loss.backward()
        request_id = f"fwd-bwd-{len(self.backward_frames)}"
        self.results[request_id] = {
            "job_id": job_id,
            # As on a server that sums the losses of two data-parallel ranks, each scaled by dp_size.
            "avg_loss": 2 * float(loss.detach()),
            "metrics": {"server_metric": 1.0, "shared": 1.0, _OBJECTIVE_SUM: float(loss.detach())},
            "post_process_outputs": {"logprobs": (logprobs.detach() + self.drift).tolist()},
        }
        return request_id

    def poll_request(self, job_id, request_id):
        return self.results[request_id]


def _client(trainer: _FakeTrainer) -> CortexTrainingClient:
    client = CortexTrainingClient(base_url="http://test.local", database="DB", schema="SCH")
    trainer.install(client)
    return client


def _pairwise_loss(data, logprobs):
    """A nonlinear objective over several items: sigmoid of a mean-logprob margin."""
    margin = logprobs[0].mean() - 0.5 * (logprobs[1].sum() + logprobs[2][1:].sum())
    loss = -torch.nn.functional.logsigmoid(2.0 * margin)
    return loss, {"margin": margin.detach(), "shared": 7}


def _direct_gradient(trainer: _FakeTrainer, data, loss_fn, tokens) -> torch.Tensor:
    """The gradient of ``loss_fn`` taken straight through the model, with no client in between."""
    reference = _FakeTrainer()
    reference.weights = trainer.weights.detach().clone().requires_grad_(True)
    logprobs = [reference.token_logprobs(torch.tensor([row]))[0, : len(row) - 1] for row in tokens]
    loss, _ = loss_fn(data, logprobs)
    (grad,) = torch.autograd.grad(loss, reference.weights)
    return grad


class TestForwardPass:
    def test_frame_pads_right_labels_the_next_token_and_targets_the_training_sub_job(self):
        trainer = _FakeTrainer()
        _client(trainer).forward_backward_custom("job-1", SEQUENCES, _pairwise_loss)

        frame = trainer.forward_frames[0]
        kwargs = frame["kwargs"]
        assert kwargs["input_ids"].tolist() == [[1, 4, 2, 7, 3], [5, 9, 6, 0, 0], [8, 2, 10, 4, 0]]
        assert kwargs["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]]
        assert kwargs["position_ids"].tolist() == [[0, 1, 2, 3, 4]] * 3
        assert kwargs["labels"].tolist() == [
            [4, 2, 7, 3, -100],
            [9, 6, -100, -100, -100],
            [2, 10, 4, -100, -100],
        ]
        assert kwargs["use_cache"] is False
        assert "context" not in frame and "processing" not in frame
        # The training sub-job is named by type, so jobs that also have sampling sub-jobs work.
        assert trainer.forward_sub_job_types == ["training"]

    def test_loss_fn_sees_one_prediction_aligned_tensor_per_item(self):
        trainer = _FakeTrainer()
        seen = []

        def loss_fn(data, logprobs):
            seen.extend(logprobs)
            return -torch.cat(logprobs).sum(), {}

        _client(trainer).forward_backward_custom("job-1", SEQUENCES, loss_fn)

        assert [row.shape for row in seen] == [torch.Size([4]), torch.Size([2]), torch.Size([3])]
        assert all(row.dtype == torch.float32 and row.requires_grad for row in seen)
        expected = trainer.token_logprobs(torch.tensor([SEQUENCES[0]]))[0, :4]
        torch.testing.assert_close(seen[0].detach(), expected.detach())

    @pytest.mark.parametrize(
        ("result", "match"),
        [
            ({}, "no per-token logprobs"),
            ({"logprobs": torch.zeros(2, 5)}, "shape"),
            ({"logprobs": torch.zeros(3, 2)}, "needs 4"),
            ({"logprobs": torch.full((3, 5), float("nan"))}, "non-finite"),
            ({"logprobs": [[0.0] * 5] * 2}, "different number of items"),
        ],
    )
    def test_rejects_a_result_that_does_not_cover_the_items(self, result, match):
        with pytest.raises(RuntimeError, match=match):
            custom_loss.logprobs_from_forward_result(result, SEQUENCES)

    def test_forward_logprobs_is_not_public(self):
        assert not hasattr(CortexTrainingClient, "forward_logprobs")


class TestData:
    def test_loss_fn_gets_the_callers_list_unchanged(self):
        trainer = _FakeTrainer()
        data = [
            {"input_ids": SEQUENCES[0], "role": "chosen", "pair": 0},
            SEQUENCES[1],
            {"input_ids": tuple(SEQUENCES[2]), "role": "rejected", "pair": 0, "extra": object()},
        ]
        snapshot = list(data)
        received = []

        def loss_fn(items, logprobs):
            received.append(items)
            return -logprobs[0].sum() + logprobs[2].sum(), {}

        _client(trainer).forward_backward_custom("job-1", data, loss_fn)

        (items,) = received
        assert items is data
        assert all(after is before for after, before in zip(items, snapshot)) and len(items) == len(snapshot)
        assert trainer.forward_frames[0]["kwargs"]["input_ids"].tolist()[2] == [8, 2, 10, 4, 0]

    def test_mapping_items_train_like_token_lists(self):
        def by_role(items, logprobs):
            signs = [1.0 if isinstance(item, dict) and item.get("role") == "chosen" else -0.5 for item in items]
            return -sum(sign * row.sum() for sign, row in zip(signs, logprobs)), {}

        data = [{"input_ids": SEQUENCES[0], "role": "chosen"}, {"input_ids": SEQUENCES[1]}, SEQUENCES[2]]
        trainer = _FakeTrainer()
        expected = _direct_gradient(trainer, data, by_role, SEQUENCES)

        _client(trainer).forward_backward_custom("job-1", data, by_role)

        torch.testing.assert_close(trainer.weights.grad, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize(
        ("data", "error", "match"),
        [
            ([], ValueError, "must not be empty"),
            ("abc", TypeError, "list of items"),
            ({"input_ids": [1, 2]}, TypeError, "list of items"),
            (None, TypeError, "list of items"),
            ([[1, 2], [3]], ValueError, "data\\[1\\] has 1 token"),
            ([[1, True]], TypeError, "integer token ids"),
            ([[1, -1]], TypeError, "integer token ids"),
            ([[1, 2.0]], TypeError, "integer token ids"),
            (["ab"], TypeError, "data\\[0\\] must be a sequence of token ids"),
            ([5], TypeError, "data\\[0\\] must be a sequence of token ids"),
            ([{"tokens": [1, 2]}], ValueError, "without 'input_ids'"),
            ([{"input_ids": [1]}], ValueError, "data\\[0\\]\\['input_ids'\\] has 1 token"),
            ([{"input_ids": "ab"}], TypeError, "data\\[0\\]\\['input_ids'\\] must be a sequence"),
        ],
    )
    def test_malformed_data_fails_before_any_request(self, data, error, match):
        trainer = _FakeTrainer()
        with pytest.raises(error, match=match):
            _client(trainer).forward_backward_custom("job-1", data, _pairwise_loss)
        assert trainer.forward_frames == [] and trainer.backward_frames == []


class TestBackwardPass:
    def test_model_gradient_is_the_gradient_of_the_callers_loss(self):
        trainer = _FakeTrainer()
        expected = _direct_gradient(trainer, SEQUENCES, _pairwise_loss, SEQUENCES)

        _client(trainer).forward_backward_custom("job-1", SEQUENCES, _pairwise_loss)

        torch.testing.assert_close(trainer.weights.grad, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("data", [SEQUENCES[:1], SEQUENCES[1:], SEQUENCES], ids=["1", "2", "3"])
    def test_gradient_scale_does_not_depend_on_the_requests_size(self, data):
        def summed_nll(items, logprobs):
            return -torch.cat(logprobs).sum(), {}

        trainer = _FakeTrainer()
        expected = _direct_gradient(trainer, data, summed_nll, data)

        _client(trainer).forward_backward_custom("job-1", data, summed_nll)

        torch.testing.assert_close(trainer.weights.grad, expected, rtol=1e-5, atol=1e-6)

    def test_frame_names_weighted_cross_entropy_with_the_negated_gradient(self):
        trainer = _FakeTrainer()
        _client(trainer).forward_backward_custom("job-1", SEQUENCES, _pairwise_loss)

        frame = trainer.backward_frames[0]
        assert frame["processing"] == {"loss_fn": "weighted_cross_entropy", "post": ["compute_logprobs"]}
        assert "config" not in frame["processing"]
        context = frame["context"]
        assert set(context) == {"input_ids", "loss_mask", "logprob_weights_shifted"}
        assert context["loss_mask"].tolist() == [[1, 1, 1, 1, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0]]
        assert context["loss_mask"].dtype == torch.float32
        assert torch.equal(context["input_ids"], frame["kwargs"]["input_ids"])
        assert set(frame["kwargs"]) == {"input_ids", "attention_mask", "position_ids", "use_cache"}

        logprobs = trainer.results["fwd-1"]["logprobs"]
        leaves = [logprobs[i, : len(row) - 1].clone().requires_grad_(True) for i, row in enumerate(SEQUENCES)]
        loss, _ = _pairwise_loss(SEQUENCES, leaves)
        grads = torch.autograd.grad(loss, leaves)
        weights = context["logprob_weights_shifted"]
        assert weights.dtype == torch.float32
        for row, (tokens, grad) in enumerate(zip(SEQUENCES, grads)):
            torch.testing.assert_close(weights[row, : len(tokens) - 1], -grad, rtol=0, atol=0)
            assert torch.count_nonzero(weights[row, len(tokens) - 1 :]) == 0

    def test_an_item_the_loss_ignores_gets_zero_weights(self):
        def first_only(items, logprobs):
            return -logprobs[0].sum(), {}

        trainer = _FakeTrainer()
        _client(trainer).forward_backward_custom("job-1", SEQUENCES, first_only)

        weights = trainer.backward_frames[0]["context"]["logprob_weights_shifted"]
        assert torch.count_nonzero(weights[1:]) == 0
        assert weights[0, :4].tolist() == [1.0, 1.0, 1.0, 1.0]

    @pytest.mark.parametrize(
        ("loss_fn", "error", "match"),
        [
            (lambda d, lp: lp[0].sum(), TypeError, "\\(loss, metrics\\) tuple"),
            (lambda d, lp: (torch.stack([row.sum() for row in lp]), {}), TypeError, "scalar"),
            (lambda d, lp: (torch.tensor(1.0), {}), ValueError, "does not depend"),
            (lambda d, lp: (lp[0].sum().detach(), {}), ValueError, "does not depend"),
            (lambda d, lp: (lp[0].sum() * math.inf, {}), ValueError, "non-finite loss"),
            (lambda d, lp: (lp[0].sum() * math.nan, {}), ValueError, "non-finite loss"),
            (lambda d, lp: (torch.sqrt(lp[0] - lp[0].detach()).sum(), {}), ValueError, "gradient is non-finite"),
            (lambda d, lp: (lp[0].sum(), {"bad": "text"}), TypeError, "metric 'bad' must be a number"),
            (lambda d, lp: (lp[0].sum(), {"bad": None}), TypeError, "metric 'bad' must be a number"),
            (lambda d, lp: (lp[0].sum(), {"bad": True}), TypeError, "metric 'bad' must be a number"),
            (lambda d, lp: (lp[0].sum(), {"bad": torch.ones(2)}), TypeError, "metric 'bad' must be a number"),
            (lambda d, lp: (lp[0].sum(), [1.0]), TypeError, "mapping"),
        ],
    )
    def test_a_bad_loss_fails_before_the_backward_request(self, loss_fn, error, match):
        trainer = _FakeTrainer()
        with pytest.raises(error, match=match):
            _client(trainer).forward_backward_custom("job-1", SEQUENCES, loss_fn)
        assert len(trainer.forward_frames) == 1
        assert trainer.backward_frames == []

    def test_rejects_gradients_that_do_not_match_the_items(self):
        with pytest.raises(ValueError, match="2 gradient tensors for 3"):
            custom_loss.build_custom_loss_backward_payload(SEQUENCES, [torch.zeros(4), torch.zeros(2)])
        with pytest.raises(ValueError, match="grads\\[1\\] has 3 values"):
            custom_loss.build_custom_loss_backward_payload(SEQUENCES, [torch.zeros(4), torch.zeros(3), torch.zeros(3)])


class TestResult:
    def test_result_reports_the_callers_loss_metrics_logprobs_and_drift(self):
        trainer = _FakeTrainer()
        trainer.drift = 0.25
        result = _client(trainer).forward_backward_custom("job-1", SEQUENCES, _pairwise_loss)

        logprobs = trainer.results["fwd-1"]["logprobs"]
        rows = [logprobs[i, : len(tokens) - 1] for i, tokens in enumerate(SEQUENCES)]
        leaves = [row.clone().requires_grad_(True) for row in rows]
        expected_loss, expected_metrics = _pairwise_loss(SEQUENCES, leaves)
        grads = torch.autograd.grad(expected_loss, leaves, allow_unused=True)
        l_surrogate = sum(float((g * row).sum()) for g, row in zip(grads, rows) if g is not None)
        assert result["loss"] == pytest.approx(float(expected_loss))
        assert result["metrics"] == {
            "server_metric": 1.0,
            "shared": 7.0,
            "margin": pytest.approx(float(expected_metrics["margin"])),
            # The summed metric is L_surrogate = sum(g * logprobs) whatever the server does with avg_loss.
            _OBJECTIVE_SUM: pytest.approx(l_surrogate, abs=1e-5),
        }
        assert l_surrogate != pytest.approx(float(expected_loss))
        assert [row.tolist() for row in result["logprobs"]] == [row.tolist() for row in rows]
        assert not any(row.requires_grad for row in result["logprobs"])
        assert result["logprob_max_abs_diff"] == pytest.approx(0.25)
        assert result["avg_loss"] == trainer.results["fwd-bwd-1"]["avg_loss"]

    def test_drift_is_absent_when_the_server_returns_no_backward_logprobs(self):
        trainer = _FakeTrainer()
        client = _client(trainer)
        poll = client.poll_request

        def without_post_outputs(job_id, request_id):
            result = dict(poll(job_id, request_id))
            result.pop("post_process_outputs", None)
            return result

        client.poll_request = without_post_outputs
        result = client.forward_backward_custom("job-1", SEQUENCES, _pairwise_loss)
        assert "logprob_max_abs_diff" not in result


class TestServerWithoutTheLoss:
    def _client_failing_backward(self, error: Exception) -> tuple[_FakeTrainer, CortexTrainingClient]:
        trainer = _FakeTrainer()
        client = _client(trainer)
        poll = client.poll_request

        def poll_request(job_id, request_id):
            if request_id.startswith("fwd-bwd-"):
                raise error
            return poll(job_id, request_id)

        client.poll_request = poll_request
        return trainer, client

    def test_a_server_without_weighted_cross_entropy_says_custom_losses_are_unsupported(self):
        server_error = RuntimeError(UNSUPPORTED_SERVER_ERROR)
        trainer, client = self._client_failing_backward(server_error)

        with pytest.raises(RuntimeError, match="does not support custom losses yet") as caught:
            client.forward_backward_custom("job-1", SEQUENCES, _pairwise_loss)

        assert caught.value.__cause__ is server_error
        assert "is not a registered BaseLoss" in str(caught.value)
        assert len(trainer.backward_frames) == 1

    @pytest.mark.parametrize(
        "server_error",
        [
            RuntimeError("Request r-1 ended with state 'failed': CUDA out of memory"),
            RuntimeError("Request r-1 ended with state 'cancelled': "),
            TimeoutError("Request r-1 did not complete within 60s"),
        ],
    )
    def test_other_failures_propagate_unchanged(self, server_error):
        _, client = self._client_failing_backward(server_error)

        with pytest.raises(type(server_error)) as caught:
            client.forward_backward_custom("job-1", SEQUENCES, _pairwise_loss)

        assert caught.value is server_error
