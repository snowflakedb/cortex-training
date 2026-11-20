"""A stand-in for the Cortex Training worker, for prototyping the request contract.

It holds only the raw TabPFN network and an optimizer, and speaks DSSST1 frames,
so whatever the client needs to send is visible here. It is not the service:
there is no data parallelism, DeepSpeed, or checkpoint storage.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.model_loading import load_model_criterion_config

from cortex_training import wire

_TASK_TYPE = {"classification": "multiclass"}


def load_tabpfn_network(task: str = "classification") -> torch.nn.Module:
    models, _, _, _ = load_model_criterion_config(
        model_path=None,
        check_bar_distribution_criterion=False,
        estimator_type="classifier" if task == "classification" else "regressor",
        version="v2",
        download_if_not_exists=True,
    )
    return models[0]


class ReferenceWorker:
    def __init__(self, network: torch.nn.Module, *, task: str = "classification", lr: float = 1e-5,
                 weight_decay: float = 0.01) -> None:
        self.network = network.float()
        self.task_type = _TASK_TYPE[task]
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        params = inspect.signature(self.network.forward).parameters
        self._accepts = {name for name in ("task_type", "estimator_ids") if name in params}

    def _call(self, kwargs: dict[str, Any]) -> torch.Tensor:
        extra = {"task_type": self.task_type, "estimator_ids": kwargs.get("estimator_ids")}
        return self.network(
            kwargs["x"],
            kwargs["y"],
            only_return_standard_out=True,
            categorical_inds=kwargs["categorical_inds"],
            performance_options=PerformanceOptions(),
            **{key: value for key, value in extra.items() if key in self._accepts},
        )

    def forward(self, frame: bytes) -> dict[str, Any]:
        request = wire.loads(frame)
        self.network.eval()
        with torch.no_grad():
            outputs = [self._call(mb["kwargs"]) for mb in request["microbatches"]]
        return {"outputs": outputs}

    def forward_backward(self, frame: bytes) -> dict[str, Any]:
        request = wire.loads(frame)
        processing = request["processing"]
        loss_fn = processing["loss_fn"]
        config = processing.get("config", {})
        microbatches = request["microbatches"]
        self.network.train()
        if loss_fn == "cross_entropy":
            num_classes = int(config["num_classes"])
            # Normalize over the whole request, however it is split into calls.
            total = sum(mb["context"]["targets"].numel() for mb in microbatches)
            loss_sum, correct = 0.0, 0
            for mb in microbatches:
                out = self._call(mb["kwargs"])[..., :num_classes]
                targets = mb["context"]["targets"].long()
                if out.shape[:-1] != targets.shape:
                    raise ValueError(f"outputs {tuple(out.shape)} do not match targets {tuple(targets.shape)}")
                loss = torch.nn.functional.cross_entropy(
                    out.reshape(-1, num_classes), targets.reshape(-1), reduction="sum"
                ) / total
                loss.backward()
                loss_sum += loss.item()
                correct += int((out.argmax(-1) == targets).sum())
            return {"avg_loss": loss_sum, "metrics": {"query_count": total, "accuracy": correct / total}}
        if loss_fn == "output_gradient":
            objective = 0.0
            for mb in microbatches:
                out = self._call(mb["kwargs"])
                grad = mb["context"]["output_grad"]
                if grad.shape != out.shape:
                    raise ValueError(f"output_grad {tuple(grad.shape)} does not match outputs {tuple(out.shape)}")
                surrogate = (out * grad.to(out.dtype)).sum()
                surrogate.backward()
                objective += float(surrogate)
            return {"avg_loss": objective, "metrics": {"output_gradient_objective_sum": objective}}
        raise ValueError(f"{loss_fn!r} is not a registered loss")

    def step(self, lr: float | None = None, grad_clip: float | None = 1.0) -> dict[str, Any]:
        if lr is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = lr
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), grad_clip or float("inf"))
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return {"metrics": {"grad_norm": float(grad_norm), "last_lr": self.optimizer.param_groups[0]["lr"]}}

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
