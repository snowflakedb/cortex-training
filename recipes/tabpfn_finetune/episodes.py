"""Client side: turn a table into TabPFN fine-tuning episodes and DSSST1 frames.

TabPFN's preprocessing (feature transforms, class shuffling, one table copy
per ensemble estimator) stays in the client. The server only runs the network,
once per estimator: ``model(x, y, categorical_inds=...)`` where ``x`` holds the
context rows followed by the query rows and ``y`` holds the context targets.

A frame carries one entry per model call in ``microbatches``. Estimators keep
different numbers of columns after preprocessing, so the calls cannot be
stacked into one tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import torch
from tabpfn import TabPFNClassifier
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.constants import ModelVersion
from tabpfn.finetuning.data_util import get_preprocessed_dataset_chunks
from tabpfn.finetuning.data_util import meta_dataset_collator
from tabpfn.finetuning.finetuned_base import _ratio_ctx_query_split
from tabpfn.preprocessing.datamodel import FeatureModality

from cortex_training import wire


@dataclass
class ModelCall:
    """The inputs of one network call (one estimator) and its query targets."""

    x: torch.Tensor  # [R_context + R_query, B, F] float32
    y: torch.Tensor  # [R_context, B] float32, labels in the estimator's shuffled class order
    categorical_inds: list[list[int]]  # one list per table in the batch
    estimator_ids: list[int | None]  # None on TabPFN versions without estimator ids
    targets: torch.Tensor  # [R_query, B] int64, in the same shuffled class order as y
    perms: list[np.ndarray]  # per table: output column perms[b][c] scores original class c


@dataclass
class Episode:
    calls: list[ModelCall]
    num_classes: int
    y_query: torch.Tensor  # [B, R_query] original labels


def make_preprocessor(n_estimators: int, *, device: str = "cpu") -> TabPFNClassifier:
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        device=device,
        n_estimators=n_estimators,
        fit_mode="batched",
        differentiable_input=False,
    )


def _class_perm(config: Any, n_classes: int) -> np.ndarray:
    if config.class_permutation is None:
        return np.arange(n_classes)
    if len(config.class_permutation) != n_classes:
        perm = np.arange(n_classes)
        perm[: len(config.class_permutation)] = config.class_permutation
        return perm
    return np.asarray(config.class_permutation)


def _shuffled_targets(y_query: torch.Tensor, perms: list[np.ndarray]) -> torch.Tensor:
    """Map original labels into each estimator's shuffled class order: ``[B, Q] -> [Q, B]``."""
    return torch.stack(
        [torch.as_tensor(perm)[y_query[table].long()] for table, perm in enumerate(perms)], dim=1
    )


def episodes(
    preprocessor: TabPFNClassifier,
    X: np.ndarray,
    y: np.ndarray,
    *,
    episode_rows: int,
    query_ratio: float,
    seed: int,
) -> list[Episode]:
    """Sample context/query episodes the same way ``FinetunedTabPFNClassifier`` does."""
    n_classes = int(len(np.unique(y)))
    split = partial(
        _ratio_ctx_query_split,
        query_ratio=query_ratio,
        min_query_size=n_classes,
        random_state=seed,
    )
    chunks = get_preprocessed_dataset_chunks(
        calling_instance=preprocessor,
        X_raw=X,
        y_raw=y,
        split_fn=split,
        max_data_size=episode_rows,
        model_type="classifier",
        equal_split_size=False,
        data_shuffle_seed=seed,
        preprocessing_random_state=seed,
    )
    out = []
    for chunk in chunks:
        batch = meta_dataset_collator([chunk])
        out.append(episode_from_batch(preprocessor, batch, n_classes))
    return out


def episode_from_batch(preprocessor: TabPFNClassifier, batch: Any, n_classes: int) -> Episode:
    """Rebuild exactly the tensors TabPFN's batched engine would hand the network."""
    preprocessor.fit_from_preprocessed(
        batch.X_context,
        batch.y_context,
        batch.cat_indices,
        batch.configs,
        performance_options=PerformanceOptions(),
    )
    engine = preprocessor.executor_
    calls = []
    for i in range(len(engine.X_trains)):
        x_full = torch.cat([engine.X_trains[i], batch.X_query[i]], dim=-2)
        configs = engine.ensemble_configs[i]
        perms = [_class_perm(config, n_classes) for config in configs]
        calls.append(
            ModelCall(
                x=x_full.transpose(0, 1).float().contiguous(),
                y=engine.y_trains[i].transpose(0, 1).float().contiguous(),
                categorical_inds=[
                    list(schema[i].indices_for(FeatureModality.CATEGORICAL))
                    for schema in engine.feature_schema_list
                ],
                estimator_ids=[getattr(config, "estimator_id", None) for config in configs],
                targets=_shuffled_targets(batch.y_query, perms),
                perms=perms,
            )
        )
    return Episode(calls=calls, num_classes=n_classes, y_query=batch.y_query.long())


def ensemble_probabilities(outputs: list[torch.Tensor], episode: Episode) -> torch.Tensor:
    """Average the estimators' softmax in the original class order: ``[B, R_query, num_classes]``."""
    probs = []
    for out, call in zip(outputs, episode.calls, strict=True):
        per_table = [
            torch.softmax(out[:, table, torch.as_tensor(perm)].float(), dim=-1)
            for table, perm in enumerate(call.perms)
        ]
        probs.append(torch.stack(per_table))
    return torch.stack(probs).mean(0)


def _call_kwargs(call: ModelCall) -> dict[str, Any]:
    kwargs = {"x": call.x, "y": call.y, "categorical_inds": call.categorical_inds}
    if None not in call.estimator_ids:
        kwargs["estimator_ids"] = call.estimator_ids
    return kwargs


def forward_backward_frame(episode: Episode) -> bytes:
    """One round trip: the server applies cross-entropy on the query rows."""
    return wire.dumps(
        {
            "microbatches": [
                {"kwargs": _call_kwargs(call), "context": {"targets": call.targets}}
                for call in episode.calls
            ],
            "processing": {"loss_fn": "cross_entropy", "config": {"num_classes": episode.num_classes}},
        }
    )


def forward_frame(episode: Episode) -> bytes:
    """No-gradient pass that returns each call's raw outputs."""
    return wire.dumps({"microbatches": [{"kwargs": _call_kwargs(call)} for call in episode.calls]})


def output_gradient_frame(episode: Episode, grads: list[torch.Tensor]) -> bytes:
    """Backpropagate a client-computed ``dL/doutputs`` through the network."""
    return wire.dumps(
        {
            "microbatches": [
                {"kwargs": _call_kwargs(call), "context": {"output_grad": grad.float().contiguous()}}
                for call, grad in zip(episode.calls, grads, strict=True)
            ],
            "processing": {"loss_fn": "output_gradient"},
        }
    )


def client_cross_entropy(outputs: list[torch.Tensor], episode: Episode) -> tuple[float, list[torch.Tensor]]:
    """Any client-side loss works here; this one reproduces TabPFN's own objective."""
    leaves = [out.detach().clone().requires_grad_(True) for out in outputs]
    total = sum(call.targets.numel() for call in episode.calls)
    loss = sum(
        torch.nn.functional.cross_entropy(
            leaf[..., : episode.num_classes].reshape(-1, episode.num_classes),
            call.targets.reshape(-1),
            reduction="sum",
        )
        for leaf, call in zip(leaves, episode.calls, strict=True)
    ) / total
    loss.backward()
    return float(loss), [leaf.grad for leaf in leaves]
