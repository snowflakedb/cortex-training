"""Prototype: fine-tune TabPFN v2 through Cortex-Training-shaped requests.

1. Parity: for one episode, the gradients from a server-side cross-entropy
   request and from a client-side loss sent as ``dL/doutputs`` must match
   TabPFN's own fine-tuning step.
2. Training: a short run through ``ReferenceWorker`` frames only, scored on a
   held-out split by loading the trained weights back into ``TabPFNClassifier``.

    python -m recipes.tabpfn_finetune.run --steps 30
"""

from __future__ import annotations

import argparse
import copy
import time
from functools import partial

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.finetuning.data_util import get_preprocessed_dataset_chunks
from tabpfn.finetuning.data_util import meta_dataset_collator
from tabpfn.finetuning.finetuned_base import _ratio_ctx_query_split

from recipes.tabpfn_finetune import episodes as ep
from recipes.tabpfn_finetune.reference_worker import ReferenceWorker
from recipes.tabpfn_finetune.reference_worker import load_tabpfn_network


def _grads(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in module.parameters()
    ])


def _rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def parity(X: np.ndarray, y: np.ndarray, n_estimators: int) -> None:
    pre = ep.make_preprocessor(n_estimators)
    n_classes = int(len(np.unique(y)))
    split = partial(_ratio_ctx_query_split, query_ratio=0.3, min_query_size=n_classes, random_state=0)
    chunk = get_preprocessed_dataset_chunks(
        calling_instance=pre, X_raw=X, y_raw=y, split_fn=split, max_data_size=300,
        model_type="classifier", equal_split_size=False, data_shuffle_seed=0, preprocessing_random_state=0,
    )[0]
    batch = meta_dataset_collator([chunk])
    episode = ep.episode_from_batch(pre, batch, n_classes)

    # TabPFN's own step, as FinetunedTabPFNClassifier._forward_with_loss runs it.
    native = pre.models_[0]
    native.train()
    native.zero_grad(set_to_none=True)
    pre.fit_from_preprocessed(batch.X_context, batch.y_context, batch.cat_indices, batch.configs,
                              performance_options=PerformanceOptions())
    logits_QBEL = pre.forward(batch.X_query, return_raw_logits=True)
    Q, B, E, L = logits_QBEL.shape
    logits = logits_QBEL.permute(1, 2, 3, 0).reshape(B * E, L, Q)
    native_loss = torch.nn.functional.cross_entropy(logits, batch.y_query.repeat(B * E, 1).long())
    native_loss.backward()
    native_grads = _grads(native)

    worker = ReferenceWorker(copy.deepcopy(load_tabpfn_network()))
    assert worker.network is not native

    frame = ep.forward_backward_frame(episode)
    result = worker.forward_backward(frame)
    server_grads = _grads(worker.network)
    print(f"[parity] frame {len(frame) / 1e6:.2f} MB, {len(episode.calls)} model calls, "
          f"x widths {[tuple(c.x.shape) for c in episode.calls]}")
    print(f"[parity] native loss {float(native_loss):.6f} | server cross_entropy {result['avg_loss']:.6f} "
          f"| grad rel diff {_rel_diff(server_grads, native_grads):.2e}")

    worker.zero_grad()
    outputs = worker.forward(ep.forward_frame(episode))["outputs"]
    client_loss, output_grads = ep.client_cross_entropy(outputs, episode)
    worker.forward_backward(ep.output_gradient_frame(episode, output_grads))
    vjp_grads = _grads(worker.network)
    print(f"[parity] client loss {client_loss:.6f} via output_gradient | grad rel diff "
          f"{_rel_diff(vjp_grads, native_grads):.2e}")


def evaluate(state: dict, X_tr, y_tr, X_te, y_te, n_estimators: int) -> tuple[float, float]:
    """Score weights the way a user would after export: inside a stock TabPFNClassifier."""
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", n_estimators=n_estimators)
    clf.fit(X_tr, y_tr)
    model = clf.models_[0]
    baseline = copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    try:
        proba = clf.predict_proba(X_te)
    finally:
        model.load_state_dict(baseline)
    return float(log_loss(y_te, proba, labels=clf.classes_)), float(accuracy_score(y_te, proba.argmax(1)))


def train(X_tr, y_tr, X_te, y_te, *, steps: int, lr: float, n_estimators: int, episode_rows: int) -> None:
    pre = ep.make_preprocessor(n_estimators)
    worker = ReferenceWorker(copy.deepcopy(load_tabpfn_network()), lr=lr)
    before = evaluate(copy.deepcopy(worker.network.state_dict()), X_tr, y_tr, X_te, y_te, n_estimators)
    print(f"[train] before: held-out log loss {before[0]:.4f}, accuracy {before[1]:.4f}")

    done, epoch, started = 0, 0, time.time()
    while done < steps:
        for episode in ep.episodes(pre, X_tr, y_tr, episode_rows=episode_rows, query_ratio=0.3, seed=epoch):
            result = worker.forward_backward(ep.forward_backward_frame(episode))
            stepped = worker.step()
            done += 1
            if done % 5 == 0 or done == 1:
                print(f"[train] step {done:3d} loss {result['avg_loss']:.4f} "
                      f"query acc {result['metrics']['accuracy']:.3f} grad_norm {stepped['metrics']['grad_norm']:.3f}")
            if done >= steps:
                break
        epoch += 1
    after = evaluate(copy.deepcopy(worker.network.state_dict()), X_tr, y_tr, X_te, y_te, n_estimators)
    print(f"[train] after {steps} steps ({time.time() - started:.0f}s): held-out log loss {after[0]:.4f}, "
          f"accuracy {after[1]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--n-estimators", type=int, default=2)
    parser.add_argument("--episode-rows", type=int, default=400)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    parity(X_tr, y_tr, args.n_estimators)
    if not args.skip_train:
        train(X_tr, y_tr, X_te, y_te, steps=args.steps, lr=args.lr, n_estimators=args.n_estimators,
              episode_rows=args.episode_rows)


if __name__ == "__main__":
    main()
