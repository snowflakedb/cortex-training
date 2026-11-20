"""Fine-tune TabPFN v2 on a Cortex Training job (``model_provider: "tabpfn"``).

The client builds TabPFN episodes and sends them as ``microbatches`` frames; the
service runs only the network. Held-out accuracy and log loss are scored with
``forward`` before and after training: context rows come from the training
split and query rows from the test split.

    python -m recipes.tabpfn_finetune.train_remote config=~/ct.json steps=30
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

# TabPFN reads this at import. Preprocessing runs on the client CPU; only the network runs on the GPU.
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from cortex_training import wire
from recipes.tabpfn_finetune import episodes as ep
from recipes.utils import make_client, running_job

logger = logging.getLogger("tabpfn_finetune")

# The service admits a job only for a model in its catalog; the tabpfn provider ignores this name's weights.
ADMISSION_MODEL = "Qwen/Qwen3-0.6B"


def job_body(*, lr: float, image_tag: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "sub_job_configs": [{
            "job_type": "training",
            "model_name": ADMISSION_MODEL,
            "training_config": {
                "model_provider": "tabpfn",
                "n_gpus": 1,
                "max_seq_len": 1,
                "train_batch_size": 1,
                "optimizer": {"name": "AdamW", "lr": lr, "weight_decay": 0.01},
                "gradient_clipping": 1.0,
            },
        }],
    }
    if image_tag:
        body["debug"] = {"job": {"image_tag": image_tag}}
    return body


def eval_episode(pre, X_ctx, y_ctx, X_te, y_te) -> ep.Episode:
    from tabpfn.finetuning.data_util import get_preprocessed_dataset_chunks
    from tabpfn.finetuning.data_util import meta_dataset_collator

    n = len(X_ctx)

    def fixed_split(X, y, **_):
        return X[:n], X[n:], y[:n], y[n:]

    chunks = get_preprocessed_dataset_chunks(
        calling_instance=pre, X_raw=np.concatenate([X_ctx, X_te]), y_raw=np.concatenate([y_ctx, y_te]),
        split_fn=fixed_split, max_data_size=None, model_type="classifier", equal_split_size=False,
        data_shuffle_seed=0, preprocessing_random_state=0,
    )
    return ep.episode_from_batch(pre, meta_dataset_collator([chunks[0]]), int(len(np.unique(y_ctx))))


def remote_outputs(client, job_id: str, episode: ep.Episode) -> list[torch.Tensor]:
    body = client.forward(job_id, ep.forward_frame(episode), sub_job_type="training")
    result = client.poll_request(job_id, body["request_id"])
    outputs = result.get("outputs")
    if outputs is None and "payload_b64" in result:
        import base64

        outputs = wire.loads(base64.b64decode(result["payload_b64"]))["outputs"]
    if outputs is None:
        raise RuntimeError(f"forward returned no outputs; keys {sorted(result)}")
    return [torch.as_tensor(o) for o in outputs]


def score(client, job_id: str, episode: ep.Episode) -> dict[str, float]:
    probs = ep.ensemble_probabilities(remote_outputs(client, job_id, episode), episode)[0]
    y = episode.y_query[0]
    log_loss = float(-torch.log(probs[torch.arange(len(y)), y].clamp_min(1e-12)).mean())
    return {"log_loss": log_loss, "accuracy": float((probs.argmax(-1) == y).float().mean())}


def main(argv: list[str]) -> None:
    args = dict(arg.split("=", 1) for arg in argv)
    steps = int(args.get("steps", 30))
    lr = float(args.get("lr", 3e-5))
    mode = args.get("mode", "cross_entropy")
    n_estimators = int(args.get("n_estimators", 2))
    episode_rows = int(args.get("episode_rows", 400))
    if args.get("image_tag"):
        os.environ["CORTEX_TRAINING_ENABLE_DEBUG_OPTIONS"] = "1"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(0)

    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    pre = ep.make_preprocessor(n_estimators)
    held_out = eval_episode(pre, X_tr[:600], y_tr[:600], X_te, y_te)

    client = make_client(
        config=args.get("config"), connection=args.get("connection"),
        poll_timeout=float(args.get("poll_timeout", 3600)),
    )
    keep = args.get("keep_job", "false").lower() == "true"
    with running_job(client, job_body(lr=lr, image_tag=args.get("image_tag")), args.get("job_id"), keep) as job_id:
        before = score(client, job_id, held_out)
        logger.info("before: held-out %s", before)
        done, epoch, started = 0, 0, time.time()
        while done < steps:
            for episode in ep.episodes(pre, X_tr, y_tr, episode_rows=episode_rows, query_ratio=0.3, seed=epoch):
                if mode == "cross_entropy":
                    frame = ep.forward_backward_frame(episode)
                else:
                    loss, grads = ep.client_cross_entropy(remote_outputs(client, job_id, episode), episode)
                    frame = ep.output_gradient_frame(episode, grads)
                result = client.poll_request(job_id, client.forward_backward(job_id, frame))
                stepped = client.poll_request(job_id, client.step(job_id, lr))
                done += 1
                metrics = result.get("metrics") or {}
                logger.info(
                    "step %d loss %.4f %s grad_norm %s (%.0fs)", done,
                    result.get("avg_loss") if mode == "cross_entropy" else loss,
                    f"query acc {metrics['tabpfn_accuracy']:.3f}" if "tabpfn_accuracy" in metrics else "",
                    stepped.get("grad_norm"), time.time() - started,
                )
                if done >= steps:
                    break
            epoch += 1
        after = score(client, job_id, held_out)
        logger.info("after %d steps: held-out %s (before %s)", steps, after, before)
        if args.get("save", "true").lower() == "true":
            saved = client.poll_request(job_id, client.save(job_id, checkpoint_type="weights-only"))
            logger.info("saved checkpoint: %s", saved)


if __name__ == "__main__":
    main(sys.argv[1:])
