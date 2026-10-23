#!/usr/bin/env python3
"""Dry-run or execute a Cortex Training catalog job file as a real job.

The request is the catalog file itself: model-catalog/models/<slug>/<context>/
<hw>-<kind>-<mode>.json for a training or inference job. An RL job, which has
no file of its own, is built from the training and inference files with
make_config(rl=True), but without router replay: the probe runs a plain
generate and forward-backward and does not perform the replay handshake, so
replay is left to the RL example's runner and its tests to validate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from validate_catalog import examples_utils
from validate_catalog import validate_catalog

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
CATALOG_DIR = Path(__file__).resolve().parent / "models"

JOBS = ("training-lora", "training-full", "inference-lora", "inference-full", "rl-lora", "rl-full")
TRAINING_JOBS = ("training-lora", "training-full", "rl-lora", "rl-full")
SAMPLING_JOBS = ("inference-lora", "inference-full", "rl-lora", "rl-full")
CONTEXTS = ("standard", "long")
HARDWARE = ("h200", "b200", "b300")
DEBUG_OPTIONS_ENV = "CORTEX_TRAINING_ENABLE_DEBUG_OPTIONS"


def _model_folder(catalog_dir: Path, model_id: str) -> Path:
    info_path = catalog_dir / examples_utils().model_slug(model_id) / "model.json"
    if not info_path.is_file():
        raise ValueError(f"unknown catalog model: {model_id}")
    # The slug is only the lowercased last segment, so a typo or wrong namespace
    # (e.g. NotQwen/Qwen3-8B) can resolve the same folder. Require the requested id
    # to match model.json exactly, so --submit never runs the wrong model.
    resolved = json.loads(info_path.read_text(encoding="utf-8")).get("model_id")
    if resolved != model_id:
        raise ValueError(
            f"unknown catalog model: {model_id}; the folder for its slug holds {resolved!r}. "
            "Pass the exact model id from the catalog."
        )
    return info_path.parent


def build_catalog_request(
    catalog_dir: Path,
    model_id: str,
    job: str,
    *,
    context: str = "standard",
    hardware: str = "h200",
) -> dict[str, Any]:
    """Load the CreateJob body of one catalog job file, or build the RL job.

    The RL job is built without router replay (``router_replay=False``), even for
    an MoE model whose real jobs enable it. The smoke probe runs a plain
    generate and forward-backward without the replay handshake (no bootstrap, no
    per-rollout ``dss_sample_id``, no ``rr_sample_ids``), so a strict-replay job
    would fail preflight here. Router replay is exercised by the RL example's
    runner and its tests, not by this probe.
    """
    if job not in JOBS:
        raise ValueError(f"invalid job: {job}")
    validate_catalog(catalog_dir)
    folder = _model_folder(catalog_dir, model_id)
    if job.startswith("rl-"):
        return examples_utils().make_config(
            model_id,
            context,
            hardware=hardware,
            lora=job == "rl-lora",
            rl=True,
            router_replay=False,
            catalog_dir=catalog_dir,
        )
    path = folder / context / f"{hardware}-{job}.json"
    if not path.is_file():
        available = sorted(
            str(found.relative_to(folder).with_suffix(""))
            for found in folder.glob("*/*.json")
        )
        raise ValueError(
            f"{model_id} has no {context}/{hardware}-{job}.json; "
            f"its catalog files are: {', '.join(available)}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def enable_training_memory_telemetry(request: dict[str, Any]) -> None:
    """Enable server-side peak memory reporting without changing placement."""
    for sub_job in request["sub_job_configs"]:
        if sub_job["job_type"] != "training":
            continue
        training = sub_job["training_config"]
        training["step_peak_memory_log"] = True
        training["training_memory_telemetry"] = True


def apply_debug_image_tag(request: dict[str, Any], image_tag: str) -> None:
    """Select a specific backend image for a live smoke test."""
    os.environ[DEBUG_OPTIONS_ENV] = "1"
    request["debug"] = {"job": {"image_tag": image_tag}}


def apply_training_sequence_parallelism(
    request: dict[str, Any],
    sp_size: int,
) -> None:
    """Apply Ulysses SP and its logical data-parallel batch-size contract."""
    if sp_size <= 1:
        raise ValueError("training sequence-parallel size must be greater than 1")
    for sub_job in request["sub_job_configs"]:
        if sub_job["job_type"] != "training":
            continue
        training = sub_job["training_config"]
        n_gpus = training["n_gpus"]
        if n_gpus % sp_size != 0:
            raise ValueError(
                f"training n_gpus {n_gpus} is not divisible by sp_size {sp_size}"
            )
        logical_dp = n_gpus // sp_size
        training["sp_size"] = sp_size
        training["train_batch_size"] = logical_dp
        ds_config = training.setdefault("ds_config", {})
        ds_config["train_batch_size"] = logical_dp
        ds_config["train_micro_batch_size_per_gpu"] = 1
        ds_config["gradient_accumulation_steps"] = 1


def build_forward_backward_probe_spec(
    request: dict[str, Any],
    job: str,
    *,
    full_context: bool = False,
) -> dict[str, Any]:
    """Build a tokenizer-independent training batch for a catalog job."""
    training = next(
        (
            sub_job["training_config"]
            for sub_job in request["sub_job_configs"]
            if sub_job["job_type"] == "training"
        ),
        None,
    )
    if training is None:
        raise ValueError("forward-backward probe requires a training sub-job")

    batch_size = training.get("train_batch_size")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("training_config.train_batch_size must be a positive integer")

    sequence_tokens = training["max_seq_len"] if full_context else 8
    token_ids = [1] * sequence_tokens if full_context else [1, 2, 3, 4, 5, 6, 7, 8]
    input_ids = [token_ids[:] for _ in range(batch_size)]
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "position_ids": "arange",
    }
    spec: dict[str, Any] = {"payload": payload}
    token_chunk_size = training.get("fused_lm_head_token_chunk_size")
    if token_chunk_size is None:
        token_chunk_size = (training.get("prime_rl") or {}).get(
            "fused_lm_head_token_chunk_size"
        )
    chunked_sft = (
        job.startswith("training-")
        and isinstance(token_chunk_size, int)
        and not isinstance(token_chunk_size, bool)
    )
    if job.startswith("rl-") or chunked_sft:
        payload.update(
            {
                "attention_mask": [[1] * len(token_ids) for _ in range(batch_size)],
                "include_attention_mask": True,
                "labels": "none",
            }
        )
        if chunked_sft:
            loss_mask = [1.0] * (sequence_tokens - 1) + [0.0]
            processing = {
                "loss_fn": "causal_cross_entropy",
                "post": ["compute_logprobs"],
                "config": {},
            }
        else:
            loss_mask = [0.0, 0.0] + [1.0] * (sequence_tokens - 3) + [0.0]
            processing = {
                "loss_fn": "grpo",
                "post": ["compute_logprobs"],
                "config": {
                    "eps_clip": 0.2,
                    "loss_agg_mode": "token-mean",
                    "entropy_coeff": 0.0,
                    "global_batch_size": batch_size,
                },
            }
        spec.update(
            {
                "context": {
                    "input_ids": {"data": input_ids, "dtype": "long"},
                    "advantages": {
                        "data": [loss_mask[:] for _ in range(batch_size)],
                        "dtype": "float32",
                    },
                    "loss_mask": {
                        "data": [loss_mask[:] for _ in range(batch_size)],
                        "dtype": "float32",
                    },
                },
                "processing": processing,
            }
        )
    else:
        payload["labels"] = {
            "strategy": "next_token",
            "mask_padding": False,
        }
    return spec


def _build_forward_backward_probe_payload(
    request: dict[str, Any],
    job: str,
    full_context: bool,
) -> bytes:
    sys.path.insert(0, str(SOURCE_ROOT))
    from cortex_training import build_forward_backward_kwargs
    from cortex_training import build_forward_backward_payload
    from cortex_training import wire

    spec = build_forward_backward_probe_spec(
        request,
        job,
        full_context=full_context,
    )
    if "processing" not in spec:
        return build_forward_backward_payload(spec)

    kwargs = build_forward_backward_kwargs(spec["payload"])
    kwargs["use_cache"] = False
    context = build_forward_backward_kwargs({"kwargs": spec["context"]})
    return wire.dumps(
        {
            "args": (),
            "kwargs": kwargs,
            "context": context,
            "processing": spec["processing"],
        },
        metadata={"response_options": {"format": "dssst1", "delivery": "chunked"}},
    )


def _sampling_max_seq_len(request: dict[str, Any]) -> int:
    sampling = next(
        (
            sub_job["inference_config"]
            for sub_job in request["sub_job_configs"]
            if sub_job["job_type"] == "sampling"
        ),
        None,
    )
    if sampling is None:
        raise ValueError("generate probe requires a sampling sub-job")
    max_seq_len = sampling.get("max_seq_len")
    if (
        isinstance(max_seq_len, bool)
        or not isinstance(max_seq_len, int)
        or max_seq_len <= 1
    ):
        raise ValueError("inference_config.max_seq_len must be an integer > 1")
    return max_seq_len


def _run_generate_probe(
    client,
    job_id: str,
    request: dict[str, Any],
    full_context_prefill: bool,
) -> dict[str, Any]:
    if full_context_prefill:
        prompt_tokens = _sampling_max_seq_len(request) - 1
        prompts: list[str | list[int]] = [[1] * prompt_tokens]
    else:
        prompt_tokens = None
        prompts = ["Reply with OK."]

    request_id = client.generate(
        job_id,
        prompts=prompts,
        sampling_params={
            "max_tokens": 1,
            "temperature": 0.0,
        },
    )
    result = client.poll_request(job_id, request_id)
    generated = result.get("results")
    if not isinstance(generated, list) or len(generated) != 1:
        raise RuntimeError(f"generate probe expected one result, got {generated!r}")
    return {
        "operation": "generate",
        "requestId": request_id,
        "resultCount": len(generated),
        "promptTokens": prompt_tokens,
    }


def _run_forward_backward_probe(
    client,
    job_id: str,
    job: str,
    request: dict[str, Any],
    full_context: bool,
) -> dict[str, Any]:
    payload = _build_forward_backward_probe_payload(
        request,
        job,
        full_context,
    )
    request_id = client.forward_backward(job_id, payload)
    result = client.poll_request(job_id, request_id)
    avg_loss = result.get("avg_loss")
    try:
        avg_loss_value = float(avg_loss)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"forward-backward probe returned invalid avg_loss: {avg_loss!r}"
        ) from exc
    if not math.isfinite(avg_loss_value):
        raise RuntimeError(
            f"forward-backward probe returned non-finite avg_loss: {avg_loss!r}"
        )
    result = {
        "operation": "forward-backward",
        "requestId": request_id,
        "avgLoss": avg_loss_value,
    }
    if full_context:
        training = next(
            sub_job["training_config"]
            for sub_job in request["sub_job_configs"]
            if sub_job["job_type"] == "training"
        )
        result.update(
            {
                "sequenceTokens": training["max_seq_len"],
                "globalTokens": (
                    training["max_seq_len"] * training["train_batch_size"]
                ),
            }
        )
    return result


def _run_optimizer_step_probe(
    client, job_id: str, request: dict[str, Any]
) -> dict[str, Any]:
    training = next(
        sub_job["training_config"]
        for sub_job in request["sub_job_configs"]
        if sub_job["job_type"] == "training"
    )
    optimizer = training.get("optimizer") or {}
    request_id = client.step(job_id, learning_rate=optimizer.get("lr"))
    result = client.poll_request(job_id, request_id)
    return {
        "operation": "step",
        "requestId": request_id,
        "globalSteps": result.get("global_steps"),
        "peakMemory": result.get("peak_memory"),
    }


def _checkpoint_id(result: dict[str, Any]) -> str:
    checkpoint_id = result.get("checkpoint_id")
    if isinstance(checkpoint_id, str) and checkpoint_id:
        return checkpoint_id
    for key in ("stage_path", "checkpoint_path"):
        path = result.get(key)
        if not isinstance(path, str):
            continue
        for part in path.split("/"):
            if part.startswith("cp_"):
                return part
    raise RuntimeError(f"checkpoint save returned no checkpoint id: {result!r}")


def _run_checkpoint_round_trip_probe(
    client,
    job_id: str,
) -> list[dict[str, Any]]:
    save_request_id = client.save(job_id, checkpoint_type="resumable")
    save_result = client.poll_request(job_id, save_request_id)
    checkpoint_id = _checkpoint_id(save_result)

    try:
        load_request_id = client.load(
            job_id,
            checkpoint_id,
            source_job_id=job_id,
            target_sub_job_id=f"{job_id}:training:0",
        )
        load_result = client.poll_request(job_id, load_request_id)
        loaded_checkpoint_id = load_result.get("checkpoint_id")
        if loaded_checkpoint_id not in (None, checkpoint_id):
            raise RuntimeError(
                "checkpoint load returned a different checkpoint id: "
                f"{loaded_checkpoint_id!r} != {checkpoint_id!r}"
            )
    finally:
        client.delete_checkpoint(job_id, checkpoint_id)
    return [
        {
            "operation": "save",
            "requestId": save_request_id,
            "checkpointId": checkpoint_id,
            "checkpointType": "resumable",
        },
        {
            "operation": "load",
            "requestId": load_request_id,
            "checkpointId": checkpoint_id,
        },
    ]


def run_data_plane_probes(
    client,
    job_id: str,
    job: str,
    request: dict[str, Any],
    *,
    full_context_prefill: bool = False,
    full_context_training: bool = False,
    training_step: bool = False,
    checkpoint_round_trip: bool = False,
) -> list[dict[str, Any]]:
    """Execute the minimal operations required by a catalog job."""
    probes = []
    if job in SAMPLING_JOBS:
        probes.append(
            _run_generate_probe(
                client,
                job_id,
                request,
                full_context_prefill,
            )
        )
    if job in TRAINING_JOBS:
        probes.append(
            _run_forward_backward_probe(
                client,
                job_id,
                job,
                request,
                full_context_training,
            )
        )
        if training_step or checkpoint_round_trip:
            probes.append(_run_optimizer_step_probe(client, job_id, request))
        if checkpoint_round_trip:
            probes.extend(_run_checkpoint_round_trip_probe(client, job_id))
    return probes


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _required_env(*names: str) -> str:
    value = _first_env(*names)
    if value is None:
        raise ValueError(f"set one of: {', '.join(names)}")
    return value


def _build_client(poll_timeout: float):
    sys.path.insert(0, str(SOURCE_ROOT))
    from cortex_training import CortexTrainingClient

    host = _required_env("SNOWFLAKE_HOST")
    host = host.removeprefix("https://").removeprefix("http://").rstrip("/")
    return CortexTrainingClient.from_pat(
        host=host,
        pat=_required_env("SNOWFLAKE_PAT"),
        database=_required_env("SNOWFLAKE_DATABASE"),
        schema=_first_env("SNOWFLAKE_SCHEMA") or "PUBLIC",
        endpoint=_first_env("CORTEX_TRAINING_ENDPOINT") or "cortex-training",
        poll_timeout=poll_timeout,
    )


def _cancel_job(client, job_id: str, active_error: BaseException | None) -> bool:
    try:
        client.cancel_job(job_id)
    except Exception as cancel_error:
        if active_error is None:
            raise
        print(
            f"WARNING: failed to cancel {job_id}: {cancel_error}",
            file=sys.stderr,
        )
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--context",
        choices=CONTEXTS,
        default="standard",
        help="Catalog context folder (default: standard).",
    )
    parser.add_argument(
        "--hardware",
        choices=HARDWARE,
        default="h200",
        help="Hardware prefix of the catalog file (default: h200).",
    )
    parser.add_argument(
        "--job",
        action="append",
        choices=JOBS,
        required=True,
        help="Catalog job file to test. Repeat to test multiple files serially.",
    )
    parser.add_argument("--catalog-dir", type=Path, default=CATALOG_DIR)
    parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Submit real jobs and execute workflow data-plane probes. "
            "Without this flag, print the request bodies."
        ),
    )
    parser.add_argument("--poll-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--full-context-prefill",
        action="store_true",
        help=(
            "For inference jobs, prefill max_seq_len - 1 token IDs and "
            "generate one token instead of running the short text probe."
        ),
    )
    parser.add_argument(
        "--full-context-training",
        action="store_true",
        help=(
            "For training jobs, execute forward-backward with "
            "train_batch_size sequences of exactly max_seq_len tokens."
        ),
    )
    parser.add_argument(
        "--training-step",
        action="store_true",
        help="Run and poll one optimizer step after forward-backward.",
    )
    parser.add_argument(
        "--checkpoint-round-trip",
        action="store_true",
        help=(
            "After forward-backward and an optimizer step, save a resumable "
            "checkpoint, load it back into the training sub-job, and delete it."
        ),
    )
    parser.add_argument(
        "--memory-telemetry",
        action="store_true",
        help="Enable server-side training peak-memory reporting for the probe.",
    )
    parser.add_argument(
        "--debug-image-tag",
        help="Run the submitted job with a specific backend image tag.",
    )
    parser.add_argument(
        "--training-sp-size",
        type=int,
        help=(
            "Test a Ulysses sequence-parallel candidate and adjust the global "
            "batch to n_gpus / sp_size."
        ),
    )
    args = parser.parse_args()

    plans = []
    for job in args.job:
        request = build_catalog_request(
            args.catalog_dir.resolve(),
            args.model_id,
            job,
            context=args.context,
            hardware=args.hardware,
        )
        if args.memory_telemetry and job in TRAINING_JOBS:
            enable_training_memory_telemetry(request)
        if args.debug_image_tag:
            apply_debug_image_tag(request, args.debug_image_tag)
        if args.training_sp_size and job in TRAINING_JOBS:
            apply_training_sequence_parallelism(request, args.training_sp_size)
        plans.append({"job": job, "request": request})

    if not args.submit:
        print(json.dumps(plans, indent=2))
        return 0

    client = _build_client(args.poll_timeout)
    results = []
    for plan in plans:
        job = plan["job"]
        job_id: str | None = None
        active_error: BaseException | None = None
        try:
            response = client.create_job_from_body(plan["request"])
            job_id = response["job_id"]
            print(f"Submitted {job} as job {job_id}", file=sys.stderr)
            status = client.wait_for_job(job_id)
            probes = run_data_plane_probes(
                client,
                job_id,
                job,
                plan["request"],
                full_context_prefill=args.full_context_prefill,
                full_context_training=args.full_context_training,
                training_step=args.training_step,
                checkpoint_round_trip=args.checkpoint_round_trip,
            )
            results.append(
                {
                    "job": job,
                    "jobId": job_id,
                    "status": str(status.get("status") or "")
                    .lower()
                    .removeprefix("job_state_"),
                    "probes": probes,
                }
            )
        except BaseException as error:
            active_error = error
            raise
        finally:
            if job_id is not None:
                if _cancel_job(client, job_id, active_error):
                    print(f"Cancelled job {job_id}", file=sys.stderr)

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
