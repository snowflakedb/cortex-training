#!/usr/bin/env python3
"""Exercise a training/sampling config pair through checkpoint restart."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable
from uuid import uuid4

# Always exercise this checkout's SDK, even when another version is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smoke_test_catalog import _build_forward_backward_probe_payload
from smoke_test_catalog import _checkpoint_id
from smoke_test_catalog import _run_forward_backward_probe
from smoke_test_catalog import _run_optimizer_step_probe
from smoke_test_catalog import build_forward_backward_probe_spec
from validate_catalog import examples_utils

CHECKS = (
    "1_create_rl_job",
    "1_sampling_probe",
    "2_train_two_steps",
    "3_weight_sync",
    "3_sampling_after_sync",
    "4_train_two_steps",
    "5_save_checkpoints",
    "6_cancel_original_job",
    "7_create_restored_rl_job",
    "7_restored_sampling_probe",
    "8_train_two_steps",
    "9_cancel_restored_job",
)


def build_request(training_config: Path, inference_config: Path) -> dict[str, Any]:
    """Compose explicit files using the examples' RL config contract."""
    utils = examples_utils()
    request = utils.make_config(
        training_config=training_config,
        inference_config=inference_config,
        rl=True,
    )
    _, info = utils.load_catalog_model(request["sub_job_configs"][0]["model_name"])
    replay = info["architecture"] == "moe"
    if replay:
        request = utils.make_config(
            training_config=training_config,
            inference_config=inference_config,
            rl=True,
            router_replay=True,
        )
    for sub in request["sub_job_configs"]:
        config = sub.get("training_config") or sub["inference_config"]
        if sub.get("source_checkpoint_info"):
            raise ValueError("base configs must not already restore a checkpoint")
        length = config.get("max_seq_len")
        if isinstance(length, bool) or not isinstance(length, int) or length < 3:
            raise ValueError("each config's max_seq_len must be an integer >= 3")
        router = config.get("router_replay") or {}
        if replay and not router.get("enabled"):
            raise ValueError("MoE lifecycle tests require enabled router replay")
        if replay and router.get("mode", "strict") != "strict":
            raise ValueError("MoE lifecycle tests require strict router replay")
        if (
            sub["job_type"] == "training"
            and config.get("load_optimizer_states") is False
        ):
            raise ValueError("checkpoint restart requires optimizer-state loading")
    if replay and (
        request["sub_job_configs"][0]["training_config"]["max_seq_len"]
        != request["sub_job_configs"][1]["inference_config"]["max_seq_len"]
    ):
        raise ValueError("full-context router replay requires matching max_seq_len")
    if replay:
        training = request["sub_job_configs"][0]["training_config"]
        # Catalog SFT defaults may use a scalar-only fused head; GRPO needs logprobs.
        if training.get("ep_size", 1) > 1:
            options = training.setdefault("model_options", {})
            options.setdefault("lm_head", {})["cross_entropy"] = False
            if "lm_head.cross_entropy" in options:
                options["lm_head.cross_entropy"] = False
            if "fused_cross_entropy" in (training.get("prime_rl") or {}):
                training["prime_rl"]["fused_cross_entropy"] = False
            if "fused_cross_entropy" in training:
                training["fused_cross_entropy"] = False
        else:
            # EP1 reads this setting only through the legacy block.
            training.setdefault("prime_rl", {})["fused_cross_entropy"] = False
    # Validate batch construction before allocating GPUs, without a large frame.
    _build_forward_backward_probe_payload(request, "rl-full", False)
    return request


def restore_request(
    request: dict[str, Any],
    source_job_id: str,
    checkpoints: dict[str, str],
) -> dict[str, Any]:
    restored = deepcopy(request)
    # Sampling needs HF assets; training needs the resumable optimizer shards.
    for sub in restored["sub_job_configs"]:
        sub["source_checkpoint_info"] = {
            "source_job_id": source_job_id,
            "checkpoint_id": checkpoints[sub["job_type"]],
        }
    return restored


class LifecycleSuite:
    """The same nine-phase sequence for every supplied config pair."""

    def __init__(
        self,
        client,
        request: dict[str, Any],
        output_dir: Path,
        *,
        cancel_timeout: float = 300.0,
        weight_sync_bucket_size: int | None = None,
    ):
        self.client = client
        self.request = deepcopy(request)
        self.training = self.request["sub_job_configs"][0]["training_config"]
        self.sampling = self.request["sub_job_configs"][1]["inference_config"]
        self.router_replay = bool(
            (self.training.get("router_replay") or {}).get("enabled")
        )
        self.output_dir = output_dir
        self.cancel_timeout = cancel_timeout
        self.weight_sync_bucket_size = weight_sync_bucket_size
        self.active_jobs: list[str] = []
        checks = list(CHECKS)
        if self.router_replay:
            checks.insert(checks.index("1_sampling_probe"), "1_router_replay_bootstrap")
            checks.insert(
                checks.index("7_restored_sampling_probe"), "7_router_replay_bootstrap"
            )
        self.report: dict[str, Any] = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "request": self.request,
            "jobs": [],
            "checkpoints": {},
            "targets": {
                "training_sequence_tokens": self.training["max_seq_len"],
                "sampling_sequence_tokens": self.sampling["max_seq_len"],
                "router_replay": self.router_replay,
            },
            "checks": [{"name": name, "status": "not_run"} for name in checks],
        }

    def _write_report(self) -> None:
        self.report["active_jobs"] = self.active_jobs[:]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "report.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _check(self, name: str, operation: Callable[[], dict]) -> dict:
        check = next(
            (item for item in self.report["checks"] if item["name"] == name), None
        )
        if check is None:
            check = {"name": name}
            self.report["checks"].append(check)
        check["status"] = "running"
        self._write_report()
        print(f"START {name}", flush=True)
        started = time.monotonic()
        try:
            result = operation()
            check.update(status="passed", result=result)
            return result
        except BaseException as exc:
            check.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            check["seconds"] = round(time.monotonic() - started, 3)
            self._write_report()
            print(f"{check['status'].upper()} {name} ({check['seconds']}s)", flush=True)

    def _create(self, request: dict[str, Any]) -> dict:
        created = self.client.create_job_from_body(deepcopy(request))
        job_id = created["job_id"]
        self.active_jobs.append(job_id)
        self.report["jobs"].append({"job_id": job_id, "request": deepcopy(request)})
        self._write_report()
        print(f"Job {job_id}: waiting for training and sampling", flush=True)
        job = self.client.wait_for_job(job_id)
        sub_jobs = job.get("sub_jobs") or []
        kinds = sorted(
            str(sub["job_type"]).lower().removeprefix("job_type_") for sub in sub_jobs
        )
        if kinds != ["sampling", "training"]:
            raise RuntimeError(
                f"expected training and sampling sub-jobs, got {kinds!r}"
            )
        return {"job_id": job_id, "status": job["status"]}

    def _train_pair(self, job_id: str, first_step: int) -> dict:
        calls = []
        for expected_step in (first_step, first_step + 1):
            forward = (
                self._replay_forward(job_id, expected_step)
                if self.router_replay
                else _run_forward_backward_probe(
                    self.client, job_id, "rl-full", self.request, True
                )
            )
            step = _run_optimizer_step_probe(self.client, job_id, self.request)
            if step["globalSteps"] != expected_step:
                raise RuntimeError(
                    f"expected global_steps={expected_step}, got {step['globalSteps']!r}; "
                    "checkpoint state must continue, not reset"
                )
            calls.append({"forward_backward": forward, "step": step})
        return {"calls": calls}

    def _generate(self, job_id: str) -> dict:
        if not self.router_replay:
            return self._sample(job_id)
        sample_ids = [f"lifecycle-{uuid4().hex}"]
        result = self._sample(job_id, sample_ids)
        result["discard"] = examples_utils().discard_router_replay(
            self.client, job_id, sample_ids
        )
        return result

    def _sample(self, job_id: str, sample_ids: list[str] | None = None) -> dict:
        prompt_tokens = self.sampling["max_seq_len"] - 1
        params = {"max_tokens": 1, "temperature": 0.0, "ignore_eos": True}
        count = len(sample_ids) if sample_ids is not None else 1
        if sample_ids is not None:
            params = examples_utils().sampling_params_with_sample_ids(
                params, sample_ids
            )
        request_id = self.client.generate(
            job_id,
            prompts=[[1] * prompt_tokens for _ in range(count)],
            sampling_params=params,
        )
        results = self.client.poll_request(job_id, request_id).get("results")
        if not isinstance(results, list) or len(results) != count:
            raise RuntimeError("generate returned an invalid result count")
        if any(not isinstance(result, dict) for result in results):
            raise RuntimeError("generate returned an invalid result row")
        token_ids = [result.get("token_ids") for result in results]
        if any(
            not isinstance(tokens, list)
            or len(tokens) != 1
            or isinstance(tokens[0], bool)
            or not isinstance(tokens[0], int)
            or tokens[0] < 0
            for tokens in token_ids
        ):
            raise RuntimeError("generate must return exactly one token per row")
        summary = {
            "operation": "generate",
            "requestId": request_id,
            "resultCount": len(results),
            "promptTokens": prompt_tokens,
            "sequenceTokens": prompt_tokens + 1,
            "tokenIds": token_ids,
        }
        if sample_ids is not None:
            summary["sampleIds"] = sample_ids
        return summary

    def _replay_forward(self, job_id: str, expected_step: int) -> dict:
        utils = examples_utils()
        sample_ids = [
            f"lifecycle-{uuid4().hex}" for _ in range(self.training["train_batch_size"])
        ]
        sampled = self._check(
            f"router_replay_generate_step_{expected_step}",
            lambda: self._sample(job_id, sample_ids),
        )
        sequences = [
            utils.sequence_from_rollout(
                [1] * sampled["promptTokens"], tokens, advantage=1.0
            )
            for tokens in sampled["tokenIds"]
        ]
        kwargs, context = utils.collate(
            sequences, pad_token_id=0, max_seq_len=self.training["max_seq_len"]
        )
        # Reuse the dummy GRPO processing contract, but train the captured tokens.
        processing = build_forward_backward_probe_spec(self.request, "rl-full")[
            "processing"
        ]
        result = utils.forward_backward(
            self.client,
            job_id,
            kwargs,
            context=context,
            processing=processing,
            rr_sample_ids=sample_ids,
            rr_discard=True,
            router_replay_sampling_job_id=f"{job_id}:sampling:0",
        )
        try:
            loss = float(result["avg_loss"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("router replay returned invalid avg_loss") from exc
        if not math.isfinite(loss):
            raise RuntimeError(f"router replay returned non-finite avg_loss: {loss!r}")
        return {
            "operation": "forward-backward",
            "avgLoss": loss,
            "sequenceTokens": self.training["max_seq_len"],
            "globalTokens": self.training["max_seq_len"] * len(sample_ids),
            "routerReplay": {
                "sampleIds": sample_ids,
                "samplingJobId": f"{job_id}:sampling:0",
                "discard": True,
            },
        }

    def _sync(self, job_id: str) -> dict:
        sampling_config = self.request["sub_job_configs"][1]["inference_config"]
        kwargs: dict[str, Any] = {}
        if sampling_config.get("peft_config"):
            kwargs["weight_format"] = "lora"
        if self.weight_sync_bucket_size is not None:
            kwargs["bucket_size"] = self.weight_sync_bucket_size
        request_id = self.client.weight_sync(
            job_id,
            source_sub_job_id=f"{job_id}:training:0",
            target_sub_job_ids=[f"{job_id}:sampling:0"],
            **kwargs,
        )
        self.client.poll_request(job_id, request_id)
        return {"request_id": request_id}

    def _save(self, job_id: str) -> dict:
        checkpoints = {}
        for kind, checkpoint_type in (
            ("training", "resumable"),
            ("sampling", "weights-only"),
        ):
            request_id = self.client.save(job_id, checkpoint_type=checkpoint_type)
            result = self.client.poll_request(job_id, request_id)
            checkpoint_id = _checkpoint_id(result)
            checkpoints[kind] = checkpoint_id
            self.report["checkpoints"][kind] = {
                "checkpoint_id": checkpoint_id,
                "checkpoint_type": checkpoint_type,
                "source_job_id": job_id,
            }
            self._write_report()
            listed = self.client.list_checkpoints(job_id)
            checkpoint = next(
                (cp for cp in listed if cp.get("checkpoint_id") == checkpoint_id), None
            )
            if (
                checkpoint is None
                or checkpoint.get("checkpoint_type") != checkpoint_type
            ):
                raise RuntimeError(
                    f"saved {checkpoint_type} checkpoint {checkpoint_id} not in checkpoint list"
                )
        return checkpoints

    def _cancel(self, job_id: str, *, cleanup: bool = False) -> dict:
        if cleanup:
            job = self.client.get_job(job_id)
            status = str(job.get("status", "")).lower().removeprefix("job_state_")
            if status in ("failed", "completed", "succeeded", "cancelled", "canceled"):
                self.active_jobs.remove(job_id)
                return {"job_id": job_id, "status": status}
        self.client.cancel_job(job_id)
        deadline = time.monotonic() + self.cancel_timeout
        while time.monotonic() < deadline:
            job = self.client.get_job(job_id)
            status = str(job.get("status", "")).lower().removeprefix("job_state_")
            if status in ("cancelled", "canceled"):
                self.active_jobs.remove(job_id)
                return {"job_id": job_id, "status": status}
            if status in ("failed", "completed", "succeeded"):
                if cleanup:
                    self.active_jobs.remove(job_id)
                    return {"job_id": job_id, "status": status}
                raise RuntimeError(f"job {job_id} ended as {status}, not cancelled")
            time.sleep(
                min(self.client.poll_interval, max(0, deadline - time.monotonic()))
            )
        raise TimeoutError(f"job {job_id} did not cancel within {self.cancel_timeout}s")

    def _delete_checkpoint(self, checkpoint: dict[str, Any]) -> dict:
        if self.active_jobs:
            raise RuntimeError("checkpoint retained because jobs are still active")
        job_id = checkpoint["source_job_id"]
        checkpoint_id = checkpoint["checkpoint_id"]
        self.client.delete_checkpoint(job_id, checkpoint_id)
        if any(
            cp.get("checkpoint_id") == checkpoint_id
            for cp in self.client.list_checkpoints(job_id)
        ):
            raise RuntimeError(
                f"checkpoint {checkpoint_id} still listed after deletion"
            )
        return {
            "source_job_id": job_id,
            "checkpoint_id": checkpoint_id,
            "status": "deleted",
        }

    def _collect_diagnostics(self) -> None:
        diagnostics = []
        for job in self.report["jobs"]:
            job_id = job["job_id"]
            entry: dict[str, Any] = {"job_id": job_id}
            try:
                files = self.client.download_execution_logs(
                    job_id, self.output_dir / "execution-logs"
                )
                entry["files"] = [
                    {"path": file["saved_path"], "bytes": file["bytes"]}
                    for file in files
                ]
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                print(f"WARNING execution logs for {job_id}: {exc}", file=sys.stderr)
            diagnostics.append(entry)
        self.report["diagnostics"] = diagnostics

    def run(self) -> dict[str, Any]:
        try:
            first = self._check("1_create_rl_job", lambda: self._create(self.request))
            source = first["job_id"]
            if self.router_replay:
                self._check(
                    "1_router_replay_bootstrap",
                    lambda: examples_utils().bootstrap_router_replay(
                        self.client, source
                    ),
                )
            self._check("1_sampling_probe", lambda: self._generate(source))
            self._check("2_train_two_steps", lambda: self._train_pair(source, 1))
            self._check("3_weight_sync", lambda: self._sync(source))
            self._check("3_sampling_after_sync", lambda: self._generate(source))
            self._check("4_train_two_steps", lambda: self._train_pair(source, 3))
            checkpoints = self._check("5_save_checkpoints", lambda: self._save(source))
            self._check("6_cancel_original_job", lambda: self._cancel(source))
            restored_body = restore_request(self.request, source, checkpoints)
            restored = self._check(
                "7_create_restored_rl_job", lambda: self._create(restored_body)
            )
            target = restored["job_id"]
            if self.router_replay:
                self._check(
                    "7_router_replay_bootstrap",
                    lambda: examples_utils().bootstrap_router_replay(
                        self.client, target
                    ),
                )
            self._check("7_restored_sampling_probe", lambda: self._generate(target))
            self._check("8_train_two_steps", lambda: self._train_pair(target, 5))
            self._check("9_cancel_restored_job", lambda: self._cancel(target))
            self.report["status"] = "passed"
            return self.report
        except BaseException as exc:
            self.report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            cleanup_errors = []
            cleanup_failure = None
            for job_id in self.active_jobs[:]:
                try:
                    self._check(
                        f"cleanup_cancel_{job_id}",
                        lambda: self._cancel(job_id, cleanup=True),
                    )
                except Exception as exc:
                    cleanup_errors.append(f"{job_id}: {exc}")
            for kind, checkpoint in self.report["checkpoints"].items():
                try:
                    self._check(
                        f"cleanup_delete_{kind}_checkpoint",
                        lambda: self._delete_checkpoint(checkpoint),
                    )
                except Exception as exc:
                    cleanup_errors.append(f"{checkpoint['checkpoint_id']}: {exc}")
            self.report["active_jobs"] = self.active_jobs[:]
            self.report["finished_at"] = datetime.now(timezone.utc).isoformat()
            if cleanup_errors:
                self.report["cleanup_errors"] = cleanup_errors
                if self.report["status"] != "failed":
                    cleanup_failure = RuntimeError(
                        "cleanup failed: " + "; ".join(cleanup_errors)
                    )
                    self.report.update(
                        status="failed",
                        error=f"{type(cleanup_failure).__name__}: {cleanup_failure}",
                    )
            if self.report["status"] == "failed":
                self._collect_diagnostics()
            self._write_report()
            if cleanup_failure is not None:
                raise cleanup_failure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--inference-config", type=Path, required=True)
    parser.add_argument("--connection-config", type=Path)
    parser.add_argument(
        "--submit", action="store_true", help="Allocate GPUs and run the suite."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--poll-timeout", type=float)
    parser.add_argument("--cancel-timeout", type=float, default=300.0)
    parser.add_argument("--weight-sync-bucket-size", type=int)
    args = parser.parse_args()
    if args.cancel_timeout <= 0 or (
        args.poll_timeout is not None and args.poll_timeout <= 0
    ):
        parser.error("timeouts must be positive")
    if args.weight_sync_bucket_size is not None and args.weight_sync_bucket_size <= 0:
        parser.error("weight-sync bucket size must be positive")
    request = build_request(args.training_config, args.inference_config)
    if not args.submit:
        print(json.dumps(request, indent=2))
        return 0
    from cortex_training.connection import resolve_connection

    overrides = {} if args.poll_timeout is None else {"poll_timeout": args.poll_timeout}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = args.output_dir or Path(__file__).resolve().parent / "results" / stamp
    # Existing evidence must not be overwritten by a rerun.
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Report: {output_dir / 'report.json'}", flush=True)
    connection = resolve_connection(
        config=str(args.connection_config) if args.connection_config else None,
        overrides=overrides,
    )
    with connection.client as client:
        suite = LifecycleSuite(
            client,
            request,
            output_dir,
            cancel_timeout=args.cancel_timeout,
            weight_sync_bucket_size=args.weight_sync_bucket_size,
        )
        suite.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
