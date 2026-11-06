#!/usr/bin/env python3
"""Run all catalog lifecycle pairings within a weighted parallel GPU budget."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

import requests

CATALOG = Path(__file__).resolve().parent
REPO = CATALOG.parent
sys.path[:0] = [str(CATALOG), str(REPO / "src")]

from job_lifecycle_suite import build_request
from run_lifecycle_config import pin_request, resolve_file_connection
from cortex_training.cli.commands import _capacity_by_hardware, _is_unsupported_hardware

TERMINAL = {"failed", "completed", "succeeded", "cancelled", "canceled", "terminated", "done"}
ACTIVE = {"pending", "placing", "initializing", "running"}
BUDGET_SCOPES = ("account", "sweep")


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(status):
    return str(status or "").lower().removeprefix("job_state_")


def gpu_count(job):
    total = 0
    for sub in job.get("sub_jobs") or job.get("sub_job_configs") or []:
        config = sub.get("training_config") or sub.get("inference_config") or {}
        count = config.get("n_gpus", sub.get("n_gpus"))
        # Get-job can echo whole JSON numbers as floats before effective config.
        if isinstance(count, float) and math.isfinite(count) and count.is_integer():
            count = int(count)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("Cannot determine GPU usage safely")
        total += count
    if not total:
        raise ValueError("Job has no GPU configuration")
    return total


def account_capacity(client):
    def lookup(hardware):
        try:
            return client.get_capacity(hardware)
        except Exception as exc:
            # The CLI classifier checks the payload; sweeps also require HTTP 409.
            if _is_unsupported_hardware(exc) and getattr(
                exc.response, "status_code", None
            ) != 409:
                raise RuntimeError("Capacity lookup failed without unsupported-hardware 409") from exc
            raise

    return _capacity_by_hardware(lookup)


def baseline_usage(client, *, stopping=lambda: False, on_timeout=None):
    for attempt in range(1, 11):
        if stopping():
            return None
        try:
            capacity = account_capacity(client)
        except requests.Timeout as exc:
            if on_timeout is not None:
                on_timeout(exc, attempt)
            if stopping():
                return None
            if attempt == 10:
                raise
            for _ in range(30):
                if stopping():
                    return None
                time.sleep(1)
        else:
            if stopping():
                return None
            return [], sum(
                pool["in_use_gpus"] + pool["pending_gpus"]
                for pool in capacity["capacity_by_hardware"].values()
            )


def pause_capacity_timeout(root, state, operation, exc, attempt=None):
    timestamp = now()
    sample = {
        "time": timestamp, "operation": operation, "error_type": type(exc).__name__,
        "admission_ready": False, "admission_ceiling_gpus": 0,
        "gpu_budget_scope": state.get("gpu_budget_scope", "account"),
        "owned_reserved_gpus": sum(e["n_gpus"] for e in state["entries"].values()
                                   if e.get("reserved")),
    }
    if attempt is not None:
        sample["attempt"] = attempt
    state.update(
        admission_ready=False, admission_status="paused", admission_ceiling_gpus=0,
        target_available_gpus=None, admission_pause_reasons=["capacity_read_timeout"],
        account_read_error={"time": timestamp, "error_type": sample["error_type"]},
    )
    state.setdefault("admission_wait_started_at", timestamp)
    state.setdefault("capacity_timeout_samples", []).append(sample)
    save(root / "ledger.json", state)


def available_gpus(pool):
    count = pool.get("available_gpus")
    if isinstance(count, float) and math.isfinite(count) and count.is_integer():
        count = int(count)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("Target hardware available_gpus must be a nonnegative whole number")
    return count


def owned_usage(client, root, state):
    owned = {}
    uncertain = {}
    for pair_id, entry in state["entries"].items():
        if not entry["reserved"]:
            continue
        report_path = root / "configs" / pair_id / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text())
            owned[pair_id] = {job["job_id"] for job in report.get("jobs", [])}
        else:
            owned[pair_id] = set()
    active = {}
    for job_id in sorted({job_id for ids in owned.values() for job_id in ids}):
        job = client.get_job(job_id)
        status = normalized(job.get("status"))
        if status in TERMINAL:
            continue
        if status not in ACTIVE:
            uncertain[job_id] = "Unrecognized owned job status"
            continue
        hardware = str(job.get("hardware") or "").upper()
        if hardware not in {"H200", "B200", "B300"}:
            uncertain[job_id] = "Cannot determine owned job hardware"
            continue
        try:
            count = gpu_count(job)
        except (ValueError, TypeError, AttributeError):
            uncertain[job_id] = "Cannot determine owned job GPU count"
            continue
        active[job_id] = {"hardware": hardware, "n_gpus": count}
    for pair_id, ids in owned.items():
        if not ids.intersection(active):
            uncertain[f"config:{pair_id}"] = "No recognized active owned allocation"
        elif sum(active[job_id]["n_gpus"] for job_id in ids.intersection(active)) > (
            state["entries"][pair_id]["n_gpus"]
        ):
            uncertain[f"config:{pair_id}"] = "Owned allocation exceeds its GPU reservation"
    return active, uncertain


def refresh_account_usage(client, root, state):
    started = time.monotonic()
    started_at = now()
    state.update(admission_ready=False, admission_status="checking")
    sweep_scope = state.get("gpu_budget_scope", "account") == "sweep"
    if sweep_scope:
        state["admission_ceiling_gpus"] = 0
    try:
        before, uncertain_before = owned_usage(client, root, state)
        capacity = account_capacity(client)
        after, uncertain_after = owned_usage(client, root, state)
    except requests.Timeout as exc:
        pause_capacity_timeout(root, state, "refresh_account_usage", exc)
        return False
    # Only unchanged live owned jobs bracket the sample. Capacity may still lag.
    stable = {job_id: job for job_id, job in before.items()
              if after.get(job_id) == job}
    stable_by_hardware = Counter()
    for job in stable.values():
        stable_by_hardware[job["hardware"]] += job["n_gpus"]
    external = {
        hardware: max(0, pool["in_use_gpus"] + pool["pending_gpus"]
                      - stable_by_hardware[hardware])
        for hardware, pool in capacity["capacity_by_hardware"].items()
    }
    count = sum(external.values())
    reservations = sum(e["n_gpus"] for e in state["entries"].values() if e["reserved"])
    observed = sum(pool["in_use_gpus"] + pool["pending_gpus"]
                   for pool in capacity["capacity_by_hardware"].values())
    uncertain = {**uncertain_before, **uncertain_after}
    changed = sorted((before.keys() | after.keys()) - stable.keys())
    reasons = []
    if uncertain:
        reasons.append("Owned metadata is uncertain")
    if changed:
        reasons.append("Owned allocations changed during sampling")
    if any(stable_by_hardware[hw] > pool["in_use_gpus"] + pool["pending_gpus"]
           for hw, pool in capacity["capacity_by_hardware"].items()):
        reasons.append("Capacity lags stable owned allocations")
    if any(hw not in capacity["capacity_by_hardware"] for hw in stable_by_hardware):
        reasons.append("Owned hardware missing from capacity sample")
    target_hardware = state.get("hardware", "").upper()
    target_available = None
    ceiling = state["limit"]
    if sweep_scope:
        ceiling = 0
        pool = capacity["capacity_by_hardware"].get(target_hardware)
        if pool is None:
            reasons.append("Target hardware missing from capacity sample")
        else:
            try:
                # Pending claims already reduce available_gpus in the API.
                target_available = available_gpus(pool)
                ceiling = min(
                    state["limit"], target_available + stable_by_hardware[target_hardware],
                )
            except ValueError as exc:
                reasons.append(str(exc))
        if reservations > ceiling:
            reasons.append("Owned reservations exceed current hardware admission ceiling")
    # Terminal owned allocations can linger in capacity. Discount them only
    # when confirming the permanent external floor, never for admission.
    possible_owned_lag = {}
    for pair_id, entry in state["entries"].items():
        report_path = root / "configs" / pair_id / "report.json"
        if report_path.is_file():
            for job in json.loads(report_path.read_text()).get("jobs", []):
                if job["job_id"] not in stable:
                    possible_owned_lag[job["job_id"]] = entry["n_gpus"]
    lag_gpus = sum(possible_owned_lag.values())
    duration = time.monotonic() - started
    if duration >= 30:
        reasons.append("Account sample expired before admission")
    floor = state.get("external_gpu_floor", state["baseline_gpus"])
    if not reasons:
        floor = max(floor, count - lag_gpus)
    state["external_gpu_floor"] = floor
    charged_external = max(count, floor)
    tentative = charged_external + reservations
    if not sweep_scope and observed > state["limit"]:
        reasons.append("Observed account usage exceeds GPU budget")
    if not sweep_scope and tentative > state["limit"]:
        reasons.append("Conservative accounting exceeds GPU budget")
    ready = not reasons
    if ready:
        state["baseline_jobs"] = []
        state["baseline_gpus"] = 0 if sweep_scope else charged_external
        state.pop("admission_wait_started_at", None)
    else:
        state.setdefault("admission_wait_started_at", now())
    state.update(
        admission_ready=ready, admission_status="ready" if ready else "paused",
        admission_pause_reasons=reasons, account_gpus=observed,
        conservative_total_gpus=tentative, tentative_external_gpus=count,
        charged_external_gpus=charged_external,
        admission_ceiling_gpus=ceiling, target_available_gpus=target_available,
        observed_account_over_limit=not sweep_scope and observed > state["limit"],
        peak_observed_account_gpus=max(state.get("peak_observed_account_gpus", 0), observed),
        peak_conservative_total_gpus=max(state.get("peak_conservative_total_gpus", 0), tentative),
    )
    sample = {
        "time": now(), "started_at": started_at, "duration_seconds": duration,
        "capacity": capacity, "owned_before": before,
        "owned_after": after, "stable_owned": stable,
        "stable_owned_by_hardware": dict(stable_by_hardware),
        "external_by_hardware": external, "external_jobs": [],
        "external_gpus": count, "account_gpus": observed,
        "external_gpu_floor": floor, "charged_external_gpus": charged_external,
        "possible_owned_lag_gpus": lag_gpus,
        "reserved_total": tentative, "conservative_total_gpus": tentative,
        "admitted_total_gpus": used_gpus(state),
        "admission_external_gpus": state["baseline_gpus"],
        "gpu_budget_scope": state.get("gpu_budget_scope", "account"),
        "target_hardware": target_hardware, "target_available_gpus": target_available,
        "admission_ceiling_gpus": ceiling,
        "owned_reserved_gpus": reservations, "uncertain_owned": uncertain,
        "changed_owned": changed, "admission_ready": ready, "pause_reasons": reasons,
    }
    state.setdefault("account_usage_samples", []).append(sample)
    state.pop("account_read_error", None)
    return ready


def source_paths(catalog=CATALOG):
    repo = catalog.parent
    paths = list(catalog.glob("*.py")) + [catalog / "rl-pairings.json"]
    paths += list((catalog / "models").rglob("*.json"))
    paths += list((repo / "src" / "cortex_training").rglob("*.py"))
    paths += list((repo / "recipes").glob("*.py"))
    paths += [path for path in (catalog / "README.md", catalog / "schema.json", repo / "pyproject.toml")
              if path.is_file()]
    return sorted(set(paths))


def source_hashes(catalog=CATALOG):
    return {str(path.relative_to(catalog.parent)): digest(path)
            for path in source_paths(catalog)}


def validate_connection_path(connection, catalog=CATALOG):
    if any(path.samefile(connection) for path in source_paths(catalog)):
        raise ValueError("Connection config must be outside the archived source/config files")


def snapshot_sources_match(root, hashes):
    try:
        return all(
            (root / "inputs" / relative).is_file()
            and digest(root / "inputs" / relative) == expected
            for relative, expected in hashes.items()
        )
    except (OSError, RuntimeError):
        return False


def plan(image_tag, budget, catalog=CATALOG, *, budget_scope="account", models=None, configs=None):
    if budget <= 0:
        raise ValueError("GPU budget must be positive")
    if budget_scope not in BUDGET_SCOPES:
        raise ValueError("GPU budget scope must be account or sweep")
    if models is not None and configs is not None:
        raise ValueError("Select either models or configs, not both")
    manifest = json.loads((catalog / "rl-pairings.json").read_text())
    if manifest.get("hardware", "").upper() not in {"H200", "H100"}:
        raise ValueError("Image verification currently supports Hopper pairings only")
    available_models = {pair["model"] for pair in manifest["pairings"]}
    selected = available_models
    if models is not None:
        if isinstance(models, str) or not models or any(
            not isinstance(model, str) or not model for model in models
        ):
            raise ValueError("Select at least one nonempty exact model name")
        selected = set(models)
        unknown = selected - available_models
        if unknown:
            raise ValueError("Unknown selected models: " + ", ".join(sorted(unknown)))
    selected_configs = None
    if configs is not None:
        if isinstance(configs, str) or not configs or any(
            not isinstance(config, str) or not config for config in configs
        ):
            raise ValueError("Select at least one nonempty exact config ID")
        selected_configs = set(configs)
        available_configs = {
            f'{pair["model"]}--{pair["context"]}--{pair["mode"]}'
            for pair in manifest["pairings"]
        }
        unknown = selected_configs - available_configs
        if unknown:
            raise ValueError("Unknown selected configs: " + ", ".join(sorted(unknown)))
    pairs, requests, seen = [], {}, set()
    for original in manifest["pairings"]:
        if original["model"] not in selected:
            continue
        pair = dict(original)
        pair_id = f'{pair["model"]}--{pair["context"]}--{pair["mode"]}'
        if selected_configs is not None and pair_id not in selected_configs:
            continue
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", pair_id) or pair_id in seen:
            raise ValueError("Unsafe or duplicate pairing ID")
        seen.add(pair_id)
        paths = []
        for key in ("training_config", "inference_config"):
            path = (catalog / pair[key]).resolve()
            if not path.is_relative_to(catalog.resolve()):
                raise ValueError("Config paths must be inside the catalog")
            paths.append(path)
        request = pin_request(build_request(*paths), image_tag)
        count = gpu_count(request)
        if count > budget:
            raise ValueError(f"{pair_id} needs {count} GPUs, exceeding budget {budget}")
        pair.update(
            id=pair_id, n_gpus=count,
            training_gpus=request["sub_job_configs"][0]["training_config"]["n_gpus"],
            sampling_gpus=request["sub_job_configs"][1]["inference_config"]["n_gpus"],
        )
        pairs.append(pair)
        requests[pair_id] = request
    if not pairs:
        raise ValueError("The pairing matrix is empty")
    manifest.update(
        pairings=pairs, image_tag=image_tag, max_parallel_gpus=budget,
        gpu_budget_scope=budget_scope,
        selected_models=list(dict.fromkeys(pair["model"] for pair in pairs)),
        selected_configs=[pair["id"] for pair in pairs],
    )
    return manifest, requests


def used_gpus(state):
    baseline = 0 if state.get("gpu_budget_scope", "account") == "sweep" else state["baseline_gpus"]
    return baseline + sum(
        entry["n_gpus"] for entry in state["entries"].values() if entry["reserved"]
    )


def reserve(state, pair):
    entry = state["entries"][pair["id"]]
    ceiling = state["limit"]
    if state.get("gpu_budget_scope", "account") == "sweep":
        ceiling = min(ceiling, state.get("admission_ceiling_gpus", 0))
    if (
        not state.get("admission_ready", True)
        or entry["status"] != "queued"
        or used_gpus(state) + pair["n_gpus"] > ceiling
    ):
        return False
    entry.update(status="running", reserved=True, started_at=now(), worker="sweep")
    total = used_gpus(state)
    state["peak_reserved_gpus"] = max(state["peak_reserved_gpus"], total)
    state["peak_admitted_gpus"] = max(state.get("peak_admitted_gpus", 0), total)
    state["events"].append({
        "action": "reserve", "id": pair["id"], "gpus": pair["n_gpus"],
        "total": total, "time": now(), "worker": "sweep",
    })
    return True


def verify_cleanup(client, output, timeout=600):
    report_path = output / "report.json"
    if not report_path.is_file():
        return False, {}, "Missing report; cannot rule out an unrecorded create"
    report = json.loads(report_path.read_text())
    jobs = report.get("jobs", [])
    # A failed create can leave an accepted job without a recorded response.
    attempts = sum(
        check["name"] in {"1_create_rl_job", "7_create_restored_rl_job"}
        and check.get("status") != "not_run"
        for check in report.get("checks", [])
    )
    if attempts > len(jobs):
        return False, {}, "Create has no recorded job ID; reconcile before release"
    statuses = {}
    deadline = time.monotonic() + timeout
    for job in jobs:
        job_id = job["job_id"]
        while True:
            status = normalized(client.get_job(job_id).get("status"))
            statuses[job_id] = status
            if status in TERMINAL:
                break
            if time.monotonic() >= deadline:
                return False, statuses, "Job remained nonterminal after cancellation"
            client.cancel_job(job_id)
            time.sleep(5)
        if client.list_checkpoints(job_id):
            return False, statuses, "Recorded job still has checkpoints"
    return True, statuses, None


def finish(state, pair, returncode, report, cleanup, statuses, error):
    entry = state["entries"][pair["id"]]
    passed = returncode == 0 and report.get("status") == "passed" and cleanup
    entry.update(
        status="passed" if passed else "failed", returncode=returncode,
        finished_at=now(), cleanup_verified=cleanup, job_statuses=statuses,
        error=error or report.get("error") or (
            None if passed else "Suite did not produce a passing report; see console.log"
        ),
    )
    if cleanup:
        entry["reserved"] = False
        if state.get("gpu_budget_scope", "account") == "sweep":
            # The last ceiling may credit this now-released owned allocation.
            state.update(
                admission_ready=False, admission_status="paused",
                admission_ceiling_gpus=0,
                admission_pause_reasons=["Ownership sample stale after cleanup release"],
            )
            state.setdefault("admission_wait_started_at", now())
        state["events"].append({
            "action": "release", "id": pair["id"], "gpus": pair["n_gpus"],
            "total": used_gpus(state), "time": now(), "worker": "sweep",
        })


def command(pair, root, connection, image_tag, request_sha256, connection_sha256):
    return [
        sys.executable, str(root / "inputs" / "model-catalog" / "run_lifecycle_config.py"),
        "--request", str(root / "inputs" / "requests" / f'{pair["id"]}.json'),
        "--request-sha256", request_sha256,
        "--connection-config", str(connection),
        "--connection-sha256", connection_sha256,
        "--output-dir", str(root / "configs" / pair["id"]),
        "--image-tag", image_tag,
    ]


def collect_result(child, root, state, client):
    pair, process, log = child
    log.close()
    output = root / "configs" / pair["id"]
    output.mkdir(exist_ok=True)
    (root / "configs" / f'{pair["id"]}.console.log').replace(output / "console.log")
    report = {}
    try:
        report_path = output / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        cleanup, statuses, error = verify_cleanup(client, output)
    except Exception as exc:
        cleanup, statuses, error = False, {}, f"{type(exc).__name__}: {exc}"
    finish(state, pair, process.returncode, report, cleanup, statuses, error)
    return cleanup


def collect_finished(active, root, state, client):
    blocked = False
    for pair_id, child in list(active.items()):
        if child[1].poll() is None:
            continue
        if not collect_result(child, root, state, client):
            blocked = True
        del active[pair_id]
    return blocked


def write_report(root, render):
    temporary = root / "index.html.tmp"
    temporary.write_text(render(root), encoding="utf-8")
    temporary.replace(root / "index.html")


@contextmanager
def exclusive_sweep():
    # Live and archived runners must coordinate independently of their location.
    path = Path("/tmp") / f"cortex-training-lifecycle-{os.getuid()}.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another lifecycle sweep is running for this operating-system user")
        yield


def snapshot(root, manifest, requests, connection):
    validate_connection_path(connection)
    inputs = root / "inputs"
    (inputs / "requests").mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    if manifest.get("planned_source_sha256", hashes) != hashes:
        raise RuntimeError("Source changed after planning; no jobs submitted")
    for relative in hashes:
        destination = inputs / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, destination)
        if digest(destination) != hashes[relative]:
            raise RuntimeError("Source changed while archiving; no jobs submitted")
    if source_hashes() != hashes:
        raise RuntimeError("Source changed while archiving; no jobs submitted")
    for pair_id, request in requests.items():
        save(inputs / "requests" / f"{pair_id}.json", request)
    manifest.update(
        started_at=now(),
        provenance={
            "source_sha256": hashes,
            "request_sha256": {
                pair_id: digest(inputs / "requests" / f"{pair_id}.json")
                for pair_id in requests
            },
            "connection_config_sha256": digest(connection),
            "python": sys.version,
            "packages": sorted(
                ({"name": dist.metadata["Name"], "version": dist.version}
                 for dist in metadata.distributions()), key=lambda dist: dist["name"] or "",
            ),
        },
    )
    save(root / "manifest.json", manifest)
    return hashes


def run_sweep(args, manifest, requests):
    from render_report import render

    scope = getattr(args, "budget_scope", manifest.get("gpu_budget_scope", "account"))
    if scope not in BUDGET_SCOPES:
        raise ValueError("GPU budget scope must be account or sweep")
    manifest["gpu_budget_scope"] = scope
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    hashes = snapshot(root, manifest, requests, args.connection_config)
    state = {
        "started_at": manifest["started_at"], "limit": args.max_parallel_gpus,
        "baseline_gpus": 0, "peak_reserved_gpus": 0, "baseline_jobs": [],
        "peak_admitted_gpus": 0, "peak_observed_account_gpus": 0,
        "gpu_budget_scope": scope, "hardware": manifest["hardware"].upper(),
        "admission_ceiling_gpus": 0 if scope == "sweep" else args.max_parallel_gpus,
        "admission_ready": False,
        "entries": {p["id"]: {"status": "queued", "n_gpus": p["n_gpus"],
                             "reserved": False} for p in manifest["pairings"]},
        "events": [],
    }
    active = {}
    stopping = False
    last_report = float("-inf")
    last_account_check = float("-inf")
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def stop_admission(signum, frame):
        nonlocal stopping
        stopping = True
        state["stop_reason"] = signal.Signals(signum).name
        print("Stopping admission; waiting for active suites and cleanup.", flush=True)

    for sig in previous_handlers:
        signal.signal(sig, stop_admission)
    try:
        with resolve_file_connection(args.connection_config).client as client:
            baseline = baseline_usage(
                client, stopping=lambda: stopping,
                on_timeout=lambda exc, attempt: pause_capacity_timeout(
                    root, state, "baseline_usage", exc, attempt),
            )
            if baseline is None:
                return 1
            state["baseline_jobs"], state["baseline_gpus"] = baseline
            state["initial_baseline_gpus"] = state["baseline_gpus"]
            state["initial_external_gpus"] = state["baseline_gpus"]
            state["external_gpu_floor"] = state["baseline_gpus"]
            state["peak_observed_account_gpus"] = state["baseline_gpus"]
            state["peak_conservative_total_gpus"] = state["baseline_gpus"]
            if scope == "sweep":
                state["baseline_gpus"] = 0
            state["peak_reserved_gpus"] = state["baseline_gpus"]
            state["peak_admitted_gpus"] = state["baseline_gpus"]
            if scope == "account" and any(p["n_gpus"] + state["baseline_gpus"] > state["limit"]
                   for p in manifest["pairings"]):
                raise ValueError("Existing account jobs leave insufficient GPU budget")
            while True:
                if collect_finished(active, root, state, client):
                    stopping = True
                if not stopping and time.monotonic() - last_account_check >= 30:
                    last_account_check = time.monotonic()
                    state.update(
                        admission_ready=False, admission_status="checking",
                        account_check_started_at=now(),
                    )
                    save(root / "ledger.json", state)
                    ready = refresh_account_usage(client, root, state)
                    pairs = sorted(manifest["pairings"], key=lambda p: -p["n_gpus"]) if ready else []
                    for pair in pairs:
                        if stopping:
                            break
                        if time.monotonic() - last_account_check >= 30:
                            break
                        if state["entries"][pair["id"]]["status"] != "queued":
                            continue
                        if collect_finished(active, root, state, client):
                            stopping = True
                        if stopping:
                            break
                        if (
                            source_hashes() != hashes
                            or not snapshot_sources_match(root, hashes)
                            or digest(args.connection_config) != (
                                manifest["provenance"]["connection_config_sha256"]
                            )
                        ):
                            stopping = True
                            state["error"] = "Live/archived source or connection changed; admission stopped"
                            break
                        if stopping:
                            break
                        if time.monotonic() - last_account_check >= 30:
                            break
                        if not reserve(state, pair):
                            continue
                        save(root / "ledger.json", state)
                        log = (root / "configs" / f'{pair["id"]}.console.log').open("w")
                        try:
                            expired = time.monotonic() - last_account_check >= 30
                            if stopping or expired:
                                log.close()
                                state["entries"][pair["id"]].update(status="queued", reserved=False)
                                reason = "Admission interrupted before child launch"
                                if expired and not stopping:
                                    reason = "Account sample expired before child launch"
                                    state.update(
                                        admission_ready=False, admission_status="paused",
                                        admission_pause_reasons=[reason],
                                    )
                                state["events"].append({
                                    "action": "release", "id": pair["id"], "gpus": pair["n_gpus"],
                                    "total": used_gpus(state), "time": now(), "worker": "sweep",
                                    "reason": reason,
                                })
                                break
                            process = subprocess.Popen(
                                command(
                                    pair, root, args.connection_config, args.image_tag,
                                    manifest["provenance"]["request_sha256"][pair["id"]],
                                    manifest["provenance"]["connection_config_sha256"],
                                ),
                                cwd=root / "inputs" / "model-catalog",
                                stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )
                        except Exception as exc:
                            log.close()
                            output = root / "configs" / pair["id"]
                            output.mkdir(exist_ok=True)
                            (root / "configs" / f'{pair["id"]}.console.log').replace(
                                output / "console.log"
                            )
                            finish(state, pair, None, {}, True, {}, str(exc))
                            continue
                        active[pair["id"]] = (pair, process, log)
                queued = [entry for entry in state["entries"].values()
                          if entry["status"] == "queued"]
                if scope == "account" and not active and queued and all(
                    entry["n_gpus"] + state["external_gpu_floor"] > state["limit"]
                    for entry in queued
                ):
                    stopping = True
                    state["error"] = "Confirmed external GPU floor leaves insufficient budget for queued configs"
                    state["stop_reason"] = "insufficient_gpu_budget"
                save(root / "ledger.json", state)
                if time.monotonic() - last_report >= 30:
                    write_report(root, render)
                    print(json.dumps({
                        "counts": dict(Counter(e["status"] for e in state["entries"].values())),
                        "reserved_gpus": used_gpus(state), "limit": state["limit"],
                        "gpu_budget_scope": scope,
                        "admission_ceiling_gpus": state["admission_ceiling_gpus"],
                        "account_gpus": state.get("account_gpus"),
                        "conservative_total_gpus": state.get("conservative_total_gpus"),
                        "admission_status": state.get("admission_status"),
                        "pause_reasons": state.get("admission_pause_reasons", []),
                    }), flush=True)
                    last_report = time.monotonic()
                if not active and (stopping or all(
                    e["status"] != "queued" for e in state["entries"].values()
                )):
                    break
                time.sleep(30 if not active and not state["admission_ready"] else 5)
    except BaseException as exc:
        state.update(
            error=f"{type(exc).__name__}: {exc}", error_traceback=traceback.format_exc(),
            error_at=now(), admission_ready=False, admission_status="draining",
            stop_reason="unexpected_exception",
        )
        save(root / "ledger.json", state)
        print(json.dumps({
            "admission_status": "draining", "error": state["error"],
            "error_at": state["error_at"],
        }), flush=True)
        try:
            write_report(root, render)
        except Exception as report_exc:
            state["report_error"] = f"{type(report_exc).__name__}: {report_exc}"
            save(root / "ledger.json", state)
        raise
    finally:
        # Failures in reporting/admission must not orphan running children.
        for pair, process, log in active.values():
            process.wait()
        if active:
            try:
                with resolve_file_connection(args.connection_config).client as client:
                    for child in active.values():
                        collect_result(child, root, state, client)
            except Exception as exc:
                state["cleanup_error"] = f"Cleanup reconciliation failed: {type(exc).__name__}: {exc}"
                state.setdefault("error", state["cleanup_error"])
            finally:
                for pair, process, log in active.values():
                    log.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        state["complete"] = all(
            e["status"] in {"passed", "failed"} and e.get("cleanup_verified")
            for e in state["entries"].values()
        )
        state["finished_at" if state["complete"] else "stopped_at"] = now()
        save(root / "ledger.json", state)
        try:
            write_report(root, render)
        except Exception as exc:
            state["report_error"] = f"{type(exc).__name__}: {exc}"
            save(root / "ledger.json", state)
    return 0 if (
        state["complete"] and not state.get("error") and not state.get("report_error")
        and all(e["status"] == "passed" for e in state["entries"].values())
    ) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--max-parallel-gpus", type=int, required=True)
    parser.add_argument("--budget-scope", choices=BUDGET_SCOPES, default="account")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model", action="append", metavar="MODEL",
                           help="Exact catalog model name; repeat to select multiple models (default: all)")
    selection.add_argument("--config", action="append", metavar="MODEL--CONTEXT--MODE",
                           help="Exact pairing ID; repeat to select multiple configs instead of models")
    parser.add_argument("--connection-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--previous-run", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", args.image_tag):
        parser.error("image tag must be a nonempty base tag")
    args.connection_config = args.connection_config.resolve()
    if not args.connection_config.is_file():
        parser.error("connection config does not exist")
    args.output_dir = args.output_dir or CATALOG / "results" / (
        "lifecycle-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    try:
        validate_connection_path(args.connection_config)
        planned_hashes = source_hashes()
        manifest, requests = plan(
            args.image_tag, args.max_parallel_gpus, budget_scope=args.budget_scope,
            models=args.model, configs=args.config,
        )
        manifest["planned_source_sha256"] = planned_hashes
        if args.previous_run:
            previous = args.previous_run.resolve()
            if previous.parent != args.output_dir.resolve().parent or not previous.is_dir():
                raise ValueError("Previous run must be an existing sibling output directory")
            manifest["previous_run"] = "../" + previous.name
        if args.dry_run:
            print(json.dumps(manifest, indent=2))
            return 0
        with exclusive_sweep():
            return run_sweep(args, manifest, requests)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
