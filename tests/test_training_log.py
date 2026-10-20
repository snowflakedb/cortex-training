# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The client's local copy of a job's training metrics (``_training_log.py``), and the TUI reading it."""

from __future__ import annotations

import base64
import json
import os
import threading
from unittest.mock import MagicMock

from cortex_training import _training_log
from cortex_training import wire
from cortex_training.client import CortexTrainingClient
from cortex_training.tui import training

JOB = "job-1"
SUB = "job-1:training:0"
CALL = {"avg_loss": 2.0419, "job_id": SUB, "metrics": {"rank": 0, "sft_model_calls": 1, "sft_packed_rows": 1}}
STEP = {"global_steps": 3.0, "grad_norm": 84.37, "job_id": SUB, "last_lr": 1e-4, "update_successful": 1.0,
        "weight_delta_norm_rank0_sample": 0.1447}


def _rows(root):
    path = _training_log.local_dir(JOB, root) / SUB / _training_log.FILE_NAME
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_each_call_and_step_is_appended_as_the_service_saves_it(tmp_path):
    log = _training_log.TrainingLog(tmp_path)
    log.record(JOB, "step", "s2", {**STEP, "global_steps": 2.0})
    log.record(JOB, "fwd_bwd", "c1", CALL)
    log.record(JOB, "fwd_bwd", "c2", CALL)
    log.record(JOB, "step", "s3", STEP)
    rows = _rows(tmp_path)
    assert [(r["op"], r["step"], r["call"], r["request_id"]) for r in rows] == [
        ("step", 2, None, "s2"), ("fwd_bwd", 3, 1, "c1"), ("fwd_bwd", 3, 2, "c2"), ("step", 3, None, "s3"),
    ]
    # Every number the result has, as it has it.
    assert rows[1]["metrics"] == {"avg_loss": 2.0419, "rank": 0.0, "sft_model_calls": 1.0, "sft_packed_rows": 1.0}
    assert rows[3]["metrics"] == {key: float(value) for key, value in STEP.items() if key != "job_id"}
    assert {r["record_version"] for r in rows} == {2} and len({r["seq"] for r in rows}) == 4


def test_calls_before_any_step_and_after_a_load_have_no_step(tmp_path):
    log = _training_log.TrainingLog(tmp_path)
    log.record(JOB, "fwd_bwd", "c1", CALL)
    log.record(JOB, "step", "s1", {**STEP, "global_steps": 1.0})
    log.forget_step(JOB)  # a load-checkpoint
    log.record(JOB, "fwd_bwd", "c2", CALL)
    log.record(JOB, "fwd_bwd", "c3", CALL)
    # No step, no call count either: it would restart at 1 in every new process.
    assert [(r["op"], r["step"], r["call"]) for r in _rows(tmp_path)] == [
        ("fwd_bwd", None, None), ("step", 1, None), ("fwd_bwd", None, None), ("fwd_bwd", None, None),
    ]


def test_each_new_process_leaves_the_call_count_empty(tmp_path):
    # The CLI runs one process per command, so each call comes from a fresh log.
    for request_id in ("c1", "c2"):
        _training_log.TrainingLog(tmp_path).record(JOB, "fwd_bwd", request_id, CALL)
    _training_log.TrainingLog(tmp_path).record(JOB, "step", "s1", {**STEP, "global_steps": 1.0})
    rows = _rows(tmp_path)
    assert [(r["step"], r["call"]) for r in rows] == [(None, None), (None, None), (1, None)]
    data = training.TrainingLoader(tmp_path / "no-stage", _training_log.local_dir(JOB, tmp_path)).load().get(SUB)
    assert data.calls == 2 and [data.step_of_call(c) for c in (1, 2)] == [1, 1]


def test_recording_never_raises_and_skips_what_it_cannot_write(tmp_path):
    log = _training_log.TrainingLog(tmp_path)
    log.record(JOB, "fwd_bwd", "c", "not a dict")
    log.record("../escape", "fwd_bwd", "c", CALL)  # not a plain id: no directory is made from it
    log.record(JOB, "generate", "g", {"x": 1.0})
    log.record(JOB, "step", "s", {"grad_norm": 1.0})  # a step row must say its step
    assert _rows(tmp_path) == []
    assert not (tmp_path / _training_log.DIR_NAME / "..").resolve().joinpath("escape").exists()


def test_a_file_stops_growing_at_its_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(_training_log, "MAX_BYTES", 400)
    log = _training_log.TrainingLog(tmp_path)
    for i in range(10):
        log.record(JOB, "fwd_bwd", f"c{i}", CALL)
    assert 1 <= len(_rows(tmp_path)) < 10


def test_the_env_turns_it_off(monkeypatch):
    monkeypatch.setenv(_training_log.DISABLE_ENV, "1")
    assert _training_log.disabled()
    assert CortexTrainingClient(base_url="http://test.local", database="DB", schema="SCH")._training_log is None
    monkeypatch.setenv(_training_log.DISABLE_ENV, "0")
    assert not _training_log.disabled()


def _client():
    client = CortexTrainingClient(base_url="http://test.local", database="DB", schema="SCH")
    client._session = MagicMock()
    return client


def test_the_client_logs_each_fwd_bwd_and_step_result_it_receives(tmp_path, monkeypatch):
    client = _client()
    client._training_log = _training_log.TrainingLog(tmp_path)
    results = {"r-call": CALL, "r-step": STEP, "r-load": {"version": 5}}
    monkeypatch.setattr(client, "get_request_status",
                        lambda job_id, request_id, cursor=None: {"status": "completed", "result": results[request_id]})
    for request_id, op in (("r-call", "fwd_bwd"), ("r-step", "step"), ("r-load", "load")):
        client._remember_training_log_op(request_id, JOB, op)
    assert client.poll_request(JOB, "r-call") == CALL  # the caller's result is unchanged
    client.poll_request(JOB, "r-step")
    client.poll_request(JOB, "r-load")
    assert [(r["op"], r["request_id"]) for r in _rows(tmp_path)] == [("fwd_bwd", "r-call"), ("step", "r-step")]
    assert client._training_log_ops == {}  # each request is logged once


def test_a_broken_log_never_fails_the_call(monkeypatch):
    client = _client()

    class Broken:
        def record(self, *args):
            raise OSError("disk full")

    client._training_log = _training_log.TrainingLog()
    monkeypatch.setattr(client._training_log, "_append", Broken().record)
    monkeypatch.setattr(client, "get_request_status",
                        lambda job_id, request_id, cursor=None: {"status": "completed", "result": CALL})
    client._remember_training_log_op("r", JOB, "fwd_bwd")
    assert client.poll_request(JOB, "r") == CALL


# ---------------------------------------------------------------- the TUI reads it first


def _stage(root, rows):
    path = root / SUB / training.FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path.write_bytes(data)
    path.with_name("." + path.name + ".manifest.json").write_text(
        json.dumps({"version": 1, "names": ["_metrics/a.gz"], "bytes": len(data)}))


def _stage_row(op, step, request_id, metrics, call=None, seq=0):
    return {"kind": "training", "record_version": 2, "sub_job_id": SUB, "seq": seq, "seq_epoch": "stage",
            "op": op, "step": step, "call": call, "request_id": request_id, "metrics": metrics}


def _stage_steps(steps, value=lambda step: float(step)):
    """The stage's copy of steps ``steps``: one call and one step row each, with request ids c<n>, s<n>."""
    rows = []
    for step in steps:
        rows.append(_stage_row("fwd_bwd", step, f"c{step}", {"avg_loss": value(step)}, call=1, seq=2 * step))
        rows.append(_stage_row("step", step, f"s{step}", {"global_steps": float(step), "grad_norm": value(step)},
                               seq=2 * step + 1))
    return rows


def _local_steps(log, steps, value=lambda step: float(step)):
    for step in steps:
        log.record(JOB, "fwd_bwd", f"c{step}", {"avg_loss": value(step), "job_id": SUB})
        log.record(JOB, "step", f"s{step}", {"global_steps": float(step), "grad_norm": value(step), "job_id": SUB})


def _grad_norms(data):
    series = {m.name: m for m in data.metrics(training.VIEW_STEP)}["grad_norm"]
    return dict(zip(series.xs, series.values))


def _loader(tmp_path):
    return training.TrainingLoader(tmp_path / "stage", _training_log.local_dir(JOB, tmp_path))


def test_the_local_log_adds_only_the_steps_the_stage_has_not_received_yet(tmp_path):
    loader = _loader(tmp_path)
    _stage(tmp_path / "stage", _stage_steps([1]))
    _local_steps(_training_log.TrainingLog(tmp_path), [1, 2, 3], value=lambda step: 100.0 + step)
    data = loader.load().get(SUB)
    # Step 1 is the stage's; 2 and 3 only the local log has so far.
    assert _grad_norms(data) == {1: 1.0, 2: 102.0, 3: 103.0} and data.calls == 3 and data.last_step == 3
    # The stage catches up: its rows replace the local ones, tags and all.
    _stage(tmp_path / "stage", _stage_steps([1, 2, 3]))
    data = loader.load().get(SUB)
    assert _grad_norms(data) == {1: 1.0, 2: 2.0, 3: 3.0} and data.calls == 3
    assert [data.step_of_call(c) for c in (1, 2, 3)] == [1, 2, 3]


def test_a_capped_local_log_never_hides_the_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(_training_log, "MAX_BYTES", 700)
    _local_steps(_training_log.TrainingLog(tmp_path), range(1, 11))
    assert 0 < len(_rows(tmp_path)) < 20  # the cap stopped it
    _stage(tmp_path / "stage", _stage_steps(range(1, 11)))
    data = _loader(tmp_path).load().get(SUB)
    assert data.last_step == 10 and len(_grad_norms(data)) == 10 and data.calls == 10


def test_steps_this_machine_did_not_poll_still_show(tmp_path):
    # Steps 1-4 ran from another machine (or were polled with get_request_status): only the stage has
    # them. This machine ran step 5, which the stage hasn't received yet.
    _stage(tmp_path / "stage", _stage_steps(range(1, 5)))
    log = _training_log.TrainingLog(tmp_path)
    log.record(JOB, "step", "s4", {"global_steps": 4.0, "job_id": SUB})  # the step it last saw
    _local_steps(log, [5])
    data = _loader(tmp_path).load().get(SUB)
    assert sorted(_grad_norms(data)) == [1, 2, 3, 4, 5] and data.calls == 5


def test_a_local_log_left_behind_after_logging_was_turned_off_hides_nothing(tmp_path):
    _local_steps(_training_log.TrainingLog(tmp_path), [1, 2])
    _stage(tmp_path / "stage", _stage_steps(range(1, 7)))  # steps 3-6 ran with the local log off
    data = _loader(tmp_path).load().get(SUB)
    assert _grad_norms(data) == {step: float(step) for step in range(1, 7)} and data.calls == 6


def test_a_result_written_twice_counts_once(tmp_path):
    log = _training_log.TrainingLog(tmp_path)
    log.record(JOB, "step", "s0", {"global_steps": 0.0, "job_id": SUB})
    log.record(JOB, "fwd_bwd", "c1", CALL)
    log.record(JOB, "fwd_bwd", "c1", CALL)  # the same request's result, polled twice
    log.record(JOB, "step", "s1", {**STEP, "global_steps": 1.0})
    assert _loader(tmp_path).load().get(SUB).calls == 1


def test_without_a_stage_copy_the_local_log_is_read_as_it_grows(tmp_path):
    loader = _loader(tmp_path)
    log = _training_log.TrainingLog(tmp_path)
    _local_steps(log, [1])
    assert loader.load().get(SUB).last_step == 1
    _local_steps(log, [2])
    data = loader.load().get(SUB)  # reads only the rows appended since
    assert data.last_step == 2 and data.calls == 2


def test_bookkeeping_a_request_never_raises_into_the_call():
    client = _client()
    client._training_log_ops = None  # anything that would raise
    client._remember_training_log_op("r", JOB, "fwd_bwd")
    client._log_training_result("r", CALL, {})


def test_threads_sharing_a_client_remember_and_log_every_request(tmp_path):
    client = _client()
    client._training_log = _training_log.TrainingLog(tmp_path)
    client._MAX_TRAINING_LOG_OPS = 8  # so eviction runs while other threads pop

    def work(n):
        for i in range(200):
            request_id = f"r{n}-{i}"
            client._remember_training_log_op(request_id, JOB, "step")
            client._log_training_result(request_id, {"global_steps": float(i), "job_id": SUB}, {})

    threads = [threading.Thread(target=work, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert client._training_log_ops == {}


def _poll(client, monkeypatch, result, request_id="r"):
    monkeypatch.setattr(client, "get_request_status",
                        lambda job_id, rid, cursor=None: {"status": "completed", "result": result})
    return client.poll_request(JOB, request_id)


def test_a_legacy_payload_b64_result_is_logged_as_its_sub_job_returned_it(tmp_path, monkeypatch):
    client = _client()
    client._training_log = _training_log.TrainingLog(tmp_path)
    payload = {"avg_loss": 1.5, "metrics": {"rank": 0}}  # no job_id inside: the envelope names it
    envelope = {"job_id": SUB, "payload_b64": base64.b64encode(wire.dumps(payload)).decode("ascii"),
                "metrics": {"transport_ms": 3.0}}
    client._forward_request_ids.add("r")
    client._remember_training_log_op("r", JOB, "fwd_bwd")
    result = _poll(client, monkeypatch, envelope)
    assert result["metrics"] == {"rank": 0, "transport_ms": 3.0}  # the caller still gets the envelope's
    (row,) = _rows(tmp_path)
    assert row["sub_job_id"] == SUB and row["metrics"] == {"avg_loss": 1.5, "rank": 0.0}


def test_an_encoded_result_is_logged_without_the_envelope_metrics(tmp_path, monkeypatch):
    client = _client()
    client._training_log = _training_log.TrainingLog(tmp_path)
    envelope = {"wire_format": wire.WIRE_FORMAT_VERSION, "encoding": "base64", "job_id": SUB,
                "payload_b64": base64.b64encode(wire.dumps({"avg_loss": 2.0})).decode("ascii"),
                "metrics": {"transport_ms": 3.0}}
    client._remember_training_log_op("r", JOB, "fwd_bwd")
    assert wire.loads(base64.b64decode(envelope["payload_b64"])) == {"avg_loss": 2.0}
    assert _poll(client, monkeypatch, envelope)["metrics"] == {"transport_ms": 3.0}
    assert [row["metrics"] for row in _rows(tmp_path)] == [{"avg_loss": 2.0}]


def test_a_result_naming_no_sub_job_of_the_job_is_skipped(tmp_path):
    log = _training_log.TrainingLog(tmp_path)
    log.record(JOB, "fwd_bwd", "c1", {"avg_loss": 1.0})  # names no sub-job
    log.record(JOB, "fwd_bwd", "c2", {"avg_loss": 1.0, "job_id": JOB})  # the job, not a sub-job
    log.record(JOB, "fwd_bwd", "c3", {"avg_loss": 1.0, "job_id": "job-2:training:0"})  # another job's
    assert not (tmp_path / _training_log.DIR_NAME / JOB).exists()
    log.record(JOB, "fwd_bwd", "c4", {"avg_loss": 1.0}, sub_job_id=SUB)  # the envelope's name
    assert [row["request_id"] for row in _rows(tmp_path)] == ["c4"]


def test_steps_and_calls_are_counted_per_sub_job(tmp_path):
    other = "job-1:training:1"
    log = _training_log.TrainingLog(tmp_path)
    log.record(JOB, "step", "s1", {"global_steps": 1.0, "job_id": SUB})
    log.record(JOB, "step", "t7", {"global_steps": 7.0, "job_id": other})
    log.record(JOB, "fwd_bwd", "c1", {"avg_loss": 1.0, "job_id": SUB})
    log.record(JOB, "fwd_bwd", "d1", {"avg_loss": 1.0, "job_id": other})
    log.forget_step(JOB, other)  # a load that targets the other sub-job only
    log.record(JOB, "fwd_bwd", "c2", {"avg_loss": 1.0, "job_id": SUB})
    log.record(JOB, "fwd_bwd", "d2", {"avg_loss": 1.0, "job_id": other})
    assert [(r["step"], r["call"]) for r in _rows(tmp_path)][1:] == [(2, 1), (2, 2)]
    path = _training_log.local_dir(JOB, tmp_path) / other / _training_log.FILE_NAME
    assert [(r["step"], r["call"]) for r in map(json.loads, path.read_text().splitlines())][1:] == [
        (8, 1), (None, None)]


def test_a_call_without_numbers_uses_no_call_number(tmp_path):
    log = _training_log.TrainingLog(tmp_path)
    log.record(JOB, "step", "s1", {"global_steps": 1.0, "job_id": SUB})
    log.record(JOB, "fwd_bwd", "c1", {"job_id": SUB, "note": "no numbers"})
    log.record(JOB, "fwd_bwd", "c2", CALL)
    assert [r["call"] for r in _rows(tmp_path) if r["op"] == "fwd_bwd"] == [1]


def test_old_and_excess_jobs_are_pruned_but_never_the_one_being_written(tmp_path):
    base = tmp_path / _training_log.DIR_NAME
    now = 1_000_000_000.0
    for name, age_days, size in (("old", 40, 10), ("mid", 3, 600), ("new", 1, 600), (JOB, 50, 10)):
        path = base / name / f"{name}:training:0" / _training_log.FILE_NAME
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x" * size)
        os.utime(path, (now - age_days * 86400, now - age_days * 86400))
    removed = _training_log.prune(tmp_path, keep=JOB, max_bytes=1000, now=now)
    # "old" is past the age limit; then the least recently written goes until the rest fit.
    assert sorted(p.name for p in removed) == ["mid", "old"]
    assert sorted(p.name for p in base.iterdir()) == sorted(["new", JOB])


def test_the_first_write_prunes_the_directory(tmp_path, monkeypatch):
    stale = tmp_path / _training_log.DIR_NAME / "stale" / "stale:training:0" / _training_log.FILE_NAME
    stale.parent.mkdir(parents=True)
    stale.write_text("{}\n")
    os.utime(stale, (0, 0))
    _training_log.TrainingLog(tmp_path).record(JOB, "fwd_bwd", "c1", CALL)
    assert not stale.exists() and len(_rows(tmp_path)) == 1


def test_requests_never_polled_are_not_remembered_forever():
    client = _client()
    for i in range(client._MAX_TRAINING_LOG_OPS + 10):
        client._remember_training_log_op(f"r{i}", JOB, "fwd_bwd")
    assert len(client._training_log_ops) == client._MAX_TRAINING_LOG_OPS
    assert "r0" not in client._training_log_ops and f"r{client._MAX_TRAINING_LOG_OPS + 9}" in client._training_log_ops
