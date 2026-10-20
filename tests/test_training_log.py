# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The client's local copy of a job's training metrics (``_training_log.py``), and the TUI reading it."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from cortex_training import _training_log
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
    assert [(r["op"], r["step"]) for r in _rows(tmp_path)] == [("fwd_bwd", None), ("step", 1), ("fwd_bwd", None)]


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
    client._training_log_ops.update({"r-call": (JOB, "fwd_bwd"), "r-step": (JOB, "step"), "r-load": (JOB, "load")})
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
    client._training_log_ops["r"] = (JOB, "fwd_bwd")
    assert client.poll_request(JOB, "r") == CALL


# ---------------------------------------------------------------- the TUI reads it first


def _stage(root, rows):
    path = root / SUB / training.FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path.write_bytes(data)
    path.with_name("." + path.name + ".manifest.json").write_text(
        json.dumps({"version": 1, "names": ["_metrics/a.gz"], "bytes": len(data)}))


def test_the_tab_reads_the_local_log_first_and_the_stage_otherwise(tmp_path):
    stage_root, local_root = tmp_path / "stage", _training_log.local_dir(JOB, tmp_path)
    stage_row = {"kind": "training", "record_version": 2, "sub_job_id": SUB, "seq": 1, "seq_epoch": "stage",
                 "op": "step", "step": 1, "call": None, "metrics": {"grad_norm": 9.0}}
    _stage(stage_root, [stage_row])
    loader = training.TrainingLoader(stage_root, local_root)
    # No local log yet (another machine's job): the stage's copy.
    assert loader.load().get(SUB).metrics(training.VIEW_STEP)[0].values == (9.0,)

    log = _training_log.TrainingLog(tmp_path)
    log.record(JOB, "step", "s1", {**STEP, "global_steps": 1.0})
    # The local log, written the moment the result arrived, wins: no wait for the stage's batches.
    assert loader.local_sub_jobs() == {SUB}
    assert loader.load().get(SUB).metrics(training.VIEW_STEP)[0].values == (84.37,)
    log.record(JOB, "fwd_bwd", "c1", CALL)
    log.record(JOB, "step", "s2", {**STEP, "global_steps": 2.0})
    data = loader.load().get(SUB)  # reads only the rows appended since
    assert data.last_step == 2 and data.calls == 1


def test_requests_never_polled_are_not_remembered_forever():
    client = _client()
    for i in range(client._MAX_TRAINING_LOG_OPS + 10):
        client._remember_training_log_op(f"r{i}", JOB, "fwd_bwd")
    assert len(client._training_log_ops) == client._MAX_TRAINING_LOG_OPS
    assert "r0" not in client._training_log_ops and f"r{client._MAX_TRAINING_LOG_OPS + 9}" in client._training_log_ops
