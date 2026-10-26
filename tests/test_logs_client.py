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

"""Client side of the TUI's Logs and Resources tabs: one sub-job's execution
log, and the order and per-file reports of a GPU metrics download."""

import gzip
import os

import pytest

from tests import test_client as _client_tests

_make_client = _client_tests._make_client
# The stage-SQL fake of the execution-log tests (bound to a name pytest won't collect).
_FakeArtifactConnection = _client_tests.TestExecutionLogDownload.FakeArtifactConnection

RUN_URI = "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/"


def _staged(monkeypatch, root, files):
    """A client whose job's run stages ``files`` (path under ``root``: content)."""
    client = _make_client()
    connection = _FakeArtifactConnection(
        {RUN_URI + root + "/": [(f"/versions/RUN_ABC/{root}/{path}",) for path in files]},
        {RUN_URI + f"{root}/{path}": content for path, content in files.items()},
    )
    monkeypatch.setattr(client, "_experiment_run_uri", lambda _: (RUN_URI, "RUN_ABC"))
    monkeypatch.setattr(client, "_open_experiment_artifact_connection", lambda: connection)
    return client, connection


def _chunks(name):
    """Gzip chunks a second apart: three for the sampling sub-job, two for training."""
    return {
        f"{sub}/{name}.20260904-12000{i}.c{i}.gz": gzip.compress(f"{sub} {i}\n".encode())
        for sub, count in (("job-1:sampling:0", 3), ("job-1:training:0", 2))
        for i in range(count)
    }


def test_fetch_execution_logs_can_fetch_one_sub_jobs_file(monkeypatch):
    client, connection = _staged(monkeypatch, "_logs", {
        "job-1:training:0/execution.jsonl": b"training\n",
        "job-1:training:0/server.log": b"server\n",
        "job-1:sampling:0/execution.jsonl": b"sampling\n",
    })
    logs = client.fetch_execution_logs("job-1", sub_job_id="job-1:training:0", filename="execution.jsonl")
    assert [(log["sub_job_id"], log["filename"], log["content"]) for log in logs] == [
        ("job-1:training:0", "execution.jsonl", "training\n")
    ]
    assert sum(command.startswith("GET ") for command in connection.commands) == 1  # the rest are only listed


def test_download_metrics_puts_priority_first_and_reports_each_commit(tmp_path, monkeypatch):
    client, connection = _staged(monkeypatch, "_metrics", _chunks("gpu"))
    commits = []

    def on_commit(entry):
        saved = tmp_path / entry["sub_job_id"] / "gpu.jsonl"
        commits.append((entry["sub_job_id"], entry["chunk_count"], saved.read_text().count("\n")))

    result = client.download_metrics("job-1", tmp_path, priority=["job-1:training:0"], on_commit=on_commit)
    # The sub-job named first is written first, and each file is complete when it is reported.
    assert commits == [("job-1:training:0", 2, 2), ("job-1:sampling:0", 3, 3)]
    assert [entry["sub_job_id"] for entry in result] == ["job-1:sampling:0", "job-1:training:0"]  # still id order
    saved = (tmp_path / "job-1:sampling:0" / "gpu.jsonl").read_text()
    assert saved == "".join(f"job-1:sampling:0 {i}\n" for i in range(3))
    assert sum(command.startswith("GET ") for command in connection.commands) == 5  # each chunk fetched once


def test_fetch_execution_logs_still_fetches_every_file_by_default(monkeypatch):
    client, connection = _staged(monkeypatch, "_logs", {
        "job-1:training:0/execution.jsonl": b"training\n",
        "job-1:sampling:0/execution.jsonl": b"sampling\n",
    })
    logs = client.fetch_execution_logs("job-1")
    assert sorted(log["sub_job_id"] for log in logs) == ["job-1:sampling:0", "job-1:training:0"]
    assert sum(command.startswith("GET ") for command in connection.commands) == 2


def test_download_metrics_refuses_a_single_string_as_priority(tmp_path, monkeypatch):
    client, _ = _staged(monkeypatch, "_metrics", _chunks("gpu"))
    with pytest.raises(TypeError, match="priority must be an iterable"):
        client.download_metrics("job-1", tmp_path, priority="job-1:training:0")


def test_download_metrics_reports_a_sub_job_with_nothing_new_on_resume(tmp_path, monkeypatch):
    client, connection = _staged(monkeypatch, "_metrics", _chunks("gpu"))
    client.download_metrics("job-1", tmp_path, resume=True)
    gets = sum(command.startswith("GET ") for command in connection.commands)
    commits = []

    client.download_metrics("job-1", tmp_path, resume=True, on_commit=lambda entry: commits.append(entry["sub_job_id"]))
    # Every sub-job is reported, though nothing new was fetched for either.
    assert commits == ["job-1:sampling:0", "job-1:training:0"]
    assert sum(command.startswith("GET ") for command in connection.commands) == gets


def test_an_error_from_on_commit_stops_the_download_and_keeps_what_is_saved(tmp_path, monkeypatch):
    client, _ = _staged(monkeypatch, "_metrics", _chunks("gpu"))

    def on_commit(entry):
        raise RuntimeError(f"stop after {entry['sub_job_id']}")

    with pytest.raises(RuntimeError, match="stop after job-1:training:0") as raised:
        client.download_metrics("job-1", tmp_path, priority=["job-1:training:0"], on_commit=on_commit)
    assert getattr(raised.value, "download_progress_saved", False)
    assert (tmp_path / "job-1:training:0" / "gpu.jsonl").read_text().count("\n") == 2
    assert not (tmp_path / "job-1:sampling:0" / "gpu.jsonl").exists()


def _minutes(sub, count):
    """Gzip chunks a minute apart, as metrics are uploaded: chunk ``i`` holds row ``"<sub> i"``."""
    return {
        f"{sub}/gpu.20260904-{12 + i // 60:02d}{i % 60:02d}00.c{i}.gz": gzip.compress(f"{sub} {i}\n".encode())
        for i in range(count)
    }


def _rows(sub, numbers):
    return "".join(f"{sub} {i}\n" for i in numbers)


TRAINING = "job-1:training:0"
SAMPLING = "job-1:sampling:0"


def test_download_metrics_previews_the_newest_chunks_first_and_fetches_each_once(tmp_path, monkeypatch):
    client, connection = _staged(monkeypatch, "_metrics", {**_minutes(TRAINING, 40), **_minutes(SAMPLING, 5)})
    events, paths = [], []

    def on_preview(entry):
        paths.append(entry["path"])
        assert not (tmp_path / TRAINING / "gpu.jsonl").exists()  # before the sub-job's own file
        events.append(("preview", entry["sub_job_id"], entry["chunk_count"], open(entry["path"]).read()))

    client.download_metrics("job-1", tmp_path, priority=[TRAINING], preview_seconds=600, on_preview=on_preview,
                            on_commit=lambda entry: events.append(("commit", entry["sub_job_id"])))
    # Ten minutes back from the newest chunk and two more (a chunk's name is when it starts): 27 to 39, in order.
    assert events == [("preview", TRAINING, 13, _rows(TRAINING, range(27, 40))),
                      ("commit", TRAINING), ("commit", SAMPLING)]
    assert (tmp_path / TRAINING / "gpu.jsonl").read_text() == _rows(TRAINING, range(40))  # the whole file, in order
    assert sum(command.startswith("GET ") for command in connection.commands) == 45  # the preview's are reused
    assert not any(map(os.path.exists, paths))  # the preview's file goes once on_preview returns


def test_download_metrics_skips_a_preview_that_would_save_no_time(tmp_path, monkeypatch):
    # The window's chunks are most of what is new: a short job's first download, then a resume.
    client, connection = _staged(monkeypatch, "_metrics", _minutes(TRAINING, 20))
    previews = []
    client.download_metrics("job-1", tmp_path, resume=True, priority=[TRAINING], preview_seconds=600,
                            on_preview=previews.append)
    assert previews == [] and sum(command.startswith("GET ") for command in connection.commands) == 20

    client, _ = _staged(monkeypatch, "_metrics", _minutes(TRAINING, 60))  # 40 more, the newest 13 in the window
    client.download_metrics("job-1", tmp_path, resume=True, priority=[TRAINING], preview_seconds=600,
                            on_preview=previews.append)
    assert [entry["chunk_count"] for entry in previews] == [13]
    client, _ = _staged(monkeypatch, "_metrics", _minutes(TRAINING, 63))  # 3 more: all of them in the window
    client.download_metrics("job-1", tmp_path, resume=True, priority=[TRAINING], preview_seconds=600,
                            on_preview=previews.append)
    assert len(previews) == 1
    assert (tmp_path / TRAINING / "gpu.jsonl").read_text() == _rows(TRAINING, range(63))


def test_a_preview_leaves_out_a_chunk_it_cannot_read_and_the_download_fetches_it_again(tmp_path, monkeypatch):
    files = _minutes(TRAINING, 40)
    damaged = next(path for path in files if ".c35.gz" in path)
    files[damaged] = b"not gzip"
    client, connection = _staged(monkeypatch, "_metrics", files)
    previews = []

    def on_preview(entry):
        previews.append(open(entry["path"]).read())

    result = client.download_metrics("job-1", tmp_path, priority=[TRAINING], preview_seconds=600,
                                     on_preview=on_preview)
    assert previews == [_rows(TRAINING, [i for i in range(27, 40) if i != 35])]
    # Its turn fetches it again, twice as for any damaged chunk, then skips it and says so.
    assert sum(command.startswith("GET ") for command in connection.commands) == 42
    assert (tmp_path / TRAINING / "gpu.jsonl").read_text() == _rows(TRAINING, [i for i in range(40) if i != 35])
    assert any("could not be decompressed" in warning for warning in result[0]["warnings"])


def test_an_error_from_on_preview_stops_the_download(tmp_path, monkeypatch):
    client, _ = _staged(monkeypatch, "_metrics", _minutes(TRAINING, 40))

    def on_preview(entry):
        raise RuntimeError("stop at the preview")

    with pytest.raises(RuntimeError, match="stop at the preview"):
        client.download_metrics("job-1", tmp_path, priority=[TRAINING], preview_seconds=600, on_preview=on_preview)
    assert not (tmp_path / TRAINING / "gpu.jsonl").exists()


@pytest.mark.parametrize("options", [
    {"preview_seconds": 600},  # nothing to call with it
    {"on_preview": print},
    {"preview_seconds": 0, "on_preview": print},
    {"preview_seconds": True, "on_preview": print},
])
def test_download_metrics_refuses_a_preview_it_cannot_make(tmp_path, monkeypatch, options):
    client, _ = _staged(monkeypatch, "_metrics", _minutes(TRAINING, 40))
    with pytest.raises(ValueError, match="preview_seconds"):
        client.download_metrics("job-1", tmp_path, priority=[TRAINING], **options)


def _console_minutes(sub, count):
    """Console chunks a minute apart: chunk ``i`` holds the line ``"<sub> line i"``."""
    return {
        f"{sub}/console.20260904-{12 + i // 60:02d}{i % 60:02d}00.c{i}.gz": gzip.compress(f"{sub} line {i}\n".encode())
        for i in range(count)
    }


def test_download_stdout_logs_previews_the_first_named_sub_job_s_newest_output(tmp_path, monkeypatch):
    client, connection = _staged(monkeypatch, "_stdout", {**_console_minutes(TRAINING, 40),
                                                          **_console_minutes(SAMPLING, 40)})
    previews = []

    def on_preview(entry):
        previews.append((entry["sub_job_id"], open(entry["path"]).read()))

    client.download_stdout_logs("job-1", tmp_path, sub_job_ids=[TRAINING], preview_seconds=600, on_preview=on_preview)
    assert previews == [(TRAINING, "".join(f"{TRAINING} line {i}\n" for i in range(27, 40)))]
    assert (tmp_path / TRAINING / "stdout.log").read_text() == "".join(f"{TRAINING} line {i}\n" for i in range(40))
    assert sum(command.startswith("GET ") for command in connection.commands) == 40  # each once; sampling's not asked
    # Without sub_job_ids, there is no first sub-job to preview.
    previews.clear()
    client.download_stdout_logs("job-1", tmp_path / "all", preview_seconds=600, on_preview=on_preview)
    assert previews == [] and (tmp_path / "all" / SAMPLING / "stdout.log").exists()


# ─── How long the first page took (preview_ms, first_commit_ms) ───────────────


def _telemetry(monkeypatch, client):
    """``client`` recording its metrics; downloads send their successes by default."""
    from unittest.mock import MagicMock

    monkeypatch.delenv(_client_tests.nc.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)
    client._metric_emitter = MagicMock()
    return client


def _outcome(client, operation):
    [call] = [call for call in client._metric_emitter.emit.call_args_list if call.args[0] == operation]
    return call.args[1]


def _slow(seconds, then=None):
    """A callback that takes ``seconds``, as the TUI's do while they draw what they were given."""
    import time

    def callback(entry):
        time.sleep(seconds)
        if then is not None:
            then(entry)

    return callback


def test_a_metrics_download_says_when_its_preview_and_first_file_were_shown(tmp_path, monkeypatch):
    client, _ = _staged(monkeypatch, "_metrics", {**_minutes(TRAINING, 40), **_minutes(SAMPLING, 5)})
    _telemetry(monkeypatch, client)
    commits = []

    def on_commit(entry):
        commits.append(entry["sub_job_id"])
        if len(commits) == 2:  # the second file is slow: first_commit_ms is the first file's
            _slow(0.2)(entry)

    client.download_metrics("job-1", tmp_path, priority=[TRAINING], preview_seconds=600,
                            on_preview=_slow(0.05), on_commit=_slow(0.05, on_commit))

    value = _outcome(client, "download_metrics")
    assert value["success"] is True and commits == [TRAINING, SAMPLING]
    assert value["preview_ms"] >= 50  # once on_preview returned, its drawing included
    assert value["preview_ms"] + 50 <= value["first_commit_ms"] <= value["duration_ms"] - 200


def test_a_download_without_callbacks_still_says_when_its_first_file_was_whole(tmp_path, monkeypatch):
    client, _ = _staged(monkeypatch, "_metrics", _minutes(TRAINING, 3))
    _telemetry(monkeypatch, client)

    client.download_metrics("job-1", tmp_path)

    value = _outcome(client, "download_metrics")
    assert 0 <= value["first_commit_ms"] <= value["duration_ms"] and "preview_ms" not in value


def test_a_training_metrics_download_says_when_its_first_graph_was_shown(tmp_path, monkeypatch):
    # The Training metrics tab has no preview: its first graph is drawn from the first file.
    files = {f"{TRAINING}/training.20260904-1200{i:02d}.c{i}.gz": gzip.compress(f'{{"step": {i}}}\n'.encode())
             for i in range(3)}
    client, _ = _staged(monkeypatch, "_metrics", files)
    _telemetry(monkeypatch, client)

    client.download_training_metrics("job-1", tmp_path, priority=[TRAINING], on_commit=_slow(0.05))

    value = _outcome(client, "download_training_metrics")
    assert value["success"] is True and "preview_ms" not in value
    assert 50 <= value["first_commit_ms"] <= value["duration_ms"]


def test_a_saved_console_download_says_when_its_newest_page_was_shown(tmp_path, monkeypatch):
    client, _ = _staged(monkeypatch, "_stdout", _console_minutes(TRAINING, 40))
    _telemetry(monkeypatch, client)

    client.download_stdout_logs("job-1", tmp_path, sub_job_ids=[TRAINING], preview_seconds=600,
                                on_preview=_slow(0.05))

    value = _outcome(client, "download_stdout_logs")
    assert 50 <= value["preview_ms"] <= value["first_commit_ms"] <= value["duration_ms"]


def test_a_preview_the_caller_could_not_show_sends_no_preview_ms(tmp_path, monkeypatch):
    client, _ = _staged(monkeypatch, "_metrics", _minutes(TRAINING, 40))
    _telemetry(monkeypatch, client)

    def on_preview(entry):
        raise RuntimeError("could not draw it")

    with pytest.raises(RuntimeError, match="could not draw it"):
        client.download_metrics("job-1", tmp_path, priority=[TRAINING], preview_seconds=600,
                                on_preview=on_preview)

    value = _outcome(client, "download_metrics")
    assert value["success"] is False and "preview_ms" not in value and "first_commit_ms" not in value
