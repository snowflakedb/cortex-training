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

"""Client side of the Logs and Resources tabs: following a job's log, its
execution logs, and the downloads of its saved logs and GPU metrics."""

import gzip
import json
from unittest.mock import MagicMock

import pytest
import requests
from tenacity import wait_none

import cortex_training.client as client_module
from cortex_training.client import CortexTrainingClient
from tests import test_client as _client_tests

_make_client = _client_tests._make_client
# The stage-SQL fake of the execution-log tests (bound to a name pytest won't collect).
_FakeArtifactConnection = _client_tests.TestExecutionLogDownload.FakeArtifactConnection

RUN_URI = "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/"
# The two answers tail-logs gives with a 409, which only the message tells apart.
_PENDING = "Cortex training operation failed: session JOB_ID is pending placement: quota_filled"
_CANCELLED = "Cortex training operation failed: session JOB_ID has been cancelled: cancelled_by_client"


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    # A retry, if one happened, must not sleep through the test.
    monkeypatch.setattr(client_module, "wait_exponential_jitter", lambda **_kwargs: wait_none())


def _response(status, body):
    """An answer to tail-logs: ``body`` is the JSON, or the service's error message."""
    resp = requests.Response()
    resp.status_code, resp.url = status, "http://test.local/operation"
    resp.headers["Content-Type"] = "application/json"
    resp._content = json.dumps(body if isinstance(body, dict) else {"message": body, "code": "510200"}).encode()
    return resp


def _client(*responses):
    client = CortexTrainingClient(base_url="http://test.local", database="DB", schema="SCH")
    client._session = MagicMock()
    client._session.post.side_effect = list(responses)
    return client


def _stream(client, follow=True):
    # A small poll_interval, not 0: `poll_interval or self.poll_interval` would
    # read 0 as "unset" and fall back to the client's half-second default.
    return client.stream_logs("JOB_ID", sub_job_id="JOB_ID:training:0", poll_interval=0.001, follow=follow)


def test_stream_logs_waits_for_a_job_that_is_not_placed_yet():
    page = {"entries": [{"_raw": "ts first line"}], "next_cursor": "C2", "eof": False}
    client = _client(_response(409, _PENDING), _response(409, _PENDING), _response(200, page))
    assert next(_stream(client)) == {"_raw": "ts first line"}
    assert client._session.post.call_count == 3  # one request each: the 409 itself is never retried


@pytest.mark.parametrize(
    ("status", "message", "follow"),
    [
        (409, _CANCELLED, True),  # the session is over: that log is complete
        (404, "session not found", True),  # and then forgotten
        (409, _PENDING, False),  # draining the log once, there is nothing to wait for
    ],
)
def test_stream_logs_raises_anything_but_that_wait_at_once(status, message, follow):
    client = _client(_response(status, message))
    with pytest.raises(requests.exceptions.HTTPError) as caught:
        next(_stream(client, follow=follow))
    assert caught.value.response.status_code == status
    assert client._session.post.call_count == 1  # tail-logs doesn't retry them either


def test_tail_logs_still_retries_a_transient_failure():
    page = {"entries": [], "next_cursor": "C2", "eof": False}
    client = _client(_response(503, "busy"), _response(200, page))
    assert client.tail_logs("JOB_ID", sub_job_id="JOB_ID:training:0", cursor="C1") == page
    assert client._session.post.call_count == 2


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
