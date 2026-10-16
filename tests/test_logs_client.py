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

"""Client side of the Logs tab: tail-logs retries and the execution-log reader."""

import json
from pathlib import Path
from pathlib import PurePosixPath
from unittest.mock import MagicMock
from urllib.parse import unquote
from urllib.parse import urlparse

import pytest
import requests
from tenacity import wait_none

import cortex_training.client as client_module
from cortex_training.client import CortexTrainingClient


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    # A retry, if one happened, must not sleep through the test.
    monkeypatch.setattr(client_module, "wait_exponential_jitter", lambda **_kwargs: wait_none())


def _response(status, body):
    resp = requests.Response()
    resp.status_code = status
    resp.url = "http://test.local/operation"
    resp.headers["Content-Type"] = "application/json"
    resp._content = json.dumps(body).encode("utf-8")
    return resp


def _client(*responses):
    client = CortexTrainingClient(base_url="http://test.local", database="DB", schema="SCH")
    client._session = MagicMock()
    client._session.post.side_effect = list(responses)
    return client


@pytest.mark.parametrize(
    "status,message",
    [
        (409, "Cortex training operation failed: session JOB_ID is pending placement: quota_filled"),
        (409, "Cortex training operation failed: session JOB_ID has been cancelled: cancelled_by_client"),
        (404, "session not found"),
        (400, "sub_job_id JOB_ID:sampling:0 does not belong to session JOB_ID"),
    ],
)
def test_tail_logs_raises_at_once_when_the_job_is_not_running(status, message):
    client = _client(_response(status, {"message": message, "code": "510200"}), _response(200, {"entries": []}))
    with pytest.raises(requests.exceptions.HTTPError) as caught:
        client.tail_logs("JOB_ID", sub_job_id="JOB_ID:training:0", cursor="C1")
    assert caught.value.response.status_code == status
    assert client._session.post.call_count == 1  # no retries
    body = client._session.post.call_args.kwargs["json"]
    assert body["operation_type"] == "tail-logs"
    assert body["payload"] == {"cursor": "C1"}


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_tail_logs_still_retries_transient_failures(status):
    page = {"entries": [{"_raw": "ts line"}], "next_cursor": "C2", "eof": False}
    client = _client(_response(status, {"message": "busy"}), _response(200, page))
    assert client.tail_logs("JOB_ID", sub_job_id="JOB_ID:training:0", cursor="C1") == page
    assert client._session.post.call_count == 2


def test_tail_logs_predicate_keeps_the_general_rules_otherwise():
    def error(status):
        return requests.exceptions.HTTPError(response=_response(status, {}))

    assert not client_module._is_tail_logs_transient(error(404))
    assert not client_module._is_tail_logs_transient(error(409))
    assert not client_module._is_tail_logs_transient(error(400))
    assert client_module._is_tail_logs_transient(error(503))
    assert client_module._is_tail_logs_transient(requests.exceptions.ConnectionError())
    # Every other call keeps retrying 404 and 409.
    assert client_module._is_transient(error(404)) and client_module._is_transient(error(409))


class _Stage:
    """A Snowflake connection that answers experiment LIST/GET from a dict."""

    def __init__(self, run_uri, files):
        self.run_uri = run_uri
        self.files = files  # run-relative path -> bytes
        self.commands = []

    def cursor(self):
        stage = self

        class Cursor:
            rows = []

            def execute(self, statement):
                stage.commands.append(statement)
                verb, _, rest = statement.partition(" ")
                if verb == "LIST":
                    prefix = rest.strip("'").removeprefix(stage.run_uri)
                    self.rows = [(f"/versions/RUN/{p}",) for p in stage.files if p.startswith(prefix)]
                elif verb == "GET":
                    source, target = (part.strip("'") for part in rest.split(" ", 1))
                    relative = source.removeprefix(stage.run_uri)
                    out = Path(unquote(urlparse(target).path))
                    out.mkdir(parents=True, exist_ok=True)
                    (out / PurePosixPath(relative).name).write_bytes(stage.files[relative])
                    self.rows = [(relative, "DOWNLOADED")]
                return self

            def fetchall(self):
                return self.rows

            def close(self):
                pass

        return Cursor()

    def close(self):
        pass


def test_fetch_execution_logs_can_read_one_sub_jobs_execution_log(monkeypatch):
    client = CortexTrainingClient(base_url="http://test.local", database="DB", schema="SCH")
    client._session = MagicMock()
    run = MagicMock(status_code=200)
    run.json.return_value = {"experiment_name": "DB.SCH.EXP", "experiment_run_name": "RUN"}
    client._session.get.return_value = run
    run_uri = "snow://experiment/DB.SCH.EXP/versions/RUN/"
    stage = _Stage(run_uri, {
        "_logs/JOB:training:0/execution.jsonl": b'{"level": "ERROR", "msg": "boom"}\n',
        "_logs/JOB:training:0/server.log": b"server line\n",
        "_logs/JOB:sampling:0/execution.jsonl": b'{"level": "INFO", "msg": "fine"}\n',
    })
    monkeypatch.setattr(client, "_open_experiment_artifact_connection", lambda: stage)

    logs = client.fetch_execution_logs("JOB", sub_job_id="JOB:training:0", filename="execution.jsonl")

    assert logs == [{
        "sub_job_id": "JOB:training:0",
        "filename": "execution.jsonl",
        "artifact_uri": run_uri + "_logs/JOB:training:0/execution.jsonl",
        "content": '{"level": "ERROR", "msg": "boom"}\n',
    }]
    gets = [c for c in stage.commands if c.startswith("GET ")]
    assert len(gets) == 1 and "JOB:training:0/execution.jsonl" in gets[0]
    # Without filters every file is still returned.
    assert len(client.fetch_execution_logs("JOB")) == 3


def _get_client(*responses):
    client = CortexTrainingClient(base_url="http://test.local", database="DB", schema="SCH")
    client._session = MagicMock()
    client._session.get.side_effect = list(responses)
    return client


@pytest.mark.parametrize("message", [
    "Linked experiment is not found for job JOB_ID",  # the experiment-run route
    "Linked experiment is not found or not authorized for the training job",
])
def test_experiment_run_lookup_raises_at_once_when_the_experiment_cant_be_resolved(message):
    # Saved consoles, execution logs and GPU metrics all start with this lookup;
    # a retry can't make the Experiment readable, so the error shows at once.
    run = {"experiment_name": "DB.SCH.EXP", "experiment_run_name": "RUN"}
    client = _get_client(_response(500, {"message": message, "code": "510200"}), _response(200, run))
    with pytest.raises(requests.exceptions.HTTPError) as caught:
        client.get_experiment_run("JOB_ID")
    assert caught.value.response.status_code == 500
    assert client._session.get.call_count == 1
    assert client._session.get.call_args.args[0].endswith("/JOB_ID/experiment-run")


def test_experiment_run_lookup_still_retries_other_failures():
    run = {"experiment_name": "DB.SCH.EXP", "experiment_run_name": "RUN"}
    client = _get_client(_response(500, {"message": "internal error"}), _response(503, {"message": "busy"}),
                         _response(200, run))
    assert client.get_experiment_run("JOB_ID") == run
    assert client._session.get.call_count == 3
