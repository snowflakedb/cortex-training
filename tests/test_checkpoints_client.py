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

"""The client's retry policy for the service's Experiment-access 500."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest
import requests

from cortex_training import client as nc

MARKER_MESSAGE = "Linked experiment is not found or not authorized for the training job"


def _response(status: int, body) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status
    resp.reason = "Internal Server Error" if status == 500 else "Error"
    resp.url = "http://test.local/api"
    resp.headers["Content-Type"] = "application/json"
    resp._content = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    return resp


def _http_error(status: int, body) -> requests.exceptions.HTTPError:
    return requests.exceptions.HTTPError(f"{status} Server Error", response=_response(status, body))


def _client(max_retries: int = 10) -> nc.CortexTrainingClient:
    client = nc.CortexTrainingClient(
        base_url="http://test.local", database="CORTEX_TRAINING_DB", schema="PUBLIC", max_retries=max_retries
    )
    client._session = MagicMock()
    return client


@pytest.mark.parametrize(
    "body",
    [
        {"message": MARKER_MESSAGE},
        {"code": "500", "message": f"Training job error: {MARKER_MESSAGE}."},
        {"error": {"details": json.dumps({"message": MARKER_MESSAGE})}},  # wrapped by a gateway
        MARKER_MESSAGE.encode("utf-8"),  # not JSON at all
    ],
)
def test_the_experiment_access_500_is_not_transient(body):
    assert not nc._is_transient(_http_error(500, body))


def test_other_failures_keep_their_retry_policy():
    assert nc._is_transient(_http_error(500, {"message": "internal error"}))
    assert nc._is_transient(_http_error(503, {"message": "overloaded"}))
    assert nc._is_transient(_http_error(404, {"message": "job not found"}))
    assert nc._is_transient(requests.exceptions.ConnectionError("reset"))
    assert not nc._is_transient(_http_error(400, {"message": "bad request"}))
    # A response whose body can't be read keeps the status-based answer.
    unreadable = MagicMock()
    unreadable.status_code = 500
    type(unreadable).text = property(lambda self: (_ for _ in ()).throw(RuntimeError("closed")))
    assert nc._is_transient(requests.exceptions.HTTPError("500", response=unreadable))


def test_list_checkpoints_fails_on_the_first_experiment_500():
    client = _client()
    client._session.get.side_effect = [_response(500, {"message": MARKER_MESSAGE})] * 11
    started = time.monotonic()
    with pytest.raises(requests.exceptions.HTTPError) as info:
        client.list_checkpoints("job-1")
    assert time.monotonic() - started < 5
    assert client._session.get.call_count == 1  # no retries: a minute of backoff saved
    assert info.value.response.json() == {"message": MARKER_MESSAGE}
    assert client._session.get.call_args.args[0].endswith("/cortex-training/job-1/checkpoints")


def test_export_checkpoint_fails_on_the_first_experiment_500():
    client = _client()
    client._session.post.side_effect = [_response(500, {"message": MARKER_MESSAGE})] * 11
    with pytest.raises(requests.exceptions.HTTPError):
        client.export_checkpoint("job-1", "cp_1")
    assert client._session.post.call_count == 1
    assert client._session.post.call_args.args[0].endswith("/cortex-training/job-1/checkpoints/cp_1:export")


def test_experiment_run_lookup_fails_on_the_first_experiment_500():
    # Saved logs and GPU metrics resolve the run this way first.
    client = _client()
    client._session.get.side_effect = [_response(500, {"message": MARKER_MESSAGE})] * 11
    with pytest.raises(requests.exceptions.HTTPError):
        client.get_experiment_run("job-1")
    assert client._session.get.call_count == 1


def test_a_plain_500_is_still_retried():
    client = _client(max_retries=1)
    client._session.get.side_effect = [
        _response(500, {"message": "internal error"}),
        _response(200, {"checkpoints": [{"checkpoint_id": "cp_1"}]}),
    ]
    assert client.list_checkpoints("job-1") == [{"checkpoint_id": "cp_1"}]
    assert client._session.get.call_count == 2


def test_export_checkpoint_returns_the_service_response():
    export = {
        "checkpoint_id": "cp_1",
        "files": [{"filename": "model.safetensors", "url": "https://storage.example/obj?sig=x", "size_bytes": 3}],
        "expires_at": "2026-07-20T23:05:00Z",
    }
    client = _client()
    client._session.post.return_value = _response(200, export)
    assert client.export_checkpoint("job-1", "cp_1") == export


# ─── reads never create the database ─────────────────────────────────────────

SCHEMA_MISSING = {"code": "517602", "message": "Schema OTHER_DB.PUBLIC is not found or not authorized"}


def _statement_posts(client) -> list:
    return [c for c in client._session.post.call_args_list if c.args and c.args[0].endswith("/api/v2/statements")]


def test_listing_checkpoints_never_creates_the_database():
    # A command copied into another account must not create the job's database there.
    client = _client(max_retries=0)
    client._session.get.return_value = _response(400, SCHEMA_MISSING)
    with pytest.raises(requests.exceptions.HTTPError) as info:
        client.list_checkpoints("job-1")
    assert info.value.response.json() == SCHEMA_MISSING  # the service's own answer
    assert _statement_posts(client) == []
    assert client._session.get.call_count == 1


def test_exporting_a_checkpoint_never_creates_the_database():
    client = _client(max_retries=0)
    client._session.post.return_value = _response(400, SCHEMA_MISSING)
    with pytest.raises(requests.exceptions.HTTPError):
        client.export_checkpoint("job-1", "cp_1")
    assert _statement_posts(client) == []
    assert [c.args[0].rsplit("/", 1)[-1] for c in client._session.post.call_args_list] == ["cp_1:export"]


def test_the_default_send_path_still_creates_a_missing_database():
    # The control for the two tests above: the same answer on a route that may create.
    client = _client(max_retries=0)
    client._session.get.side_effect = [_response(400, SCHEMA_MISSING), _response(200, {"ok": True})]
    client._session.post.return_value = _response(200, {"data": []})
    client._send("GET", f"{client._prefix}/job-1")
    assert [c.kwargs["json"]["statement"] for c in _statement_posts(client)] == [
        "CREATE DATABASE IF NOT EXISTS CORTEX_TRAINING_DB"]
