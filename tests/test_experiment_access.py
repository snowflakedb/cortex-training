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

"""A role that can't read the job's Experiment: the client fails at once, and
the CLI says why in one error line and a hint."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock

import pytest
import requests
from snowflake.connector.errors import DatabaseError
from snowflake.connector.errors import ProgrammingError
from tenacity import wait_none

import cortex_training._cli as cli
import cortex_training.client as client_module
from cortex_training.client import CortexTrainingClient

RUN = {"experiment_name": "DB.SCH.EXP", "experiment_run_name": "RUN"}
# The service's two wordings of its 500 for a role that can't resolve the job's Experiment.
EXPERIMENT_RUN_WORDING = "Linked experiment is not found for job JOB_ID"
CHECKPOINT_WORDING = "Linked experiment is not found or not authorized for the training job"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "no-login.json"))
    monkeypatch.delenv("CORTEX_TRAINING_CONFIG", raising=False)
    # A retry, if one happened, must not sleep through the test.
    monkeypatch.setattr(client_module, "wait_exponential_jitter", lambda **_kwargs: wait_none())


def _response(status: int, body: dict) -> requests.Response:
    resp = requests.Response()
    resp.status_code, resp.url = status, "http://test.local/api"
    resp.headers["Content-Type"] = "application/json"
    resp._content = json.dumps(body).encode("utf-8")
    return resp


def _client(*get_responses) -> CortexTrainingClient:
    client = CortexTrainingClient(base_url="http://test.local", database="DB", schema="SCH")
    client._session = MagicMock()
    client._session.get.side_effect = list(get_responses)
    return client


def _run(argv: list[str], client_factory) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    base = ["--base-url", "http://test.local", "--database", "DB", "--schema", "SCH"]
    rc = cli.main(base + argv, client_factory=client_factory, stdout=stdout, stderr=stderr)
    return rc, stdout.getvalue(), stderr.getvalue()


# --- The client ---


@pytest.mark.parametrize(
    ("read", "message"),
    [
        # Saved logs and GPU metrics start with the experiment-run lookup.
        ("get_experiment_run", EXPERIMENT_RUN_WORDING),
        ("list_checkpoints", f"Training job error: {CHECKPOINT_WORDING}."),
    ],
)
def test_a_read_the_role_isnt_allowed_fails_on_the_first_500(read, message):
    client = _client(_response(500, {"code": "510200", "message": message}), _response(200, RUN))
    with pytest.raises(requests.exceptions.HTTPError):
        getattr(client, read)("JOB_ID")
    assert client._session.get.call_count == 1  # not retried: a minute of backoff saved


def test_other_failures_are_still_retried():
    client = _client(
        _response(500, {"message": "internal error"}), _response(503, {"message": "busy"}), _response(200, RUN)
    )
    assert client.get_experiment_run("JOB_ID") == RUN
    assert client._session.get.call_count == 3


# --- The CLI ---


def test_the_cli_keeps_copies_of_the_clients_markers():
    # Its own copies, so that formatting an error doesn't import the client.
    assert cli._EXPERIMENT_ACCESS_DENIED == client_module._EXPERIMENT_ACCESS_DENIED
    assert cli._LINKED_EXPERIMENT_NOT_FOUND == client_module._LINKED_EXPERIMENT_NOT_FOUND


@pytest.mark.parametrize(
    ("message", "hint"),
    [
        (EXPERIMENT_RUN_WORDING, cli._EXPERIMENT_GONE_HINT),
        (CHECKPOINT_WORDING, cli._EXPERIMENT_ACCESS_HINT),
        ("no such job", None),
    ],
)
def test_each_wording_gets_its_own_hint(message, hint):
    error = requests.exceptions.HTTPError("500 Server Error", response=_response(500, {"message": message}))
    formatted = cli._format_error(error)
    assert message in formatted  # the service's body is still shown
    if hint is None:
        assert "hint:" not in formatted
    else:
        assert formatted.endswith("\n" + hint) and "use the role that created the job" in hint


@pytest.mark.parametrize(
    ("message", "hinted"),
    [
        ("SQL compilation error:\nExperiment 'DB.SCH.EXP' does not exist or not authorized.", True),
        ("SQL access control error:\nInsufficient privileges to operate on experiment 'EXP'", True),
        # An error about a schema that only mentions experiments is not one.
        ("SQL compilation error:\nSchema 'DB.EXPERIMENTS' does not exist or not authorized.", False),
        ("SQL access control error:\nInsufficient privileges to operate on schema 'EXPERIMENTS'", False),
    ],
)
def test_refused_sql_on_the_experiment_is_one_error_line_and_a_hint(tmp_path, monkeypatch, message, hinted):
    error = ProgrammingError(msg=message, errno=2003, sqlstate="02000")
    client = _client(_response(200, RUN))
    connection = MagicMock()
    connection.cursor.return_value.execute.side_effect = error
    monkeypatch.setattr(client, "_open_experiment_artifact_connection", lambda: connection)

    rc, out, err = _run(["download-metrics", "JOB_ID", "--output-dir", str(tmp_path)], lambda _args: client)

    assert (rc, out) == (1, "")
    assert err == f"error: {error}\n" + (f"{cli._EXPERIMENT_SQL_HINT}\n" if hinted else "")


def test_a_failed_login_is_one_error_line_but_a_bug_still_raises():
    def raising(error):
        def client_factory(_args):
            raise error

        return client_factory

    login = DatabaseError(
        msg="Failed to connect to DB: ACCOUNT.snowflakecomputing.com:443", errno=250001, sqlstate="08001"
    )
    rc, out, err = _run(["get", "JOB_ID"], raising(login))
    assert (rc, out) == (1, "") and err.startswith("error: 250001 (08001): Failed to connect to DB")
    with pytest.raises(KeyError):
        _run(["get", "JOB_ID"], raising(KeyError("a bug, not a refusal")))
