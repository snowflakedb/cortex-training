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

"""CLI errors: the Experiment hints, refused SQL, and the account-list messages."""

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

BASE = ["--base-url", "http://test.local", "--database", "DB", "--schema", "SCH"]
RUN = {"experiment_name": "DB.SCH.EXP", "experiment_run_name": "RUN"}
# The service's two wordings of "this role can't resolve the job's Experiment".
EXPERIMENT_RUN_WORDING = "Linked experiment is not found for job JOB_ID"
CHECKPOINT_WORDING = "Linked experiment is not found or not authorized for the training job"
GONE_HINT = ("hint: this role can't read the job's Experiment, or it no longer exists; the job's saved "
             "logs, GPU metrics and checkpoints live there. Ask for access to the Experiment, or use the "
             "role that created the job.")
ACCESS_HINT = ("hint: this role can't read the job's Experiment, where its saved logs, GPU metrics and "
               "checkpoints live. Ask for access to the Experiment, or use the role that created the job.")
SQL_HINT = ("hint: this role has no privileges on the job's Experiment, where its saved logs and GPU "
            "metrics are read with SQL. Ask for access to the Experiment, or use the role that created "
            "the job; the checkpoints command doesn't need it.")


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "no-login.json"))
    monkeypatch.delenv("CORTEX_TRAINING_CONFIG", raising=False)
    # A retry, if one happened, must not sleep through the test.
    monkeypatch.setattr(client_module, "wait_exponential_jitter", lambda **_kwargs: wait_none())


def _response(status: int, body: dict) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status
    resp.url = "http://test.local/api/v2/databases/DB/schemas/SCH/cortex-training/JOB_ID/experiment-run"
    resp.headers["Content-Type"] = "application/json"
    resp._content = json.dumps(body).encode("utf-8")
    return resp


def _run(argv: list[str], client) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    rc = cli.main(BASE + argv, client_factory=lambda _args: client, stdout=stdout, stderr=stderr)
    return rc, stdout.getvalue(), stderr.getvalue()


def _real_client(*get_responses) -> CortexTrainingClient:
    client = CortexTrainingClient(base_url="http://test.local", database="DB", schema="SCH")
    client._session = MagicMock()
    client._session.get.side_effect = list(get_responses)
    return client


DOWNLOADS = [
    ["download-log", "JOB_ID"],
    ["download-log", "JOB_ID", "--log-type", "stdout"],
    ["download-metrics", "JOB_ID"],
]


# ─── R1-81: both wordings of the service's Experiment answer get a hint ─────


@pytest.mark.parametrize("command", DOWNLOADS)
def test_the_experiment_run_wording_gets_a_hint_at_once(tmp_path, command):
    client = _real_client(_response(500, {"code": "510200", "message": EXPERIMENT_RUN_WORDING}),
                          _response(200, RUN))
    rc, out, err = _run(command + ["--output-dir", str(tmp_path)], client)
    assert rc == 1 and out == ""
    assert client._session.get.call_count == 1, "not retried"
    assert err.startswith("error: ") and "response body: " in err and EXPERIMENT_RUN_WORDING in err
    assert err.rstrip("\n").endswith(GONE_HINT)
    assert err.count("hint:") == 1


@pytest.mark.parametrize("message, hint", [(EXPERIMENT_RUN_WORDING, GONE_HINT), (CHECKPOINT_WORDING, ACCESS_HINT)])
def test_each_wording_gets_its_own_hint(message, hint):
    error = requests.exceptions.HTTPError("500 Server Error", response=_response(500, {"message": message}))
    formatted = cli._format_error(error)
    assert formatted.endswith("\n" + hint) and formatted.count("hint:") == 1


def test_the_markers_match_the_clients():
    # The CLI keeps its own copies so that formatting an error doesn't import the client.
    assert cli._LINKED_EXPERIMENT_NOT_FOUND == client_module._LINKED_EXPERIMENT_NOT_FOUND
    assert cli._EXPERIMENT_ACCESS_DENIED == client_module._EXPERIMENT_ACCESS_DENIED
    assert CHECKPOINT_WORDING.startswith(cli._LINKED_EXPERIMENT_NOT_FOUND)
    assert EXPERIMENT_RUN_WORDING.startswith(cli._LINKED_EXPERIMENT_NOT_FOUND)


# ─── R1-82: Snowflake refusing the SQL on the Experiment ─────────────────────


class _RefusingStage:
    """A Snowflake connection whose every statement on the Experiment is refused."""

    def __init__(self, error: Exception):
        self.error = error

    def cursor(self):
        stage = self

        class Cursor:
            def execute(self, _statement):
                raise stage.error

            def close(self):
                pass

        return Cursor()

    def close(self):
        pass


DOES_NOT_EXIST = ProgrammingError(
    msg="SQL compilation error:\nExperiment 'DB.SCH.EXP' does not exist or not authorized.",
    errno=2003, sqlstate="02000")
INSUFFICIENT = ProgrammingError(
    msg="SQL access control error:\nInsufficient privileges to operate on experiment 'EXP'",
    errno=3001, sqlstate="42501")


@pytest.mark.parametrize("error", [DOES_NOT_EXIST, INSUFFICIENT], ids=["does-not-exist", "insufficient"])
@pytest.mark.parametrize("command", DOWNLOADS)
def test_refused_sql_is_one_error_line_with_a_hint_not_a_traceback(tmp_path, monkeypatch, command, error):
    client = _real_client(_response(200, RUN))
    monkeypatch.setattr(client, "_open_experiment_artifact_connection", lambda: _RefusingStage(error))
    rc, out, err = _run(command + ["--output-dir", str(tmp_path)], client)
    assert rc == 1 and out == ""
    assert err == f"error: {error}\n{SQL_HINT}\n"
    assert "Traceback" not in err


@pytest.mark.parametrize("message", [
    "SQL access control error:\nInsufficient privileges to operate on schema 'EXPERIMENTS'",
    "SQL compilation error:\nSchema 'TEAM_DB.EXPERIMENTS' does not exist or not authorized.",
    "No active warehouse selected in the current session.",
])
def test_other_sql_errors_get_no_experiment_hint(tmp_path, monkeypatch, message):
    error = ProgrammingError(msg=message, errno=2003, sqlstate="02000")
    client = _real_client(_response(200, RUN))
    monkeypatch.setattr(client, "_open_experiment_artifact_connection", lambda: _RefusingStage(error))
    rc, out, err = _run(["download-metrics", "JOB_ID", "--output-dir", str(tmp_path)], client)
    assert rc == 1 and out == ""
    assert err == f"error: {error}\n"


def test_a_connector_error_before_any_request_is_one_error_line():
    def refuse(_args):
        raise DatabaseError(msg="Failed to connect to DB: ACCOUNT.snowflakecomputing.com:443", errno=250001,
                            sqlstate="08001")

    stdout, stderr = io.StringIO(), io.StringIO()
    rc = cli.main(BASE + ["get", "JOB_ID"], client_factory=refuse, stdout=stdout, stderr=stderr)
    assert rc == 1 and stdout.getvalue() == ""
    assert stderr.getvalue().startswith("error: 250001 (08001): Failed to connect to DB")


def test_a_bug_still_raises():
    class Broken:
        def get_job(self, _job_id):
            raise KeyError("a bug, not a refusal")

    with pytest.raises(KeyError):
        cli.main(BASE + ["get", "JOB_ID"], client_factory=lambda _args: Broken(),
                 stdout=io.StringIO(), stderr=io.StringIO())


# ─── R1-103: the account-list messages ──────────────────────────────────────


def _account_list(argv: list[str], service) -> tuple[int, str, str]:
    from tests.test_acct_contract import BASE_URL
    from tests.test_acct_contract import ServiceAdapter

    def build(args):
        client = cli.build_client(args, client_module.CortexTrainingClient)
        client._session.mount("https://", ServiceAdapter(service))
        return client

    stdout, stderr = io.StringIO(), io.StringIO()
    rc = cli.main(["--base-url", BASE_URL, "--database", "CORTEX_TRAINING_DB", "list", *argv],
                  client_factory=build, stdout=stdout, stderr=stderr)
    return rc, stdout.getvalue(), stderr.getvalue()


def test_a_token_refused_again_after_the_restart_says_to_list_again():
    from tests.test_acct_contract import FakeAccountService
    from tests.test_acct_contract import account_rows

    service = FakeAccountService(account_rows(6), scan_budget=2)
    service.after_page = lambda service: service.rotate_key()
    rc, out, err = _account_list(["--all"], service)
    assert rc == 1 and out == ""
    first_line = err.splitlines()[0]
    assert first_line.endswith(
        "the service could not open page_token (HTTP 400 Invalid page_token) a second time, after the "
        "walk had started over from the first page; list the jobs again")
    assert "start again from the first page" not in err, "that was tried already"
    assert len(service.requests) == 4  # page one, its token, page one again, that token


def test_the_second_refusal_is_still_an_invalid_page_token_with_the_services_answer():
    from tests.test_acct_contract import FakeAccountService
    from tests.test_acct_contract import account_rows
    from tests.test_acct_contract import client_for

    service = FakeAccountService(account_rows(6), scan_budget=2)
    service.after_page = lambda service: service.rotate_key()
    with pytest.raises(client_module.InvalidPageToken) as raised:
        list(client_for(service).iter_jobs())
    assert raised.value.response.status_code == 400
    assert isinstance(raised.value.__cause__, client_module.InvalidPageToken)
    assert "a second time" in str(raised.value)


def test_a_single_page_refusal_still_says_to_start_again():
    from tests.test_acct_contract import FakeAccountService
    from tests.test_acct_contract import client_for

    service = FakeAccountService(scan_budget=2)
    client = client_for(service)
    token = client.list_jobs_page()["next_page_token"]
    service.rotate_key()
    with pytest.raises(client_module.InvalidPageToken, match="start again from the first page"):
        client.list_jobs_page(page_token=token)


def test_a_404_page_from_a_proxy_is_not_called_an_undeployed_route():
    from tests.test_acct_contract import FakeAccountService
    from tests.test_acct_contract import answer_every_request

    service = FakeAccountService()
    answer_every_request(service, 404, b"<html><body>Not Found (proxy)</body></html>")
    rc, out, err = _account_list(["--all"], service)
    assert rc == 1 and out == ""
    assert ("answered 404 without the service's not-enabled error; the route may not be deployed yet, or a "
            "proxy in front of the service answered") in err
    assert "so the route is not deployed" not in err


def test_an_empty_404_still_says_the_route_is_not_deployed():
    from tests.test_acct_contract import FakeAccountService
    from tests.test_acct_contract import answer_every_request

    service = FakeAccountService()
    answer_every_request(service, 404, b"")
    rc, out, err = _account_list(["--all"], service)
    assert rc == 1
    assert "answered 404 with no error body, so the route is not deployed" in err
