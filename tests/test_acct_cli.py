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

"""``cortex-training list`` against the account-level list contract.

The CLI builds a real client, whose session talks to the fake service from
``tests/test_acct_contract.py``; only the transport is fake.
"""

from __future__ import annotations

import io
import json

import pytest

import cortex_training._cli as cli
from cortex_training import client as client_module
from tests.test_acct_contract import BASE_URL
from tests.test_acct_contract import FORBIDDEN_CODE
from tests.test_acct_contract import NOT_ENABLED
from tests.test_acct_contract import SUBMITTER_NOT_FOUND
from tests.test_acct_contract import FakeAccountService
from tests.test_acct_contract import ServiceAdapter
from tests.test_acct_contract import account_rows
from tests.test_acct_contract import answer_every_request
from tests.test_acct_contract import rest_error
from tests.test_acct_contract import server_row
from tests.test_cli import FakeClient
from tests.test_cli import _base_args


def _list(argv: list[str], service: FakeAccountService) -> tuple[int, str, str]:
    """Run ``cortex-training list ARGV`` with a real client on ``service``."""

    def build(args):
        client = cli.build_client(args, client_module.CortexTrainingClient)
        client._session.mount("https://", ServiceAdapter(service))
        return client

    stdout = io.StringIO()
    stderr = io.StringIO()
    rc = cli.main(
        ["--base-url", BASE_URL, "--database", "CORTEX_TRAINING_DB", "list", *argv],
        client_factory=build,
        stdout=stdout,
        stderr=stderr,
    )
    return rc, stdout.getvalue(), stderr.getvalue()


def _printed_ids(stdout: str) -> list[str]:
    return [job["job_id"] for job in json.loads(stdout)["jobs"]]


@pytest.fixture
def no_backoff(monkeypatch):
    """Retry at once instead of sleeping between attempts."""
    monkeypatch.setattr(
        client_module, "wait_exponential_jitter", lambda **kwargs: lambda retry_state: 0
    )


def test_list_all_prints_one_page_oldest_first():
    service = FakeAccountService(account_rows(3))

    rc, stdout, stderr = _list(["--all"], service)

    assert (rc, stderr) == (0, "")
    assert json.loads(stdout) == {"jobs": list(reversed(service.jobs))}
    assert service.queries == [{"limit": "100"}]


def test_list_all_follows_short_and_empty_pages_to_the_end():
    service = FakeAccountService(scan_budget=3)

    rc, stdout, _ = _list(["--all", "--status", "cancelled"], service)

    assert rc == 0
    assert _printed_ids(stdout) == ["job-0004"]
    assert [query.get("page_token", "")[:2] for query in service.queries] == ["", "1+", "1+", "1+"]


def test_list_all_reads_several_pages():
    service = FakeAccountService(scan_budget=2)

    rc, stdout, _ = _list(["--all"], service)

    assert rc == 0
    assert _printed_ids(stdout) == [f"job-{n:04d}" for n in reversed(range(10))]
    assert len(service.requests) == 5


def test_list_all_limit_reads_only_the_pages_it_prints():
    service = FakeAccountService(scan_budget=2)

    rc, stdout, _ = _list(["--all", "--limit", "3"], service)

    assert rc == 0
    assert _printed_ids(stdout) == ["job-0002", "job-0001", "job-0000"]
    assert [query["limit"] for query in service.queries] == ["3", "3"]


def test_list_all_status_terminal_reaches_the_service():
    service = FakeAccountService()

    rc, stdout, _ = _list(["--all", "--status", "terminal"], service)

    assert rc == 0
    assert service.queries == [{"status": "terminal", "limit": "100"}]
    assert {job["status"] for job in json.loads(stdout)["jobs"]} == {
        "CANCELLED",
        "TERMINATED",
        "FAILED",
    }


def test_list_all_status_done_sends_cancelled_and_terminated():
    service = FakeAccountService()

    rc, stdout, stderr = _list(["--all", "--status", "done,running"], service)

    assert (rc, stderr) == (0, "")
    # The service answers 400 for done; the client sends what done meant.
    assert service.queries == [{"status": "cancelled,terminated,running", "limit": "100"}]
    assert {job["status"] for job in json.loads(stdout)["jobs"]} == {
        "CANCELLED",
        "TERMINATED",
        "RUNNING",
    }


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("terminal", {"cancelled", "terminated", "failed"}),
        ("terminal,active", {"cancelled", "terminated", "failed", "pending", "placing", "initializing", "running"}),
        ("done", {"cancelled", "terminated", "legacy"}),
    ],
)
def test_list_without_all_filters_terminal_and_done_itself(status, expected):
    client = FakeClient()
    client.jobs = [
        {"job_id": word.lower(), "status": word}
        for word in (
            "PENDING",
            "PLACING",
            "INITIALIZING",
            "RUNNING",
            "FAILED",
            "CANCELLED",
            "TERMINATED",
            "UNKNOWN",
        )
    ] + [{"job_id": "legacy", "status": "DONE"}]
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["list", "--status", status],
        client_factory=lambda _args: client,
        stdout=stdout,
    )

    assert rc == 0
    assert client.status_filter is None
    assert set(_printed_ids(stdout.getvalue())) == expected


def test_list_help_explains_terminal_and_done(capsys):
    with pytest.raises(SystemExit):
        cli.main(_base_args() + ["list", "--help"])

    help_text = " ".join(capsys.readouterr().out.split())
    assert "terminal, done" in help_text
    assert "terminal is cancelled, terminated, and failed" in help_text
    assert "done is cancelled and terminated, plus, without --all, the DONE" in help_text


# ─── --user ──────────────────────────────────────────────────────────────


def _response_body(line: str) -> dict:
    assert line.startswith("response body: "), line
    return json.loads(line.removeprefix("response body: "))


def test_list_all_unknown_user_prints_a_clear_error():
    service = FakeAccountService()

    rc, stdout, stderr = _list(["--all", "--user", "nobody"], service)

    assert rc == 1
    assert stdout == ""
    first, body = stderr.splitlines()
    # The request id and the service's body, as for every other failed request.
    assert first == (
        "error: [snowflake request id: request-id] --user nobody: no active user "
        "in this account has that name (HTTP 400: submitted_by user not found or "
        "not authorized); an unquoted name means NOBODY, so quote a lowercase or "
        "mixed-case name to match it exactly: --user '\"nobody\"'"
    )
    assert _response_body(body)["message"] == SUBMITTER_NOT_FOUND
    assert service.queries == [{"submitted_by": "nobody", "limit": "100"}]


@pytest.mark.parametrize("user", ["NOBODY", '"nobody"'])
def test_list_all_unknown_user_needs_no_quoting_hint_when_already_exact(user):
    rc, _, stderr = _list(["--all", "--user", user], FakeAccountService())

    assert rc == 1
    first, body = stderr.splitlines()
    assert first == (
        f"error: [snowflake request id: request-id] --user {user}: no active user "
        "in this account has that name (HTTP 400: submitted_by user not found or "
        "not authorized)"
    )
    assert _response_body(body)["message"] == SUBMITTER_NOT_FOUND


def test_list_all_known_user_is_sent_by_sql_identifier_rules():
    service = FakeAccountService(
        [server_row(1, submitted_by="JDOE"), server_row(2, submitted_by="jdoe")],
        users=("JDOE", "jdoe"),
    )

    rc, stdout, _ = _list(["--all", "--user", '"jdoe"'], service)

    assert rc == 0
    assert _printed_ids(stdout) == ["job-0002"]
    assert service.queries == [{"submitted_by": '"jdoe"', "limit": "100"}]


def test_list_without_all_unknown_user_lists_nothing():
    client = FakeClient()
    client.jobs = [{"job_id": "j1", "submitted_by": "JDOE"}]
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["list", "--user", "nobody"],
        client_factory=lambda _args: client,
        stdout=stdout,
    )

    assert rc == 0
    assert json.loads(stdout.getvalue()) == {"jobs": []}


# ─── An unavailable route ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("gate", "first_line", "body"),
    [
        (
            "not_deployed",
            "error: [snowflake request id: request-id] account-level job listing is "
            "not available on this service yet: GET /api/v2/cortex-training "
            "answered 404 with no error body, so the route is not deployed",
            None,
        ),
        (
            "disabled",
            "error: [snowflake request id: request-id] account-level job listing is "
            "not enabled for this account (HTTP 404); Snowflake enables it per account",
            NOT_ENABLED,
        ),
        (
            "forbidden",
            "error: [snowflake request id: request-id] account-level job listing was "
            "refused (HTTP 403): Cortex Training is not enabled for this account",
            "Cortex Training is not enabled for this account",
        ),
        (
            "rest_disabled",
            "error: [snowflake request id: request-id] account-level job listing is "
            "unavailable: the Cortex Training REST API is turned off for this account "
            "(HTTP 405)",
            None,
        ),
    ],
)
def test_list_all_says_why_the_route_is_unavailable(gate, first_line, body):
    service = FakeAccountService(gate=gate)

    rc, stdout, stderr = _list(["--all", "--status", "active"], service)

    assert rc == 1
    assert stdout == ""
    lines = stderr.splitlines()
    assert lines[0] == first_line
    if body is None:
        assert len(lines) == 1
    else:
        assert lines[1].startswith("response body: ")
        assert json.loads(lines[1].removeprefix("response body: "))["message"] == body
    # Not retried.
    assert len(service.requests) == 1


def test_list_all_restarts_after_an_expired_token():
    service = FakeAccountService(account_rows(6), scan_budget=2)
    rotated = []

    def rotate_once(service):
        if not rotated:
            rotated.append(True)
            service.rotate_key()

    service.after_page = rotate_once

    rc, stdout, stderr = _list(["--all"], service)

    assert (rc, stderr) == (0, "")
    assert _printed_ids(stdout) == [f"job-{n:04d}" for n in reversed(range(6))]
    assert len(service.requests) == 5


def test_list_all_says_when_a_403_carries_no_reason():
    service = FakeAccountService()
    answer_every_request(service, 403, None)

    rc, stdout, stderr = _list(["--all"], service)

    assert (rc, stdout) == (1, "")
    assert stderr == (
        "error: [snowflake request id: request-id] account-level job listing was "
        "refused (HTTP 403) with no error message: a network policy or a proxy may "
        "have blocked the request, or Cortex Training is not enabled for this "
        "account or its trial has ended\n"
    )
    assert len(service.requests) == 1


def test_list_all_with_status_retries_a_failing_page_twice(no_backoff):
    service = FakeAccountService()
    # The service fails a filtered page when its status lookup fails.
    service.status_lookup_failure = (503, rest_error("Control Plane is unavailable"))

    rc, stdout, stderr = _list(["--all", "--status", "active"], service)

    assert (rc, stdout) == (1, "")
    assert stderr.startswith("error: [snowflake request id: request-id] 503 Server Error")
    assert len(service.requests) == 3


def test_list_all_with_status_quotes_a_refused_status_lookup():
    service = FakeAccountService()
    service.status_lookup_failure = (
        403,
        rest_error("Control Plane permission denied", FORBIDDEN_CODE),
    )

    rc, _, stderr = _list(["--all", "--status", "terminal"], service)

    assert rc == 1
    assert stderr.splitlines()[0] == (
        "error: [snowflake request id: request-id] account-level job listing was "
        "refused (HTTP 403): Control Plane permission denied"
    )
    assert len(service.requests) == 1


def test_list_all_without_status_keeps_the_clients_retries(no_backoff):
    service = FakeAccountService()
    service.failures = [503] * 10

    rc, stdout, _ = _list(["--all"], service)

    assert rc == 0
    assert _printed_ids(stdout) == [f"job-{n:04d}" for n in reversed(range(10))]
    assert len(service.requests) == 11
