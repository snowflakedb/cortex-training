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

"""Contract tests for the account-level job list, ``GET /api/v2/cortex-training``.

The client talks to :class:`FakeAccountService` through a transport adapter
mounted on its real session, so ``requests`` encodes every URL and the client's
own retry, timeout and error code runs unchanged. The fake answers the way the
service does (see ``docs/reference/rest-api.md`` section 5.3): filters and
status groups, sealed page tokens bound to the filters, a per-call scan budget
that makes short and empty pages, UNKNOWN rows, and the 403/404/405 gates.
"""

from __future__ import annotations

import copy
import hashlib
import http.client
import json
import pickle
import re
from collections.abc import Callable
from datetime import date
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import parse_qs
from urllib.parse import urlsplit

import pytest
import requests

import cortex_training
from cortex_training import client as client_module

CortexTrainingClient = client_module.CortexTrainingClient
AccountJobListUnavailable = client_module.AccountJobListUnavailable
InvalidPageToken = client_module.InvalidPageToken

BASE_URL = "https://ACCOUNT.snowflakecomputing.com"
# The route as sent: requests lowercases the host.
SENT_ROUTE = "https://account.snowflakecomputing.com/api/v2/cortex-training"

# The service's messages for this route.
NOT_ENABLED = "Account-level job listing is not enabled for this account"
INVALID_SCOPE = "Invalid database or schema scope"
SCHEMA_WITHOUT_DATABASE = "schema must be given together with database"
SUBMITTER_NOT_FOUND = "submitted_by user not found or not authorized"
INVALID_STATUS = (
    "status must be a comma-separated list of pending, placing, initializing, "
    "running, failed, cancelled, terminated, active, terminal"
)
INVALID_PAGE_TOKEN = "Invalid page_token"
PAGE_TOKEN_MISMATCH = "page_token does not match the request filters"
STATUS_UNAVAILABLE = "Job status unavailable: Control Plane is unavailable"

# Placeholder error codes. The service sends numeric codes, which the client
# does not read: it goes by the HTTP status and the message.
INVALID_REQUEST_CODE = "000400"
NOT_ENABLED_CODE = "000404"
FORBIDDEN_CODE = "000403"

STATUS_VALUES = {
    "pending": {"PENDING"},
    "placing": {"PLACING"},
    "initializing": {"INITIALIZING"},
    "running": {"RUNNING"},
    "failed": {"FAILED"},
    "cancelled": {"CANCELLED"},
    "terminated": {"TERMINATED"},
    "active": {"PENDING", "PLACING", "INITIALIZING", "RUNNING"},
    "terminal": {"CANCELLED", "TERMINATED", "FAILED"},
}

# One newest-first walk over the account. The service lists by its own job id,
# which follows creation order.
_STATUSES = [
    "RUNNING",
    "PENDING",
    "FAILED",
    "UNKNOWN",
    "CANCELLED",
    "TERMINATED",
    "PLACING",
    "RUNNING",
    "FAILED",
    "TERMINATED",
]
_LOCATIONS = [
    ("CORTEX_TRAINING_DB", "PUBLIC"),
    ("CORTEX_TRAINING_DB", "EXPERIMENTS"),
    ("ml", "Exp 1"),
]


def server_row(
    n: int,
    status: str = "RUNNING",
    *,
    database: str = "CORTEX_TRAINING_DB",
    schema: str = "PUBLIC",
    created_at: str | None = None,
    submitted_by: str = "JDOE",
) -> dict:
    """A row as the account-level list renders one: a job record plus its location."""
    job_id = f"job-{n:04d}"
    created_at = created_at or (
        datetime(2026, 10, 1, tzinfo=timezone.utc) - timedelta(hours=n)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    sub_job = {
        "job_type": "TRAINING",
        "status": status,
        "model_name": "Qwen/Qwen3-1.7B",
        "training_config": {"max_seq_len": 2048, "n_gpus": 8},
    }
    row = {
        "job_id": job_id,
        "status": status,
        "created_at": created_at,
        "submitted_by": submitted_by,
        "owner_role": "ML_ENGINEER",
        "comment": None,
        "sub_jobs": [sub_job],
        "database_name": database,
        "schema_name": schema,
    }
    if status == "UNKNOWN":
        # The service could not read the live status: no sub-job ids, a reason,
        # and the creation time it recorded.
        row["reason"] = STATUS_UNAVAILABLE
    else:
        row["updated_at"] = created_at
        sub_job["sub_job_id"] = f"{job_id}:training:0"
    return row


def account_rows(count: int = 10) -> list[dict]:
    return [
        server_row(
            n,
            _STATUSES[n % len(_STATUSES)],
            database=_LOCATIONS[n % len(_LOCATIONS)][0],
            schema=_LOCATIONS[n % len(_LOCATIONS)][1],
        )
        for n in range(count)
    ]


def rest_error(message: str, code: str = INVALID_REQUEST_CODE) -> dict:
    return {"code": code, "error_code": code, "message": message, "request_id": "request-id"}


class _BadRequest(Exception):
    pass


class _Answer(Exception):
    """Answer with this status and body instead of a page."""

    def __init__(self, status: int, body: Any = None):
        super().__init__(status)
        self.status = status
        self.body = body


def _identifier(value: str | None) -> str | None:
    """SQL identifier rules: unquoted folds to uppercase, double-quoted is exact."""
    if not value:
        return None
    if len(value) > 1 and value[0] == value[-1] == '"':
        return value[1:-1].replace('""', '"')
    return value.upper()


def _instant(value: str) -> datetime:
    text = value[:-1] + "+00:00" if value[-1:] in ("Z", "z") else value
    return datetime.fromisoformat(text)


def _created_bound(name: str, value: str | None) -> datetime | None:
    """A bare date is midnight UTC; anything longer needs a 'T' and an offset."""
    if not value:
        return None
    try:
        if len(value) == 10:
            day = date.fromisoformat(value)
            return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        if value[10:11] not in ("T", "t"):
            raise ValueError(value)
        moment = _instant(value[:10] + "T" + value[11:])
        if moment.tzinfo is None:
            raise ValueError(value)
        return moment
    except ValueError:
        raise _BadRequest(f"{name} must be an RFC 3339 timestamp or a YYYY-MM-DD date")


class FakeAccountService:
    """``GET /api/v2/cortex-training``, answered the way the service answers it.

    ``jobs`` is the account newest first; a row whose ``status`` is ``UNKNOWN``
    is one whose live status the service could not read. ``gate`` names the
    answer of a gate in front of the route, as in ``AccountJobListUnavailable``.
    ``failures`` is consumed one entry per request before anything else: an
    exception is raised by the transport, an int is answered as that status.
    ``status_lookup_failure``, a ``(status, body)`` pair, is how a request
    with a status filter is answered once it passes validation: the service
    asks the control plane for live statuses then, and a failed lookup fails
    the page. ``after_page`` runs after each page served.
    """

    def __init__(
        self,
        jobs: list[dict] | None = None,
        *,
        users: tuple[str, ...] = ("JDOE", "ASMITH"),
        scan_budget: int = 5000,
        gate: str | None = None,
    ):
        self.jobs = account_rows() if jobs is None else jobs
        self.users = set(users)
        self.scan_budget = scan_budget
        self.gate = gate
        self.key = 1
        self.failures: list[BaseException | int] = []
        self.status_lookup_failure: tuple[int, Any] | None = None
        self.after_page: Callable[[FakeAccountService], None] | None = None
        self.requests: list[tuple[str, Any]] = []

    # ─── What the client sent ─────────────────────────────────────────────

    @property
    def urls(self) -> list[str]:
        return [url for url, _ in self.requests]

    @property
    def timeouts(self) -> list[Any]:
        return [timeout for _, timeout in self.requests]

    @property
    def queries(self) -> list[dict[str, str]]:
        """Each request's query, decoded as a servlet decodes it ('+' is a space)."""
        return [_decoded_query(url) for url in self.urls]

    def rotate_key(self) -> None:
        """Change the key that seals page tokens, as a key rotation does."""
        self.key += 1

    # ─── Serving ──────────────────────────────────────────────────────────

    def respond(self, request: requests.PreparedRequest, timeout: Any) -> requests.Response:
        self.requests.append((request.url, timeout))
        if self.failures:
            failure = self.failures.pop(0)
            if isinstance(failure, BaseException):
                raise failure
            return _response(request, failure, rest_error("service unavailable"))
        url = urlsplit(request.url)
        if request.method != "GET" or url.path != "/api/v2/cortex-training":
            return _response(request, 404)
        if self.gate == "rest_disabled":
            return _response(request, 405)
        if self.gate == "forbidden":
            return _response(
                request,
                403,
                rest_error("Cortex Training is not enabled for this account", FORBIDDEN_CODE),
            )
        if self.gate == "not_deployed":
            return _response(request, 404)
        if self.gate == "disabled":
            return _response(request, 404, rest_error(NOT_ENABLED, NOT_ENABLED_CODE))
        try:
            body = self._page(_decoded_query(request.url))
        except _BadRequest as exc:
            return _response(request, 400, rest_error(str(exc)))
        except _Answer as answer:
            return _response(request, answer.status, answer.body)
        if self.after_page is not None:
            self.after_page(self)
        return _response(request, 200, body)

    def _page(self, query: dict[str, str]) -> dict:
        after = _created_bound("created_after", query.get("created_after"))
        before = _created_bound("created_before", query.get("created_before"))
        if after is not None and before is not None and after >= before:
            raise _BadRequest("created_after must be earlier than created_before")
        database = _identifier(query.get("database"))
        schema = _identifier(query.get("schema"))
        if schema is not None and database is None:
            raise _BadRequest(SCHEMA_WITHOUT_DATABASE)
        try:
            limit = int(query.get("limit") or 100)
        except ValueError:
            raise _BadRequest("limit must be an integer")
        if not 1 <= limit <= 1000:
            raise _BadRequest("limit must be between 1 and 1000")
        statuses = None
        if query.get("status"):
            statuses = set()
            for word in query["status"].split(","):
                if word.strip().lower() not in STATUS_VALUES:
                    raise _BadRequest(INVALID_STATUS)
                statuses |= STATUS_VALUES[word.strip().lower()]
        submitted_by = _identifier(query.get("submitted_by"))
        fingerprint = hashlib.sha256(
            json.dumps(
                [
                    database,
                    schema,
                    None if statuses is None else sorted(statuses),
                    submitted_by,
                    None if after is None else after.timestamp(),
                    None if before is None else before.timestamp(),
                ]
            ).encode()
        ).hexdigest()[:16]
        position = self._open(query["page_token"], fingerprint) if query.get("page_token") else 0
        if database is not None and not any(
            row["database_name"] == database
            and (schema is None or row["schema_name"] == schema)
            for row in self.jobs
        ):
            raise _BadRequest(INVALID_SCOPE)
        if submitted_by is not None and submitted_by not in self.users:
            raise _BadRequest(SUBMITTER_NOT_FOUND)
        if statuses is not None and self.status_lookup_failure is not None:
            raise _Answer(*self.status_lookup_failure)

        page: list[dict] = []
        examined = 0
        while position < len(self.jobs) and len(page) < limit and examined < self.scan_budget:
            row = self.jobs[position]
            created = _instant(row["created_at"])
            if after is not None and created < after - timedelta(seconds=60):
                # Ids follow creation order, so the walk ends past created_after.
                position = len(self.jobs)
                break
            position += 1
            examined += 1
            if (
                (database is None or row["database_name"] == database)
                and (schema is None or row["schema_name"] == schema)
                and (submitted_by is None or row.get("submitted_by") == submitted_by)
                and (after is None or created >= after)
                and (before is None or created < before)
                # A job whose status could not be read never matches a filter.
                and (statuses is None or row["status"] in statuses)
            ):
                page.append(row)
        body: dict[str, Any] = {"jobs": page}
        if position < len(self.jobs):
            # Sealed, opaque, and full of characters a URL must encode.
            body["next_page_token"] = f"{self.key}+{position}/{fingerprint}="
        return body

    def _open(self, token: str, fingerprint: str) -> int:
        match = re.fullmatch(r"([0-9]+)\+([0-9]+)/([0-9a-f]+)=", token)
        if match is None or int(match[1]) != self.key:
            raise _BadRequest(INVALID_PAGE_TOKEN)
        if match[3] != fingerprint:
            raise _BadRequest(PAGE_TOKEN_MISMATCH)
        return int(match[2])


def _decoded_query(url: str) -> dict[str, str]:
    query = parse_qs(urlsplit(url).query, keep_blank_values=True)
    return {key: values[-1] for key, values in query.items()}


def _response(request: requests.PreparedRequest, status: int, body: Any = None) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.reason = http.client.responses.get(status, "")
    response.request = request
    response.url = request.url
    response.encoding = "utf-8"
    response.headers["x-snowflake-request-id"] = "request-id"
    if body is None:
        response._content = b""
    else:
        response._content = json.dumps(body).encode("utf-8")
        response.headers["Content-Type"] = "application/json"
    return response


class ServiceAdapter(requests.adapters.BaseAdapter):
    """A transport that hands each prepared request to a fake service."""

    def __init__(self, service: FakeAccountService):
        super().__init__()
        self.service = service

    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
        return self.service.respond(request, timeout)

    def close(self):
        pass


def client_for(service: FakeAccountService, **kwargs: Any) -> CortexTrainingClient:
    client = CortexTrainingClient(
        base_url=BASE_URL, database="CORTEX_TRAINING_DB", schema="PUBLIC", **kwargs
    )
    client._session.mount("https://", ServiceAdapter(service))
    return client


@pytest.fixture
def no_backoff(monkeypatch):
    """Retry at once instead of sleeping between attempts."""
    monkeypatch.setattr(
        client_module, "wait_exponential_jitter", lambda **kwargs: lambda retry_state: 0
    )


def _ids(jobs: list[dict]) -> list[str]:
    return [job["job_id"] for job in jobs]


# ─── Route and URL encoding ──────────────────────────────────────────────


def test_reads_the_fixed_account_route_whatever_the_endpoint():
    service = FakeAccountService()
    client = client_for(service, endpoint="training-v2")

    page = client.list_jobs_page()

    assert service.urls == [f"{SENT_ROUTE}?limit=100"]
    assert _ids(page["jobs"]) == _ids(service.jobs)
    # Per-job calls keep the configured endpoint.
    assert client.for_job(page["jobs"][0])._prefix.endswith("/training-v2")


def test_url_encodes_every_filter_exactly():
    service = FakeAccountService(
        [server_row(1, database="ml", schema="Exp 1", submitted_by="j doe")],
        users=("j doe",),
    )
    client = client_for(service)

    page = client.list_jobs_page(
        database='"ml"',
        schema='"Exp 1"',
        status=["active", "FAILED"],
        submitted_by='"j doe"',
        created_after="2026-09-01T09:30:00+02:00",
        created_before=datetime(2026, 10, 2, 7, 30, 0, 250000, tzinfo=timezone.utc),
        page_size=50,
    )

    assert service.urls == [
        f"{SENT_ROUTE}?database=%22ml%22&schema=%22Exp+1%22"
        "&status=active%2CFAILED&submitted_by=%22j+doe%22"
        "&created_after=2026-09-01T07%3A30%3A00Z"
        "&created_before=2026-10-02T07%3A30%3A00.250000Z&limit=50"
    ]
    # What the service reads back: '+' is a space, so no timestamp may carry one.
    assert service.queries == [
        {
            "database": '"ml"',
            "schema": '"Exp 1"',
            "status": "active,FAILED",
            "submitted_by": '"j doe"',
            "created_after": "2026-09-01T07:30:00Z",
            "created_before": "2026-10-02T07:30:00.250000Z",
            "limit": "50",
        }
    ]
    assert _ids(page["jobs"]) == ["job-0001"]


def test_page_token_is_sent_back_url_encoded_and_unchanged():
    service = FakeAccountService(scan_budget=4)
    client = client_for(service)

    first = client.list_jobs_page(status="terminal")
    second = client.list_jobs_page(status="terminal", page_token=first["next_page_token"])

    token = first["next_page_token"]
    assert set("+/=") <= set(token)
    assert service.urls[1] == (
        f"{SENT_ROUTE}?status=terminal&limit=100&page_token="
        + token.replace("+", "%2B").replace("/", "%2F").replace("=", "%3D")
    )
    assert service.queries[1]["page_token"] == token
    # Four rows examined per call: the second page continues where the first stopped.
    assert _ids(first["jobs"]) == ["job-0002"]
    assert _ids(second["jobs"]) == ["job-0004", "job-0005"]


# ─── Rows and pages ──────────────────────────────────────────────────────


def test_rows_are_returned_as_the_service_renders_them():
    service = FakeAccountService()
    client = client_for(service)

    page = client.list_jobs_page()

    assert page == {"jobs": service.jobs, "next_page_token": None}
    statuses = {job["status"] for job in page["jobs"]}
    assert statuses == {"RUNNING", "PENDING", "FAILED", "UNKNOWN", "CANCELLED", "TERMINATED", "PLACING"}
    for job in page["jobs"]:
        assert job["database_name"] and job["schema_name"]


def test_for_job_addresses_each_listed_job_in_its_own_schema():
    service = FakeAccountService()
    client = client_for(service)

    prefixes = {
        (job["database_name"], job["schema_name"]): client.for_job(job)._prefix
        for job in client.iter_jobs()
    }

    assert prefixes == {
        ("CORTEX_TRAINING_DB", "PUBLIC"): (
            f"{BASE_URL}/api/v2/databases/CORTEX_TRAINING_DB/schemas/PUBLIC/cortex-training"
        ),
        ("CORTEX_TRAINING_DB", "EXPERIMENTS"): (
            f"{BASE_URL}/api/v2/databases/CORTEX_TRAINING_DB/schemas/EXPERIMENTS/cortex-training"
        ),
        ("ml", "Exp 1"): (
            f"{BASE_URL}/api/v2/databases/%22ml%22/schemas/%22Exp%201%22/cortex-training"
        ),
    }


def test_unknown_rows_come_back_only_without_a_status_filter():
    service = FakeAccountService()
    client = client_for(service)

    unfiltered = list(client.iter_jobs())
    active = list(client.iter_jobs(status="active"))

    unknown = [job for job in unfiltered if job["status"] == "UNKNOWN"]
    assert _ids(unknown) == ["job-0003"]
    assert unknown[0]["reason"] == STATUS_UNAVAILABLE
    assert unknown[0]["created_at"] == "2026-09-30T21:00:00Z"
    assert "sub_job_id" not in unknown[0]["sub_jobs"][0]
    # Under any status filter the service leaves such a job out, uncounted.
    assert _ids(active) == ["job-0000", "job-0001", "job-0006", "job-0007"]
    assert "UNKNOWN" not in {job["status"] for job in active}


def test_short_and_empty_pages_that_carry_a_token_are_followed():
    service = FakeAccountService(scan_budget=3)
    client = client_for(service)

    pages = []
    token = None
    while True:
        page = client.list_jobs_page(status="cancelled", page_token=token)
        pages.append(page)
        token = page["next_page_token"]
        if token is None:
            break

    # Each call examines three rows; only one row in the account matches.
    assert [_ids(page["jobs"]) for page in pages] == [[], ["job-0004"], [], []]
    assert [page["next_page_token"] is not None for page in pages] == [True, True, True, False]
    assert _ids(client.iter_jobs(status="cancelled")) == ["job-0004"]


def test_iter_jobs_reads_a_page_only_when_iterated():
    service = FakeAccountService(scan_budget=2)
    client = client_for(service)

    jobs = client.iter_jobs(page_size=2)

    assert service.requests == []
    assert next(jobs)["job_id"] == "job-0000"
    assert len(service.requests) == 1
    assert _ids(jobs) == _ids(service.jobs)[1:]
    assert len(service.requests) == 5


def test_a_page_after_the_last_has_no_token():
    service = FakeAccountService(account_rows(4))
    client = client_for(service)

    first = client.list_jobs_page(page_size=2)
    last = client.list_jobs_page(page_size=3, page_token=first["next_page_token"])

    # page_size may change between pages; the token binds only the filters.
    assert _ids(first["jobs"]) == ["job-0000", "job-0001"]
    assert last == {"jobs": service.jobs[2:], "next_page_token": None}


def test_a_token_sent_with_other_filters_is_the_services_400():
    service = FakeAccountService(scan_budget=2)
    client = client_for(service)
    token = client.list_jobs_page(status="active")["next_page_token"]

    with pytest.raises(requests.HTTPError) as raised:
        client.list_jobs_page(status="terminal", page_token=token)

    # A caller bug, not an expired token: it is not mapped or retried.
    assert not isinstance(raised.value, InvalidPageToken)
    assert raised.value.response.status_code == 400
    assert raised.value.response.json()["message"] == PAGE_TOKEN_MISMATCH
    assert len(service.requests) == 2


def test_created_after_bounds_the_walk():
    service = FakeAccountService(scan_budget=2)
    client = client_for(service)

    jobs = list(client.iter_jobs(created_after=datetime(2026, 9, 30, 21, tzinfo=timezone.utc)))

    assert _ids(jobs) == ["job-0000", "job-0001", "job-0002", "job-0003"]
    # Two full pages and the one that runs past created_after; nothing older is read.
    assert len(service.requests) == 3


# ─── A1: status words and groups ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("status", "sent", "expected"),
    [
        ("terminal", "terminal", ["job-0002", "job-0004", "job-0005", "job-0008", "job-0009"]),
        # done is not a status the service takes any more; its older meaning is sent.
        ("done", "cancelled,terminated", ["job-0004", "job-0005", "job-0009"]),
        ("DONE", "cancelled,terminated", ["job-0004", "job-0005", "job-0009"]),
        (["done", "terminated", "Failed"], "cancelled,terminated,Failed",
         ["job-0002", "job-0004", "job-0005", "job-0008", "job-0009"]),
        ("active, done", "active,cancelled,terminated",
         ["job-0000", "job-0001", "job-0004", "job-0005", "job-0006", "job-0007", "job-0009"]),
    ],
)
def test_status_groups_and_done(status, sent, expected):
    service = FakeAccountService()
    client = client_for(service)

    jobs = list(client.iter_jobs(status=status))

    assert service.queries == [{"status": sent, "limit": "100"}]
    assert _ids(jobs) == expected


def test_an_unknown_status_reaches_the_service_and_is_its_400():
    service = FakeAccountService()
    client = client_for(service)

    with pytest.raises(requests.HTTPError) as raised:
        list(client.iter_jobs(status="paused"))

    assert raised.value.response.status_code == 400
    assert raised.value.response.json()["message"] == INVALID_STATUS
    assert len(service.requests) == 1


# ─── A2: submitted_by ────────────────────────────────────────────────────


def test_an_unknown_user_is_the_services_400_and_creates_nothing():
    service = FakeAccountService()
    client = client_for(service)

    with pytest.raises(requests.HTTPError) as raised:
        list(client.iter_jobs(submitted_by="nobody"))

    assert raised.value.response.status_code == 400
    assert raised.value.response.json()["message"] == SUBMITTER_NOT_FOUND
    assert service.queries == [{"submitted_by": "nobody", "limit": "100"}]
    # The message says "not found or not authorized", which on a per-schema
    # route would run CREATE DATABASE: here nothing else is sent.
    assert client._is_database_not_found(raised.value.response)
    assert service.urls == [f"{SENT_ROUTE}?submitted_by=nobody&limit=100"]
    assert client._db_ensured is False


def test_submitted_by_follows_sql_identifier_rules():
    service = FakeAccountService(
        [server_row(1, submitted_by="JDOE"), server_row(2, submitted_by="jdoe")],
        users=("JDOE", "jdoe"),
    )
    client = client_for(service)

    assert _ids(client.iter_jobs(submitted_by="jdoe")) == ["job-0001"]
    assert _ids(client.iter_jobs(submitted_by='"jdoe"')) == ["job-0002"]


# ─── A3: the route is unavailable ────────────────────────────────────────


@pytest.mark.parametrize(
    ("gate", "status_code", "reason", "message"),
    [
        (
            "not_deployed",
            404,
            "not_deployed",
            "account-level job listing is not available on this service yet: "
            "GET /api/v2/cortex-training answered 404 with no error body, so the "
            "route is not deployed",
        ),
        (
            "disabled",
            404,
            "disabled",
            "account-level job listing is not enabled for this account (HTTP 404); "
            "Snowflake enables it per account",
        ),
        (
            "forbidden",
            403,
            "forbidden",
            # The service's own reason, quoted.
            "account-level job listing was refused (HTTP 403): Cortex Training is "
            "not enabled for this account",
        ),
        (
            "rest_disabled",
            405,
            "rest_disabled",
            "account-level job listing is unavailable: the Cortex Training REST "
            "API is turned off for this account (HTTP 405)",
        ),
    ],
)
@pytest.mark.parametrize("call", ["list_jobs_page", "iter_jobs"])
def test_each_unavailable_answer_raises_one_typed_error_without_retrying(
    gate, status_code, reason, message, call
):
    service = FakeAccountService(gate=gate)
    client = client_for(service)

    with pytest.raises(AccountJobListUnavailable) as raised:
        if call == "iter_jobs":
            list(client.iter_jobs(status="active"))
        else:
            client.list_jobs_page(status="active")

    error = raised.value
    assert error.reason == reason
    assert error.reason in AccountJobListUnavailable.REASONS
    assert str(error) == f"[snowflake request id: request-id] {message}"
    assert error.response.status_code == status_code
    # What callers caught before still catches it.
    assert isinstance(error, requests.HTTPError)
    assert isinstance(error, OSError)
    assert cortex_training.AccountJobListUnavailable is AccountJobListUnavailable
    assert client.max_retries == 10
    assert len(service.requests) == 1


_NOT_DEPLOYED = (
    "account-level job listing is not available on this service yet: "
    "GET /api/v2/cortex-training answered 404 with no error body, so the "
    "route is not deployed"
)
_NOT_DEPLOYED_WITH_A_BODY = (
    "account-level job listing is not available: GET /api/v2/cortex-training "
    "answered 404 without the service's not-enabled error; the route may not "
    "be deployed yet, or a proxy in front of the service answered"
)
_DISABLED = (
    "account-level job listing is not enabled for this account (HTTP 404); "
    "Snowflake enables it per account"
)
_REFUSED = "account-level job listing was refused (HTTP 403)"
_REFUSED_WITHOUT_A_MESSAGE = (
    f"{_REFUSED} with no error message: a network policy or a proxy may have "
    "blocked the request, or Cortex Training is not enabled for this account "
    "or its trial has ended"
)


def answer_every_request(service: FakeAccountService, status: int, body: Any) -> None:
    """Answer every request with ``status`` and ``body``: JSON, raw bytes, or nothing."""

    def answer(request, timeout):
        service.requests.append((request.url, timeout))
        if isinstance(body, bytes):
            response = _response(request, status)
            response._content = body
            response.headers["Content-Type"] = "text/html"
            return response
        return _response(request, status, body)

    service.respond = answer


@pytest.mark.parametrize(
    ("body", "reason", "message"),
    [
        # The service's answer that the route is not enabled, in any case,
        # wrapped or not.
        (rest_error(NOT_ENABLED, NOT_ENABLED_CODE), "disabled", _DISABLED),
        ({"error": {"message": NOT_ENABLED.upper() + "."}}, "disabled", _DISABLED),
        # Anything else is a missing route; only an empty body is called one.
        (None, "not_deployed", _NOT_DEPLOYED),
        (b" \n", "not_deployed", _NOT_DEPLOYED),
        (b"<html><body>Not Found</body></html>", "not_deployed", _NOT_DEPLOYED_WITH_A_BODY),
        (rest_error("Not found"), "not_deployed", _NOT_DEPLOYED_WITH_A_BODY),
        # The 403's wording: not enough to call a 404 the disabled route.
        (
            rest_error("Cortex Training is not enabled for this account"),
            "not_deployed",
            _NOT_DEPLOYED_WITH_A_BODY,
        ),
    ],
)
def test_a_404_is_disabled_only_when_the_service_says_so(body, reason, message):
    service = FakeAccountService()
    client = client_for(service)
    answer_every_request(service, 404, body)

    with pytest.raises(AccountJobListUnavailable) as raised:
        client.list_jobs_page()

    assert raised.value.reason == reason
    assert str(raised.value) == f"[snowflake request id: request-id] {message}"
    assert len(service.requests) == 1


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            rest_error("Cortex Training is not enabled for this account", FORBIDDEN_CODE),
            f"{_REFUSED}: Cortex Training is not enabled for this account",
        ),
        (
            rest_error("The trial for this account has ended", FORBIDDEN_CODE),
            f"{_REFUSED}: The trial for this account has ended",
        ),
        ({"error": {"message": "  Access\n  denied "}}, f"{_REFUSED}: Access denied"),
        # No reason from the service: say so, and what else sends a 403.
        (None, _REFUSED_WITHOUT_A_MESSAGE),
        (b"<html><body>Forbidden</body></html>", _REFUSED_WITHOUT_A_MESSAGE),
        ({"code": FORBIDDEN_CODE, "message": ""}, _REFUSED_WITHOUT_A_MESSAGE),
    ],
)
def test_a_403_quotes_the_services_reason_or_says_it_gave_none(body, message):
    service = FakeAccountService()
    client = client_for(service)
    answer_every_request(service, 403, body)

    with pytest.raises(AccountJobListUnavailable) as raised:
        client.list_jobs_page()

    assert raised.value.reason == "forbidden"
    assert str(raised.value) == f"[snowflake request id: request-id] {message}"
    assert len(service.requests) == 1


def test_a_403_under_a_status_filter_quotes_the_refused_status_lookup():
    service = FakeAccountService()
    service.status_lookup_failure = (
        403,
        rest_error("Control Plane permission denied", FORBIDDEN_CODE),
    )
    client = client_for(service)

    with pytest.raises(AccountJobListUnavailable) as raised:
        list(client.iter_jobs(status="active"))

    assert raised.value.reason == "forbidden"
    assert str(raised.value) == (
        f"[snowflake request id: request-id] {_REFUSED}: Control Plane permission denied"
    )
    assert len(service.requests) == 1
    # Without a status filter there is no status lookup to refuse.
    assert _ids(client.iter_jobs()) == _ids(service.jobs)


def test_a_long_service_reason_is_quoted_on_one_bounded_line():
    service = FakeAccountService()
    client = client_for(service)
    reason = "refused\nbecause " + "x" * 1000
    answer_every_request(service, 403, rest_error(reason, FORBIDDEN_CODE))

    with pytest.raises(AccountJobListUnavailable) as raised:
        client.list_jobs_page()

    quoted = str(raised.value).removeprefix(f"[snowflake request id: request-id] {_REFUSED}: ")
    assert "\n" not in str(raised.value)
    assert quoted == ("refused because " + "x" * 1000)[:297] + "..."


@pytest.mark.parametrize("gate", ["not_deployed", "disabled", "forbidden", "rest_disabled"])
def test_unavailable_errors_survive_copy_and_pickle(gate):
    service = FakeAccountService(gate=gate)
    client = client_for(service)
    with pytest.raises(AccountJobListUnavailable) as raised:
        client.list_jobs_page()
    error = raised.value

    for copied in (copy.copy(error), copy.deepcopy(error), pickle.loads(pickle.dumps(error))):
        assert type(copied) is AccountJobListUnavailable
        assert copied.reason == gate
        assert str(copied) == str(error)
        assert copied.response.status_code == error.response.status_code
        assert copied.response.headers["x-snowflake-request-id"] == "request-id"


def test_an_expired_token_error_survives_copy_and_pickle():
    service = FakeAccountService(scan_budget=2)
    client = client_for(service)
    token = client.list_jobs_page()["next_page_token"]
    service.rotate_key()
    with pytest.raises(InvalidPageToken) as raised:
        client.list_jobs_page(page_token=token)

    for copied in (copy.copy(raised.value), pickle.loads(pickle.dumps(raised.value))):
        assert type(copied) is InvalidPageToken
        assert str(copied) == str(raised.value)
        assert copied.response.json()["message"] == INVALID_PAGE_TOKEN


def test_an_unavailable_route_never_creates_a_database():
    service = FakeAccountService(gate="disabled")
    client = client_for(service)

    with pytest.raises(AccountJobListUnavailable):
        client.list_jobs_page(database="NOPE")

    assert [urlsplit(url).path for url in service.urls] == ["/api/v2/cortex-training"]
    assert client._db_ensured is False


# ─── A4: an expired page token ───────────────────────────────────────────


def test_an_expired_token_restarts_the_walk_once_and_skips_jobs_already_yielded():
    service = FakeAccountService(account_rows(6), scan_budget=2)
    client = client_for(service)
    rotated = []

    def rotate_once(service):
        if not rotated:
            rotated.append(True)
            service.rotate_key()
            # A job created while the walk was paused: newest, so first.
            service.jobs.insert(0, server_row(99, "PENDING"))

    service.after_page = rotate_once

    jobs = list(client.iter_jobs())

    assert _ids(jobs) == [
        "job-0000",
        "job-0001",
        "job-0099",
        "job-0002",
        "job-0003",
        "job-0004",
        "job-0005",
    ]
    # Page one, the rejected token, then page one again and onward.
    assert [query.get("page_token", "")[:2] for query in service.queries] == [
        "",
        "1+",
        "",
        "2+",
        "2+",
        "2+",
    ]


def test_a_second_expired_token_raises():
    service = FakeAccountService(account_rows(6), scan_budget=2)
    client = client_for(service)
    service.after_page = lambda service: service.rotate_key()

    yielded = []
    with pytest.raises(InvalidPageToken) as raised:
        for job in client.iter_jobs():
            yielded.append(job["job_id"])

    assert yielded == ["job-0000", "job-0001"]
    assert isinstance(raised.value, requests.HTTPError)
    assert raised.value.response.status_code == 400
    assert raised.value.response.json()["message"] == INVALID_PAGE_TOKEN
    assert cortex_training.InvalidPageToken is InvalidPageToken
    # Page one, its token rejected, page one again, that token rejected too.
    assert len(service.requests) == 4


def test_list_jobs_page_raises_an_expired_token_for_the_caller_to_restart():
    service = FakeAccountService(scan_budget=2)
    client = client_for(service)
    token = client.list_jobs_page()["next_page_token"]
    service.rotate_key()

    with pytest.raises(InvalidPageToken, match="start again from the first page"):
        client.list_jobs_page(page_token=token)

    assert len(service.requests) == 2


@pytest.mark.parametrize(
    ("body", "expired"),
    [
        (rest_error(INVALID_PAGE_TOKEN), True),
        # Detail after the service's words, any case, or a wrapped body.
        (rest_error("Invalid page_token: sealed with a retired key"), True),
        ({"error": rest_error("INVALID PAGE_TOKEN")}, True),
        # Other 400s sent with a token are the caller's to fix.
        (rest_error(PAGE_TOKEN_MISMATCH), False),
        (rest_error(INVALID_SCOPE), False),
        (rest_error("limit must be between 1 and 1000"), False),
    ],
)
def test_an_expired_token_is_known_by_the_start_of_the_message(body, expired):
    service = FakeAccountService()
    client = client_for(service)
    answer_every_request(service, 400, body)

    with pytest.raises(requests.HTTPError) as raised:
        client.list_jobs_page(page_token="opaque")

    assert isinstance(raised.value, InvalidPageToken) is expired
    assert len(service.requests) == 1


# ─── A5: bounded retries and timeouts ────────────────────────────────────


def test_a_page_uses_the_clients_retries_and_timeout_by_default(no_backoff):
    service = FakeAccountService()
    service.failures = [503] * 10
    client = client_for(service)

    page = client.list_jobs_page()

    assert _ids(page["jobs"]) == _ids(service.jobs)
    assert len(service.requests) == 11
    assert set(service.timeouts) == {(30.0, 600.0)}


@pytest.mark.parametrize(
    ("request_timeout", "sent"),
    [(45, (45.0, 45.0)), (2.5, (2.5, 2.5)), ((5, 45), (5.0, 45.0))],
)
def test_max_retries_and_request_timeout_bound_one_page(no_backoff, request_timeout, sent):
    service = FakeAccountService()
    service.failures = [503] * 5
    client = client_for(service)

    with pytest.raises(requests.HTTPError) as raised:
        client.list_jobs_page(max_retries=2, request_timeout=request_timeout)

    assert raised.value.response.status_code == 503
    assert len(service.requests) == 3
    assert service.timeouts == [sent] * 3
    # The client's own settings are unchanged.
    assert (client.max_retries, client.request_timeout) == (10, (30.0, 600.0))


def test_bounds_apply_to_every_page_of_iter_jobs(no_backoff):
    service = FakeAccountService(scan_budget=4)
    client = client_for(service)
    served = []

    def fail_the_third_page(service):
        served.append(True)
        if len(served) == 2:
            service.failures = [504, 504, 504]

    service.after_page = fail_the_third_page

    yielded = []
    with pytest.raises(requests.HTTPError) as raised:
        for job in client.iter_jobs(max_retries=1, request_timeout=(3, 30)):
            yielded.append(job["job_id"])

    assert raised.value.response.status_code == 504
    assert yielded == _ids(service.jobs[:8])
    # Two pages, then two attempts at the third.
    assert len(service.requests) == 4
    assert set(service.timeouts) == {(3.0, 30.0)}


@pytest.mark.parametrize(
    "error",
    [
        requests.exceptions.ReadTimeout("read timed out"),
        requests.exceptions.ConnectionError("connection reset"),
    ],
)
def test_bounds_cover_timeouts_and_connection_errors(no_backoff, error):
    service = FakeAccountService()
    service.failures = [error, error, error]
    client = client_for(service)

    with pytest.raises(type(error)):
        client.list_jobs_page(max_retries=1, request_timeout=30)

    assert len(service.requests) == 2


def test_max_retries_zero_is_one_attempt(no_backoff):
    service = FakeAccountService()
    service.failures = [429]
    client = client_for(service)

    with pytest.raises(requests.HTTPError):
        client.list_jobs_page(max_retries=0)

    assert len(service.requests) == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_retries": -1}, "max_retries must be an int >= 0"),
        ({"max_retries": True}, "max_retries must be an int >= 0"),
        ({"max_retries": 1.5}, "max_retries must be an int >= 0"),
        ({"request_timeout": 0}, "request_timeout must be a positive finite number"),
        ({"request_timeout": (5, None)}, "request_timeout must be a positive finite number"),
        ({"request_timeout": "60"}, "request_timeout must be a positive finite number"),
    ],
)
@pytest.mark.parametrize("call", ["list_jobs_page", "iter_jobs"])
def test_bad_bounds_are_refused_before_any_request(kwargs, message, call):
    service = FakeAccountService()
    client = client_for(service)

    with pytest.raises(ValueError, match=message):
        getattr(client, call)(**kwargs)

    assert service.requests == []


# ─── A8: input validation ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("page_size", "message"),
    [
        (True, "page_size must be an int, got bool True"),
        (1.5, "page_size must be an int, got float 1.5"),
        (100.0, "page_size must be an int, got float 100.0"),
        ("100", "page_size must be an int, got str '100'"),
        (0, "page_size must be between 1 and 1000, got 0"),
        (1001, "page_size must be between 1 and 1000, got 1001"),
    ],
)
@pytest.mark.parametrize("call", ["list_jobs_page", "iter_jobs"])
def test_page_size_must_be_an_int_from_1_to_1000(page_size, message, call):
    service = FakeAccountService()
    client = client_for(service)

    with pytest.raises(ValueError, match=re.escape(message)):
        getattr(client, call)(page_size=page_size)

    assert service.requests == []


@pytest.mark.parametrize("page_size", [1, 1000])
def test_page_size_bounds_are_accepted(page_size):
    service = FakeAccountService()
    client = client_for(service)

    client.list_jobs_page(page_size=page_size)

    assert service.queries == [{"limit": str(page_size)}]


def test_page_size_and_max_retries_take_other_integral_types(no_backoff):
    numpy = pytest.importorskip("numpy")
    service = FakeAccountService()
    service.failures = [503]
    client = client_for(service)

    client.list_jobs_page(page_size=numpy.int64(50), max_retries=numpy.int32(1))

    assert service.queries == [{"limit": "50"}] * 2


def test_empty_filters_are_not_sent():
    service = FakeAccountService()
    client = client_for(service)

    client.list_jobs_page(
        database="",
        schema="",
        status="",
        submitted_by="",
        created_after="",
        created_before="",
        page_token="",
    )
    list(client.iter_jobs(status=[]))
    list(client.iter_jobs(status=["", " "], schema=""))
    list(client.iter_jobs(status=" , "))

    assert service.queries == [{"limit": "100"}] * 4


def test_an_empty_database_with_a_schema_is_still_a_schema_without_one():
    client = client_for(FakeAccountService())

    with pytest.raises(ValueError, match="schema=... needs database=... with it"):
        client.list_jobs_page(database="", schema="PUBLIC")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"database": 7}, "database must be a string, got int"),
        ({"status": 7}, "status must be a string or a list of strings, got int"),
        ({"status": ["running", None]}, "status must be a string or a list of strings"),
        ({"submitted_by": ["JDOE"]}, "submitted_by must be a string, got list"),
        ({"page_token": 7}, "page_token must be a string, got int"),
    ],
)
def test_filters_of_the_wrong_type_are_refused(kwargs, message):
    service = FakeAccountService()
    client = client_for(service)

    with pytest.raises(ValueError, match=re.escape(message)):
        client.list_jobs_page(**kwargs)

    assert service.requests == []


@pytest.mark.parametrize(
    ("value", "sent"),
    [
        (date(2026, 10, 1), "2026-10-01T00:00:00Z"),
        (datetime(2026, 10, 1, 8, 30), "2026-10-01T08:30:00Z"),
        (datetime(2026, 10, 1, 8, 30, tzinfo=timezone(timedelta(hours=2))), "2026-10-01T06:30:00Z"),
        (datetime(2026, 10, 1, 8, 30, 0, 1500, tzinfo=timezone.utc), "2026-10-01T08:30:00.001500Z"),
        ("2026-10-01", "2026-10-01T00:00:00Z"),
        ("2026-10-01T08:30:00Z", "2026-10-01T08:30:00Z"),
        ("2026-10-01T08:30:00+02:00", "2026-10-01T06:30:00Z"),
        ("2026-10-01t08:30:00-0700", "2026-10-01T15:30:00Z"),
        ("2026-10-01T08:30+05", "2026-10-01T03:30:00Z"),
        ("2026-10-01T08:30:00+05:59", "2026-10-01T02:31:00Z"),
        # The service refuses these two as they are; the client renders them.
        ("2026-10-01 08:30:00", "2026-10-01T08:30:00Z"),
        ("2026-10-01T08:30:00", "2026-10-01T08:30:00Z"),
        ("2026-10-01T08:30:00.123456789z", "2026-10-01T08:30:00.123456Z"),
        ("  2026-10-01T08:30:00,5Z ", "2026-10-01T08:30:00.500000Z"),
    ],
)
def test_created_bounds_are_sent_as_rfc_3339_in_utc_with_z(value, sent):
    service = FakeAccountService([server_row(1, created_at="2026-10-01T12:00:00Z")])
    client = client_for(service)

    client.list_jobs_page(created_after=value)
    client.list_jobs_page(created_before=value)

    assert service.queries == [
        {"created_after": sent, "limit": "100"},
        {"created_before": sent, "limit": "100"},
    ]


@pytest.mark.parametrize(
    "value",
    [
        "yesterday",
        "2026-13-01",
        "2026-10-01T25:00:00Z",
        "20261001",
        "20261001T083000Z",
        "2026-10-01T08",
        "2026-10-01T08:30:00+24:00",
        # Offset minutes stop at 59.
        "2026-10-01T08:30:00+05:99",
        "2026-10-01T08:30:00-0560",
        1759276800,
    ],
)
def test_created_bounds_that_are_not_dates_or_timestamps_are_refused(value):
    service = FakeAccountService()
    client = client_for(service)

    with pytest.raises(ValueError, match="created_after must be"):
        client.list_jobs_page(created_after=value)

    assert service.requests == []


def test_a_refused_created_bound_says_which_forms_it_takes():
    client = client_for(FakeAccountService())

    with pytest.raises(ValueError) as raised:
        client.list_jobs_page(created_before="2026-10-01T08:30:00+05:99")

    assert str(raised.value) == (
        "created_before must be YYYY-MM-DD, or YYYY-MM-DDTHH:MM[:SS[.fraction]] "
        "(a space may replace the T) with an optional Z or +HH[:MM] / -HH[:MM] "
        "offset; got '2026-10-01T08:30:00+05:99'"
    )


def test_a_page_whose_body_is_not_a_job_list_fails_clearly():
    service = FakeAccountService()
    client = client_for(service)
    service.respond = lambda request, timeout: _response(request, 200, {"jobs": {"job_id": "x"}})

    with pytest.raises(ValueError, match="non-list 'jobs'"):
        client.list_jobs_page()


# ─── A7: telemetry keeps filter values out ───────────────────────────────

_FILTERS = {
    "database": '"ml"',
    "schema": '"Exp 1"',
    "status": "active,done",
    "submitted_by": "nobody",
    "created_after": "2026-09-01T09:30:00+02:00",
    "created_before": date(2026, 10, 2),
}
_FILTER_TEXT = ["ml", "Exp", "active", "cancelled", "nobody", "2026-09", "2026-10", "09:30", "07:30"]


def _emitted(client) -> str:
    return repr(client._metric_emitter.emit.call_args_list)


@pytest.mark.parametrize(
    "failures",
    [
        [400],
        [503] * 3,
        [requests.exceptions.ConnectionError(
            "HTTPSConnectionPool(host='ACCOUNT.snowflakecomputing.com', port=443): Max retries "
            "exceeded with url: /api/v2/cortex-training?database=%22ml%22&schema=%22Exp+1%22"
            "&status=active%2Ccancelled%2Cterminated&submitted_by=nobody (Caused by reset)"
        )] * 3,
    ],
)
def test_failure_events_carry_no_filter_values(no_backoff, monkeypatch, failures):
    monkeypatch.delenv(client_module.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)
    service = FakeAccountService([server_row(1, database="ml", schema="Exp 1")])
    service.failures = list(failures)
    client = client_for(service)
    client._metric_emitter = MagicMock()

    with pytest.raises(requests.RequestException) as raised:
        list(client.iter_jobs(**_FILTERS, max_retries=2))

    # The raised error still shows the caller what was sent.
    assert "nobody" in str(raised.value)
    operation, value = client._metric_emitter.emit.call_args.args
    assert operation == "iter_jobs"
    assert value["success"] is False
    assert "?<redacted>" in value["error_message"]
    for text in _FILTER_TEXT:
        assert text not in _emitted(client), text


def test_a_rejected_user_and_an_expired_token_leave_no_filter_values(no_backoff, monkeypatch):
    monkeypatch.delenv(client_module.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)
    service = FakeAccountService([server_row(1, database="ml", schema="Exp 1")], users=())
    client = client_for(service)
    client._metric_emitter = MagicMock()

    with pytest.raises(requests.HTTPError):
        client.list_jobs_page(**_FILTERS)

    service.users = {"NOBODY"}
    service.scan_budget = 0
    token = client.list_jobs_page(**_FILTERS)["next_page_token"]
    service.rotate_key()
    with pytest.raises(InvalidPageToken):
        client.list_jobs_page(**_FILTERS, page_token=token)

    calls = client._metric_emitter.emit.call_args_list
    assert [call.args[0] for call in calls] == ["list_jobs_page", "list_jobs_page"]
    assert [call.kwargs["attributes"]["error.type"] for call in calls] == [
        "HTTPError",
        "InvalidPageToken",
    ]
    assert [call.kwargs["attributes"]["error.http_status"] for call in calls] == [400, 400]
    for text in _FILTER_TEXT + [token, token.split("/")[1]]:
        assert text not in _emitted(client), text


def test_success_events_name_the_call_that_read_the_page(monkeypatch):
    monkeypatch.setenv(client_module.ENABLE_SUCCESS_TELEMETRY_ENV, "1")
    service = FakeAccountService(scan_budget=5)
    client = client_for(service)
    client._metric_emitter = MagicMock()

    list(client.iter_jobs(status="active,done", submitted_by="JDOE"))
    client.list_jobs_page(status="active,done", submitted_by="JDOE")

    calls = client._metric_emitter.emit.call_args_list
    assert [call.args[0] for call in calls] == ["iter_jobs", "iter_jobs", "list_jobs_page"]
    assert [call.kwargs["attributes"] for call in calls] == [
        {"snowflake.request_ids": ["request-id"]}
    ] * 3
    for text in ("JDOE", "active", "cancelled"):
        assert text not in _emitted(client), text


@pytest.mark.parametrize(
    ("message", "redacted"),
    [
        (
            "400 Client Error: Bad Request for url: "
            "https://ACCOUNT.snowflakecomputing.com/api/v2/cortex-training?submitted_by=nobody&limit=100",
            "400 Client Error: Bad Request for url: "
            "https://ACCOUNT.snowflakecomputing.com/api/v2/cortex-training?<redacted>",
        ),
        (
            "Max retries exceeded with url: /api/v2/cortex-training?status=active&page_token=a+b/c= "
            "(Caused by NewConnectionError('refused'))",
            "Max retries exceeded with url: /api/v2/cortex-training?<redacted> "
            "(Caused by NewConnectionError('refused'))",
        ),
        ("HTTP://HOST/PATH?Q=1#frag tail", "HTTP://HOST/PATH?<redacted>#frag tail"),
        ("no query in https://host/path here", "no query in https://host/path here"),
        ("Is this a question? yes", "Is this a question? yes"),
    ],
)
def test_metric_error_messages_cut_url_queries(message, redacted):
    assert client_module._safe_metric_error_message(RuntimeError(message)) == redacted


def test_metric_error_messages_stay_bounded_for_long_messages():
    message = "/" * 100_000 + "?secret=1"

    redacted = client_module._safe_metric_error_message(RuntimeError(message))

    assert len(redacted) == client_module._METRIC_ERROR_MESSAGE_LIMIT
    assert "secret" not in redacted
