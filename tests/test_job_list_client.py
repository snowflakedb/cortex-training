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

"""The client's job lists: ``list_jobs_page`` and ``iter_jobs`` on the
account-level ``GET /api/v2/cortex-training``, and the per-call bounds they
share with the per-schema ``list_jobs``."""

from __future__ import annotations

import copy
import json
import pickle
from urllib.parse import parse_qsl
from urllib.parse import urlsplit

import pytest
import requests
from requests.adapters import BaseAdapter
from tenacity import wait_none

import cortex_training.client as client_module
from cortex_training import AccountJobListUnavailable
from cortex_training import CortexTrainingClient
from cortex_training import InvalidPageToken

NOT_ENABLED = "Account-level job listing is not enabled for this account"
INVALID_PAGE_TOKEN = (400, {"message": "Invalid page_token"})


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(client_module, "wait_exponential_jitter", lambda **_kwargs: wait_none())


class _Service(BaseAdapter):
    """Answers the n-th request with the n-th ``(status, body)`` (the last one
    from then on), the body JSON, raw bytes or None; records what was sent."""

    def __init__(self, answers):
        super().__init__()
        self.answers, self.urls, self.timeouts = answers, [], []

    @property
    def queries(self) -> list[dict[str, str]]:
        return [dict(parse_qsl(urlsplit(url).query)) for url in self.urls]

    def send(self, request, timeout=None, **_kwargs):
        self.urls.append(request.url)
        self.timeouts.append(timeout)
        status, body = self.answers[min(len(self.urls), len(self.answers)) - 1]
        resp = requests.Response()
        resp.status_code, resp.request, resp.url, resp.encoding = status, request, request.url, "utf-8"
        resp._content = body if isinstance(body, bytes) else b"" if body is None else json.dumps(body).encode()
        return resp

    def close(self):
        pass


def _client(*answers, **kwargs) -> tuple[CortexTrainingClient, _Service]:
    service = _Service(list(answers))
    client = CortexTrainingClient(base_url="http://test.local", database="DB", schema="SCH", **kwargs)
    client._session.mount("http://", service)
    return client, service


def _page(*job_ids, token=None):
    return 200, {"jobs": [{"job_id": job_id} for job_id in job_ids], "next_page_token": token}


# --- list_jobs_page and its arguments ---


def test_list_jobs_page_reads_one_page_and_takes_its_token_back():
    client, service = _client(_page("b", token="t+/="), (200, {"next_page_token": ""}), endpoint="training-v2")

    first = client.list_jobs_page(status="active", page_size=1)
    last = client.list_jobs_page(status="active", page_size=1, page_token=first["next_page_token"])

    assert first == {"jobs": [{"job_id": "b"}], "next_page_token": "t+/="}
    assert last == {"jobs": [], "next_page_token": None}
    # One fixed path, whatever the endpoint; the token goes back as it came.
    assert {urlsplit(url).path for url in service.urls} == {"/api/v2/cortex-training"}
    assert service.queries == [
        {"status": "active", "limit": "1"},
        {"status": "active", "limit": "1", "page_token": "t+/="},
    ]
    assert service.timeouts == [client.request_timeout] * 2  # the client's own, unless bounded


@pytest.mark.parametrize("body", [[{"job_id": "b"}], {"jobs": {"job_id": "b"}}])
def test_a_page_that_is_not_a_job_list_is_refused(body):
    client, _ = _client((200, body))
    with pytest.raises(ValueError, match="the account-level job list answered with a non-"):
        client.list_jobs_page()


def test_status_done_is_sent_as_what_it_stood_for():
    # The service no longer takes done. Repeats go, whatever their case.
    client, service = _client(_page())
    client.list_jobs_page(status="active, DONE,Terminated")
    assert service.queries == [{"status": "active,cancelled,terminated", "limit": "100"}]


def test_none_or_empty_filters_are_not_sent():
    client, service = _client(_page())
    client.list_jobs_page(
        database="", schema="", status=" , ", submitted_by="", created_after="", created_before="", page_token=""
    )
    list(client.iter_jobs(status=[]))
    assert service.queries == [{"limit": "100"}] * 2


@pytest.mark.parametrize(
    ("value", "sent"),
    [
        ("2026-10-01", "2026-10-01T00:00:00Z"),
        ("2026-10-01 08:30:00", "2026-10-01T08:30:00Z"),  # no offset is UTC
        ("2026-10-01t08:30:00-0700", "2026-10-01T15:30:00Z"),
        ("2026-10-01T08:30+05", "2026-10-01T03:30:00Z"),
        (" 2026-10-01T08:30:00,5z ", "2026-10-01T08:30:00.500000Z"),
        ("2026-10-01T08:30:00.123456789Z", "2026-10-01T08:30:00.123456Z"),
    ],
)
def test_created_bounds_are_sent_as_rfc_3339_in_utc(value, sent):
    client, service = _client(_page())
    client.list_jobs_page(created_after=value, created_before=value)
    assert service.queries == [{"created_after": sent, "created_before": sent, "limit": "100"}]


@pytest.mark.parametrize(
    ("call", "kwargs"),
    [
        ("list_jobs_page", {"database": 7}),
        ("list_jobs_page", {"status": ["running", None]}),
        ("list_jobs_page", {"page_token": 7}),
        ("list_jobs_page", {"page_size": True}),
        ("list_jobs_page", {"page_size": 1.5}),
        ("list_jobs_page", {"created_after": "yesterday"}),
        ("list_jobs_page", {"created_after": "2026-13-01"}),
        ("list_jobs_page", {"created_after": "2026-10-01T08:30:00+05:99"}),  # offset minutes stop at 59
        ("list_jobs_page", {"created_before": 1759276800}),
        ("list_jobs_page", {"request_timeout": 0}),
        ("iter_jobs", {"max_retries": True}),
        ("list_jobs", {"max_retries": -1}),
    ],
)
def test_bad_arguments_are_refused_before_any_request(call, kwargs):
    client, service = _client(_page())
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        getattr(client, call)(**kwargs)
    assert service.urls == []


def test_page_size_and_max_retries_take_numpy_integers():
    numpy = pytest.importorskip("numpy")
    client, service = _client((503, None), _page())
    client.list_jobs_page(page_size=numpy.int64(50), max_retries=numpy.int32(1))
    assert service.queries == [{"limit": "50"}] * 2


# --- An unavailable list, and a page token the service can't open ---


@pytest.mark.parametrize(
    ("status", "body", "reason", "words"),
    [
        (404, None, "not_deployed", "answered 404 with no error body"),
        (404, b"<html>Not Found</html>", "not_deployed", "or a proxy in front of the service answered"),
        (404, {"code": "000404", "message": NOT_ENABLED}, "disabled", "not enabled for this account (HTTP 404)"),
        # The service's reason, quoted on one line and cut short.
        (
            403,
            {"message": "refused\nbecause " + "x" * 1000},
            "forbidden",
            "(HTTP 403): refused because " + "x" * 281 + "...",
        ),
        (403, None, "forbidden", "(HTTP 403) with no error message"),
        (405, None, "rest_disabled", "REST API is turned off for this account (HTTP 405)"),
    ],
)
def test_an_unavailable_list_raises_why_at_once(status, body, reason, words):
    client, service = _client((status, body))
    with pytest.raises(AccountJobListUnavailable) as raised:
        client.list_jobs_page()
    assert (raised.value.reason, raised.value.response.status_code) == (reason, status)
    assert words in str(raised.value) and "\n" not in str(raised.value)
    assert len(service.urls) == 1  # not retried


def test_iter_jobs_raises_it_too_and_it_survives_copy_and_pickle():
    client, _ = _client((404, {"message": NOT_ENABLED}))
    with pytest.raises(requests.HTTPError) as raised:  # what callers caught before still catches it
        list(client.iter_jobs())
    error = raised.value
    for copied in (copy.copy(error), copy.deepcopy(error), pickle.loads(pickle.dumps(error))):
        assert (type(copied), copied.reason, str(copied)) == (AccountJobListUnavailable, "disabled", str(error))


@pytest.mark.parametrize(
    ("message", "invalid"),
    [
        ("Invalid page_token", True),
        ("Invalid page_token: sealed with a retired key", True),
        # A token sent with other filters is the caller's to fix: starting again would hide it.
        ("page_token does not match the request filters", False),
    ],
)
def test_list_jobs_page_raises_an_invalid_page_token_for_the_caller_to_start_again(message, invalid):
    client, service = _client((400, {"message": message}))
    with pytest.raises(requests.HTTPError) as raised:
        client.list_jobs_page(page_token="opaque")
    assert isinstance(raised.value, InvalidPageToken) is invalid
    assert len(service.urls) == 1


def test_iter_jobs_starts_again_once_after_an_invalid_token_and_skips_jobs_already_yielded():
    # c was created while the walk was paused, so it comes first the second time.
    client, service = _client(
        _page("b", "a", token="t1"), INVALID_PAGE_TOKEN, _page("c", "b", token="t2"), INVALID_PAGE_TOKEN
    )
    yielded = []
    with pytest.raises(InvalidPageToken):  # the second time
        for job in client.iter_jobs():
            yielded.append(job["job_id"])
    assert yielded == ["b", "a", "c"]
    assert [query.get("page_token") for query in service.queries] == [None, "t1", None, "t2"]


# --- Per-call bounds ---


@pytest.mark.parametrize("call", ["list_jobs", "list_jobs_page"])
def test_max_retries_and_request_timeout_bound_one_call(call):
    client, service = _client((503, None))
    with pytest.raises(requests.HTTPError):
        getattr(client, call)(max_retries=2, request_timeout=(5, 45))
    assert service.timeouts == [(5.0, 45.0)] * 3  # one attempt and two retries, not eleven
    assert (client.max_retries, client.request_timeout) == (10, (30.0, 600.0))


def test_iter_jobs_bounds_each_page():
    client, service = _client(_page("b", token="t1"), (504, None))
    with pytest.raises(requests.HTTPError):
        list(client.iter_jobs(max_retries=1, request_timeout=3))
    assert service.timeouts == [(3.0, 3.0)] * 3  # the first page, then two attempts at the second


# --- Telemetry ---


@pytest.mark.parametrize(
    ("message", "sent"),
    [
        (
            "400 Client Error: Bad Request for url: http://test.local/api/v2/cortex-training?submitted_by=jdoe",
            "400 Client Error: Bad Request for url: http://test.local/api/v2/cortex-training?<redacted>",
        ),
        (
            "Max retries exceeded with url: /api/v2/cortex-training?status=active&page_token=a+b/c= (Caused by reset)",
            "Max retries exceeded with url: /api/v2/cortex-training?<redacted> (Caused by reset)",
        ),
        ("Is this a question? yes", "Is this a question? yes"),
    ],
)
def test_telemetry_leaves_url_queries_out_of_error_messages(message, sent):
    # A query carries the filters and the page token the caller sent.
    assert client_module._safe_metric_error_message(RuntimeError(message)) == sent
