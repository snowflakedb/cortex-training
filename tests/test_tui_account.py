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

"""The job list's account-wide list: its fallback to the configured schema, and when it is asked again."""

import json
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import pytest
import requests

from cortex_training import AccountJobListUnavailable
from cortex_training import InvalidPageToken
from cortex_training.tui import views
from cortex_training.tui.account import ACCOUNT
from cortex_training.tui.account import ACCOUNT_LIST_CAP
from cortex_training.tui.account import ACCOUNT_LIST_MAX_PAGES
from cortex_training.tui.account import ACCOUNT_MANUAL_REPROBE_SECONDS
from cortex_training.tui.account import ACCOUNT_PAGE_SIZE
from cortex_training.tui.account import ACCOUNT_REPROBE_SECONDS
from cortex_training.tui.account import ACTIVE_NEXT
from cortex_training.tui.account import CAPPED
from cortex_training.tui.account import FAILED
from cortex_training.tui.account import FALLBACK
from cortex_training.tui.account import OLDER
from cortex_training.tui.account import PAGE_LIMIT
from cortex_training.tui.account import JobLister
from cortex_training.tui.account import JobListError
from cortex_training.tui.account import Listing
from cortex_training.tui.account import ListScope
from cortex_training.tui.account import _page
from cortex_training.tui.account import empty_text
from cortex_training.tui.account import loaded_text
from cortex_training.tui.account import scope_text
from cortex_training.tui.account import stopped_scope

NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)
NOT_DEPLOYED = AccountJobListUnavailable.NOT_DEPLOYED
DISABLED = AccountJobListUnavailable.DISABLED
FORBIDDEN = AccountJobListUnavailable.FORBIDDEN
REST_DISABLED = AccountJobListUnavailable.REST_DISABLED


def _response(status: int, body=None) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.reason = {403: "Forbidden", 404: "Not Found", 405: "Method Not Allowed"}.get(status, "")
    response._content = json.dumps(body).encode() if body is not None else b""
    return response


def _unavailable(reason: str, body=None) -> AccountJobListUnavailable:
    status = {NOT_DEPLOYED: 404, DISABLED: 404, FORBIDDEN: 403, REST_DISABLED: 405}[reason]
    return AccountJobListUnavailable(f"the account-level job list is unavailable ({reason})", reason=reason,
                                     response=_response(status, body))


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class _Client:
    """``list_jobs_page`` raises ``account_error`` or answers one page; ``list_jobs`` raises ``schema_error``
    or lists the configured schema. Each counts its calls."""

    def __init__(self, account_error=None, schema_error=None) -> None:
        self.account_error, self.schema_error = account_error, schema_error
        self.account_asks = self.schema_asks = 0

    def list_jobs_page(self, **filters) -> dict:
        self.account_asks += 1
        if self.account_error is not None:
            raise self.account_error
        return {"jobs": [{"job_id": "account-job", "database_name": "DB", "schema_name": "S"}], "next_page_token": None}

    def list_jobs(self, status=None) -> list:  # the real client's signature: bounds passed here fail
        self.schema_asks += 1
        if self.schema_error is not None:
            raise self.schema_error
        return [{"job_id": "schema-job"}, "not a job"]


def test_an_unavailable_account_list_falls_back_and_is_asked_again_only_when_due():
    clock, client = _Clock(), _Client(account_error=_unavailable(NOT_DEPLOYED))
    lister = JobLister(client, account_wide=True, clock=clock)
    listing = lister.load(NOW)
    assert listing.scope == ListScope(FALLBACK, reason=NOT_DEPLOYED) and listing.jobs == [{"job_id": "schema-job"}]
    assert lister._fallback == NOT_DEPLOYED and client.account_asks == 1

    clock.now += ACCOUNT_MANUAL_REPROBE_SECONDS / 2
    lister.load(NOW, manual=True)  # r, too soon after the last ask
    clock.now += ACCOUNT_MANUAL_REPROBE_SECONDS / 2
    lister.load(NOW)  # a timed refresh waits longer than r does
    assert client.account_asks == 1
    lister.load(NOW, manual=True)
    assert client.account_asks == 2  # still unavailable: that ask restarts the wait

    clock.now += ACCOUNT_REPROBE_SECONDS
    client.account_error = None
    listing = lister.load(NOW)
    assert client.account_asks == 3 and listing.scope == ListScope(ACCOUNT) and lister._fallback is None


def test_asking_again_while_falling_back_never_raises_and_restarts_the_wait():
    clock, client = _Clock(), _Client(account_error=_unavailable(DISABLED))
    lister = JobLister(client, account_wide=True, clock=clock)
    lister.load(NOW)
    clock.now += ACCOUNT_REPROBE_SECONDS
    client.account_error = requests.exceptions.ConnectionError("connection reset")
    assert lister.load(NOW).scope == ListScope(FALLBACK, reason=DISABLED)  # the earlier reason stands
    clock.now += ACCOUNT_REPROBE_SECONDS / 2
    lister.load(NOW)
    assert client.account_asks == 2
    clock.now += ACCOUNT_REPROBE_SECONDS
    client.account_error = _unavailable(REST_DISABLED)  # refused now: the list says so, still without an error
    assert lister.load(NOW).scope == ListScope(FALLBACK, reason=REST_DISABLED) and client.account_asks == 3


def test_a_asks_again_once_the_last_answer_is_a_minute_old_and_then_raises_what_it_gets():
    clock, client = _Clock(), _Client(account_error=_unavailable(NOT_DEPLOYED))
    lister = JobLister(client, account_wide=True, clock=clock)
    lister.load(NOW)
    assert lister.load(NOW, manual=True, asked=True).scope.reason == NOT_DEPLOYED  # the answer it just had
    assert client.account_asks == 1
    clock.now += ACCOUNT_MANUAL_REPROBE_SECONDS
    client.account_error = requests.exceptions.ConnectionError("connection reset")
    with pytest.raises(JobListError) as failed:
        lister.load(NOW, manual=True, asked=True)
    assert not failed.value.refused and str(failed.value).startswith("the account-wide job list failed:")


@pytest.mark.parametrize("refusal", [_unavailable(REST_DISABLED), _unavailable(FORBIDDEN, {"message": "not enabled"})])
def test_a_refusal_from_a_gate_the_schema_list_shares_is_the_lists_error(refusal):
    client = _Client(account_error=refusal)
    with pytest.raises(JobListError) as refused:
        JobLister(client, account_wide=True).load(NOW)
    assert refused.value.refused and client.schema_asks == 0


def test_a_403_without_the_services_message_asks_the_configured_schema_once():
    client = _Client(account_error=_unavailable(FORBIDDEN))  # a proxy or a network policy may have sent it
    lister = JobLister(client, account_wide=True)
    assert lister.load(NOW).scope == ListScope(FALLBACK, reason=FORBIDDEN) and client.schema_asks == 1
    assert lister._fallback == FORBIDDEN

    refused = requests.exceptions.HTTPError("403 Client Error", response=_response(403))
    client = _Client(account_error=_unavailable(FORBIDDEN), schema_error=refused)
    with pytest.raises(JobListError) as failed:
        JobLister(client, account_wide=True).load(NOW)
    assert not failed.value.refused and str(failed.value).endswith("; and list_jobs answered HTTP 403 Forbidden")


# ─── the account-level walk, a page at a time ────────────────────────────────


class _Paged:
    """The account-level list of ``total`` jobs, newest first; every ``active_every``-th job is
    running (the active walk lists those). ``fail_at``: the page of the newest jobs' walk (1 is
    the first) that fails; ``bad_token_at``: the one whose token the service can't open, once;
    ``unavailable``: what every page raises. Each page asked for is recorded as
    ``(status or "newest", page_token)``."""

    def __init__(self, total, *, active_every=7, fail_at=None, bad_token_at=None):
        start = datetime(2026, 9, 30, tzinfo=timezone.utc)
        self.jobs = [{"job_id": f"job-{i:05d}", "database_name": "DB", "schema_name": f"S{i % 3}",
                      "status": "RUNNING" if i % active_every == 0 else "DONE",
                      "created_at": (start - timedelta(minutes=i)).isoformat()} for i in range(total)]
        self.fail_at, self.bad_token_at = fail_at, bad_token_at
        self.unavailable = None
        self.calls = []

    def list_jobs_page(self, *, created_after, page_size, page_token=None, status=None):
        if self.unavailable is not None:
            raise self.unavailable
        self.calls.append((status or "newest", page_token))
        rows = [job for job in self.jobs if not status or job["status"] == "RUNNING"]
        start = int(page_token or 0)
        page = start // page_size + 1
        if not status and page == self.bad_token_at:
            self.bad_token_at = None
            raise InvalidPageToken("400 Invalid page_token", response=_response(400))
        if not status and page == self.fail_at:
            raise requests.exceptions.ConnectionError("connection reset")
        end = start + page_size
        return {"jobs": [dict(job) for job in rows[start:end]], "next_page_token": str(end) if end < len(rows) else None}

    def list_jobs(self, status=None) -> list:
        return [{"job_id": "schema-job"}]


def _old_walk(client) -> Listing:
    """The account-level list as it was read all at once, before it showed its pages: the
    newest jobs' walk, then (when it stopped short) the active jobs not among them."""
    def pages(status):
        jobs, seen, token = [], set(), None
        for _ in range(ACCOUNT_LIST_MAX_PAGES):
            rows, token = _page(client.list_jobs_page(
                created_after=NOW - timedelta(days=views.RETENTION_DAYS), page_size=ACCOUNT_PAGE_SIZE,
                page_token=token, **({"status": status} if status else {})))
            for job in rows:
                if views.job_key(job) not in seen:
                    seen.add(views.job_key(job))
                    jobs.append(job)
            if len(jobs) >= ACCOUNT_LIST_CAP:
                return jobs[:ACCOUNT_LIST_CAP], CAPPED if token is not None or len(jobs) > ACCOUNT_LIST_CAP else None
            if token is None:
                return jobs, None
        return jobs, PAGE_LIMIT

    jobs, stopped = pages(None)
    if stopped is None:
        return Listing(jobs, ListScope(ACCOUNT))
    rows, active_stopped = pages("active")
    listed = {views.job_key(job) for job in jobs}
    return Listing(jobs + [job for job in rows if views.job_key(job) not in listed],
                   ListScope(ACCOUNT, stopped=stopped, walked=len(jobs), active_missing=active_stopped is not None))


@pytest.mark.parametrize("total", [ACCOUNT_LIST_CAP + 300, 1_200, 400])
def test_the_walk_shows_each_page_and_ends_with_the_list_it_read_all_at_once(total):
    old_client = _Paged(total)
    old = _old_walk(old_client)
    client, partials = _Paged(total), []
    listing = JobLister(client, account_wide=True).load(NOW, on_partial=partials.append)
    assert listing == old and client.calls == old_client.calls  # the same list, from the same requests
    if total > ACCOUNT_LIST_CAP:
        assert len(listing.jobs) > ACCOUNT_LIST_CAP  # and the active jobs older than the newest 2,000

    capped = total > ACCOUNT_LIST_CAP
    pages = -(-min(total, ACCOUNT_LIST_CAP) // ACCOUNT_PAGE_SIZE)
    # One after each page but the last one read (a page that makes the list whole is the result).
    assert [len(p.jobs) for p in partials] == [ACCOUNT_PAGE_SIZE * n for n in range(1, pages + capped)]
    for partial in partials:
        assert partial.jobs == listing.jobs[:len(partial.jobs)]  # each the start of the list, newest first
        assert partial.scope.walked == len(partial.jobs) and partial.scope.truncated
    assert all(p.scope.loading == OLDER for p in partials[:pages - 1])
    if capped:  # the newest 2,000 are read: the active jobs come next
        assert partials[-1].scope == ListScope(ACCOUNT, stopped=CAPPED, walked=ACCOUNT_LIST_CAP, loading=ACTIVE_NEXT)
    assert len({id(p.jobs) for p in partials}) == len(partials)  # each its own list
    assert not listing.scope.loading


def test_a_page_that_fails_after_the_first_raises_after_showing_the_pages_before_it():
    client, partials = _Paged(ACCOUNT_LIST_CAP, fail_at=3), []
    with pytest.raises(JobListError) as failed:
        JobLister(client, account_wide=True).load(NOW, on_partial=partials.append)
    assert str(failed.value).startswith("the account-wide job list failed:")
    assert [len(p.jobs) for p in partials] == [500, 1_000]
    stopped = stopped_scope(partials[-1].scope)
    assert stopped == ListScope(ACCOUNT, stopped=FAILED, walked=1_000, active_missing=True)
    assert scope_text(stopped, location="") == (
        "newest 1,000 (the rest didn't load; r retries; older active jobs may be missing)")


def test_a_walk_read_again_from_its_start_shows_no_fewer_jobs_than_before():
    client, partials = _Paged(1_700, bad_token_at=3), []
    listing = JobLister(client, account_wide=True).load(NOW, on_partial=partials.append)
    assert listing == _old_walk(_Paged(1_700))
    # Pages 1 and 2, then from the first page again: 500 is not shown again, 1,000 is.
    assert [len(p.jobs) for p in partials] == [500, 1_000, 1_000, 1_500]


def test_a_cancelled_walk_stops_before_its_next_page():
    client, partials = _Paged(ACCOUNT_LIST_CAP), []
    listing = JobLister(client, account_wide=True).load(NOW, on_partial=partials.append,
                                                        is_cancelled=lambda: len(partials) == 2)
    assert listing is None and len(client.calls) == 2


def test_asking_again_in_fallback_shows_no_pages():
    clock, client = _Clock(), _Paged(1_200)
    client.unavailable = _unavailable(NOT_DEPLOYED)
    lister = JobLister(client, account_wide=True, clock=clock)
    assert lister.load(NOW).scope.kind == FALLBACK
    client.unavailable = None
    clock.now += ACCOUNT_REPROBE_SECONDS
    partials = []
    listing = lister.load(NOW, on_partial=partials.append)
    assert listing.scope == ListScope(ACCOUNT) and len(listing.jobs) == 1_200
    assert partials == []  # the configured schema's list stands until the account-wide one is whole


def test_a_loading_list_says_how_far_it_is():
    older = ListScope(ACCOUNT, walked=1_000, loading=OLDER)
    assert scope_text(older, location="") == "loading older jobs… 1,000 so far"
    assert loaded_text(older) == "the newest 1,000 so far; older jobs are still loading"
    assert "older jobs are still loading" in empty_text(older)
    active = ListScope(ACCOUNT, stopped=CAPPED, walked=2_000, loading=ACTIVE_NEXT)
    assert scope_text(active, location="") == "newest 2,000; loading older active jobs…"
    # Superseded while it read the active jobs: the newest 2,000 stand, saying what may be missing.
    assert scope_text(stopped_scope(active), location="") == (
        "newest 2,000 (more exist; older active jobs may be missing)")
    assert stopped_scope(ListScope(ACCOUNT)) == ListScope(ACCOUNT)

