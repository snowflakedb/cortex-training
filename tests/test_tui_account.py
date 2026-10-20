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
import threading
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import pytest
import requests

from cortex_training import AccountJobListUnavailable
from cortex_training import InvalidPageToken
from cortex_training.tui import account
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
    running (the active walk lists those). Pages hold ``rows`` jobs (the size asked for when
    None), each starting with the last job of the page before when ``repeat`` (as a service
    examining a bounded number of jobs per call may answer), and the ``empty_at``-th page of the
    newest jobs is empty but carries a token. ``fail_at``: the page of the newest jobs' walk (1
    is the first) that fails; ``bad_token_at``: the one whose token the service can't open,
    once (``bad_active_token_at``: in the active walk); ``active_fail``: the active walk's
    second page fails; ``unavailable``: what every page raises. Each page asked for is recorded
    as ``(status or "newest", page_token)``."""

    def __init__(self, total, *, active_every=7, rows=None, repeat=False, empty_at=None, fail_at=None,
                 bad_token_at=None, bad_active_token_at=None, active_fail=False):
        start = datetime(2026, 9, 30, tzinfo=timezone.utc)
        self.jobs = [{"job_id": f"job-{i:05d}", "database_name": "DB", "schema_name": f"S{i % 3}",
                      "status": "RUNNING" if i % active_every == 0 else "DONE",
                      "created_at": (start - timedelta(minutes=i)).isoformat()} for i in range(total)]
        self.rows, self.repeat, self.empty_at, self.fail_at = rows, repeat, empty_at, fail_at
        self.bad = {"newest": bad_token_at, "active": bad_active_token_at}
        self.active_fail = active_fail
        self.unavailable = None
        self.calls = []

    def list_jobs_page(self, *, created_after, page_size, page_token=None, status=None):
        if self.unavailable is not None:
            raise self.unavailable
        walk = "active" if status else "newest"
        self.calls.append((walk, page_token))
        rows = [job for job in self.jobs if not status or job["status"] == "RUNNING"]
        start, page = map(int, (page_token or "0:0").split(":"))
        page += 1
        if page == self.bad[walk]:
            self.bad[walk] = None
            raise InvalidPageToken("400 Invalid page_token", response=_response(400))
        if (walk == "newest" and page == self.fail_at) or (walk == "active" and self.active_fail and page == 2):
            raise requests.exceptions.ConnectionError("connection reset")
        if walk == "newest" and page == self.empty_at:
            return {"jobs": [], "next_page_token": f"{start}:{page}"}
        end = start + (self.rows or page_size)
        first = max(0, start - 1) if self.repeat and start else start
        return {"jobs": [dict(job) for job in rows[first:end]],
                "next_page_token": f"{end}:{page}" if end < len(rows) else None}

    def list_jobs(self, status=None) -> list:
        return [{"job_id": "schema-job"}]


def _old_walk(client) -> Listing:
    """The account-level list as the job list read it all at once, before it showed its pages
    (the walk as it was, step for step): the newest jobs' walk, then (when it stopped short)
    the active jobs not among them."""
    created_after = NOW - timedelta(days=views.RETENTION_DAYS)

    def read_pages(status):
        filters = {"created_after": created_after, **({"status": status} if status else {})}
        jobs, seen, token = [], set(), None
        for _ in range(ACCOUNT_LIST_MAX_PAGES):
            rows, token = _page(client.list_jobs_page(**filters, page_size=ACCOUNT_PAGE_SIZE, page_token=token))
            for job in rows:
                key = views.job_key(job)
                if views.job_id_of(job) and key in seen:
                    continue
                seen.add(key)
                jobs.append(job)
            if len(jobs) >= ACCOUNT_LIST_CAP:
                more = token is not None or len(jobs) > ACCOUNT_LIST_CAP
                return jobs[:ACCOUNT_LIST_CAP], CAPPED if more else None
            if token is None:
                return jobs, None
        return jobs, PAGE_LIMIT

    def pages(status):
        try:
            return read_pages(status)
        except InvalidPageToken:
            return read_pages(status)

    jobs, stopped = pages(None)
    if stopped is None:
        return Listing(jobs, ListScope(ACCOUNT))
    try:
        active = pages("active")
    except Exception:  # noqa: BLE001
        active = ([], "failed")
    rows, active_stopped = active
    listed = {views.job_key(job) for job in jobs}
    added = [job for job in rows if views.job_key(job) not in listed]
    scope = ListScope(ACCOUNT, stopped=stopped, walked=len(jobs),
                      pages=ACCOUNT_LIST_MAX_PAGES if stopped == PAGE_LIMIT else 0,
                      active_missing=active_stopped is not None)
    return Listing(jobs + added, scope)


_SHAPES = {
    "capped": dict(total=ACCOUNT_LIST_CAP + 300),
    "whole in 3 pages": dict(total=1_200),
    "whole in 1 page": dict(total=400),
    "page limit": dict(total=3_000, rows=50),
    "page limit, pages repeating a job": dict(total=3_000, rows=50, repeat=True),
    "capped, an empty page with a token": dict(total=ACCOUNT_LIST_CAP + 300, empty_at=2),
    "capped, the active walk fails": dict(total=ACCOUNT_LIST_CAP + 300, active_every=2, active_fail=True),
    "capped, a bad token in the active walk": dict(total=ACCOUNT_LIST_CAP + 300, active_every=2,
                                                   bad_active_token_at=2),
    "capped, a bad token in the newest walk": dict(total=ACCOUNT_LIST_CAP + 300, bad_token_at=3),
}


def _walk_calls(calls: list, walk: str) -> list:
    return [call for call in calls if call[0] == walk]


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_a_refresh_reads_the_list_as_it_did_all_at_once(shape):
    old_client, client = _Paged(**_SHAPES[shape]), _Paged(**_SHAPES[shape])
    assert JobLister(client, account_wide=True).load(NOW) == _old_walk(old_client)
    assert client.calls == old_client.calls  # the same requests, in the same order


@pytest.mark.parametrize("shape", list(_SHAPES))
@pytest.mark.parametrize("run", range(3))  # the two walks interleave as their threads go
def test_the_walk_shows_each_page_and_ends_with_the_list_it_read_all_at_once(shape, run):
    old_client = _Paged(**_SHAPES[shape])
    old = _old_walk(old_client)
    client, partials = _Paged(**_SHAPES[shape]), []
    listing = JobLister(client, account_wide=True).load(NOW, on_partial=partials.append)
    assert listing == old and not listing.scope.loading
    # The same requests of each walk; the active jobs' one, read beside, maybe begun where the old one wasn't.
    assert _walk_calls(client.calls, "newest") == _walk_calls(old_client.calls, "newest")
    active, old_active = _walk_calls(client.calls, "active"), _walk_calls(old_client.calls, "active")
    assert active == old_active if old_active else active == _walk_calls(_active_calls(shape), "active")[:len(active)]

    sizes = [p.scope.walked for p in partials]
    assert sizes == sorted(sizes)
    final = {views.job_key(job) for job in listing.jobs}
    for partial in partials:
        assert partial.jobs[:partial.scope.walked] == listing.jobs[:partial.scope.walked]  # the newest, in order
        assert {views.job_key(job) for job in partial.jobs} <= final and partial.scope.truncated
    assert all(p.scope.loading == OLDER for p in partials[:-1])
    if partials and partials[-1].scope.loading == ACTIVE_NEXT:  # the newest jobs' walk stopped short first
        assert partials[-1].scope == ListScope(ACCOUNT, stopped=listing.scope.stopped, walked=listing.scope.walked,
                                               pages=listing.scope.pages, loading=ACTIVE_NEXT)
    # Once the active jobs are read, every list after holds them.
    read = [p.scope.active_read for p in partials if p.scope.loading == OLDER]
    assert read == sorted(read)
    assert len({id(p.jobs) for p in partials}) == len(partials)  # each its own list


def _active_calls(shape: str) -> list:
    """The requests a whole active jobs' walk of ``shape`` makes."""
    client = _Paged(**_SHAPES[shape])
    JobLister(client)._active(NOW - timedelta(days=views.RETENTION_DAYS), lambda: False)
    return client.calls


@pytest.mark.parametrize("total", [ACCOUNT_LIST_CAP + 300, 1_200, 400])
def test_the_walk_shows_one_list_after_each_page_but_the_last(total):
    partials = []
    JobLister(_Paged(total), account_wide=True).load(NOW, on_partial=partials.append)
    capped = total > ACCOUNT_LIST_CAP
    pages = -(-min(total, ACCOUNT_LIST_CAP) // ACCOUNT_PAGE_SIZE)
    # A page that makes the list whole is the result; the active jobs, once read, may come between pages,
    # and the newest 2,000 of more show while the active jobs are still being read, if they are.
    walked = {p.scope.walked for p in partials}
    assert sorted(walked - {ACCOUNT_LIST_CAP} if capped else walked) == [ACCOUNT_PAGE_SIZE * n for n in range(1, pages)]


class _Held(_Paged):
    """A ``_Paged`` whose walks each answer only while their event (``hold["newest"]``,
    ``hold["active"]``) is set; ``entered`` counts the pages each walk asked for."""

    def __init__(self, total, **kwargs):
        super().__init__(total, **kwargs)
        self.hold = {"newest": threading.Event(), "active": threading.Event()}
        self.entered = {"newest": 0, "active": 0}

    def list_jobs_page(self, *, status=None, **kwargs):
        walk = "active" if status else "newest"
        self.entered[walk] += 1
        assert self.hold[walk].wait(10)
        return super().list_jobs_page(status=status, **kwargs)


def _until(predicate, seconds=5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _load_in_thread(lister, **kwargs) -> tuple:
    """Run ``lister.load`` on a thread: (the thread, the dict its answer lands in under "listing" or "error")."""
    answer = {}

    def run():
        try:
            answer["listing"] = lister.load(NOW, **kwargs)
        except Exception as exc:  # noqa: BLE001
            answer["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, answer


@pytest.mark.parametrize("fails", [False, True], ids=["read", "failed"])
def test_the_active_jobs_read_beside_the_newest_show_before_the_newest_are_read(fails):
    client, partials = _Held(ACCOUNT_LIST_CAP + 300, active_every=2, active_fail=fails), []
    client.hold["newest"].set()
    client.rows = None
    held_page = threading.Event()
    real = client.list_jobs_page

    def newest_waits_at_page_3(**kwargs):  # the newest jobs' walk waits at its third page
        if not kwargs.get("status") and (kwargs.get("page_token") or "").endswith(":2"):
            held_page.wait(10)
        return real(**kwargs)

    client.list_jobs_page = newest_waits_at_page_3
    thread, answer = _load_in_thread(JobLister(client, account_wide=True), on_partial=partials.append)
    assert _until(lambda: len(partials) == 2)  # pages 1 and 2
    client.hold["active"].set()  # the active jobs' walk, all of it
    assert _until(lambda: len(partials) == 3), [p.scope for p in partials]
    shown = partials[-1]
    assert shown.scope == ListScope(ACCOUNT, walked=1_000, loading=OLDER, active_read=True, active_missing=fails)
    assert shown.scope.active_whole is not fails
    if fails:
        assert len(shown.jobs) == 1_000
        assert scope_text(shown.scope, location="") == "loading older jobs… 1,000 so far (older active jobs may be missing)"
    else:  # the active jobs older than the first two pages are listed now, with them
        assert len(shown.jobs) > 1_000
        assert scope_text(shown.scope, location="") == "loading older jobs… 1,000 so far + every active job"
    held_page.set()
    thread.join(10)
    assert answer["listing"] == _old_walk(_Paged(ACCOUNT_LIST_CAP + 300, active_every=2, active_fail=fails))
    assert answer["listing"].scope.active_missing is fails
    assert [p.scope.walked for p in partials[3:]] == [1_500]  # one more page, and then the result


def test_a_walk_cancelled_while_it_waits_for_the_active_jobs_answers_at_once():
    client, partials, cancelled = _Held(ACCOUNT_LIST_CAP + 300), [], threading.Event()
    client.hold["newest"].set()  # the active jobs' first page takes its time
    thread, answer = _load_in_thread(JobLister(client, account_wide=True), on_partial=partials.append,
                                     is_cancelled=cancelled.is_set)
    assert _until(lambda: partials and partials[-1].scope.loading == ACTIVE_NEXT)
    cancelled.set()
    thread.join(1.0)
    assert not thread.is_alive() and answer == {"listing": None}
    shown = len(partials)
    client.hold["active"].set()  # its page answers: the walk asks for no other, and shows nothing more
    time.sleep(0.2)
    assert client.entered["active"] == 1 and len(partials) == shown


def test_a_list_whole_before_the_active_jobs_are_read_answers_without_them():
    client, partials = _Held(1_200), []
    client.hold["newest"].set()
    thread, answer = _load_in_thread(JobLister(client, account_wide=True), on_partial=partials.append)
    thread.join(2.0)
    assert not thread.is_alive() and answer["listing"] == _old_walk(_Paged(1_200))  # no wait for the active jobs
    shown = len(partials)
    client.hold["active"].set()
    time.sleep(0.2)
    assert len(partials) == shown  # nothing after the result
    assert all(not p.scope.active_read for p in partials)


def test_a_cancelled_load_whose_page_was_refused_leaves_the_fallback_to_the_load_after_it():
    clock, client = _Clock(), _Paged(1_200)
    lister = JobLister(client, account_wide=True, clock=clock)
    cancelled = []
    client.unavailable = _unavailable(NOT_DEPLOYED)
    schema_asks = []
    client.list_jobs = lambda status=None: schema_asks.append(1) or []
    # The load was superseded while its page was read: it answers nothing and changes nothing.
    assert lister.load(NOW, is_cancelled=lambda: bool(cancelled.append(1) or len(cancelled) > 1)) is None
    assert lister._fallback is None and lister._asked_at == 0.0 and schema_asks == []
    client.unavailable = requests.exceptions.ConnectionError("connection reset")
    cancelled.clear()
    assert lister.load(NOW, is_cancelled=lambda: bool(cancelled.append(1) or len(cancelled) > 1)) is None


def test_a_page_that_fails_after_the_first_raises_after_showing_the_pages_before_it():
    client, partials = _Held(ACCOUNT_LIST_CAP, fail_at=3), []
    client.hold["newest"].set()  # the active jobs' walk is still being read when page 3 fails
    with pytest.raises(JobListError) as failed:
        JobLister(client, account_wide=True).load(NOW, on_partial=partials.append)
    client.hold["active"].set()
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
    sizes = [p.scope.walked for p in partials]
    assert sizes == sorted(sizes) and sorted(set(sizes)) == [500, 1_000, 1_500]


def test_a_cancelled_walk_stops_before_its_next_page():
    client, partials = _Paged(ACCOUNT_LIST_CAP), []
    listing = JobLister(client, account_wide=True).load(NOW, on_partial=partials.append,
                                                        is_cancelled=lambda: len(partials) >= 2)
    assert listing is None and len(_walk_calls(client.calls, "newest")) <= 3


def test_a_load_while_an_earlier_active_jobs_walk_still_reads_reads_them_after_the_newest():
    client, cancelled = _Held(ACCOUNT_LIST_CAP + 300), threading.Event()
    lister = JobLister(client, account_wide=True)
    client.hold["newest"].set()  # the active jobs' first page takes its time
    first, answer = _load_in_thread(lister, on_partial=lambda listing: None, is_cancelled=cancelled.is_set)
    assert _until(lambda: client.entered["active"] == 1)
    cancelled.set()  # a: superseded while its active jobs' page is in flight
    first.join(1.0)
    assert answer == {"listing": None} and lister._reader.is_alive()
    seen = []  # the active jobs' pages asked for so far, at each newest page the next load asks for
    real = client.list_jobs_page

    def watched(**kwargs):
        if not kwargs.get("status"):
            seen.append(client.entered["active"])
        return real(**kwargs)

    client.list_jobs_page = watched
    partials = []
    second, answer = _load_in_thread(lister, on_partial=partials.append)
    assert _until(lambda: partials and partials[-1].scope.loading == ACTIVE_NEXT)
    assert set(seen) == {1}  # no second active jobs' walk beside the first: it comes after the newest
    client.hold["active"].set()
    second.join(5)
    assert answer["listing"] == _old_walk(_Paged(ACCOUNT_LIST_CAP + 300))
    assert _until(lambda: not lister._reader.is_alive())


def test_a_list_whole_when_the_active_jobs_come_shows_no_list_of_them(monkeypatch):
    # The active jobs answer while the newest jobs' last page is read but before the walk has made
    # its list of it (as when its thread waits for the lock): no "loading older active jobs" list.
    real_step = account._Walk.step

    def step_then_wait(self, is_cancelled):
        done = real_step(self, is_cancelled)
        if done and self.done and "status" not in self._filters:
            time.sleep(0.05)
        return done

    monkeypatch.setattr(account._Walk, "step", step_then_wait)
    for _ in range(10):
        client, partials = _Paged(900), []
        real = client.list_jobs_page
        client.list_jobs_page = lambda **kw: (time.sleep(0.02) if kw.get("status") else None) or real(**kw)
        listing = JobLister(client, account_wide=True).load(NOW, on_partial=partials.append)
        assert listing == _old_walk(_Paged(900))
        assert all(p.scope.loading == OLDER for p in partials), [p.scope for p in partials]


def test_no_list_is_shown_once_the_load_is_cancelled():
    for _ in range(20):
        client, cancelled = _Held(ACCOUNT_LIST_CAP + 300), threading.Event()
        client.hold["active"].set()
        real = client.list_jobs_page

        def slow_third_page(**kwargs):  # cancelled while the third newest page is read
            if not kwargs.get("status") and (kwargs.get("page_token") or "").endswith(":2"):
                cancelled.set()
                time.sleep(0.02)
            return real(**kwargs)

        client.list_jobs_page = slow_third_page
        client.hold["newest"].set()
        late = []
        JobLister(client, account_wide=True).load(
            NOW, is_cancelled=cancelled.is_set, on_partial=lambda listing: late.append(cancelled.is_set()))
        time.sleep(0.05)
        assert not any(late)


def test_a_failure_reading_the_active_jobs_beside_goes_to_the_log_not_the_thread_hook(monkeypatch, caplog):
    def broken(self, *args):
        raise RuntimeError("broken")

    monkeypatch.setattr(JobLister, "_active", broken)
    hooked = []
    monkeypatch.setattr(threading, "excepthook", hooked.append)
    lister = JobLister(_Paged(ACCOUNT_LIST_CAP + 300), account_wide=True)
    listing = lister.load(NOW, on_partial=lambda listing: None)
    assert _until(lambda: lister._reader is not None and not lister._reader.is_alive())
    assert hooked == [] and "reading the active jobs beside the newest failed" in caplog.text
    # The newest jobs stand, saying active ones may be missing, as when the walk fails in a request.
    assert listing.scope == ListScope(ACCOUNT, stopped=CAPPED, walked=ACCOUNT_LIST_CAP, active_missing=True)


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

