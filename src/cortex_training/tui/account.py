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

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Callable
from typing import Optional

from cortex_training import AccountJobListUnavailable
from cortex_training import InvalidPageToken
from cortex_training.tui import views

logger = logging.getLogger(__name__)

# Account-list bounds for one refresh.
ACCOUNT_LIST_CAP = 2_000
ACCOUNT_PAGE_SIZE = 500
ACCOUNT_LIST_MAX_PAGES = 20
# Fallback reprobe cadence (manual refreshes may ask sooner).
ACCOUNT_REPROBE_SECONDS = 600
ACCOUNT_MANUAL_REPROBE_SECONDS = 60
_ASK_SLACK_SECONDS = 5.0
ACTIVE = "active"

ACCOUNT = "account"  # the account-level list answered
SCHEMA = "schema"  # the configured schema's list: the default, while the account-level one isn't asked for
FALLBACK = "fallback"  # the account-level list was asked for and isn't there: the configured schema's list

CAPPED = "cap"  # a walk read ACCOUNT_LIST_CAP jobs, and more exist
PAGE_LIMIT = "pages"  # a walk read ACCOUNT_LIST_MAX_PAGES pages, and the last one carried a token
FAILED = "failed"  # the walk's first pages were shown, and a later one failed (or was never read)

OLDER = "older"  # loading: the newest jobs' walk has more pages to read
ACTIVE_NEXT = "active"  # loading: that walk is done; the active jobs' walk is read next

# These failures affect only the account route, so schema listing can continue.
_FALLBACK_REASONS = frozenset({AccountJobListUnavailable.NOT_DEPLOYED, AccountJobListUnavailable.DISABLED})
_FALLBACK_WHY = {
    AccountJobListUnavailable.NOT_DEPLOYED: "no account-wide list yet",
    AccountJobListUnavailable.DISABLED: "account-wide list not enabled",
    AccountJobListUnavailable.FORBIDDEN: "account-wide list: HTTP 403",
    AccountJobListUnavailable.REST_DISABLED: "account-wide list: HTTP 405",
}
_FAILURE_LIMIT = 300  # cells of a failure's one-line text
_ACCOUNT_ROUTE, _SCHEMA_ROUTE = "the account-wide job list", "list_jobs"


class JobListError(Exception):
    """A one-line list failure; ``refused`` means the schema was not attempted."""

    def __init__(self, text: str, *, refused: bool = False):
        super().__init__(text)
        self.refused = refused


@dataclass(frozen=True)
class ListScope:
    """What one listing covers."""

    kind: str  # ACCOUNT, SCHEMA or FALLBACK
    stopped: Optional[str] = None
    walked: int = 0  # ACCOUNT, stopped: the jobs that walk read, the newest
    pages: int = 0  # ACCOUNT, stopped at PAGE_LIMIT: the pages it read
    active_missing: bool = False  # ACCOUNT, stopped: the active-job walk failed or stopped early; some may be missing
    reason: Optional[str] = None  # FALLBACK: why the account-level list is unavailable
    # ACCOUNT, shown while it is read: OLDER, or ACTIVE_NEXT (the newest jobs' walk
    # stopped as ``stopped`` says). "" once the list is whole.
    loading: str = ""
    # ACCOUNT, OLDER: the active jobs' walk, read beside the newest jobs', is done and its
    # jobs listed (``active_missing``: it failed or stopped early).
    active_read: bool = False

    @property
    def truncated(self) -> bool:
        """The list holds only the newest jobs: more exist, may exist, or are still loading."""
        return self.stopped is not None or bool(self.loading)

    @property
    def active_whole(self) -> bool:
        """Every active job is listed: the list is whole, or its active jobs' walk is read in full.
        (Not every live one: an ``unknown`` job, which no status filter lists, may still come with
        a later page of the newest jobs.)"""
        return not self.loading or (self.active_read and not self.active_missing)


def stopped_scope(scope: ListScope) -> ListScope:
    """A loading scope as its list stands once its walk stopped short (a page failed, or the load was superseded)."""
    if not scope.loading:
        return scope
    return replace(scope, loading="", stopped=scope.stopped if scope.loading == ACTIVE_NEXT else FAILED,
                   active_read=False, active_missing=not scope.active_whole)


@dataclass
class Listing:
    """One refresh's jobs, as a plain list, and what they cover."""

    jobs: list
    scope: ListScope


def _more(scope: ListScope) -> str:
    more = {CAPPED: "more exist", FAILED: "the rest didn't load; r retries"}.get(scope.stopped or "", "more may exist")
    return more + ("; older active jobs may be missing" if scope.active_missing else "")


def scope_text(scope: ListScope, *, location: str, only: str = "") -> str:
    """A short header for this scope; ``only`` names its sole location."""
    if scope.kind == ACCOUNT:
        if scope.loading == OLDER:
            active = "" if not scope.active_read else (
                " (older active jobs may be missing)" if scope.active_missing else " + every active job")
            return f"loading older jobs… {scope.walked:,} so far{active}"
        if scope.loading:
            return f"newest {scope.walked:,}; loading older active jobs…"
        if scope.stopped is None:
            return f"all schemas (all in {only})" if only else "all schemas"
        held = f"newest {scope.walked:,}" if scope.stopped in (CAPPED, FAILED) else (
            f"first {scope.pages} pages: {scope.walked:,} job{'' if scope.walked == 1 else 's'}")
        return held + ("" if scope.active_missing else " + every active job") + f" ({_more(scope)})"
    if scope.kind == SCHEMA:
        return f"Current schema: {location}"
    return f"Current schema only: {location} ({unavailable_text(scope)})"


def unavailable_text(scope: ListScope) -> str:
    """Why a :data:`FALLBACK` listing holds the configured schema's jobs, in a few words."""
    return _FALLBACK_WHY.get(scope.reason or "", "")


def loaded_text(scope: Optional[ListScope]) -> str:
    """What a list cut short holds, for a search without results; "" when it holds every job."""
    if scope is None or not scope.truncated:
        return ""
    if scope.loading:
        what = "older jobs are" if scope.loading == OLDER else "older active jobs are"
        active = " and every active job" if scope.active_read and not scope.active_missing else ""
        return f"the newest {scope.walked:,} so far{active}; {what} still loading"
    held = (f"the newest {scope.walked:,} of the last 6 months" if scope.stopped in (CAPPED, FAILED)
            else f"what the first {scope.pages} pages held")
    return held + ("" if scope.active_missing else " and every active job") + f"; {_more(scope)}"


def empty_text(scope: ListScope) -> str:
    """The list's message when a walk cut short (or still loading) left no job to show."""
    if scope.loading:
        return (f"The account-wide list has no jobs to show among its newest {scope.walked:,} yet; "
                f"{'older' if scope.loading == OLDER else 'older active'} jobs are still loading.")
    held = (f"in its first {scope.pages} pages" if scope.stopped == PAGE_LIMIT
            else f"among its newest {scope.walked:,}")
    return f"The account-wide list returned no jobs to show {held}; more may exist. r reads it again."


def _one_line(text: str) -> str:
    return views.fit_cells(views.collapse(text), _FAILURE_LIMIT)


def _failure_text(route: str, exc: BaseException) -> str:
    """What failed, on one line: the route, then the answer's status and message or the error's own text."""
    answered = isinstance(getattr(getattr(exc, "response", None), "status_code", None), int)
    return _one_line(f"{route} {'answered' if answered else 'failed:'} {views.describe_error(exc)}")


def _page(page: Any) -> tuple:
    """``(jobs, next_page_token)`` of one ``list_jobs_page`` answer, as it documents them."""
    if not isinstance(page, dict) or not isinstance(page.get("jobs"), list):
        raise ValueError("the account-level job list answered a page that is not a job list")
    token = page.get("next_page_token")
    return [job for job in page["jobs"] if isinstance(job, dict)], token if isinstance(token, str) and token else None


def _has_message(value: Any, depth: int = 0) -> bool:
    if depth > 4:
        return False
    if isinstance(value, dict):
        message = value.get("message")
        if isinstance(message, str) and message.strip():
            return True
        return any(_has_message(item, depth + 1) for item in value.values())
    if isinstance(value, list):
        return any(_has_message(item, depth + 1) for item in value)
    return False


def _service_explained(exc: BaseException) -> bool:
    """Whether a refusal carries the service's own JSON error message."""
    try:
        body = exc.response.json()  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - no response, or no JSON in it
        return False
    return _has_message(body)


class JobLister:
    def __init__(self, client: Any, *, account_wide: bool = False, clock: Callable[[], float] = time.monotonic):
        self._client = client
        self.account_wide = account_wide
        self._clock = clock
        self._fallback: Optional[str] = None  # why the account-level list is unavailable, while it is
        self._asked_at = 0.0  # in fallback: when the account-level list last answered (or failed)
        self._reader: Optional[threading.Thread] = None  # the last active jobs' walk read beside the newest
        self._reader_lock = threading.Lock()

    def load(self, now: datetime, *, is_cancelled: Callable[[], bool] = lambda: False,
             manual: bool = False, account: Optional[bool] = None, asked: bool = False,
             on_partial: Optional[Callable[[Listing], None]] = None) -> Optional[Listing]:
        """Load one refresh; flags override scope, reprobe, and fallback behavior.

        ``on_partial`` is given the account-level list as it is read (see :meth:`_walk`), except while
        asking again in fallback: the configured schema's list stands until the account-level one is whole.
        """
        if not (self.account_wide if account is None else account):
            return self._schema(ListScope(SCHEMA))
        if self._fallback is not None and not self._ask_again(manual):
            return self._schema(ListScope(FALLBACK, reason=self._fallback))
        if asked:
            self._fallback = None  # asked again for a: the earlier answer no longer counts
        probing = self._fallback is not None
        try:
            listing = self._walk(now, is_cancelled, None if probing else on_partial)
        except AccountJobListUnavailable as exc:
            if is_cancelled():
                return None  # superseded while its page was read: the load in its place decides
            return self._unavailable(exc, probing)
        except Exception as exc:
            if is_cancelled():
                return None
            if not probing:
                raise JobListError(_failure_text(_ACCOUNT_ROUTE, exc)) from exc
            # Asking again failed some other way (a timeout, a 503): keep listing
            # the configured schema, and ask again later.
            self._asked_at = self._clock()
            logger.info("job list: the account-level list failed again; listing the configured schema", exc_info=True)
            return self._schema(ListScope(FALLBACK, reason=self._fallback))
        if listing is not None and not is_cancelled():
            self._fallback = None
        return listing

    def _ask_again(self, manual: bool) -> bool:
        waited = self._clock() - self._asked_at + _ASK_SLACK_SECONDS
        return waited >= ACCOUNT_REPROBE_SECONDS or (manual and waited >= ACCOUNT_MANUAL_REPROBE_SECONDS)

    def _fall_back(self, exc: AccountJobListUnavailable) -> None:
        self._fallback = exc.reason
        self._asked_at = self._clock()
        logger.info("job list: %s; listing the configured schema instead", exc)

    def _unavailable(self, exc: AccountJobListUnavailable, probing: bool) -> Listing:
        if exc.reason in _FALLBACK_REASONS or probing:
            self._fall_back(exc)
            return self._schema(ListScope(FALLBACK, reason=exc.reason))
        refused = JobListError(_one_line(str(exc)), refused=True)
        # An unexplained 403 may affect only this route; probe the schema once.
        if exc.reason != AccountJobListUnavailable.FORBIDDEN or _service_explained(exc):
            raise refused from exc
        try:
            listing = self._schema(ListScope(FALLBACK, reason=exc.reason))
        except JobListError as schema_failed:
            raise JobListError(_one_line(f"{exc}; and {schema_failed}")) from exc
        self._fall_back(exc)
        return listing

    def _schema(self, scope: ListScope) -> Listing:
        try:
            jobs = self._client.list_jobs()
        except Exception as exc:
            raise JobListError(_failure_text(_SCHEMA_ROUTE, exc)) from exc
        return Listing([job for job in (jobs or []) if isinstance(job, dict)], scope)

    def _walk(self, now: datetime, is_cancelled: Callable[[], bool],
              on_partial: Optional[Callable[[Listing], None]] = None) -> Optional[Listing]:
        """The newest jobs of the account-level list, and the active ones when more exist; None when cancelled.

        Without ``on_partial`` (a refresh of a list that is whole), the active jobs' walk follows the
        newest jobs', as only a walk that stopped short needs it. With it, ``on_partial`` is given what
        is read so far (a :class:`Listing` whose scope is ``loading``) after each page of the newest
        jobs that isn't the last one read, and the active jobs' walk is read beside the newest jobs'
        from their second page on (on a busy account it is pages long too, and its first page the
        slowest), its jobs listed once it is done. The result is the same either way.
        """
        created_after = now - timedelta(days=views.RETENTION_DAYS)
        if on_partial is not None:
            return _Beside(self, created_after, is_cancelled, on_partial).walk()
        newest = _Walk(self._client, created_after, None)
        while not newest.done:
            if not newest.step(is_cancelled):
                return None
        if newest.stopped is None:
            return Listing(newest.jobs, ListScope(ACCOUNT))
        active = self._active(created_after, is_cancelled)
        if active is None:
            return None
        return _listing(newest, active)

    def _start_reader(self, read: Callable[[], None]) -> Optional[threading.Thread]:
        """A thread reading the active jobs beside the newest (``read``), started; None while the
        last one started is still reading (a load cancelled in its request): no second beside it."""
        with self._reader_lock:
            if self._reader is not None and self._reader.is_alive():
                return None
            self._reader = threading.Thread(target=read, name="active-jobs-walk", daemon=True)
            self._reader.start()
            return self._reader

    def _active(self, created_after: datetime, is_cancelled: Callable[[], bool]) -> Optional[tuple]:
        """``(jobs, stopped)`` of the active-job walk (stopped "failed" when it failed); None when cancelled."""
        walk = _Walk(self._client, created_after, ACTIVE)
        try:
            while not walk.done:
                if not walk.step(is_cancelled):
                    return None
        except Exception:  # noqa: BLE001 - the newest jobs stand; the scope says active ones may be missing
            logger.info("job list: reading the active jobs failed; older active jobs may be missing", exc_info=True)
            return [], "failed"
        return walk.jobs, walk.stopped


def _listing(newest: "_Walk", active: tuple) -> Listing:
    """The list of a newest jobs' walk that stopped short, and of the active jobs' walk ``(jobs, stopped)``."""
    rows, active_stopped = active
    scope = ListScope(ACCOUNT, stopped=newest.stopped, walked=len(newest.jobs),
                      pages=ACCOUNT_LIST_MAX_PAGES if newest.stopped == PAGE_LIMIT else 0,
                      active_missing=active_stopped is not None)
    return Listing(_merge(newest.jobs, rows), scope)


class _Beside:
    """One walk of the account-level list that shows its pages: the newest jobs' walk on the
    caller's thread and, from its second page on, the active jobs' walk on another (at most two
    requests in flight), each list shown made under one lock, so they reach ``on_partial`` in
    the order they were made, and none once the walk has answered or was cancelled.

    An active jobs' walk of an earlier load that is still reading (it was cancelled during a
    request) keeps its thread until that request answers; while it does, a new load reads the
    active jobs after the newest ones instead, so loads in a row don't pile requests up."""

    _POLL_SECONDS = 0.1  # how soon a walk waiting for the active jobs sees it was cancelled

    def __init__(self, lister: "JobLister", created_after: datetime, is_cancelled: Callable[[], bool],
                 on_partial: Callable[[Listing], None]) -> None:
        self._lister, self._created_after = lister, created_after
        self._is_cancelled, self._on_partial = is_cancelled, on_partial
        self._lock = threading.Lock()
        self._stop = threading.Event()  # the walk answered: the active jobs' walk asks for no other page
        self._newest = _Walk(lister._client, created_after, None)
        self._shown: Optional[list] = None  # the newest jobs last shown (a walk read again shows none fewer)
        self._active: Optional[tuple] = None  # the active jobs' walk, once done
        # The newest jobs' walk as of the last list made: whether it is done, and how it stopped.
        self._newest_done = False
        self._newest_stopped: Optional[str] = None

    def walk(self) -> Optional[Listing]:
        newest = self._newest
        reader = None
        beside = True  # the active jobs may be read beside the newest
        try:
            while not newest.done:
                if not newest.step(self._is_cancelled):
                    return None
                if reader is None and beside and not newest.done:
                    reader = self._lister._start_reader(self._read_active)
                    beside = reader is not None
                with self._lock:
                    self._newest_done, self._newest_stopped = newest.done, newest.stopped
                    if newest.done and newest.stopped is None:
                        break  # whole: the result comes next
                    if self._shown is None or len(newest.jobs) >= len(self._shown):
                        self._shown = list(newest.jobs)
                        if not (newest.done and self._active is not None):  # else the result comes next
                            self._report()
            if newest.stopped is None:
                return Listing(newest.jobs, ListScope(ACCOUNT))
            if reader is None:  # the first page stopped it short, or an earlier walk still reads
                active = self._lister._active(self._created_after, self._is_cancelled)
            else:
                while reader.is_alive():
                    reader.join(self._POLL_SECONDS)
                    if self._is_cancelled():
                        return None
                # A walk that ended with no answer failed past its own handling (it was logged).
                active = self._active if self._active is not None else ([], "failed")
            if active is None or self._is_cancelled():
                return None
            return _listing(newest, active)
        finally:
            with self._lock:
                self._stop.set()

    def _read_active(self) -> None:
        try:
            active = self._lister._active(self._created_after, lambda: self._is_cancelled() or self._stop.is_set())
            with self._lock:
                if active is None or self._stop.is_set() or self._is_cancelled():
                    return
                self._active = active
                # Shown with the newest jobs: now, or with their next list if none is made yet; once
                # the newest jobs' walk is done, the walk answers with them.
                if not self._newest_done and self._shown is not None:
                    self._report()
        except Exception:  # noqa: BLE001 - not to the thread's hook, which would print over the TUI
            logger.warning("job list: reading the active jobs beside the newest failed", exc_info=True)

    def _report(self) -> None:
        """Show what is read so far (called under the lock, from the walk's state as of the last list made);
        nothing once the load was cancelled (a page that answers after it is not shown)."""
        if self._is_cancelled():
            return
        active = self._active
        if self._newest_done:  # stopped short, the active jobs not read yet
            stopped = self._newest_stopped
            scope = ListScope(ACCOUNT, stopped=stopped, walked=len(self._shown), loading=ACTIVE_NEXT,
                              pages=ACCOUNT_LIST_MAX_PAGES if stopped == PAGE_LIMIT else 0)
        else:
            scope = ListScope(ACCOUNT, walked=len(self._shown), loading=OLDER, active_read=active is not None,
                              active_missing=active is not None and active[1] is not None)
        jobs = _merge(self._shown, active[0]) if active is not None else list(self._shown)
        try:
            self._on_partial(Listing(jobs, scope))
        except Exception:  # noqa: BLE001 - showing it failed; the walk goes on
            logger.warning("job list: showing the jobs read so far failed", exc_info=True)


def _merge(jobs: list, active: list) -> list:
    """``jobs``, then the active jobs not among them (a new list)."""
    listed = {views.job_key(job) for job in jobs}
    return jobs + [job for job in active if views.job_key(job) not in listed]


class _Walk:
    """One walk of the account-level list, a page at a time: up to the cap or the page limit, newest first.
    It starts again from the first page, once, when the service can't open its token."""

    def __init__(self, client: Any, created_after: datetime, status: Optional[str]) -> None:
        self._client = client
        self._filters: dict = {"created_after": created_after, **({"status": status} if status else {})}
        self._restarted = False
        self._start()

    def _start(self) -> None:
        self.jobs: list = []
        self.done = False
        self.stopped: Optional[str] = None
        self._seen: set = set()
        self._token: Optional[str] = None
        self._pages = 0

    def step(self, is_cancelled: Callable[[], bool]) -> bool:
        """Read the next page; False when cancelled first."""
        if is_cancelled():
            return False
        try:
            page = self._read()
        except InvalidPageToken:
            if self._restarted:
                raise
            logger.info("job list: the service could not open its page token; reading again from the first page")
            self._restarted = True
            self._start()
            if is_cancelled():
                return False
            page = self._read()
        rows, self._token = _page(page)
        self._pages += 1
        for job in rows:
            key = views.job_key(job)
            if views.job_id_of(job) and key in self._seen:
                continue
            self._seen.add(key)
            self.jobs.append(job)
        if len(self.jobs) >= ACCOUNT_LIST_CAP:
            more = self._token is not None or len(self.jobs) > ACCOUNT_LIST_CAP
            self.jobs = self.jobs[:ACCOUNT_LIST_CAP]
            self.done, self.stopped = True, CAPPED if more else None
        elif self._token is None:
            self.done = True
        elif self._pages >= ACCOUNT_LIST_MAX_PAGES:
            self.done, self.stopped = True, PAGE_LIMIT  # stopped at the page limit with pages left
        return True

    def _read(self) -> Any:
        return self._client.list_jobs_page(**self._filters, page_size=ACCOUNT_PAGE_SIZE, page_token=self._token)


class JobClients:
    """The base client plus one cached child per account-list location."""

    def __init__(self, client: Any):
        self.base = client
        self.home = views.client_location(client)
        self._children: dict = {}
        self._lock = threading.Lock()

    def elsewhere(self, job: Any) -> bool:
        """Whether ``job`` names a location other than the configured one (or the client doesn't say which)."""
        names = views.location_names(job)
        return names is not None and names != self.home

    def for_job(self, job: Any) -> Any:
        """The location-bound client for ``job``."""
        location = views.location_names(job)
        if location is None:  # the configured schema's own list, or a job opened by id
            return self.base
        with self._lock:
            child = self._children.get(location)
            if child is None:
                child = self._children[location] = self.base.for_job(job)
            return child
