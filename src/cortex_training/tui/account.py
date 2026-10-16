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

"""Where the TUI's job list comes from, and which client each job's calls use.

The job list shows the configured schema's jobs (``list_jobs``) until ``a``
asks for the account-level list (``GET /api/v2/cortex-training``): the jobs
this role can see in every database and schema (:attr:`JobLister.account_wide`;
``a`` again goes back). Where the service doesn't have that route yet, or
hasn't enabled it for the account, a load of it lists the configured schema
instead (:data:`FALLBACK`); the list screen then goes back to the configured
schema and says why when ``a`` had just asked (the answer to that ask: an
earlier one counts only until the list is asked again). A list already showing
every location keeps the configured schema in its place and asks the
account-level list again at most every :data:`ACCOUNT_REPROBE_SECONDS` (``r``
or ``a`` asks once the last ask is :data:`ACCOUNT_MANUAL_REPROBE_SECONDS` old).
``--schema-only`` keeps to the configured schema and never asks.

One refresh reads the newest jobs, at most :data:`ACCOUNT_LIST_CAP` of them or
:data:`ACCOUNT_LIST_MAX_PAGES` pages. When more exist, it also reads the active
jobs, so a job still running that was created before the newest ones is listed.

A row of the account-level list names its database and schema. Every call
about that job (``get_job``, logs, GPU metrics) goes through a
client bound to that location (:class:`JobClients`); a job without one (the
configured schema's own list, or a job opened by id) uses the TUI's client.

Nothing here imports ``textual``: the job list runs :meth:`JobLister.load` in a
thread worker, and every rule is unit-testable on its own.
"""

from __future__ import annotations

import inspect
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Callable
from typing import Optional

from cortex_training import AccountJobListUnavailable
from cortex_training import InvalidPageToken
from cortex_training.tui import views

logger = logging.getLogger(__name__)

# The newest jobs one refresh reads from the account-level list. A refresh never
# reads further back than this, so where more jobs match it lists these and
# says that more exist.
ACCOUNT_LIST_CAP = 2_000
ACCOUNT_PAGE_SIZE = 500
# The service examines a bounded number of jobs per call, so a page can be short,
# or empty, and still carry a token: a walk also stops after this many pages.
ACCOUNT_LIST_MAX_PAGES = 20
# Once the account-level list answered that it isn't deployed or enabled, the
# list shows the configured schema and asks it again at most this often...
ACCOUNT_REPROBE_SECONDS = 600
# ...or, when someone presses r, once the last ask is this old.
ACCOUNT_MANUAL_REPROBE_SECONDS = 60
# The list refreshes every 60 s, so the tenth refresh after an ask comes ten
# minutes on, give or take a fraction of a second: this much slack keeps it from
# waiting for the eleventh.
_ASK_SLACK_SECONDS = 5.0
# Each call of a refresh: the client's own defaults retry a failing call ten
# times with a ten-minute read timeout, far too long for a list on screen.
REFRESH_MAX_RETRIES = 2
REFRESH_TIMEOUT = (10.0, 45.0)
# The account-level list's status group for pending, placing, initializing and
# running jobs. Under any status filter the service leaves out the jobs whose
# status it can't read, so only the walk for the jobs a cut-short list may miss
# uses it.
ACTIVE = "active"

ACCOUNT = "account"  # the account-level list answered
SCHEMA = "schema"  # the configured schema's list: the default, while the account-level one isn't asked for
FALLBACK = "fallback"  # the account-level list was asked for and isn't there: the configured schema's list
SCHEMA_ONLY = "schema-only"  # --schema-only: the configured schema's list, never the account-level one

CAPPED = "cap"  # a walk read ACCOUNT_LIST_CAP jobs, and more exist
PAGE_LIMIT = "pages"  # a walk read ACCOUNT_LIST_MAX_PAGES pages, and the last one carried a token

# Reasons that concern only the account-level route: list_jobs still lists the
# configured schema. The other two (forbidden, rest_disabled) come from gates the
# per-schema routes share, so the schema's list would most likely fail the same way.
_FALLBACK_REASONS = frozenset({AccountJobListUnavailable.NOT_DEPLOYED, AccountJobListUnavailable.DISABLED})
# Short: the header holds the refresh status, then "Current schema only: DB.SCHEMA (why)",
# and that fits a 140-column terminal with the documented CORTEX_TRAINING_DB.PUBLIC.
_FALLBACK_WHY = {
    AccountJobListUnavailable.NOT_DEPLOYED: "no account-wide list yet",
    AccountJobListUnavailable.DISABLED: "account-wide list not enabled",
    AccountJobListUnavailable.FORBIDDEN: "account-wide list: HTTP 403",
    AccountJobListUnavailable.REST_DISABLED: "account-wide list: HTTP 405",
}
_NO_ACCOUNT_LIST = "no account-wide list"  # a client without one
_FAILURE_LIMIT = 300  # cells of a failure's one-line text
_ACCOUNT_ROUTE, _SCHEMA_ROUTE = "the account-wide job list", "list_jobs"


class JobListError(Exception):
    """A refresh of the job list failed.

    ``str()`` says what failed, on one line, and ``__cause__`` is the error.
    ``refused``: the account-level list refused this caller (HTTP 403 or 405)
    and the configured schema's list (the job list's default) wasn't asked.
    """

    def __init__(self, text: str, *, refused: bool = False):
        super().__init__(text)
        self.refused = refused


@dataclass(frozen=True)
class ListScope:
    """What one listing covers."""

    kind: str  # ACCOUNT, SCHEMA, FALLBACK or SCHEMA_ONLY
    # ACCOUNT: why the walk stopped with more jobs left to read (CAPPED or
    # PAGE_LIMIT), or None when it read them all.
    stopped: Optional[str] = None
    walked: int = 0  # ACCOUNT, stopped: the jobs that walk read, the newest
    pages: int = 0  # ACCOUNT, stopped at PAGE_LIMIT: the pages it read
    active_added: int = 0  # ACCOUNT, stopped: older active jobs the walk for active jobs added
    active_missing: bool = False  # ACCOUNT, stopped: that walk failed or stopped early, so some may be missing
    reason: Optional[str] = None  # FALLBACK: why the account-level list is unavailable

    @property
    def truncated(self) -> bool:
        """The list holds only the newest jobs: more exist, or may exist."""
        return self.stopped is not None


@dataclass
class Listing:
    """One refresh's jobs, as a plain list, and what they cover."""

    jobs: list
    scope: ListScope


def _jobs(count: int) -> str:
    return f"{count:,} job" + ("" if count == 1 else "s")


def _more(scope: ListScope) -> str:
    more = "more exist" if scope.stopped == CAPPED else "more may exist"
    return more + ("; older active jobs may be missing" if scope.active_missing else "")


def scope_text(scope: ListScope, *, location: str, only: str = "") -> str:
    """The list's scope in a few words, for the header.

    ``location`` is the configured ``DB.SCHEMA``; ``only`` is the one
    ``DB.SCHEMA`` every job of an account-wide listing is in, if they share one
    (the list shows no Location column then, so the header names it). For
    example "Current schema: CORTEX_TRAINING_DB.PUBLIC" (the default), "all
    schemas", "newest 2,000 + every active job (more exist)" or "Current schema
    only: CORTEX_TRAINING_DB.PUBLIC (--schema-only)".
    """
    if scope.kind == ACCOUNT:
        if scope.stopped is None:
            return f"all schemas (all in {only})" if only else "all schemas"
        held = f"newest {scope.walked:,}" if scope.stopped == CAPPED else (
            f"first {scope.pages} pages: {_jobs(scope.walked)}")
        return held + ("" if scope.active_missing else " + every active job") + f" ({_more(scope)})"
    if scope.kind == SCHEMA:
        return f"Current schema: {location}"
    why = "--schema-only" if scope.kind == SCHEMA_ONLY else unavailable_text(scope)
    return f"Current schema only: {location} ({why})"


def unavailable_text(scope: ListScope) -> str:
    """Why a listing of the account-level list holds the configured schema's jobs instead (a
    :data:`FALLBACK` scope), in a few words: "no account-wide list yet", "account-wide list not
    enabled", "account-wide list: HTTP 403" or "HTTP 405", or "no account-wide list" (a client
    without one)."""
    return _FALLBACK_WHY.get(scope.reason or "", _NO_ACCOUNT_LIST)


def window_text(scope: Optional[ListScope], *, recent: int, hidden: int) -> str:
    """The jobs the list holds, for the line under the header.

    Short: that line also carries the live and historical counts, the sort and
    a search's matches, within a 140-column terminal.
    """
    if scope is None or scope.stopped is None:
        return f"last 6 months ({recent:,} jobs" + (f", {hidden:,} older hidden" if hidden else "") + ")"
    held = (f"newest {_jobs(scope.walked)}" if scope.stopped == CAPPED
            else f"first {scope.pages} pages ({_jobs(scope.walked)})")
    return held + (f" + {scope.active_added:,} older active" if scope.active_added else "")


def loaded_text(scope: Optional[ListScope]) -> str:
    """What a list cut short holds, for a search without results; "" when it holds every job."""
    if scope is None or scope.stopped is None:
        return ""
    held = (f"the newest {scope.walked:,} of the last 6 months" if scope.stopped == CAPPED
            else f"what the first {scope.pages} pages held")
    return held + ("" if scope.active_missing else " and every active job") + f"; {_more(scope)}"


def empty_text(scope: ListScope) -> str:
    """The list's message when a walk cut short left no job to show."""
    held = (f"in its first {scope.pages} pages" if scope.stopped == PAGE_LIMIT
            else f"among its newest {scope.walked:,}")
    return (f"The account-wide list returned no jobs to show {held}; more may exist. r reads it again; "
            "--database DB --schema SCHEMA --schema-only lists one schema's jobs.")


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
    """Whether a refusal carries the service's own JSON error message.

    A 403 without one may come from a network policy or a proxy in front of the
    account-level route, which may let the per-schema routes through.
    """
    try:
        body = exc.response.json()  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - no response, or no JSON in it
        return False
    return _has_message(body)


def _takes_bounds(method: Any) -> bool:
    """Whether ``list_jobs`` takes ``max_retries`` and ``request_timeout`` (this package's client
    does; an older or a custom one may not)."""
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False
    return "max_retries" in parameters and "request_timeout" in parameters


class JobLister:
    """Reads the job list for each refresh of the job list screen.

    Call :meth:`load` from a worker thread, one call at a time (the list screen
    never starts a second refresh while one is in flight). Between calls it
    remembers which list the job list shows (:attr:`account_wide`), whether the
    account-level list was unavailable, and when it last asked.
    """

    def __init__(self, client: Any, *, schema_only: bool = False, account_wide: bool = False,
                 clock: Callable[[], float] = time.monotonic):
        self._client = client
        self.schema_only = schema_only
        # The list the job list shows: the configured schema's (False, the default)
        # or the account-level one (True), which ``a`` switches between. Shared by
        # every job list the app shows. --schema-only overrides it.
        self.account_wide = account_wide
        self._clock = clock
        self._fallback: Optional[str] = None  # why the account-level list is unavailable, while it is
        self._asked_at = 0.0  # in fallback: when the account-level list last answered (or failed)

    @property
    def fallback_reason(self) -> Optional[str]:
        """The ``AccountJobListUnavailable`` reason while the list shows the configured schema instead."""
        return self._fallback

    def load(self, now: datetime, *, is_cancelled: Callable[[], bool] = lambda: False,
             manual: bool = False, account: Optional[bool] = None, asked: bool = False) -> Optional[Listing]:
        """Worker thread. One refresh's jobs, or None when ``is_cancelled`` turned true.

        ``account``: read the account-level list (True) or the configured
        schema's (False); None reads the one :attr:`account_wide` names (the
        list screen passes the one it asked for, as ``a`` can switch while a
        load runs). ``manual``: someone pressed r or a, so while the
        account-level list is unavailable it is asked again sooner. ``asked``:
        this is the load ``a`` asked for, so once the account-level list is
        asked again, what it answers is what ``a``'s notice says: a failure
        or a refusal raises, rather than keeping the configured schema with
        the earlier reason (that rule is for a list already showing every
        location). Within :data:`ACCOUNT_MANUAL_REPROBE_SECONDS` of an
        unavailable answer, ``a`` still gets that answer without asking.
        Raises :class:`JobListError` when the refresh fails.
        """
        if self.schema_only:
            return self._schema(ListScope(SCHEMA_ONLY))
        if not (self.account_wide if account is None else account):
            return self._schema(ListScope(SCHEMA))
        if not callable(getattr(self._client, "list_jobs_page", None)):
            # A client without the account-level list (an older or a custom one).
            return self._schema(ListScope(FALLBACK))
        if self._fallback is not None and not self._ask_again(manual):
            return self._schema(ListScope(FALLBACK, reason=self._fallback))
        if asked:
            self._fallback = None  # asked again for a: the earlier answer no longer counts
        probing = self._fallback is not None
        try:
            listing = self._walk(now, is_cancelled)
        except AccountJobListUnavailable as exc:
            return self._unavailable(exc, probing)
        except Exception as exc:
            if not probing:
                raise JobListError(_failure_text(_ACCOUNT_ROUTE, exc)) from exc
            # Asking again failed some other way (a timeout, a 503): keep listing
            # the configured schema, and ask again later.
            self._asked_at = self._clock()
            logger.info("job list: the account-level list failed again; listing the configured schema", exc_info=True)
            return self._schema(ListScope(FALLBACK, reason=self._fallback))
        if listing is not None:
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
            # Not there for this account; or, while the configured schema's list is
            # shown, refused now: keep showing it, and say why.
            self._fall_back(exc)
            return self._schema(ListScope(FALLBACK, reason=exc.reason))
        refused = JobListError(_one_line(str(exc)), refused=True)
        # 403 and 405 come from gates the per-schema routes share, so their list
        # would most likely be refused too: the refusal is the list's error. A 403
        # without the service's own message may come from a network policy or a
        # proxy in front of this route alone, so the configured schema is asked once.
        if exc.reason != AccountJobListUnavailable.FORBIDDEN or _service_explained(exc):
            raise refused from exc
        try:
            listing = self._schema(ListScope(FALLBACK, reason=exc.reason))
        except JobListError as schema_failed:
            # Refused there too: whatever refuses is in front of both, so the configured schema's list wouldn't help.
            raise JobListError(_one_line(f"{exc}; and {schema_failed}")) from exc
        self._fall_back(exc)
        return listing

    def _schema(self, scope: ListScope) -> Listing:
        method = self._client.list_jobs
        try:
            if _takes_bounds(method):
                jobs = method(max_retries=REFRESH_MAX_RETRIES, request_timeout=REFRESH_TIMEOUT)
            else:
                jobs = method()
        except Exception as exc:
            raise JobListError(_failure_text(_SCHEMA_ROUTE, exc)) from exc
        return Listing([job for job in (jobs or []) if isinstance(job, dict)], scope)

    def _walk(self, now: datetime, is_cancelled: Callable[[], bool]) -> Optional[Listing]:
        """The newest jobs of the account-level list, and the active ones when more exist; None when cancelled."""
        # Once per refresh: a page token only opens with the filters it was issued for.
        created_after = now - timedelta(days=views.RETENTION_DAYS)
        walked = self._pages(created_after, None, is_cancelled)
        if walked is None:
            return None
        jobs, stopped = walked
        if stopped is None:
            return Listing(jobs, ListScope(ACCOUNT))
        # More jobs exist than one refresh reads, so a job still running that was
        # created before the newest ones would be missing: read the active jobs
        # too (a short list) and add those not listed yet.
        try:
            active = self._pages(created_after, ACTIVE, is_cancelled)
        except Exception:  # noqa: BLE001 - the newest jobs stand; the scope says active ones may be missing
            logger.info("job list: reading the active jobs failed; older active jobs may be missing", exc_info=True)
            active = ([], "failed")
        if active is None:
            return None
        rows, active_stopped = active
        listed = {views.job_id_of(job) for job in jobs}
        added = [job for job in rows if views.job_id_of(job) not in listed]
        scope = ListScope(ACCOUNT, stopped=stopped, walked=len(jobs),
                          pages=ACCOUNT_LIST_MAX_PAGES if stopped == PAGE_LIMIT else 0,
                          active_added=len(added), active_missing=active_stopped is not None)
        return Listing(jobs + added, scope)

    def _pages(self, created_after: datetime, status: Optional[str],
               is_cancelled: Callable[[], bool]) -> Optional[tuple]:
        """One walk; it starts again from the first page, once, when the service can't open its token."""
        try:
            return self._read_pages(created_after, status, is_cancelled)
        except InvalidPageToken:
            # The service could no longer open its token (it rotated the key that
            # seals them, say): read again from the first page, once.
            logger.info("job list: the service could not open its page token; reading again from the first page")
            return self._read_pages(created_after, status, is_cancelled)

    def _read_pages(self, created_after: datetime, status: Optional[str],
                    is_cancelled: Callable[[], bool]) -> Optional[tuple]:
        """``(jobs, stopped)``: up to the cap or the page limit, newest first; None when cancelled."""
        # The main walk has no status filter, so the jobs whose status the service
        # can't read (UNKNOWN, which may well still be running) are listed too.
        filters: dict = {"created_after": created_after, **({"status": status} if status else {})}
        jobs: list = []
        seen: set = set()
        token = None
        for _ in range(ACCOUNT_LIST_MAX_PAGES):
            if is_cancelled():
                return None
            page = self._client.list_jobs_page(
                **filters,
                page_size=ACCOUNT_PAGE_SIZE,
                page_token=token,
                max_retries=REFRESH_MAX_RETRIES,
                request_timeout=REFRESH_TIMEOUT,
            )
            rows, token = _page(page)
            for job in rows:
                job_id = views.job_id_of(job)
                if job_id and job_id in seen:
                    continue
                seen.add(job_id)
                jobs.append(job)
            if len(jobs) >= ACCOUNT_LIST_CAP:
                more = token is not None or len(jobs) > ACCOUNT_LIST_CAP
                return jobs[:ACCOUNT_LIST_CAP], CAPPED if more else None
            if token is None:
                return jobs, None
        return jobs, PAGE_LIMIT  # stopped at the page limit with pages left


class JobClients:
    """The TUI's client, and one client per job location for the calls about a job.

    A row of the account-level list names its ``database_name`` and
    ``schema_name``; :meth:`for_job` returns a client bound to them, made once
    per location with ``client.for_job(row)`` (which never creates a database).
    A job without a location uses the TUI's own client.
    """

    def __init__(self, client: Any):
        self.base = client
        # The configured (database_name, schema_name); None when the client doesn't say.
        self.home = views.client_location(client)
        self._children: dict = {}
        self._lock = threading.Lock()

    def elsewhere(self, job: Any) -> bool:
        """Whether ``job`` names a location other than the configured one (or the client doesn't say which)."""
        names = views.location_names(job)
        return names is not None and names != self.home

    def for_job(self, job: Any) -> Any:
        """The client for the calls about ``job``.

        Raises what ``client.for_job(row)`` raises: the TUI's own client would
        send the job's calls to the configured location, which isn't the job's.
        """
        location = views.location_names(job)
        make = getattr(self.base, "for_job", None)
        if location is None or not callable(make):
            # No location (the configured schema's own list, a job opened by id), or
            # an older or a custom client without per-location clients: the TUI's own.
            return self.base
        with self._lock:
            child = self._children.get(location)
            if child is None:
                child = self._children[location] = make(job)
            return child
