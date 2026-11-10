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

import bisect
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
ACCOUNT_LIST_CAP = 1_000  # one page of the list: page 1 is the newest, page 2 the ones before
ACCOUNT_PAGE_SIZE = 100
# 100 a page: enough pages that ACCOUNT_LIST_CAP, not the page limit, is what a
# full walk stops at (a short or empty page with a token still counts as one).
ACCOUNT_LIST_MAX_PAGES = 40
# The most jobs a search covers: a list's first page, and the rest of the list read behind it
# (see JobLister.read_rest), which stops there on a very large account.
ARCHIVE_CAP = 25_000
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


class _BadNarrowing(Exception):
    """The service refused the database or schema a narrowed walk asked for (HTTP 400).

    A name this role cannot see is answered 400, which carries no typed reason of its own, so
    without this it would read as the whole account list failing. :meth:`JobLister.load` reads
    the account whole instead and lets the search say that nothing matches.
    """


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
    walked: int = 0  # ACCOUNT, stopped: the jobs in history that walk read, the newest (see _in_history)
    pages: int = 0  # ACCOUNT, stopped at PAGE_LIMIT: the pages it read
    active_missing: bool = False  # ACCOUNT, stopped: the active-job walk failed or stopped early; some may be missing
    reason: Optional[str] = None  # FALLBACK: why the account-level list is unavailable
    # ACCOUNT, shown while it is read: OLDER, or ACTIVE_NEXT (the newest jobs' walk
    # stopped as ``stopped`` says). "" once the list is whole.
    loading: str = ""
    # ACCOUNT, OLDER: the active jobs' walk, read beside the newest jobs', is done and its
    # jobs listed (``active_missing``: it failed or stopped early).
    active_read: bool = False
    # ACCOUNT: the location the walk asked the server for (``TEAM_DB.EXPERIMENTS``, or
    # ``TEAM_DB`` for a database alone), so the header can say the list is not the whole
    # account. "" when it read every schema. Set from what the search box pinned with ``=``
    # (see :func:`cortex_training.tui.views.pushdown_location`); the rows a listing holds were
    # read under it, so the header must read it from here and never from the live search text.
    narrowed: str = ""
    # Which page of the list this is (see PageBound): 1 for the newest jobs, 2 for the ones
    # before those, and so on. Every page lists every active job.
    page: int = 1

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


# How much later than a page's oldest job the next page reaches: created_at is often whole
# seconds, so the next page takes in that second too, less the jobs this page holds.
_BOUND_SLACK = timedelta(seconds=1)


@dataclass(frozen=True)
class PageBound:
    """Where a page past the first starts: the newest jobs created before ``before`` plus
    :data:`_BOUND_SLACK`, less ``skip`` (the :func:`views.job_key` of jobs the pages before
    hold), so jobs near the boundary are neither lost nor listed twice. A bound, not
    a page token: each load reads from the first page again, and a token is only good for the
    walk that made it."""

    page: int
    before: datetime
    skip: frozenset = frozenset()

    @property
    def created_before(self) -> datetime:
        return self.before + _BOUND_SLACK


@dataclass
class Listing:
    """One refresh's jobs, as a plain list, and what they cover."""

    jobs: list
    scope: ListScope
    # A schema's whole list cut as a walk cuts one (see cut_whole_listing): every job it held.
    everything: Optional[list] = None
    # Where the page after this one starts, when older jobs exist (or may); None on the last page.
    next_page: Optional[PageBound] = None


@dataclass
class Rest:
    """One request's worth of what :meth:`JobLister.read_rest` reads behind a list."""

    jobs: list  # the jobs this request read, newest first
    done: bool = False  # nothing more comes: the list is read to its end, stopped at ARCHIVE_CAP, or failed
    capped: bool = False  # stopped at ARCHIVE_CAP (or the request limit) with more left
    failed: str = ""  # why it stopped short


def _page_of(before: Optional[PageBound]) -> int:
    return before.page if before is not None else 1


def _next_bound(newest: list, before: Optional[PageBound]) -> Optional[PageBound]:
    """The bound of the page after one whose newest jobs (not the active ones merged in) are
    ``newest``: before its oldest dated job, skipping every job it holds; None when it holds no
    dated job.

    Every one, not only those of its oldest second: the service bounds by the time it records,
    which can be a second off the ``created_at`` it reports, so a page's last jobs can come back
    on the next one. Its undated jobs, which a bound can't place, are skipped with the rest."""
    dated = [stamp for stamp in (views.parse_ts(job.get("created_at")) for job in newest) if stamp is not None]
    if not dated:
        return None
    oldest = min(dated)
    skip = {views.job_key(job) for job in newest}
    if before is not None and oldest < before.created_before:
        skip |= before.skip  # the whole page within the bound's own second: what it skipped still is
    return PageBound(_page_of(before) + 1, oldest, frozenset(skip))


def _in_history(job: dict) -> bool:
    """Whether ``job`` counts toward a page's ACCOUNT_LIST_CAP: one the job list files under
    history. Its deployments (shown apart, its ``d``) and live jobs (every page lists the active
    ones) come with the page beside its 1,000, so every page but the last holds 1,000 in history."""
    return not views.is_deployment(job) and not views.is_live(job)


def _job_count(jobs: list) -> int:
    """The jobs among ``jobs`` that count toward a page (see :func:`_in_history`)."""
    return sum(1 for job in jobs if _in_history(job))


def _cap_cut(jobs: list, counts: Callable[[dict], bool] = _in_history, cap: Optional[int] = None) -> Optional[int]:
    """Where a page of ``jobs`` (newest first) ends: just past its ``cap``-th job (by default
    ACCOUNT_LIST_CAP) that ``counts``; None while it holds fewer."""
    cap = ACCOUNT_LIST_CAP if cap is None else cap
    count = 0
    for at, job in enumerate(jobs):
        if counts(job):
            count += 1
            if count == cap:
                return at + 1
    return None


def _created_at(job: dict) -> float:
    """``created_at`` as epoch seconds; a job without one counts as the oldest."""
    stamp = views.parse_ts(job.get("created_at"))
    return stamp.timestamp() if stamp is not None else float("-inf")


def cut_whole_listing(listing: Listing, before: Optional[PageBound] = None) -> Listing:
    """A whole list of more than ACCOUNT_LIST_CAP jobs, held as the account-level walk holds
    one: its newest ACCOUNT_LIST_CAP jobs and every active job older than them, its scope
    stopped at CAPPED, and every job in ``everything``. With ``before``, the page it starts:
    the newest ACCOUNT_LIST_CAP of the jobs before it, and every active job of the list.

    ``list_jobs`` answers a schema whole, so where the account-level list is not there a
    schema of tens of thousands came whole too, and drawing it froze the list for seconds.
    The job list shows ``everything`` when asked (its ``f``).
    """
    jobs = listing.jobs
    if before is None and _job_count(jobs) <= ACCOUNT_LIST_CAP:
        return listing
    if before is None:
        candidates = jobs
    else:
        edge = before.created_before.timestamp()
        candidates = [job for job in jobs if _created_at(job) < edge and views.job_key(job) not in before.skip]
    ordered = sorted(candidates, key=_created_at, reverse=True)
    cut = _cap_cut(ordered)
    newest = ordered if cut is None else ordered[:cut]
    held = {id(job) for job in newest}
    active = [job for job in jobs if id(job) not in held
              and views.status_word(job.get("status")) in views.ACTIVE_STATUSES]
    more = len(newest) < len(ordered)
    scope = replace(listing.scope, stopped=CAPPED if more else None, walked=_job_count(newest), active_read=True,
                    page=_page_of(before))
    return Listing(newest + active, scope, everything=jobs, next_page=_next_bound(newest, before) if more else None)


def sorted_for_pages(jobs) -> tuple:
    """``(ordered, keys)``: ``jobs`` newest first, and their negated creation times (ascending),
    which :func:`cut_sorted` bisects to find where a page starts. Made once for many pages: a page
    cut from them costs its own jobs, not the list's."""
    ordered = sorted(jobs, key=_created_at, reverse=True)
    return ordered, [-_created_at(job) for job in ordered]


def cut_sorted(ordered: list, keys: list, before: PageBound, scope: ListScope) -> Listing:
    """The page ``before`` starts, cut from ``ordered`` and ``keys`` (see :func:`sorted_for_pages`)
    as :func:`cut_whole_listing` cuts a whole list: its newest ACCOUNT_LIST_CAP jobs in history
    created before the bound and not skipped, the live jobs and deployments among them, then every
    active job of the list; ``scope`` stopped at CAPPED while older jobs are left."""
    at = bisect.bisect_right(keys, -before.created_before.timestamp())  # the first job created before it
    page: list = []
    held = 0
    while at < len(ordered) and held < ACCOUNT_LIST_CAP:
        job = ordered[at]
        at += 1
        if views.job_key(job) in before.skip:
            continue
        page.append(job)
        held += _in_history(job)
    # Older jobs left: the next page's (the skipped ones are the pages' before, undated ones last among them).
    more = any(views.job_key(job) not in before.skip for job in ordered[at:])
    taken = {id(job) for job in page}
    active = [job for job in ordered if id(job) not in taken
              and views.status_word(job.get("status")) in views.ACTIVE_STATUSES]
    scope = replace(scope, stopped=CAPPED if more else None, walked=held, active_read=True, page=before.page)
    return Listing(page + active, scope, next_page=_next_bound(page, before) if more else None)


def _more(scope: ListScope) -> str:
    more = {CAPPED: "more exist", FAILED: "the rest didn't load; r retries"}.get(scope.stopped or "", "more may exist")
    return more + ("; older active jobs may be missing" if scope.active_missing else "")


def page_text(scope: ListScope) -> str:
    """The line under the job list's header: which page of the list shows, ``page 2 + active
    jobs`` (every page lists every active job), or how far it is read, ``page 1 · loading… 300 so
    far``; "" for a list read whole on one page, which needs no line.

    A page short of ACCOUNT_LIST_CAP that isn't the last says how many jobs it holds, and why."""
    if not _shows_walk(scope):
        return ""
    page = f"page {scope.page}"
    missing = " (older active jobs may be missing)"
    if scope.loading == OLDER:
        active = "" if not scope.active_read else (missing if scope.active_missing else " + active jobs")
        return f"{page} · loading… {scope.walked:,} so far{active}"
    if scope.loading:
        return f"{page} · loading active jobs…"
    active = missing if scope.active_missing else " + active jobs"
    if scope.stopped == FAILED:
        return f"{page}: {scope.walked:,} jobs{active} · the rest didn't load; r retries"
    if scope.stopped == PAGE_LIMIT:
        return f"{page}: {scope.walked:,} jobs{active} · more may exist"
    return page + active


def _shows_walk(scope: ListScope) -> bool:
    """Whether the header says how far the walk got: cut short or loading, or any page past the first."""
    return scope.truncated or scope.page > 1


def scope_name(scope: ListScope, *, location: str, declined: bool = False) -> str:
    """The words of :func:`scope_text` that name which list it is: ``all schemas``, else
    ``Current schema: DB.SCHEMA`` (``Current schema only:`` where it stands in for all schemas,
    ``declined`` or unavailable). The job list's header draws them as a chip."""
    if scope.kind == ACCOUNT:
        return "all schemas"
    return f"Current schema{' only' if declined or scope.kind != SCHEMA else ''}: {location}"


def scope_text(scope: ListScope, *, location: str, only: str = "", declined: str = "") -> str:
    """A short header for this scope; ``only`` names its sole location, and ``declined`` why
    ``a`` couldn't show all schemas, for the configured schema's list shown in their place. Which
    page shows, and how far it is read, is :func:`page_text`'s, on a line of its own.

    A narrowed listing (:attr:`ListScope.narrowed`) says so in place of ``only``: the rows do
    share one location, but because the walk asked for it, not because the account holds no other.
    """
    name = scope_name(scope, location=location, declined=bool(declined))
    if scope.kind == ACCOUNT:
        if scope.narrowed:
            return f"{name}, narrowed to {scope.narrowed}"
        # A page of the list sharing one location doesn't make it the account's only one.
        return f"{name} (all in {only})" if only and not _shows_walk(scope) else name
    if scope.kind == SCHEMA:
        return f"{name} ({declined})" if declined else name
    return f"{name} ({unavailable_text(scope)})"


def unavailable_text(scope: ListScope) -> str:
    """Why a :data:`FALLBACK` listing holds the configured schema's jobs, in a few words."""
    return _FALLBACK_WHY.get(scope.reason or "", "")


def loaded_text(scope: Optional[ListScope]) -> str:
    """What a list cut short holds, for a search without results; "" when it holds every job."""
    if scope is None or not _shows_walk(scope):
        return ""
    where = f" in {scope.narrowed}" if scope.narrowed else ""
    if scope.page > 1:
        if scope.loading:
            return f"the first {scope.walked:,} jobs of page {scope.page}{where} so far; the rest are still loading"
        held = f"the {scope.walked:,} jobs of page {scope.page}{where}"
        more = f"; {_more(scope)}" if scope.stopped else ""
        return held + ("" if scope.active_missing else " and every active job") + more
    if scope.loading:
        what = "older jobs are" if scope.loading == OLDER else "older active jobs are"
        active = " and every active job" if scope.active_read and not scope.active_missing else ""
        return f"the newest {scope.walked:,}{where} so far{active}; {what} still loading"
    held = (f"the newest {scope.walked:,}{where}" if scope.stopped in (CAPPED, FAILED)
            else f"what the first {scope.pages} pages held{where}")
    return held + ("" if scope.active_missing else " and every active job") + f"; {_more(scope)}"


def empty_text(scope: ListScope, what: str = "jobs") -> str:
    """The list's message when a walk cut short (or still loading) left no job (or ``what``) to show."""
    where = f" in {scope.narrowed}" if scope.narrowed else ""
    which = "The account-wide list" if scope.kind == ACCOUNT else "This schema's list"
    if scope.page > 1:
        return (f"Page {scope.page} of {which[0].lower()}{which[1:]} has no {what} to show{where}"
                f"{' yet; it is still loading' if scope.loading else ''}. [ shows the newer jobs.")
    if scope.loading:
        return (f"{which} has no {what} to show among its newest {scope.walked:,}{where} yet; "
                f"{'older' if scope.loading == OLDER else 'older active'} jobs are still loading.")
    held = (f"in its first {scope.pages} pages" if scope.stopped == PAGE_LIMIT
            else f"among its newest {scope.walked:,}")
    return f"{which} returned no {what} to show {held}{where}; more may exist. r reads it again."


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


def _configured_location(client: Any) -> Optional[tuple]:
    """``(database, schema)`` the client is bound to, to scope a walk to its schema; None when the
    client doesn't say which (a test client's attributes may be anything).

    Each name goes out as a SQL identifier, as ``CortexTrainingClient.for_job`` builds them, so a
    quoted or lowercase configured name resolves back to the same schema at the service.
    """
    location = views.client_location(client)
    if location is None:
        return None
    return views.sql_identifier(location[0]), views.sql_identifier(location[1])


# The fields the account-level list adds to the per-schema list's rows.
_LOCATION_KEYS = ("database_name", "schema_name")


def _unlocated(job: dict) -> dict:
    """``job`` as the per-schema list returns it: the configured schema's list keeps one row
    shape whichever route read it, so its records match across the two."""
    return {key: value for key, value in job.items() if key not in _LOCATION_KEYS}


class JobLister:
    def __init__(self, client: Any, *, account_wide: bool = False, clock: Callable[[], float] = time.monotonic):
        self._client = client
        self.account_wide = account_wide
        self._clock = clock
        self._fallback: Optional[str] = None  # why the account-level list is unavailable, while it is
        self._asked_at = 0.0  # in fallback: when the account-level list last answered (or failed)
        # When the configured schema's scoped walk last refused or failed, while it does; None while
        # it answers. Kept apart from ``_fallback``, which is a's: a schema refresh that found the
        # route refused must not answer a's own ask, which judges the refusal for itself.
        self._scoped_failed_at: Optional[float] = None
        self._reader: Optional[threading.Thread] = None  # the last active jobs' walk read beside the newest
        self._reader_lock = threading.Lock()

    def load(self, now: datetime, *, is_cancelled: Callable[[], bool] = lambda: False,
             manual: bool = False, account: Optional[bool] = None, asked: bool = False,
             on_partial: Optional[Callable[[Listing], None]] = None,
             location: Optional[tuple] = None, before: Optional[PageBound] = None) -> Optional[Listing]:
        """Load one refresh; flags override scope, reprobe, and fallback behavior. The list has
        no lower bound on creation time, so it reaches every job ever made (``now`` bounds nothing
        any more; callers pass their clock).

        ``on_partial`` is given the account-level list as it is read (see :meth:`_walk`), except while
        asking again in fallback: the configured schema's list stands until the account-level one is whole.

        ``location`` is a ``(database, schema)`` the account-level walk asks the service for, so the
        jobs of one schema are read whole rather than filtered out of the newest
        :data:`ACCOUNT_LIST_CAP` of the account (see
        :func:`cortex_training.tui.views.pushdown_location`). It does not narrow the configured
        schema's list, which is already scoped to that one schema (see :meth:`_schema_listing`).

        ``before`` is the page to read past the first (see :class:`PageBound`), of whichever list
        answers; every page lists every active job.
        """
        if not (self.account_wide if account is None else account):
            return self._schema_listing(is_cancelled, manual, on_partial, before)
        if self._fallback is not None and not self._ask_again(manual):
            return self._schema(ListScope(FALLBACK, reason=self._fallback), before)
        if asked:
            self._fallback = None  # asked again for a: the earlier answer no longer counts
        probing = self._fallback is not None
        try:
            listing = self._walk(is_cancelled, None if probing else on_partial, location, before=before)
        except _BadNarrowing as exc:
            if is_cancelled():
                return None
            # The service refused the narrowing itself (a name this role can't see). Read the whole
            # account instead and let the search say "no jobs match": the fallback state is about the
            # route, not the filter, so it is deliberately left alone here.
            logger.info("job list: %s; reading every schema instead", exc)
            listing = self._walk(is_cancelled, None if probing else on_partial, before=before)
        except AccountJobListUnavailable as exc:

            if is_cancelled():
                return None  # superseded while its page was read: the load in its place decides
            return self._unavailable(exc, probing, before)
        except Exception as exc:
            if is_cancelled():
                return None
            if not probing:
                raise JobListError(_failure_text(_ACCOUNT_ROUTE, exc)) from exc
            # Asking again failed some other way (a timeout, a 503): keep listing
            # the configured schema, and ask again later.
            self._asked_at = self._clock()
            logger.info("job list: the account-level list failed again; listing the configured schema", exc_info=True)
            return self._schema(ListScope(FALLBACK, reason=self._fallback), before)
        if listing is not None and not is_cancelled():
            self._fallback = None
        return listing

    def _ask_again(self, manual: bool) -> bool:
        return self._due(self._asked_at, manual)

    def _due(self, since: float, manual: bool) -> bool:
        """Whether a route that last refused or failed at ``since`` is worth asking again now."""
        waited = self._clock() - since + _ASK_SLACK_SECONDS
        return waited >= ACCOUNT_REPROBE_SECONDS or (manual and waited >= ACCOUNT_MANUAL_REPROBE_SECONDS)

    def _fall_back(self, exc: AccountJobListUnavailable) -> None:
        self._fallback = exc.reason
        self._asked_at = self._clock()
        logger.info("job list: %s; listing the configured schema instead", exc)

    def _unavailable(self, exc: AccountJobListUnavailable, probing: bool,
                     before: Optional[PageBound] = None) -> Listing:
        if exc.reason in _FALLBACK_REASONS or probing:
            self._fall_back(exc)
            return self._schema(ListScope(FALLBACK, reason=exc.reason), before)
        refused = JobListError(_one_line(str(exc)), refused=True)
        # An unexplained 403 may affect only this route; probe the schema once.
        if exc.reason != AccountJobListUnavailable.FORBIDDEN or _service_explained(exc):
            raise refused from exc
        try:
            listing = self._schema(ListScope(FALLBACK, reason=exc.reason), before)
        except JobListError as schema_failed:
            raise JobListError(_one_line(f"{exc}; and {schema_failed}")) from exc
        self._fall_back(exc)
        return listing

    def _schema_listing(self, is_cancelled: Callable[[], bool], manual: bool,
                        on_partial: Optional[Callable[[Listing], None]],
                        before: Optional[PageBound] = None) -> Optional[Listing]:
        """The configured schema's jobs: the account-level list scoped to it, a page at a time.

        TODO: remove the fallback to ``list_jobs`` below once every account answers the
        account-level route. Until then the configured schema's list tries that route (the
        per-schema ``list_jobs`` has no paging) and reads ``list_jobs`` where it refuses, fails,
        or answers no job at all: a schema it does not index yet, or one its scoping resolves
        elsewhere, would otherwise read as empty. The fallback is invisible, as ``list_jobs``
        returns a superset of what a walk reads and only the paging is lost. A refusal or failure
        is asked again on the account-level list's cadence. When this goes, so do the ``except``
        arm, the empty-answer read and ``_scoped_failed_at``.
        """
        location = _configured_location(self._client)
        if location is None:  # the client doesn't name its schema: nothing to scope by
            return self._schema(ListScope(SCHEMA), before)
        if self._scoped_failed_at is not None and not self._due(self._scoped_failed_at, manual):
            return self._schema(ListScope(SCHEMA), before)  # it failed lately: don't ask every refresh
        try:
            listing = self._walk(is_cancelled, on_partial, location, kind=SCHEMA, before=before)
        except Exception as exc:
            if is_cancelled():
                return None  # superseded while its page was read: the load in its place decides
            self._scoped_failed_at = self._clock()
            logger.info("job list: the account-level list failed for the configured schema; "
                        "listing it with %s instead", _SCHEMA_ROUTE, exc_info=True)
            try:
                return self._schema(ListScope(SCHEMA), before)
            except JobListError as schema_failed:
                # Both routes failed: say so, as either alone would leave the other unexplained.
                why = str(exc) if isinstance(exc, _BadNarrowing) else _failure_text(_ACCOUNT_ROUTE, exc)
                raise JobListError(_one_line(f"{why}; and {schema_failed}")) from exc
        if listing is None:
            return None  # cancelled before a page: the load in its place decides
        if not is_cancelled():
            self._scoped_failed_at = None
            if not listing.jobs:
                return self._schema(ListScope(SCHEMA), before)  # no job from the walk: read list_jobs
        return listing

    def _schema(self, scope: ListScope, before: Optional[PageBound] = None) -> Listing:
        try:
            jobs = self._client.list_jobs()
        except Exception as exc:
            raise JobListError(_failure_text(_SCHEMA_ROUTE, exc)) from exc
        return cut_whole_listing(Listing([job for job in (jobs or []) if isinstance(job, dict)], scope), before)

    def _walk(self, is_cancelled: Callable[[], bool],
              on_partial: Optional[Callable[[Listing], None]] = None,
              location: Optional[tuple] = None, *, kind: str = ACCOUNT,
              before: Optional[PageBound] = None) -> Optional[Listing]:

        """The newest jobs of the account-level list, and the active ones when more exist; None when cancelled.

        Without ``on_partial`` (a refresh of a list that is whole), the active jobs' walk follows the
        newest jobs', as only a walk that stopped short needs it. With it, ``on_partial`` is given what
        is read so far (a :class:`Listing` whose scope is ``loading``) after each page of the newest
        jobs that isn't the last one read, and the active jobs' walk is read beside the newest jobs'
        from their second page on (on a busy account it is pages long too, and its first page the
        slowest), its jobs listed once it is done. The result is the same either way.

        ``kind`` :data:`SCHEMA` reads the configured schema's list, ``location`` naming that schema:
        its scopes are that list's, and its rows take the per-schema list's shape (see
        :func:`_unlocated`). Only an :data:`ACCOUNT` walk is narrowed by its location.

        ``before`` bounds the newest jobs' walk to a page past the first; the active jobs' walk
        is never bounded, so every page lists every active job.
        """
        # Snapshotted here and passed down, never re-read from the lister: the newest jobs' walk
        # and the active one must ask for the same scope.
        narrowed = views.location_filter_text(location) if kind == ACCOUNT else ""
        unlocated = kind == SCHEMA
        if on_partial is not None:
            return _Beside(self, is_cancelled, on_partial, location, kind, before).walk()
        newest = _Walk(self._client, None, location, unlocated, before)
        while not newest.done:
            if not newest.step(is_cancelled):
                return None
        if newest.stopped is None and before is None:
            return Listing(newest.jobs, ListScope(kind, narrowed=narrowed))
        active = self._active(is_cancelled, location, unlocated)
        if active is None:
            return None
        return _listing(newest, active, narrowed, kind, before)

    def _start_reader(self, read: Callable[[], None]) -> Optional[threading.Thread]:
        """A thread reading the active jobs beside the newest (``read``), started; None while the
        last one started is still reading (a load cancelled in its request): no second beside it."""
        with self._reader_lock:
            if self._reader is not None and self._reader.is_alive():
                return None
            self._reader = threading.Thread(target=read, name="active-jobs-walk", daemon=True)
            self._reader.start()
            return self._reader

    def _active(self, is_cancelled: Callable[[], bool],
                location: Optional[tuple] = None, unlocated: bool = False) -> Optional[tuple]:
        """``(jobs, stopped)`` of the active-job walk (stopped "failed" when it failed); None when cancelled.

        ``location`` and ``unlocated`` are the newest jobs' walk's own, so the two walks cover one
        scope and :func:`_merge` unions two sets read the same way."""
        walk = _Walk(self._client, ACTIVE, location, unlocated)

        try:
            while not walk.done:
                if not walk.step(is_cancelled):
                    return None
        except Exception:  # noqa: BLE001 - the newest jobs stand; the scope says active ones may be missing
            logger.info("job list: reading the active jobs failed; older active jobs may be missing", exc_info=True)
            return [], "failed"
        return walk.jobs, walk.stopped

    def read_rest(self, first: Listing, *, is_cancelled: Callable[[], bool],
                  on_rest: Callable[[Rest], None]) -> None:
        """Read the rest of ``first``'s list, the jobs a search covers past its first page: a
        request at a time, to the list's end (its first job), with up to
        :data:`ARCHIVE_CAP` jobs in all (``first``'s own among them). Each request's jobs go to
        ``on_rest``; the last :class:`Rest` is ``done``. Nothing comes once ``is_cancelled``.

        ``first`` is a whole first page of the account-wide list (not narrowed to a location) or
        of the configured schema's, which the walk reads the same way. A list that came whole
        (``list_jobs``) or fits on one page has nothing more to read. A failure ends the read
        with the jobs read so far, saying why."""
        if first.everything is not None or first.next_page is None:
            on_rest(Rest(list(first.everything or ()), done=True))
            return
        room = ARCHIVE_CAP - len(first.jobs)
        if room <= 0:
            on_rest(Rest([], done=True, capped=True))
            return
        account = first.scope.kind == ACCOUNT
        location = None if account else _configured_location(self._client)
        # Every job counts toward the cap, and short or empty answers may come on the way.
        walk = _Walk(self._client, None, location, not account, first.next_page, cap=room, counts=lambda _job: True,
                     max_pages=room // ACCOUNT_PAGE_SIZE * 2 + ACCOUNT_LIST_MAX_PAGES)
        while not walk.done:
            held = len(walk.jobs)
            try:
                if not walk.step(is_cancelled):
                    return
            except Exception as exc:  # noqa: BLE001 - the jobs read so far stand; the search says the rest didn't load
                if is_cancelled():
                    return
                logger.info("job list: reading the rest of the list for search failed", exc_info=True)
                on_rest(Rest([], done=True, failed=_failure_text(_ACCOUNT_ROUTE, exc)))
                return
            if is_cancelled():
                return
            # A walk read again from its first page (its token went stale) holds fewer than before.
            batch = walk.jobs[held:] if len(walk.jobs) >= held else list(walk.jobs)
            on_rest(Rest(batch, done=walk.done, capped=walk.stopped is not None))


def _listing(newest: "_Walk", active: tuple, narrowed: str = "", kind: str = ACCOUNT,
             before: Optional[PageBound] = None) -> Listing:
    """The list of a newest jobs' walk that stopped short (or read a page past the first), and of
    the active jobs' walk ``(jobs, stopped)``."""
    rows, active_stopped = active
    scope = ListScope(kind, stopped=newest.stopped, walked=_job_count(newest.jobs),
                      pages=ACCOUNT_LIST_MAX_PAGES if newest.stopped == PAGE_LIMIT else 0,
                      active_missing=active_stopped is not None, narrowed=narrowed, page=_page_of(before))
    next_page = _next_bound(newest.jobs, before) if newest.stopped is not None else None
    return Listing(_merge(newest.jobs, rows), scope, next_page=next_page)


class _Beside:
    """One walk of the account-level list that shows its pages: the newest jobs' walk on the
    caller's thread and, from its second page on, the active jobs' walk on another (at most two
    requests in flight), each list shown made under one lock, so they reach ``on_partial`` in
    the order they were made, and none once the walk has answered or was cancelled.

    An active jobs' walk of an earlier load that is still reading (it was cancelled during a
    request) keeps its thread until that request answers; while it does, a new load reads the
    active jobs after the newest ones instead, so loads in a row don't pile requests up."""

    _POLL_SECONDS = 0.1  # how soon a walk waiting for the active jobs sees it was cancelled

    def __init__(self, lister: "JobLister", is_cancelled: Callable[[], bool],
                 on_partial: Callable[[Listing], None], location: Optional[tuple] = None,
                 kind: str = ACCOUNT, before: Optional[PageBound] = None) -> None:
        self._lister = lister
        self._is_cancelled, self._on_partial = is_cancelled, on_partial
        self._location = location  # both walks ask for this one scope; its text goes on every scope made
        # The list these pages are of (see JobLister._walk): only an account-wide one is narrowed.
        self._kind, self._unlocated = kind, kind == SCHEMA
        self._narrowed = views.location_filter_text(location) if kind == ACCOUNT else ""
        self._lock = threading.Lock()
        self._stop = threading.Event()  # the walk answered: the active jobs' walk asks for no other page
        self._before = before  # the page read, past the first: bounds the newest jobs' walk only
        self._newest = _Walk(lister._client, None, location, self._unlocated, before)

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
            if newest.stopped is None and self._before is None:
                return Listing(newest.jobs, ListScope(self._kind, narrowed=self._narrowed))
            if reader is None:  # the first page stopped it short, or an earlier walk still reads
                active = self._lister._active(self._is_cancelled, self._location, self._unlocated)

            else:
                while reader.is_alive():
                    reader.join(self._POLL_SECONDS)
                    if self._is_cancelled():
                        return None
                # A walk that ended with no answer failed past its own handling (it was logged).
                active = self._active if self._active is not None else ([], "failed")
            if active is None or self._is_cancelled():
                return None
            return _listing(newest, active, self._narrowed, self._kind, self._before)

        finally:
            with self._lock:
                self._stop.set()

    def _read_active(self) -> None:
        try:
            active = self._lister._active(lambda: self._is_cancelled() or self._stop.is_set(), self._location,
                                          self._unlocated)

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
        page = _page_of(self._before)
        if self._newest_done:  # stopped short, the active jobs not read yet
            stopped = self._newest_stopped
            scope = ListScope(self._kind, stopped=stopped, walked=_job_count(self._shown), loading=ACTIVE_NEXT,
                              pages=ACCOUNT_LIST_MAX_PAGES if stopped == PAGE_LIMIT else 0,
                              narrowed=self._narrowed, page=page)
        else:
            scope = ListScope(self._kind, walked=_job_count(self._shown), loading=OLDER, active_read=active is not None,
                              active_missing=active is not None and active[1] is not None,
                              narrowed=self._narrowed, page=page)

        jobs = _merge(self._shown, active[0]) if active is not None else list(self._shown)
        # Where a page after these would start, should the rest not load: older jobs may exist.
        next_page = _next_bound(self._shown, self._before)
        try:
            self._on_partial(Listing(jobs, scope, next_page=next_page))
        except Exception:  # noqa: BLE001 - showing it failed; the walk goes on
            logger.warning("job list: showing the jobs read so far failed", exc_info=True)


def _merge(jobs: list, active: list) -> list:
    """``jobs``, then the active jobs not among them (a new list)."""
    listed = {views.job_key(job) for job in jobs}
    return jobs + [job for job in active if views.job_key(job) not in listed]


class _Walk:
    """One walk of the account-level list, a page at a time: up to the cap or the page limit, newest first.
    It starts again from the first page, once, when the service can't open its token. ``unlocated``:
    the walk reads the configured schema's list, whose rows take the per-schema list's shape."""

    def __init__(self, client: Any, status: Optional[str],
                 location: Optional[tuple] = None, unlocated: bool = False,
                 before: Optional[PageBound] = None, *, cap: Optional[int] = None,
                 counts: Optional[Callable[[dict], bool]] = None,
                 max_pages: Optional[int] = None) -> None:
        self._client = client
        self._unlocated = unlocated
        database, schema = location if location else (None, None)
        # Fixed for this walk's life: the service refuses a page token sent with filters other than
        # the ones the token was made under. No lower bound on creation time: every job, ever.
        self._filters: dict = {**({"status": status} if status else {}),
                               **({"database": database} if database else {}),
                               **({"schema": schema} if schema else {}),
                               **({"created_before": before.created_before} if before else {})}
        self._skip = before.skip if before else frozenset()  # jobs the pages before list
        # The newest jobs' walk ends a page at its 1,000th job in history; the active jobs' walk,
        # every one of them live, at its 1,000th job (``cap`` and ``counts`` say otherwise for
        # the read behind a list: see JobLister.read_rest).
        self._counts = counts or (_in_history if status is None else (lambda _job: True))
        # Read when the walk starts, as ACCOUNT_LIST_CAP and ACCOUNT_LIST_MAX_PAGES are by default.
        self._cap = ACCOUNT_LIST_CAP if cap is None else cap
        self._max_pages = ACCOUNT_LIST_MAX_PAGES if max_pages is None else max_pages
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
            page = self._read_page()
        except InvalidPageToken:
            if self._restarted:
                raise
            logger.info("job list: the service could not open its page token; reading again from the first page")
            self._restarted = True
            self._start()
            if is_cancelled():
                return False
            page = self._read_page()
        rows, self._token = _page(page)
        self._pages += 1
        for job in rows:
            if self._unlocated:
                job = _unlocated(job)
            key = views.job_key(job)
            if key in self._skip or (views.job_id_of(job) and key in self._seen):
                continue
            self._seen.add(key)
            self.jobs.append(job)
        cut = _cap_cut(self.jobs, self._counts, self._cap)
        if cut is not None:
            more = self._token is not None or len(self.jobs) > cut
            self.jobs = self.jobs[:cut]
            self.done, self.stopped = True, CAPPED if more else None
        elif self._token is None:
            self.done = True
        elif self._pages >= self._max_pages:
            self.done, self.stopped = True, PAGE_LIMIT  # stopped at the page limit with pages left
        return True

    def _read(self) -> Any:
        return self._client.list_jobs_page(**self._filters, page_size=ACCOUNT_PAGE_SIZE, page_token=self._token)

    def _read_page(self) -> Any:
        """One page, with a 400 about a narrowing surfaced as :class:`_BadNarrowing`."""
        try:
            return self._read()
        except InvalidPageToken:
            raise  # about the token, not the narrowing: :meth:`step` reads again from the first page
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 400 and "database" in self._filters:
                raise _BadNarrowing(_failure_text(_ACCOUNT_ROUTE, exc)) from exc
            raise


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
