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

"""Bounds and stale-artifact cleanup for the TUI's local job cache.

Each opened job caches its log tail (``logs/``), finished console (``console/``),
and telemetry (``metrics/``) under ``<cache_root>/jobs/<job>/``. :func:`enforce_cache_limit`
evicts least-recently-used jobs past ``DEFAULT_CACHE_LIMIT`` while skipping non-cache
entries, symlinks, leased open jobs (:class:`JobLease`), and active downloads
(:class:`DownloadLock`). :func:`sweep_stale` removes leftovers from interrupted TUIs.
"""

from __future__ import annotations

import atexit
import contextlib
import heapq
import os
import re
import shutil
import socket
import stat
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

try:  # POSIX only; elsewhere a lease is its marker file alone, and stages aren't swept.
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None  # type: ignore[assignment]

from cortex_training._download_lock import DownloadLock
from cortex_training.tui.log_cache import cache_root
from cortex_training.tui.log_cache import unsafe_source_name

DEFAULT_CACHE_LIMIT = 5 * 1024 ** 3
# How often an open job screen refreshes its marker, and how old a marker may be before
# it stops protecting its job from a TUI on another machine (one that can't see the lock).
LEASE_HEARTBEAT_SECONDS = 5 * 60
LEASE_STALE_SECONDS = 30 * 60
# How long a closed screen's lease outlives it: its threads can't be stopped mid-request,
# and a read that answers late (every retry of a 45 s read) then writes into the job.
LEASE_GRACE_SECONDS = 5 * 60
_LEASE_WAIT_SECONDS = 2.0
_LEASE_RETRY_SECONDS = 5.0
# A half-written console copy or cursor file older than this belongs to no running TUI:
# writing one takes seconds.
_STALE_PARTIAL_SECONDS = 60 * 60
_LEGACY_STAGE_SECONDS = 7 * 24 * 3600

STAGE_PREFIX = "cortex-training-stage-"
_HOST = re.sub(r"[^A-Za-z0-9.-]", "-", socket.gethostname()) or "host"
_STAGE_NAME = re.compile(re.escape(STAGE_PREFIX) + r"(?P<host>[A-Za-z0-9.-]+)\+(?P<pid>\d+)\+")
_STAGE_OWNER = ".owner"

_KEYED_JOB = re.compile(r"[0-9a-f]{16}%3A[A-Za-z0-9._%-]+")
_BARE_JOB = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_JOB_ENTRIES = frozenset({"logs", "console", "metrics", "cursors.json", ".lease", ".leases", ".last-used",
                          ".metrics.lock", ".console.lock", ".cursors.lock"})
_JOB_ENTRY_PREFIXES = (".console-", ".tmp-")
_LEASE_LOCK = ".lease"
_LEASES = ".leases"
_LAST_USED = ".last-used"
_REMOVING = ".removing-"
_PASS_LOCK = ".cleanup.lock"


def format_size(size: float) -> str:
    """``1.2 GiB``, ``340.0 MiB``: what a person reads in a notice."""
    for unit in ("B", "KiB", "MiB"):
        if abs(size) < 1024:
            return f"{size:.0f} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def freed_text(removed: list, limit: Optional[int]) -> str:
    """The notice for the jobs a pass removed (``(job_id, bytes)`` pairs)."""
    freed = sum(size for _job, size in removed)
    jobs = f"{len(removed)} job{'s' if len(removed) != 1 else ''}"
    keeps = f" (the cache keeps under {format_size(limit)})" if limit is not None else ""
    return f"Freed {format_size(freed)} of cached data from {jobs} not used lately{keeps}."


def _disk_bytes(path: Path) -> int:
    """Regular-file bytes under ``path``, ignoring symlinks and vanishing files."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            with contextlib.suppress(OSError):
                info = os.lstat(os.path.join(dirpath, name))
                if stat.S_ISREG(info.st_mode):
                    total += info.st_size
    return total


def _last_used(job_dir: Path) -> float:
    """When the job was last open: its lease's last refresh, else its newest non-lock file.

    A download can write files into a job after a screen last had it open, and lock probes
    create ``.lease`` / ``*.lock`` files, so ``.last-used`` wins and lock files are ignored.
    """
    with contextlib.suppress(OSError):
        return os.lstat(job_dir / _LAST_USED).st_mtime
    newest = 0.0
    for dirpath, _dirnames, filenames in os.walk(job_dir):
        for name in filenames:
            if name == _LEASE_LOCK or name.endswith(".lock"):
                continue
            with contextlib.suppress(OSError):
                newest = max(newest, os.lstat(os.path.join(dirpath, name)).st_mtime)
    return newest


def _jobs_dir(root: Path) -> Optional[Path]:
    """``<root>/jobs`` when it is a real directory; a link is never followed."""
    jobs = root / "jobs"
    with contextlib.suppress(OSError):
        if not jobs.is_symlink() and jobs.is_dir():
            return jobs
    return None


def _is_job_dir(path: Path) -> bool:
    """Whether ``path`` is a real TUI job directory containing only TUI-owned entries."""
    if not (_KEYED_JOB.fullmatch(path.name) or _BARE_JOB.fullmatch(path.name)):
        return False
    try:
        if path.is_symlink() or not path.is_dir():
            return False
        entries = os.listdir(path)
    except OSError:
        return False
    return all(entry in _JOB_ENTRIES or entry.startswith(_JOB_ENTRY_PREFIXES) for entry in entries)


def _jobs(root: Path) -> list:
    jobs = _jobs_dir(root)
    with contextlib.suppress(OSError):
        return [p for p in sorted(jobs.iterdir()) if _is_job_dir(p)] if jobs else []
    return []


def _job_label(job_dir: Path) -> str:
    """The job id a cache directory holds (its name is ``<location digest>:<job id>``, escaped)."""
    try:
        name = unsafe_source_name(job_dir.name)
    except Exception:  # noqa: BLE001 - not a name the cache writes
        return job_dir.name
    label = name.split(":", 1)[-1]
    return f"{label} (name shortened)" if len(job_dir.name) >= 200 else label


def _open_lock(path: Path) -> Optional[int]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        return None


def _same_file(fd: int, path: Path) -> bool:
    """Whether ``fd`` still refers to the file currently at ``path``."""
    try:
        return os.fstat(fd).st_ino == os.stat(path).st_ino
    except OSError:
        return False


def _try_exclusive(fd: int) -> Optional[bool]:
    """Try non-blocking ``LOCK_EX``: True taken, False held by another, None unsupported."""
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    except OSError:
        return None
    return True


class _Scheduler:
    """Single daemon thread for delayed lease retries and grace-period releases."""

    def __init__(self) -> None:
        self._heap: list = []
        self._cv = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._seq = 0

    def after(self, delay: float, fn) -> None:
        with self._cv:
            self._seq += 1
            heapq.heappush(self._heap, (time.monotonic() + delay, self._seq, fn))
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="cortex-tui-leases", daemon=True)
                self._thread.start()
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._heap:
                    self._cv.wait()
                when, _seq, fn = self._heap[0]
                wait = when - time.monotonic()
                if wait > 0:
                    self._cv.wait(wait)
                    continue
                heapq.heappop(self._heap)
            with contextlib.suppress(Exception):
                fn()


_SCHEDULER = _Scheduler()


class JobLease:
    """Marks a job's cache directory as open via a shared ``.lease`` flock and a
    periodically refreshed ``.leases/<holder>`` marker."""

    def __init__(self, job_dir: Path, *, holder: Optional[str] = None) -> None:
        self._job_dir = Path(job_dir)
        self._lock_path = self._job_dir / _LEASE_LOCK
        name = holder or f"{_HOST}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._marker = self._job_dir / _LEASES / name
        self._fd: Optional[int] = None
        self._held = False
        self._mutex = threading.Lock()

    @property
    def held(self) -> bool:
        """Whether the lease still protects the job (acquired, and not yet released)."""
        return self._held

    def acquire(self) -> None:
        with self._mutex:
            self._held = True
        # Touch the marker before waiting on the lock so a concurrent removal checking
        # markers sees the job open immediately, and again after in case a removal moved
        # the job aside while _lock waited.
        self._touch()
        self._retry_lock()

    def _retry_lock(self) -> None:
        if not self._held or self._fd is not None:
            return
        self._lock()
        self._touch()  # refresh in case a removal moved the directory aside before _lock
        if self._held and self._fd is None and fcntl is not None:
            _SCHEDULER.after(_LEASE_RETRY_SECONDS, self._retry_lock)

    def _lock(self) -> None:
        if fcntl is None:
            return
        deadline = time.monotonic() + _LEASE_WAIT_SECONDS
        while self._held:
            fd = _open_lock(self._lock_path)
            if fd is None:
                return
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                if time.monotonic() >= deadline:
                    return
                time.sleep(0.02)
                continue
            except OSError:
                os.close(fd)
                return
            if _same_file(fd, self._lock_path):
                with self._mutex:
                    if self._held and self._fd is None:
                        self._fd = fd
                        return
                os.close(fd)
                return
            os.close(fd)  # locked a file a removal just moved aside; reopen at the path
            if time.monotonic() >= deadline:
                return
            time.sleep(0.01)

    def heartbeat(self) -> None:
        """Refresh the marker and last-used timestamp, reacquiring the lock if needed."""
        if not self._held:
            return
        with self._mutex:
            stale = self._fd is not None and not _same_file(self._fd, self._lock_path)
            fd, self._fd = (self._fd, None) if stale else (None, self._fd)
        if fd is not None:
            os.close(fd)
        if self._fd is None:
            self._lock()
        self._touch()

    def _touch(self) -> None:
        with self._mutex:
            if not self._held:
                return
            for _attempt in range(3):
                try:
                    self._marker.parent.mkdir(parents=True, exist_ok=True)
                    self._marker.touch()
                    (self._job_dir / _LAST_USED).touch()
                    return
                except FileNotFoundError:
                    continue
                except OSError:
                    return

    def release_later(self, delay: float = LEASE_GRACE_SECONDS) -> None:
        """Release after ``delay``, or at process exit if sooner."""
        _PENDING.add(self)
        _SCHEDULER.after(delay, self.release)

    def release(self) -> None:
        with self._mutex:
            if not self._held and self._fd is None:
                return
            self._held, fd, self._fd = False, self._fd, None
        _PENDING.discard(self)
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        _unlink(self._marker)
        with contextlib.suppress(OSError):
            self._marker.parent.rmdir()


_PENDING: set = set()


@atexit.register
def _release_pending() -> None:
    for lease in list(_PENDING):
        lease.release()


def _fresh_marker(job_dir: Path, now: float) -> bool:
    """Whether any lease, on any machine, refreshed its marker recently."""
    try:
        markers = list((Path(job_dir) / _LEASES).iterdir())
    except OSError:
        return False
    for marker in markers:
        with contextlib.suppress(OSError):
            if now - marker.lstat().st_mtime < LEASE_STALE_SECONDS:
                return True
    return False


def job_in_use(job_dir: Path, *, now: Optional[float] = None) -> bool:
    """Whether a running TUI has the job open via a fresh marker or held lease lock."""
    now = time.time() if now is None else now
    if _fresh_marker(job_dir, now):
        return True
    path = Path(job_dir) / _LEASE_LOCK
    if fcntl is None or not path.exists():
        return False
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return False
    try:
        return _try_exclusive(fd) is False
    finally:
        os.close(fd)


def _remove_unless_in_use(job_dir: Path, now: float) -> tuple:
    """``("removed", bytes)``, ``("in_use", 0)`` or ``("failed", 0)`` for one job.

    Under the lease and download locks the job is renamed aside before deletion, so an
    arriving screen or download never writes into a directory being deleted.
    """
    if _fresh_marker(job_dir, now):
        return "in_use", 0
    fd = None
    downloads: list = []
    try:
        if fcntl is not None:
            fd = _open_lock(job_dir / _LEASE_LOCK)
            if fd is None or _try_exclusive(fd) is False:
                return "in_use", 0
        for part in ("metrics", "console"):
            path = job_dir / part
            if path.is_symlink():
                continue
            lock = DownloadLock(path)
            if not lock.try_acquire():
                return "in_use", 0
            downloads.append(lock)
        if _fresh_marker(job_dir, now):
            return "in_use", 0
        size = _disk_bytes(job_dir)
        aside = job_dir.parent / f"{_REMOVING}{uuid.uuid4().hex}"
        try:
            os.rename(job_dir, aside)
        except OSError:
            return "failed", 0
    finally:
        for lock in downloads:
            lock.release()
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
    shutil.rmtree(aside, ignore_errors=True)
    return "removed", size


@contextlib.contextmanager
def _pass_lock(root: Path):
    """Serialize cleanup passes across TUIs so two processes don't both evict the excess."""
    if fcntl is None:
        yield True
        return
    fd = _open_lock(root / _PASS_LOCK)
    try:
        yield fd is not None and _try_exclusive(fd) is not False
    finally:
        if fd is not None:
            os.close(fd)


def enforce_cache_limit(root: Optional[Path] = None, limit: Optional[int] = DEFAULT_CACHE_LIMIT) -> list:
    """Remove least-recently-used jobs until the cache fits ``limit`` (None: no limit),
    skipping jobs open in a TUI or being downloaded into. Returns ``(job_id, bytes)`` pairs."""
    if limit is None:
        return []
    root = Path(root) if root is not None else cache_root()
    if not _jobs(root):
        return []
    with _pass_lock(root) as ours:
        if not ours:
            return []
        jobs = _jobs(root)
        sizes = {job: _disk_bytes(job) for job in jobs}
        total = sum(sizes.values())
        removed = []
        now = time.time()
        for job_dir in sorted(jobs, key=_last_used):
            if total <= limit:
                break
            try:
                outcome, size = _remove_unless_in_use(job_dir, now)
            except OSError:
                continue
            if outcome == "removed":
                total -= sizes[job_dir]
                removed.append((_job_label(job_dir), size))
        return removed


def _unlink(path: Path) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


def _stage_abandoned(stage: Path, now: float) -> bool:
    """Whether a staging directory's TUI is gone (owner lock free, or legacy > 1 week old)."""
    match = _STAGE_NAME.match(stage.name)
    if match is None:
        return now - stage.lstat().st_mtime > _LEGACY_STAGE_SECONDS
    if match.group("host") != _HOST or fcntl is None:
        return False
    try:
        fd = os.open(f"{stage}{_STAGE_OWNER}", os.O_RDWR)
    except FileNotFoundError:
        return now - stage.lstat().st_mtime > _LEGACY_STAGE_SECONDS
    except OSError:
        return False
    try:
        return _try_exclusive(fd) is True
    finally:
        os.close(fd)


def sweep_stale(root: Optional[Path] = None, *, temp_dir: Optional[str] = None,
                now: Optional[float] = None) -> list:
    """Remove abandoned staging directories, unfinished removals, stale partial writes,
    and expired lease markers. Returns the paths removed."""
    root = Path(root) if root is not None else cache_root()
    now = time.time() if now is None else now
    removed = []

    def remove(path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            _unlink(path)
        if not os.path.lexists(path):
            removed.append(path)

    temp = Path(temp_dir if temp_dir is not None else tempfile.gettempdir())
    with contextlib.suppress(OSError):
        for stage in list(temp.iterdir()):
            if not stage.name.startswith(STAGE_PREFIX) or stage.is_symlink():
                continue
            if stage.name.endswith(_STAGE_OWNER):
                if not os.path.lexists(str(stage)[:-len(_STAGE_OWNER)]):
                    _unlink(stage)
                continue
            with contextlib.suppress(OSError):
                if stage.is_dir() and _stage_abandoned(stage, now):
                    remove(stage)
                    _unlink(Path(f"{stage}{_STAGE_OWNER}"))

    jobs = _jobs_dir(root)
    if jobs is not None:
        with contextlib.suppress(OSError):
            for entry in list(jobs.iterdir()):
                if entry.name.startswith(_REMOVING) and not entry.is_symlink():
                    remove(entry)
    for job_dir in _jobs(root):
        for dirpath, dirnames, filenames in os.walk(job_dir):
            here = Path(dirpath)
            for name in (*dirnames, *filenames):
                if name.startswith(".console-") or (name.startswith(".tmp-") and name.endswith(".swp")):
                    path = here / name
                    with contextlib.suppress(OSError):
                        if not path.is_symlink() and now - path.lstat().st_mtime > _STALE_PARTIAL_SECONDS:
                            remove(path)
            dirnames[:] = [d for d in dirnames if not d.startswith(".console-")]
        with contextlib.suppress(OSError):
            for marker in list((job_dir / _LEASES).iterdir()):
                with contextlib.suppress(OSError):
                    if now - marker.lstat().st_mtime > LEASE_STALE_SECONDS:
                        remove(marker)
            (job_dir / _LEASES).rmdir()
    return removed


def make_stage() -> tuple:
    """Create a staging directory and its locked ``.owner`` file: ``(TemporaryDirectory, fd)``."""
    stage = tempfile.TemporaryDirectory(prefix=f"{STAGE_PREFIX}{_HOST}+{os.getpid()}+")
    fd = None
    if fcntl is not None:
        try:
            fd = os.open(f"{stage.name}{_STAGE_OWNER}", os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            if fd is not None:
                os.close(fd)
            fd = None
    return stage, fd


def release_stage(stage, fd: Optional[int]) -> None:
    """Remove a staging directory and its owner file, then drop the owner lock."""
    owner = Path(f"{stage.name}{_STAGE_OWNER}")
    stage.cleanup()
    _unlink(owner)
    if fd is not None:
        with contextlib.suppress(OSError):
            os.close(fd)
