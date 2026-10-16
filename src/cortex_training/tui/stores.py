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

"""Client-side store for job display names.

A job has no server-side name: the ``comment`` it was created with is shown as
its comment. Local aliases name a job on this machine. They live next to the
log cache on the client:

    <cache_root>/aliases.json                          job_id -> display name
    <cache_root>/aliases.json.lock                     taken while a rename writes it

Pure stdlib, NO ``textual`` import, so the store is unit-testable like ``format.py``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from typing import Optional

from cortex_training.tui.log_cache import _atomic_write_text
from cortex_training.tui.log_cache import cache_root

try:  # a lock every process on this machine sees (POSIX)
    import fcntl
except ImportError:  # pragma: no cover - Windows: a rename locks out this process's others only
    fcntl = None

# A display name keeps at most this many characters (the rename prompt takes no more).
ALIAS_MAX_CHARS = 80
_MAX_ALIAS = ALIAS_MAX_CHARS
# A rename waits this long at most for another session's to finish (it holds the
# lock while it writes a few hundred bytes), then writes without the lock rather
# than hold up the screen it runs on.
_LOCK_WAIT_SECONDS = 2.0
_LOCK_RETRY_SECONDS = 0.01
_RENAMING = threading.Lock()  # this process's renames, one at a time


class AliasStore:
    """Persistent ``job_id -> display name`` map, shared by every TUI session.

    An alias wins over the job's server ``comment`` as its display name, on
    this machine only; the comment stays visible and searchable beside it.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        base = root if root is not None else cache_root()
        self._path = base / "aliases.json"
        # Beside the names, not on them: a rename replaces aliases.json, and a lock
        # on the file it replaced would no longer keep anyone out.
        self._lock_path = base / "aliases.json.lock"

    def _read(self) -> dict:
        try:
            doc = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        names = doc.get("aliases") if isinstance(doc, dict) else None
        return {str(k): str(v) for k, v in names.items() if v} if isinstance(names, dict) else {}

    def all(self) -> dict:
        return self._read()

    def get(self, job_id: str) -> Optional[str]:
        return self._read().get(job_id)

    def set(self, job_id: str, name: Optional[str]) -> None:
        """Set (or, with an empty name, remove) a job's display name.

        The names are read and written back under the store's lock (see
        :meth:`_locked`), so this one change is merged into what is on disk then:
        another session renaming a job at the same moment keeps its name too."""
        clean = " ".join(str(name or "").split())[:_MAX_ALIAS]
        with self._locked():
            names = self._read()
            if clean:
                names[job_id] = clean
            else:
                names.pop(job_id, None)
            _atomic_write_text(self._path,
                               json.dumps({"version": 1, "aliases": names}, indent=1, sort_keys=True) + "\n")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold the store's lock: ``aliases.json.lock``, locked with ``flock``, which
        every process on this machine sees. Without ``fcntl`` (Windows), on a
        filesystem that has no locks, or after ``_LOCK_WAIT_SECONDS`` of another
        session holding it, only this process's renames are kept apart."""
        with _RENAMING:
            fd = self._take_lock()
            try:
                yield
            finally:
                if fd is not None:
                    os.close(fd)  # which lets the lock go

    def _take_lock(self) -> Optional[int]:
        """The lock file, open and locked; None where it can't be (see :meth:`_locked`)."""
        if fcntl is None:
            return None
        try:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError:
            return None  # the write that follows says why, if it fails too
        deadline = time.monotonic() + _LOCK_WAIT_SECONDS
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:  # another session is renaming
                if time.monotonic() >= deadline:
                    break
                time.sleep(_LOCK_RETRY_SECONDS)
            except OSError:  # no locks here (some network filesystems)
                break
        os.close(fd)
        return None
