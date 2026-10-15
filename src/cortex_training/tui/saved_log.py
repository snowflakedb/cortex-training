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

"""Saved-console helpers for the TUI: a bounded tail reader and a download lock.

A finished job's console can be gigabytes, so the log pane only ever reads the
end of ``stdout.log``. Downloads into the shared per-job cache are serialized
per output directory, within this process and (where ``fcntl`` exists) across
TUI processes.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from cortex_training.tui.format import entry_matches

try:  # POSIX only; elsewhere the lock is in-process only.
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

TAIL_MAX_LINES = 20000
TAIL_CHUNK_BYTES = 1 << 20
TAIL_MAX_BYTES = 64 << 20


@dataclass(frozen=True)
class SavedTail:
    """Matching lines from the end of a saved console, oldest first."""

    lines: list[str]
    # True when reading stopped before the start of the file.
    truncated: bool
    size: int


def _split_universal(raw: bytes) -> list[str]:
    # Match text-mode universal newlines: "\r\n" and a lone "\r" both end a line.
    text = raw.decode("utf-8", errors="replace")
    if text.endswith("\r"):
        text = text[:-1]
    return text.split("\r")


def read_saved_tail(
    path: Path,
    query: str = "",
    *,
    max_lines: int = TAIL_MAX_LINES,
    chunk_bytes: int = TAIL_CHUNK_BYTES,
    max_bytes: int = TAIL_MAX_BYTES,
) -> SavedTail | None:
    """Return up to ``max_lines`` lines matching ``query`` from the end of ``path``.

    Reads backwards in ``chunk_bytes`` steps and stops once enough lines match
    or ``max_bytes`` have been read, so cost is bounded by the window rather
    than the file. The line cut by the window start is dropped. Returns
    ``None`` when the file is missing or empty.
    """
    try:
        handle = path.open("rb")
    except OSError:
        return None
    with handle:
        size = handle.seek(0, 2)
        if size == 0:
            return None
        end = size
        handle.seek(end - 1)
        if handle.read(1) == b"\n":
            end -= 1  # the final newline terminates the last line; it starts none
        kept: list[str] = []  # newest first
        carry = b""  # start of a line whose beginning lies before ``pos``
        pos = end
        read = 0
        capped = False
        while pos > 0 and not capped and read < max_bytes:
            start = max(0, pos - chunk_bytes, end - max_bytes)
            handle.seek(start)
            data = handle.read(pos - start) + carry
            read += pos - start
            pos = start
            parts = data.split(b"\n")
            if pos > 0:
                carry = parts[0]
                parts = parts[1:]
            else:
                carry = b""
            texts = [text for raw in parts for text in _split_universal(raw)]
            for i in range(len(texts) - 1, -1, -1):
                if entry_matches(texts[i], query):
                    if len(kept) >= max_lines:
                        capped = True
                        break
                    kept.append(texts[i])
        kept.reverse()
        return SavedTail(lines=kept, truncated=capped or pos > 0, size=size)


class DownloadLock:
    """One downloader per output directory.

    Holds an in-process lock keyed on the directory plus, on POSIX, an
    advisory ``flock`` on a sibling ``.<name>.lock`` file so two TUI processes
    do not write the same files at once. If the file lock cannot be used at all
    (no ``fcntl``, or a filesystem without lock support) it degrades to the
    in-process lock.
    """

    _GUARD = threading.Lock()
    _LOCKS: dict[str, threading.Lock] = {}

    def __init__(self, root: Path):
        self._root = Path(root)
        key = str(self._root.resolve())
        with self._GUARD:
            self._thread_lock = self._LOCKS.setdefault(key, threading.Lock())
        self._lock_path = self._root.with_name(f".{self._root.name}.lock")
        self._handle = None
        self._held = False

    def acquire(
        self,
        keep_waiting: Callable[[], bool],
        *,
        on_wait: Callable[[], None] | None = None,
        poll_seconds: float = 0.1,
    ) -> bool:
        """Block until held; return False if ``keep_waiting()`` turns false first."""
        waited = False
        while True:
            if not keep_waiting():
                return False
            if self._thread_lock.acquire(timeout=poll_seconds):
                break
            if not waited and on_wait is not None:
                waited = True
                on_wait()
        while True:
            if self._try_file_lock():
                self._held = True
                return True
            if not waited and on_wait is not None:
                waited = True
                on_wait()
            if not keep_waiting():
                self._thread_lock.release()
                return False
            time.sleep(poll_seconds)

    def _try_file_lock(self) -> bool:
        if fcntl is None:
            return True
        try:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(self._lock_path, "a+b")
        except OSError as exc:
            logger.debug("download lock file unavailable (%s); using in-process lock", exc)
            return True
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        except OSError as exc:
            handle.close()
            logger.debug("flock unsupported (%s); using in-process lock", exc)
            return True
        self._handle = handle
        return True

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        handle = self._handle
        self._handle = None
        if handle is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            handle.close()
        self._thread_lock.release()
