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

"""Tests for the TUI's saved-console tail reader and download lock."""

import threading
import time

import pytest

from cortex_training.tui.saved_log import DownloadLock
from cortex_training.tui.saved_log import read_saved_tail


def _write(tmp_path, data: bytes):
    path = tmp_path / "stdout.log"
    path.write_bytes(data)
    return path


def test_missing_or_empty_file_returns_none(tmp_path):
    assert read_saved_tail(tmp_path / "absent.log") is None
    assert read_saved_tail(_write(tmp_path, b"")) is None


@pytest.mark.parametrize("chunk_bytes", [1, 3, 7, 64, 1 << 20])
def test_last_lines_across_chunk_boundaries(tmp_path, chunk_bytes):
    lines = [f"line-{i:03d}" for i in range(200)]
    path = _write(tmp_path, ("\n".join(lines) + "\n").encode())
    tail = read_saved_tail(path, max_lines=25, chunk_bytes=chunk_bytes)
    assert tail.lines == lines[-25:]
    assert tail.truncated is True
    assert tail.size == path.stat().st_size


@pytest.mark.parametrize("chunk_bytes", [2, 5, 1 << 20])
def test_file_smaller_than_window_returns_every_line(tmp_path, chunk_bytes):
    path = _write(tmp_path, b"first\n\nthird\nno-newline-at-end")
    tail = read_saved_tail(path, chunk_bytes=chunk_bytes)
    assert tail.lines == ["first", "", "third", "no-newline-at-end"]
    assert tail.truncated is False


def test_window_start_drops_the_partial_first_line(tmp_path):
    path = _write(tmp_path, b"aaaaaaaaaa\nbbbb\ncccc\n")
    # 12 bytes back from the final newline starts inside "aaaaaaaaaa".
    tail = read_saved_tail(path, chunk_bytes=4, max_bytes=12)
    assert tail.lines == ["bbbb", "cccc"]
    assert tail.truncated is True


def test_window_exactly_at_line_start_keeps_that_line(tmp_path):
    path = _write(tmp_path, b"aaaa\nbbbb\ncccc\n")
    # "bbbb\ncccc" is 9 bytes, but the reader cannot see the preceding newline
    # without reading one more byte, so the line at the cut is dropped.
    assert read_saved_tail(path, max_bytes=9).lines == ["cccc"]
    assert read_saved_tail(path, max_bytes=10).lines == ["bbbb", "cccc"]


def test_filter_is_applied_and_keeps_the_newest_matches(tmp_path):
    lines = [f"{'ERROR' if i % 10 == 0 else 'info'} {i}" for i in range(100)]
    path = _write(tmp_path, ("\n".join(lines) + "\n").encode())
    tail = read_saved_tail(path, "error", max_lines=3, chunk_bytes=16)
    assert tail.lines == ["ERROR 70", "ERROR 80", "ERROR 90"]
    assert tail.truncated is True
    every = read_saved_tail(path, "error", chunk_bytes=16)
    assert every.lines == [f"ERROR {i}" for i in range(0, 100, 10)]
    assert every.truncated is False
    assert read_saved_tail(path, "no-such-text").lines == []


def test_universal_newlines_and_invalid_utf8(tmp_path):
    path = _write(tmp_path, b"crlf\r\nprogress 1\rprogress 2\r\nbad \xff byte\n")
    tail = read_saved_tail(path, chunk_bytes=3)
    assert tail.lines == ["crlf", "progress 1", "progress 2", "bad � byte"]


def test_multibyte_characters_split_across_chunks(tmp_path):
    lines = ["héllo wörld ✓"] * 5
    path = _write(tmp_path, ("\n".join(lines) + "\n").encode())
    assert read_saved_tail(path, chunk_bytes=3).lines == lines


def test_download_lock_serializes_threads(tmp_path):
    root = tmp_path / "stdout"
    first = DownloadLock(root)
    assert first.acquire(lambda: True)
    waits = []
    acquired = threading.Event()

    def contender():
        second = DownloadLock(root)
        if second.acquire(lambda: True, on_wait=lambda: waits.append(1), poll_seconds=0.01):
            acquired.set()
            second.release()

    thread = threading.Thread(target=contender)
    thread.start()
    time.sleep(0.1)
    assert not acquired.is_set()
    assert waits == [1]
    first.release()
    thread.join(timeout=5)
    assert acquired.is_set()


def test_download_lock_gives_up_when_told_to_stop(tmp_path):
    root = tmp_path / "stdout"
    holder = DownloadLock(root)
    assert holder.acquire(lambda: True)
    tries = iter([True, True, False])
    assert DownloadLock(root).acquire(lambda: next(tries), poll_seconds=0.01) is False
    holder.release()
    again = DownloadLock(root)
    assert again.acquire(lambda: True)
    again.release()


def test_download_lock_waits_on_a_file_lock_held_elsewhere(tmp_path):
    fcntl = pytest.importorskip("fcntl")
    root = tmp_path / "stdout"
    lock_path = tmp_path / ".stdout.lock"
    # A separate open file description stands in for another TUI process.
    with open(lock_path, "a+b") as other:
        fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        tries = iter([True] * 5 + [False])
        waits = []
        lock = DownloadLock(root)
        assert lock.acquire(lambda: next(tries), on_wait=lambda: waits.append(1), poll_seconds=0.01) is False
        assert waits == [1]
        fcntl.flock(other.fileno(), fcntl.LOCK_UN)
    lock = DownloadLock(root)
    assert lock.acquire(lambda: True)
    lock.release()
