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

"""Keeping the TUI's local cache within bounds (tui/cache_cleanup.py)."""

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from cortex_training._download_lock import DownloadLock
from cortex_training.tui import cache_cleanup
from cortex_training.tui.cache_cleanup import JobLease
from cortex_training.tui.log_cache import LogCache
from cortex_training.tui.log_cache import append_error_log
from cortex_training.tui.log_cache import cached_pages
from cortex_training.tui.log_cache import job_cache_dir

MIB = 1024 * 1024
DIGEST = "0123456789abcdef"


def _job(root, name, *, mib=1, age=0.0):
    job_dir = job_cache_dir(f"{DIGEST}:{name}", root=root)
    for part in ("logs", "metrics"):
        (job_dir / part).mkdir(parents=True, exist_ok=True)
    (job_dir / "metrics" / "gpu.jsonl").write_bytes(b"x" * int(mib * MIB))
    (job_dir / "logs" / "s.jsonl").write_bytes(b"{}\n")
    _age(job_dir, age)
    return job_dir


def _age(job_dir, age):
    when = time.time() - age
    for path in (*job_dir.rglob("*"), job_dir):
        os.utime(path, (when, when), follow_symlinks=False)


def _names(removed):
    return [name for name, _size in removed]


@pytest.fixture(autouse=True)
def _release_leases():
    yield
    cache_cleanup._release_pending()


def test_only_the_cache_s_own_job_directories_are_counted_or_removed(tmp_path):
    cached = _job(tmp_path, "cached", mib=1, age=9000)
    mine = tmp_path / "jobs" / "mine"
    mine.mkdir()
    (mine / "thesis.tex").write_text("years of work")
    shaped = tmp_path / "jobs" / "0123abcd-0000-4000-8000-000000000000"
    shaped.mkdir()
    (shaped / "notes.txt").write_text("mine too")
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "big.bin").write_bytes(b"x" * (3 * MIB))
    sessions = tmp_path / ".sessions"
    sessions.mkdir()
    (sessions / "notes.txt").write_text("not the cache's")

    assert cache_cleanup.enforce_cache_limit(tmp_path, limit=int(1.5 * MIB)) == []
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=1)) == ["cached"]
    cache_cleanup.sweep_stale(tmp_path, temp_dir=str(tmp_path / "no-temp"))
    assert cache_cleanup.enforce_cache_limit(tmp_path, limit=1) == []
    assert not cached.exists()
    for kept in (mine / "thesis.tex", shaped / "notes.txt", tmp_path / "datasets" / "big.bin", sessions / "notes.txt"):
        assert kept.exists(), f"{kept} was removed"


def test_a_jobs_directory_that_is_a_link_is_left_alone(tmp_path):
    outside = tmp_path / "outside"
    _job(outside, "linked")
    canary = outside / "jobs" / "precious.txt"
    canary.write_text("keep")
    root = tmp_path / "cache"
    root.mkdir()
    os.symlink(outside / "jobs", root / "jobs")
    assert cache_cleanup.enforce_cache_limit(root, limit=1) == []
    cache_cleanup.sweep_stale(root, temp_dir=str(tmp_path / "no-temp"))
    assert canary.exists() and cache_cleanup._jobs(outside)


def test_nothing_outside_the_cache_is_reached_or_counted_through_a_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("must survive")
    big = outside / "big.bin"
    big.write_bytes(b"\0" * (5 * MIB))
    root = tmp_path / "cache"
    (root / "jobs").mkdir(parents=True)
    os.symlink(outside, root / "jobs" / f"{DIGEST}%3Alinked-job")
    real = job_cache_dir(f"{DIGEST}:real-job", root=root)
    real.mkdir(parents=True)
    os.symlink(outside, real / "metrics")
    (real / "logs").mkdir()
    os.symlink(big, real / "logs" / "link.jsonl")
    temp = tmp_path / "tmp"
    temp.mkdir()
    os.symlink(outside, temp / f"{cache_cleanup.STAGE_PREFIX}{cache_cleanup._HOST}+1+x")
    assert cache_cleanup._disk_bytes(root) == 0
    cache_cleanup.enforce_cache_limit(root, limit=1)
    cache_cleanup.sweep_stale(root, temp_dir=str(temp))
    assert (outside / "precious.txt").read_text() == "must survive" and big.exists()


def test_a_job_directory_the_tui_cannot_read_is_skipped_not_fatal(tmp_path):
    unreadable = _job(tmp_path, "unreadable", mib=1, age=9000)
    _job(tmp_path, "readable", mib=1, age=8000)
    os.chmod(unreadable, 0)
    try:
        assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=1)) == ["readable"]
        cache_cleanup.sweep_stale(tmp_path, temp_dir=str(tmp_path / "no-temp"))
    finally:
        os.chmod(unreadable, 0o755)


def test_the_cache_keeps_under_its_limit_by_dropping_the_jobs_used_least_recently(tmp_path):
    oldest = _job(tmp_path, "oldest", mib=3, age=3000)
    older = _job(tmp_path, "older", mib=3, age=2000)
    recent = _job(tmp_path, "recent", mib=3, age=10)
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=5 * MIB)) == ["oldest", "older"]
    assert not oldest.exists() and not older.exists() and recent.exists()
    assert cache_cleanup.enforce_cache_limit(tmp_path, limit=5 * MIB) == []


def test_the_last_use_a_lease_records_decides_the_order(tmp_path):
    opened_long_ago = _job(tmp_path, "opened-long-ago", mib=1, age=10)
    opened_recently = _job(tmp_path, "opened-recently", mib=1, age=9000)
    for job, age in ((opened_long_ago, 9000), (opened_recently, 10)):
        (job / ".last-used").touch()
        os.utime(job / ".last-used", (time.time() - age,) * 2)
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=int(1.5 * MIB))) == ["opened-long-ago"]


def test_no_limit_removes_nothing(tmp_path):
    job = _job(tmp_path, "a", mib=2, age=9999)
    assert cache_cleanup.enforce_cache_limit(tmp_path, limit=None) == []
    assert job.exists()


def test_two_passes_at_once_remove_the_excess_once(tmp_path, monkeypatch):
    for i, age in enumerate((4000, 3000, 2000, 1000)):
        _job(tmp_path, f"j{i}", mib=1, age=age)
    entered, go = threading.Event(), threading.Event()
    real = cache_cleanup._remove_unless_in_use

    def slow(job_dir, now):
        entered.set()
        go.wait(5)
        return real(job_dir, now)

    monkeypatch.setattr(cache_cleanup, "_remove_unless_in_use", slow)
    first, limit = [], int(3.5 * MIB)
    thread = threading.Thread(target=lambda: first.extend(cache_cleanup.enforce_cache_limit(tmp_path, limit=limit)))
    thread.start()
    assert entered.wait(5)
    second = cache_cleanup.enforce_cache_limit(tmp_path, limit=limit)
    go.set()
    thread.join(5)
    assert second == [], "a second pass ran alongside the first"
    assert _names(first) == ["j0"] and len(cache_cleanup._jobs(tmp_path)) == 3


def test_a_pass_runs_without_its_lock_file_but_skips_a_job_without_its_lease(tmp_path, monkeypatch):
    # No free inodes or a quota: freeing nothing is worse than two TUIs evicting at once. A
    # job whose .lease can't be created still stays: removing it could race a screen opening it.
    old = _job(tmp_path, "old", mib=1, age=9000)
    no_lease = _job(tmp_path, "no-lease", mib=1, age=8000)
    real_open_lock = cache_cleanup._open_lock

    def open_lock(path):
        if path.name == cache_cleanup._PASS_LOCK or path.parent == no_lease:
            return None  # as when the file can't be created
        return real_open_lock(path)

    monkeypatch.setattr(cache_cleanup, "_open_lock", open_lock)
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=1)) == ["old"]
    assert not old.exists() and no_lease.exists()


def test_a_job_open_in_a_running_tui_is_never_removed(tmp_path):
    open_job = _job(tmp_path, "open", mib=3, age=5000)
    _job(tmp_path, "other", mib=3, age=100)
    JobLease(open_job).acquire()
    os.utime(open_job / ".last-used", (time.time() - 5000,) * 2)
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=1 * MIB)) == ["other"]
    assert open_job.exists()


def test_a_job_being_downloaded_is_never_removed(tmp_path):
    busy = _job(tmp_path, "busy", mib=3, age=5000)
    newer = _job(tmp_path, "newer", mib=3, age=1000)
    lock = DownloadLock(busy / "metrics")
    assert lock.try_acquire()
    try:
        # 6 MiB over a 4 MiB limit: busy is skipped because of its download lock, so newer goes.
        assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=4 * MIB)) == ["newer"]
        assert busy.exists() and not newer.exists()
    finally:
        lock.release()
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=1 * MIB)) == ["busy"]


def test_probing_a_busy_job_s_locks_does_not_make_it_look_newer(tmp_path):
    busy = _job(tmp_path, "busy", mib=3, age=5000)
    middle = _job(tmp_path, "middle", mib=3, age=2000)
    newest = _job(tmp_path, "newest", mib=3, age=100)
    lock = DownloadLock(busy / "metrics")
    assert lock.try_acquire()
    try:
        # 9 MiB over a 7 MiB limit: probes busy (skipped), then removes middle to reach 6 MiB.
        assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=7 * MIB)) == ["middle"]
    finally:
        lock.release()
    # busy was probed during the first pass, but its fallback age is still older than newest's.
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=4 * MIB)) == ["busy"]
    assert newest.exists()


def test_a_first_download_is_protected_before_its_directory_exists(tmp_path):
    job = _job(tmp_path, "first-download", mib=1, age=5000)
    shutil.rmtree(job / "metrics")
    (job / "logs" / "s.jsonl").write_bytes(b"x" * MIB)
    lock = DownloadLock(job / "metrics")
    assert lock.try_acquire()
    try:
        assert cache_cleanup.enforce_cache_limit(tmp_path, limit=1) == []
        assert job.exists()
    finally:
        lock.release()


def test_a_removal_moves_the_job_aside_so_a_screen_opening_meanwhile_never_writes_into_it(tmp_path, monkeypatch):
    job = _job(tmp_path, "racing", mib=1)
    moved, finish = threading.Event(), threading.Event()
    real_rmtree = cache_cleanup.shutil.rmtree

    def slow_rmtree(path, *args, **kwargs):
        moved.set()
        finish.wait(5)
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(cache_cleanup.shutil, "rmtree", slow_rmtree)
    outcome = {}
    remover = threading.Thread(target=lambda: outcome.update(r=cache_cleanup._remove_unless_in_use(job, time.time())))
    remover.start()
    assert moved.wait(5)
    lease = JobLease(job)
    lease.acquire()
    (job / "logs").mkdir(parents=True, exist_ok=True)
    (job / "logs" / "new.jsonl").write_text("{}\n")
    finish.set()
    remover.join(5)
    assert outcome["r"][0] == "removed"
    assert (job / "logs" / "new.jsonl").exists(), "the delete reached what the new screen wrote"
    assert lease._fd is not None and cache_cleanup._same_file(lease._fd, job / ".lease")
    assert not list(job.parent.glob(".removing-*"))
    lease.release()


def test_a_lease_protects_while_held_and_not_after_release(tmp_path):
    job = _job(tmp_path, "j")
    lease = JobLease(job)
    lease.acquire()
    assert cache_cleanup.job_in_use(job)
    assert cache_cleanup.enforce_cache_limit(tmp_path, limit=1) == []
    lease.release()
    assert not cache_cleanup.job_in_use(job)
    assert not (job / ".leases").exists()


def test_a_tui_on_another_machine_is_seen_by_its_marker_whatever_its_pid(tmp_path):
    job = _job(tmp_path, "shared", mib=1, age=5000)
    marker = job / ".leases" / "other-host-4242"
    marker.parent.mkdir()
    marker.touch()
    assert cache_cleanup.job_in_use(job)
    assert cache_cleanup.enforce_cache_limit(tmp_path, limit=1) == []
    stale = time.time() - cache_cleanup.LEASE_STALE_SECONDS - 60
    os.utime(marker, (stale, stale))
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=1)) == ["shared"]


def test_a_killed_tui_leaves_no_lock_and_its_marker_expires(tmp_path):
    job = _job(tmp_path, "k")
    code = ("import os, sys; from pathlib import Path; from cortex_training.tui.cache_cleanup import JobLease; "
            "JobLease(Path(sys.argv[1])).acquire(); os._exit(0)")
    subprocess.run([sys.executable, "-c", code, str(job)], check=True)
    fd = os.open(job / ".lease", os.O_RDWR)
    try:
        cache_cleanup.fcntl.flock(fd, cache_cleanup.fcntl.LOCK_EX | cache_cleanup.fcntl.LOCK_NB)
    finally:
        os.close(fd)
    assert cache_cleanup.job_in_use(job)
    later = time.time() + cache_cleanup.LEASE_STALE_SECONDS + 60
    assert not cache_cleanup.job_in_use(job, now=later)


def test_a_closed_screen_s_lease_outlives_it_for_the_grace_then_goes(tmp_path):
    job = _job(tmp_path, "closing", mib=1, age=5000)
    lease = JobLease(job)
    lease.acquire()
    lease.release_later(0.3)
    assert lease.held and cache_cleanup.job_in_use(job)
    assert cache_cleanup.enforce_cache_limit(tmp_path, limit=1) == []
    deadline = time.time() + 5
    while lease.held and time.time() < deadline:
        time.sleep(0.05)
    assert not lease.held and not cache_cleanup.job_in_use(job)
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=1)) == ["closing"]


def test_leases_still_in_their_grace_are_released_when_the_process_exits(tmp_path):
    job = _job(tmp_path, "exiting")
    code = ("import sys; from pathlib import Path; from cortex_training.tui.cache_cleanup import JobLease; "
            "lease = JobLease(Path(sys.argv[1])); lease.acquire(); lease.release_later(3600)")
    subprocess.run([sys.executable, "-c", code, str(job)], check=True)
    assert not cache_cleanup.job_in_use(job), "the exited TUI's lease outlived it"


def test_a_job_reopened_in_its_grace_keeps_the_new_screen_s_marker(tmp_path):
    job = _job(tmp_path, "reopened")
    first = JobLease(job)
    first.acquire()
    first.release_later(3600)
    second = JobLease(job)
    second.acquire()
    first.release()
    assert cache_cleanup.job_in_use(job), "the reopened screen lost its protection"
    second.release()
    assert not cache_cleanup.job_in_use(job)


def test_the_sweep_removes_only_what_no_running_tui_owns(tmp_path):
    temp = tmp_path / "tmp"
    temp.mkdir()
    host = cache_cleanup._HOST

    def stage(name, *, owner=False, age=0.0):
        path = temp / name
        path.mkdir()
        if owner:
            Path(f"{path}.owner").touch()
        os.utime(path, (time.time() - age,) * 2)
        return path

    mine, mine_fd = cache_cleanup.make_stage()
    os.rename(mine.name, temp / Path(mine.name).name)
    os.rename(f"{mine.name}.owner", f"{temp / Path(mine.name).name}.owner")
    alive = temp / Path(mine.name).name
    gone = stage(f"{cache_cleanup.STAGE_PREFIX}{host}+4242+abc", owner=True)
    huge = stage(f"{cache_cleanup.STAGE_PREFIX}{host}+99999999999999999999+abc", owner=True)
    other_host = stage(f"{cache_cleanup.STAGE_PREFIX}other-machine+4242+abc", owner=True)
    legacy_old = stage(f"{cache_cleanup.STAGE_PREFIX}oldname1", age=8 * 24 * 3600)
    legacy_new = stage(f"{cache_cleanup.STAGE_PREFIX}oldname2")
    unrelated = stage("something-else")
    orphan_owner = temp / f"{cache_cleanup.STAGE_PREFIX}{host}+4242+orphan.owner"
    orphan_owner.touch()

    root = tmp_path / "cache"
    job = _job(root, "j")
    aside = job.parent / ".removing-cut-short"
    aside.mkdir()
    (aside / "f").write_text("x")
    stale_copy, fresh_copy = job / ".console-old", job / ".console-new"
    stale_tmp, fresh_tmp = job / "logs" / ".tmp-a.swp", job / "logs" / ".tmp-b.swp"
    for path in (stale_copy, fresh_copy):
        path.mkdir()
    for path in (stale_tmp, fresh_tmp):
        path.write_text("partial")
    stale_marker, fresh_marker = job / ".leases" / "gone-host-1-x", job / ".leases" / "live-host-2-y"
    stale_marker.parent.mkdir()
    for path in (stale_marker, fresh_marker):
        path.touch()
    old = time.time() - 2 * 3600
    for path in (stale_copy, stale_tmp):
        os.utime(path, (old, old))
    os.utime(stale_marker, (time.time() - cache_cleanup.LEASE_STALE_SECONDS - 60,) * 2)

    removed = set(cache_cleanup.sweep_stale(root, temp_dir=str(temp)))
    assert removed == {gone, huge, legacy_old, aside, stale_copy, stale_tmp, stale_marker}
    for kept in (alive, other_host, legacy_new, unrelated, fresh_copy, fresh_tmp, fresh_marker):
        assert kept.exists(), f"{kept} was swept"
    assert not Path(f"{gone}.owner").exists() and not orphan_owner.exists()
    os.close(mine_fd)


def test_a_stage_made_for_a_screen_is_swept_once_its_tui_is_gone(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_cleanup.tempfile, "tempdir", str(tmp_path))
    stage, fd = cache_cleanup.make_stage()
    assert cache_cleanup.sweep_stale(tmp_path / "cache") == []
    os.close(fd)
    assert [Path(stage.name)] == cache_cleanup.sweep_stale(tmp_path / "cache")
    assert not Path(f"{stage.name}.owner").exists()


def test_a_job_that_cannot_be_moved_aside_is_skipped_and_the_pass_goes_on(tmp_path, monkeypatch):
    stuck = _job(tmp_path, "stuck", mib=1, age=9000)
    _job(tmp_path, "next", mib=1, age=8000)
    _job(tmp_path, "recent", mib=1, age=10)
    real_rename = cache_cleanup.os.rename

    def rename(src, dst):
        if Path(src) == stuck:
            raise PermissionError("can't move it")
        return real_rename(src, dst)

    monkeypatch.setattr(cache_cleanup.os, "rename", rename)
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=int(1.5 * MIB))) == ["next", "recent"]
    assert stuck.exists()


def test_the_error_log_rotates_instead_of_growing_without_end(tmp_path):
    path = str(tmp_path / "errors.log")
    for _ in range(50):
        append_error_log("x" * 100 + "\n", path=path, max_bytes=1000)
    assert os.path.getsize(path) <= 1000 + 101 and os.path.getsize(path + ".1") <= 1000 + 101
    append_error_log("y\n", path=str(tmp_path / "no-such-dir" / "errors.log"))


def test_two_writers_rotating_at_once_keep_the_older_lines(tmp_path, monkeypatch):
    import cortex_training.tui.log_cache as log_cache

    path = str(tmp_path / "errors.log")
    Path(path).write_text("old history\n" * 1000)
    real_replace = log_cache.os.replace

    def slow_replace(src, dst):
        time.sleep(0.2)
        return real_replace(src, dst)

    monkeypatch.setattr(log_cache.os, "replace", slow_replace)
    writers = [threading.Thread(target=append_error_log, args=(f"new {i}\n",), kwargs={"path": path, "max_bytes": 100})
               for i in range(2)]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join()
    assert Path(path + ".1").read_text().startswith("old history"), "the older lines were lost"
    assert sorted(Path(path).read_text().split("\n")[:-1]) == ["new 0", "new 1"]


def test_a_traceback_goes_through_the_rotation(tmp_path, monkeypatch):
    import cortex_training.tui.widgets as widgets

    path = tmp_path / "errors.log"
    path.write_bytes(b"x" * (6 * MIB))
    monkeypatch.setattr(widgets, "_ERROR_LOG", str(path))
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        widgets.log_traceback("a failure")
    assert (tmp_path / "errors.log.1").stat().st_size == 6 * MIB
    assert "RuntimeError: boom" in path.read_text()


def test_two_tuis_updating_cursors_of_one_job_lose_neither(tmp_path):
    def writer(source):
        cache = LogCache("shared-job", root=tmp_path)
        for n in range(150):
            cache.set_cursor(source, f"{source}-{n}")

    threads = [threading.Thread(target=writer, args=(f"sub-{i}",)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    cache = LogCache("shared-job", root=tmp_path)
    assert {f"sub-{i}": cache.get_cursor(f"sub-{i}") for i in range(4)} == {f"sub-{i}": f"sub-{i}-149" for i in range(4)}


def test_marking_a_log_s_head_missing_never_rolls_its_cursor_back(tmp_path, monkeypatch):
    advancing = LogCache("j", root=tmp_path)
    marking = LogCache("j", root=tmp_path)
    monkeypatch.setattr(marking, "head_missing", lambda source_id: False)

    def advance():
        for n in range(200):
            advancing.set_cursor("s", f"c{n}")

    def mark():
        for _ in range(200):
            marking.set_head_missing("s")

    threads = [threading.Thread(target=advance), threading.Thread(target=mark)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert advancing.get_cursor("s") == "c199"
    assert LogCache("j", root=tmp_path).head_missing("s")


def test_a_cancelled_tail_persists_nothing(tmp_path):
    cache = LogCache("late", root=tmp_path)
    cancelled = []

    def fetch(_cursor):
        cancelled.append(True)
        return {"entries": [{"msg": "late line"}], "next_cursor": "after-late"}

    pages = list(cached_pages(cache, "s", fetch, entries_key="entries", follow=True, poll_interval=0.01,
                              is_cancelled=lambda: bool(cancelled), sleep=lambda _s: None))
    assert pages == []
    assert cache.get_cursor("s") is None and cache.load_entries("s") == []


def test_a_cancelled_tail_persists_neither_an_empty_page_nor_a_missing_head(tmp_path):
    # The screen closed during the read: an empty page's new cursor, and the
    # "head missing" mark from a cursorless window, must not write the job back.
    cache = LogCache("late", root=tmp_path)
    cache.set_cursor("empty", "old")
    cancelled = []

    def fetch(cursor):
        cancelled.append(True)
        if cursor == "old":
            return {"entries": [], "next_cursor": "new"}
        return {"entries": [{"n": 1}, {"n": 2}], "next_cursor": "window-end"}

    pages = list(cached_pages(cache, "empty", fetch, entries_key="entries", follow=True, poll_interval=0.01,
                              is_cancelled=lambda: bool(cancelled), sleep=lambda _s: None))
    assert pages == [] and cache.get_cursor("empty") == "old"

    cancelled.clear()
    pages = list(cached_pages(cache, "window", fetch, entries_key="entries", follow=False, poll_interval=0.01,
                              tail_window=2, is_cancelled=lambda: bool(cancelled), sleep=lambda _s: None))
    assert pages == []
    assert cache.get_cursor("window") is None and cache.load_entries("window") == []
    assert not cache.head_missing("window")
