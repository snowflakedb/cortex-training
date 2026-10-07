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

"""Smoke tests for the Textual app (job picker -> log screen). Skipped unless
the optional ``textual`` extra is installed (``pip install 'cortex-training[tui]'``)."""

import asyncio
import glob
import os
import re
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from textual.widgets import Input  # noqa: E402

from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402
from cortex_training.tui.app import LogScreen  # noqa: E402


def _client():
    c = MagicMock()
    c.list_jobs.return_value = [
        {"job_id": "7", "status": "RUNNING", "sub_jobs": [{"job_type": "training"}]},
        {"job_id": "old", "status": "FAILED"},
    ]
    # Sources come from get_job().sub_jobs now (one per sub-job). RUNNING keeps
    # the live tail path; terminal-job behavior is covered by overriding get_job.
    c.get_job.return_value = {
        "status": "RUNNING",
        "sub_jobs": [
            {"sub_job_id": "7:training:0", "job_type": "training"},
            {"sub_job_id": "7:sampling:0", "job_type": "sampling"},
        ],
    }
    # Finite results so the tail worker doesn't loop forever.
    c.tail_logs.return_value = {"entries": [], "next_cursor": "", "eof": True}
    return c


async def _wait(pilot, app, predicate, tries=100):
    for _ in range(tries):
        await pilot.pause()
        if predicate():
            return True
    return False


async def _settle(app, pilot):
    """Stop background workers and drain before ``run_test`` tears down, so a
    thread worker (ours, or Textual's internal Log resize worker) can't call into
    a half-destroyed app and surface a NoActiveAppError as a WorkerError. Without
    this the live-tail/resize workers make these tests flaky."""
    try:
        app.workers.cancel_all()
    except Exception:  # noqa: BLE001
        pass
    for _ in range(5):
        await pilot.pause()


async def _run_with_job_id():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot, app, lambda: isinstance(app.screen, LogScreen) and len(app.screen.query("#sources ListItem")) == 2
        )
        assert ok, "log screen / sources did not load"
        screen = app.screen
        assert screen._source_by_item["src-0"] == "7:training:0"
        assert screen._source_by_item["src-1"] == "7:sampling:0"
        await _settle(app, pilot)


async def _run_with_picker():
    app = CortexTrainingLogTUI(_client(), poll_interval=0.01)  # no job_id -> picker
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot, app, lambda: isinstance(app.screen, JobListScreen) and len(app.screen.query("#jobs ListItem")) >= 2
        )
        assert ok, "job picker did not load jobs"
        screen = app.screen
        # active job (RUNNING) sorted first
        assert screen._job_by_item["job-0"] == "7"
        assert screen._job_by_item["job-1"] == "old"
        assert re.search(r"last refreshed \d{2}:\d{2}:\d{2}", app.sub_title or "")
        await _settle(app, pilot)


def test_app_with_job_id_opens_logs():
    asyncio.run(_run_with_job_id())


def test_app_without_job_id_shows_job_picker():
    asyncio.run(_run_with_picker())


async def _run_refresh_picker():
    c = _client()
    jobs = c.list_jobs.return_value
    refresh_started = threading.Event()
    finish_refresh = threading.Event()
    calls = 0

    def list_jobs():
        nonlocal calls
        calls += 1
        if calls > 1:
            refresh_started.set()
            finish_refresh.wait(timeout=5)
        return jobs

    c.list_jobs.side_effect = list_jobs
    app = CortexTrainingLogTUI(c, poll_interval=0.01)
    try:
        async with app.run_test() as pilot:
            ok = await _wait(
                pilot,
                app,
                lambda: isinstance(app.screen, JobListScreen)
                and len(app.screen.query("#jobs ListItem")) == 2,
            )
            assert ok, "initial jobs did not load"
            previous_refresh = app.screen._last_refreshed
            # Refresh re-populates with the same item IDs (job-0, job-1). Must
            # show that work is underway without discarding the last success.
            app.screen.action_refresh()
            ok = await _wait(pilot, app, refresh_started.is_set)
            assert ok, "refresh did not start"
            assert "refreshing…" in (app.sub_title or "")
            assert f"last refreshed {previous_refresh}" in (app.sub_title or "")
            finish_refresh.set()
            ok = await _wait(
                pilot,
                app,
                lambda: "refreshing…" not in (app.sub_title or "")
                and len(app.screen.query("#jobs ListItem")) == 2,
            )
            assert ok, "refresh did not finish cleanly"
            await _settle(app, pilot)
    finally:
        finish_refresh.set()


async def _run_refresh_sources():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot, app, lambda: isinstance(app.screen, LogScreen) and len(app.screen.query("#sources ListItem")) == 2
        )
        assert ok, "initial sources did not load"
        app.screen.action_refresh_sources()
        ok = await _wait(pilot, app, lambda: len(app.screen.query("#sources ListItem")) == 2)
        assert ok, "source refresh changed item count (DuplicateIds regression)"
        await _settle(app, pilot)


def test_picker_refresh_no_duplicate_ids():
    asyncio.run(_run_refresh_picker())


def test_source_refresh_no_duplicate_ids():
    asyncio.run(_run_refresh_sources())


async def _run_offline_fallback():
    c = _client()
    c.get_job.side_effect = RuntimeError("network down")  # can't fetch sub-jobs
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and "7:training:0" in (app.screen._source_by_item or {}).values(),
        )
        assert ok, "cached sub-job not shown after get_job failure"
        await _settle(app, pilot)


def test_offline_shows_cached_sources(tmp_path, monkeypatch):
    # Seed a cached sub-job for job "7", then make get_job fail: the source list
    # must fall back to the cached sub-job so logs stay reachable.
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    asyncio.run(_run_offline_fallback())


async def _run_terminal_cache_only():
    c = _client()
    c.get_job.return_value = {  # terminal -> zone gone; metadata still lists sub-jobs
        "status": "FAILED",
        "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
    }
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and "7:training:0" in (app.screen._source_by_item or {}).values(),
        )
        assert ok, "sub-job source not shown for terminal job"
        # A terminal job is served from cache only — no tail-logs operation call.
        c.tail_logs.assert_not_called()
        await _settle(app, pilot)


def test_terminal_job_is_cache_only(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    asyncio.run(_run_terminal_cache_only())


async def _run_resize_sources():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen))
        assert ok, "log screen did not open"
        s = app.screen
        w0 = s._sources_width
        s.action_grow_sources()
        assert s._sources_width == w0 + s._SOURCES_STEP
        s.action_shrink_sources()
        s.action_shrink_sources()
        assert s._sources_width == w0 - s._SOURCES_STEP
        for _ in range(50):
            s.action_shrink_sources()
        assert s._sources_width == s._SOURCES_MIN  # clamps low
        for _ in range(50):
            s.action_grow_sources()
        assert s._sources_width == s._SOURCES_MAX  # clamps high
        await _settle(app, pilot)


def test_sources_panel_resize():
    asyncio.run(_run_resize_sources())


async def _run_picker_filter():
    app = CortexTrainingLogTUI(_client(), poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot, app, lambda: isinstance(app.screen, JobListScreen) and len(app.screen.query("#jobs ListItem")) >= 2
        )
        assert ok, "jobs did not load"
        s = app.screen
        s.query_one("#jobfilter", Input).value = "old"  # fires Input.Changed → re-render
        ok = await _wait(pilot, app, lambda: list(s._job_by_item.values()) == ["old"])
        assert ok, f"filter did not narrow the list: {s._job_by_item}"
        await _settle(app, pilot)


def test_picker_filter():
    asyncio.run(_run_picker_filter())


async def _run_logscreen_controls():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._logview is not None)
        assert ok, "log screen did not open"
        s = app.screen
        # Following is managed manually (write-time), so the widget's blanket
        # auto_scroll stays off; pause just toggles the follow flag.
        assert s._logview.auto_scroll is False
        s.action_toggle_pause()
        assert s._paused is True
        s.action_toggle_pause()
        assert s._paused is False
        assert s._min_level is None
        s.action_cycle_level()
        assert s._min_level == "INFO"
        s.action_cycle_level()
        assert s._min_level == "WARNING"
        s.action_cycle_level()
        assert s._min_level == "ERROR"
        s.action_cycle_level()
        assert s._min_level is None
        await _settle(app, pilot)


def test_logscreen_controls():
    asyncio.run(_run_logscreen_controls())


async def _run_always_tails_bottom():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and app.screen._current_source is not None
            and len(app.screen.query("#sources ListItem")) == 2,
        )
        assert ok, "log screen / initial tail did not start"
        s = app.screen
        lv = s._logview
        # Writing past the viewport keeps the view pinned to the tail — every
        # viewer lands on the newest lines, no manual scroll needed.
        for i in range(200):
            s._write_line(f"line {i}")
        await pilot.pause()
        assert lv.is_vertical_scroll_end, "log should stick to the bottom"
        # A new live line keeps us at the bottom.
        s._write_line("newest live line")
        await pilot.pause()
        assert lv.is_vertical_scroll_end, "live tail should keep following the bottom"
        # Pause is the escape hatch: scroll up to read history and new lines
        # must not yank the user back down.
        s.action_toggle_pause()
        assert s._paused is True
        lv.scroll_home(animate=False)
        await pilot.pause()
        s._write_line("live while paused")
        await pilot.pause()
        assert not lv.is_vertical_scroll_end, "paused tail must not follow"
        # Un-pausing snaps back to the tail.
        s.action_toggle_pause()
        await pilot.pause()
        assert lv.is_vertical_scroll_end, "un-pause should jump back to the bottom"
        await _settle(app, pilot)


def test_log_always_tails_bottom():
    asyncio.run(_run_always_tails_bottom())


async def _run_replay_lands_at_bottom():
    c = _client()
    c.get_job.return_value = {  # terminal → cache-only replay, no live poll
        "status": "FAILED",
        "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
    }
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source == "7:training:0"
        )
        assert ok, "log screen / cache replay did not start"
        s = app.screen
        lv = s._logview
        # The cached backlog is far taller than the viewport — exactly the
        # reopen-a-long-running-job case. The old code wrote the replay with
        # scroll_end=False and stranded the viewer at the oldest line; the fix
        # lands them on the newest. Wait for the replay, then assert the tail.
        ok = await _wait(pilot, app, lambda: any("cache line 299" in ln for ln in s._shown_lines))
        assert ok, "cache replay did not render"
        await pilot.pause()
        assert lv.is_vertical_scroll_end, "cache replay must land at the tail, not the top"
        await _settle(app, pilot)


def test_cache_replay_lands_at_bottom(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": f"cache line {i}"} for i in range(300)])
    asyncio.run(_run_replay_lands_at_bottom())


async def _run_source_switch_resets_pause():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and app.screen._current_source is not None
            and len(app.screen.query("#sources ListItem")) == 2,
        )
        assert ok, "log screen / initial tail did not start"
        s = app.screen
        lv = s._logview
        # Pause on the first source (the user is reading history).
        s.action_toggle_pause()
        assert s._paused is True
        # Switching to another source is a fresh follow context: a stale pause
        # must clear so the new source lands on — and follows — its own tail
        # instead of being stranded at the top.
        other = s._source_by_item["src-1"]
        assert other != s._current_source
        s._start_tail(other)
        await pilot.pause()
        assert s._paused is False, "switching sources must clear a stale pause"
        assert s._current_source == other
        for i in range(200):
            s._write_line(f"line {i}")
        await pilot.pause()
        assert lv.is_vertical_scroll_end, "new source should follow its tail"
        await _settle(app, pilot)


def test_source_switch_resets_pause():
    asyncio.run(_run_source_switch_resets_pause())


async def _run_last_updated():
    c = _client()
    calls = {"n": 0}

    def tail(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"entries": [{"_raw": "live line"}], "next_cursor": "C1", "eof": False}
        return {"entries": [], "next_cursor": "C1", "eof": False}

    c.tail_logs.side_effect = tail
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        # Wait for the subtitle to pick up the freshness stamp from a live batch.
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and "updated" in (app.sub_title or ""))
        assert ok, "subtitle never showed a last-updated stamp"
        assert app.screen._last_update is not None
        assert "read-only" not in app.sub_title  # dropped from the subtitle
        assert "RUNNING" in app.sub_title  # job status now on this line
        # The summary line is reserved for static config — no status word there.
        from textual.widgets import Static

        summary = str(app.screen.query_one("#summary", Static).render())
        assert "RUNNING" not in summary
        await _settle(app, pilot)


def test_subtitle_last_updated():
    asyncio.run(_run_last_updated())


async def _run_export():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._logview is not None)
        assert ok, "log screen did not open"
        s = app.screen
        s._current_source = "head:server"
        s.action_save_log()  # background worker writes the file
        ok = await _wait(
            pilot,
            app,
            lambda: bool(glob.glob(os.path.expanduser("~/cortex-training-7-*.log"))),
        )
        assert ok, "export file was not written"
        content = open(
            glob.glob(os.path.expanduser("~/cortex-training-7-*.log"))[0],
            encoding="utf-8",
        ).read()
        assert "alpha" in content and "beta" in content
        await _settle(app, pilot)


def test_log_export(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))  # so ~/ writes into tmp
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("head:server", [{"_raw": "alpha"}, {"_raw": "beta"}])
    asyncio.run(_run_export())


async def _run_wrap_and_select():
    from textual.widgets import Log

    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        # Wait until the initial source is selected (its tail started + banner
        # written) so _start_tail won't reset _shown_lines under us.
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and app.screen._logview is not None
            and app.screen._current_source is not None
            and len(app.screen.query("#sources ListItem")) == 2,
        )
        assert ok, "log screen / initial tail did not start"
        s = app.screen
        # The log pane is a Log (not RichLog): it participates in Textual's
        # native text selection, and the screen can copy the selection.
        assert isinstance(s._logview, Log)
        assert s._logview.ALLOW_SELECT is True
        assert hasattr(s, "action_copy_text")  # `c` → copy selection

        # Wrapping: a long line is split to the pane width on write.
        s._content_width = lambda: 10
        before = len(s._logview._lines)
        s._write_line("x" * 60)  # 60 / 10 = 6 visual lines
        assert len(s._logview._lines) - before == 6
        assert s._shown_lines[-1] == "x" * 60  # buffer keeps the unwrapped line

        # Re-wrap on width change: widening collapses each buffered line to one.
        s._content_width = lambda: 200
        s._rewrap()
        assert len(s._logview._lines) == len(s._shown_lines)
        assert s._wrap_width == 200
        await _settle(app, pilot)


def test_log_wrap_and_selectable():
    asyncio.run(_run_wrap_and_select())


def _write_stdout(output_dir, sub_job_id, text):
    path = Path(output_dir) / sub_job_id / "stdout.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _terminal_job():
    return {
        "status": "FAILED",
        "sub_jobs": [
            {"sub_job_id": "7:training:0", "job_type": "training"},
            {"sub_job_id": "7:sampling:0", "job_type": "sampling"},
        ],
    }


async def _run_terminal_shows_saved_file():
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir):
        _write_stdout(output_dir, "7:training:0", "saved line\n")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "saved line" in app.screen._shown_lines)
        assert ok, "saved console did not replace the pane"
        c.tail_logs.assert_not_called()
        await _settle(app, pilot)


def test_terminal_open_shows_downloaded_file(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_terminal_shows_saved_file())


async def _run_keeps_cache_when_download_fails():
    c = _client()
    c.get_job.return_value = _terminal_job()
    c.download_stdout_logs.side_effect = RuntimeError("unavailable")
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: "cached line" in app.screen._shown_lines and not app.screen._stdout_inflight,
        )
        assert ok, "cache line missing after a failed download"
        assert "No log output is available." not in app.screen._shown_lines
        downloads = c.download_stdout_logs.call_count
        app.screen._refilter()
        for _ in range(5):
            await pilot.pause()
        assert c.download_stdout_logs.call_count == downloads
        await _settle(app, pilot)


def test_failed_download_keeps_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    asyncio.run(_run_keeps_cache_when_download_fails())


async def _run_empty_file_keeps_cache():
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir):
        _write_stdout(output_dir, "7:training:0", "")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: app.screen._stdout_ready and "cached line" in app.screen._shown_lines,
        )
        assert ok, "empty saved file did not leave the cache on screen"
        await _settle(app, pilot)


def test_empty_saved_file_keeps_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    asyncio.run(_run_empty_file_keeps_cache())


async def _run_hidden_filter_shows_no_note(release, started):
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir):
        started.set()
        assert release.wait(timeout=5)
        _write_stdout(output_dir, "7:training:0", "")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "cached line" in app.screen._shown_lines and started.is_set())
        assert ok, "cache was not shown before the filter"
        app.screen._filter = "no-such-line"
        release.set()
        ok = await _wait(pilot, app, lambda: app.screen._stdout_ready)
        assert ok, "download did not finish"
        for _ in range(5):
            await pilot.pause()
        assert "cached line" not in app.screen._shown_lines
        assert "No log output is available." not in app.screen._shown_lines
        await _settle(app, pilot)


def test_filter_that_hides_cache_shows_no_empty_note(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    release = threading.Event()
    try:
        asyncio.run(_run_hidden_filter_shows_no_note(release, threading.Event()))
    finally:
        release.set()


async def _run_running_tail_error_does_not_download():
    c = _client()
    c.tail_logs.side_effect = RuntimeError("zone down")
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: any(line.startswith("[error] tail") for line in app.screen._shown_lines))
        assert ok, "running-job tail error was not shown"
        c.download_stdout_logs.assert_not_called()
        await _settle(app, pilot)


def test_running_tail_error_does_not_download():
    asyncio.run(_run_running_tail_error_does_not_download())


async def _run_tail_error_after_finish_shows_file():
    c = _client()
    calls = {"get": 0, "tail": 0}

    def get_job(_job_id):
        calls["get"] += 1
        if calls["get"] == 1:
            return {
                "status": "RUNNING",
                "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
            }
        return {"status": "FAILED", "reason": "cancelled", "sub_jobs": []}

    def tail_logs(*_args, **_kwargs):
        calls["tail"] += 1
        raise RuntimeError("zone gone")

    def download(job_id, output_dir):
        _write_stdout(output_dir, "7:training:0", "after finish\n")
        return []

    c.get_job.side_effect = get_job
    c.tail_logs.side_effect = tail_logs
    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "after finish" in app.screen._shown_lines)
        assert ok, "terminal tail error did not show the saved console"
        assert app.screen._job_status == "FAILED"
        tails = calls["tail"]
        app.screen._refilter()
        await _wait(pilot, app, lambda: "after finish" in app.screen._shown_lines)
        assert calls["tail"] == tails
        await _settle(app, pilot)


def test_tail_error_after_terminal_status_shows_file_once():
    asyncio.run(_run_tail_error_after_finish_shows_file())


async def _run_terminal_download_failure_keeps_cache():
    c = _client()
    calls = {"get": 0}

    def get_job(_job_id):
        calls["get"] += 1
        if calls["get"] == 1:
            return {
                "status": "RUNNING",
                "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
            }
        return {"status": "FAILED", "sub_jobs": []}

    c.get_job.side_effect = get_job
    c.tail_logs.side_effect = RuntimeError("zone gone")
    c.download_stdout_logs.side_effect = RuntimeError("unavailable")
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: c.download_stdout_logs.called and "cached line" in app.screen._shown_lines,
        )
        assert ok, "failed saved-console download did not leave the cache"
        for _ in range(5):
            await pilot.pause()
        assert not any(line.startswith("[error] tail") for line in app.screen._shown_lines)
        await _settle(app, pilot)


def test_terminal_download_failure_keeps_cache_without_tail_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    asyncio.run(_run_terminal_download_failure_keeps_cache())


async def _run_healthy_tail_does_not_download():
    c = _client()
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source == "7:training:0")
        assert ok, "live tail did not start"
        await _settle(app, pilot)
        c.download_stdout_logs.assert_not_called()


def test_healthy_running_tail_does_not_download():
    asyncio.run(_run_healthy_tail_does_not_download())


async def _run_filter_during_download_once(release, started, calls):
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir):
        calls.append(output_dir)
        started.set()
        assert release.wait(timeout=5)
        _write_stdout(output_dir, "7:training:0", "from file\n")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, started.is_set)
        assert ok, "download did not start"
        app.screen._refilter()
        assert len(calls) == 1
        ok = await _wait(pilot, app, lambda: "cached line" in app.screen._shown_lines)
        assert ok, "cache replay did not paint during the download"
        release.set()
        ok = await _wait(pilot, app, lambda: "from file" in app.screen._shown_lines)
        assert ok, "finished download did not replace the current source"
        assert "cached line" not in app.screen._shown_lines
        for _ in range(5):
            await pilot.pause()
        assert "cached line" not in app.screen._shown_lines
        assert len(calls) == 1
        await _settle(app, pilot)


def test_filter_during_download_does_not_start_another(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    release = threading.Event()
    started = threading.Event()
    try:
        asyncio.run(_run_filter_during_download_once(release, started, []))
    finally:
        release.set()


async def _run_switch_source_does_not_download_again():
    c = _client()
    c.get_job.return_value = _terminal_job()
    calls = []

    def download(job_id, output_dir):
        calls.append(1)
        _write_stdout(output_dir, "7:training:0", "training line\n")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._stdout_ready)
        assert ok, "first download did not finish"
        app.screen._start_tail("7:sampling:0")
        await _wait(pilot, app, lambda: "sampling cache" in app.screen._shown_lines)
        assert len(calls) == 1
        await _settle(app, pilot)


def test_source_switch_after_download_does_not_download_again(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:sampling:0", [{"_raw": "sampling cache"}])
    asyncio.run(_run_switch_source_does_not_download_again())


async def _run_missing_sibling_shows_note():
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir):
        _write_stdout(output_dir, "7:training:0", "training line\n")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._stdout_ready)
        assert ok, "first download did not finish"
        app.screen._start_tail("7:sampling:0")
        ok = await _wait(pilot, app, lambda: "No log output is available." in app.screen._shown_lines)
        assert ok, "empty sibling did not show the finished-view note"
        assert "training line" not in app.screen._shown_lines
        assert c.download_stdout_logs.call_count == 1
        await _settle(app, pilot)


def test_missing_sibling_after_download_shows_empty_note():
    asyncio.run(_run_missing_sibling_shows_note())


async def _run_save_exports_file(tmp_path):
    c = _client()
    c.get_job.return_value = _terminal_job()

    payload = b"file body\r\nsecond\r"

    def download(job_id, output_dir):
        path = _write_stdout(output_dir, "7:training:0", "")
        path.write_bytes(payload)
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "file body" in app.screen._shown_lines)
        assert ok, "saved console was not shown"
        app.screen._filter = "file"
        copied: list[str] = []
        app.copy_to_clipboard = copied.append
        app.screen.action_copy_log()
        ok = await _wait(pilot, app, lambda: bool(copied))
        assert ok, "copy did not export the saved console"
        assert copied == ["file body\r\nsecond\r"]
        home = tmp_path / "home"
        home.mkdir()
        import cortex_training.tui.app as app_module

        original = app_module.os.path.expanduser
        app_module.os.path.expanduser = lambda path: str(home / os.path.basename(path))
        try:
            app.screen.action_save_log()
            saved = home / "cortex-training-7-7-training-0.log"
            ok = await _wait(pilot, app, saved.is_file)
            assert ok, "save did not write the console file"
            assert saved.read_bytes() == b"file body\r\nsecond\r"
        finally:
            app_module.os.path.expanduser = original
        await _settle(app, pilot)


def test_save_exports_saved_console_not_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "cache"))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached only"}])
    asyncio.run(_run_save_exports_file(tmp_path))


async def _run_pause_holds_replacement(release, started):
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir):
        started.set()
        assert release.wait(timeout=5)
        _write_stdout(output_dir, "7:training:0", "held line\n")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "cached line" in app.screen._shown_lines and started.is_set())
        assert ok, "cache was not on screen before the file arrived"
        app.screen.action_toggle_pause()
        release.set()
        ok = await _wait(pilot, app, lambda: app.screen._held is not None)
        assert ok, "paused view did not hold the saved console"
        assert "held line" not in app.screen._shown_lines
        app.screen.action_toggle_pause()
        ok = await _wait(pilot, app, lambda: "held line" in app.screen._shown_lines)
        assert ok, "unpause did not show the saved console"
        app.screen.action_toggle_pause()
        app.screen._held = (["stale line"], False)
        app.screen._start_tail("7:training:0")
        assert app.screen._held is None
        assert "stale line" not in app.screen._shown_lines
        assert "held line" in app.screen._shown_lines
        await _settle(app, pilot)


def test_pause_holds_saved_console_until_unpause(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    release = threading.Event()
    try:
        asyncio.run(_run_pause_holds_replacement(release, threading.Event()))
    finally:
        release.set()


async def _run_leave_during_download(release, started, dirs):
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir):
        dirs.append(output_dir)
        started.set()
        assert release.wait(timeout=5)
        _write_stdout(output_dir, "7:training:0", "too late\n")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, started.is_set)
        assert ok, "download did not start"
        app.pop_screen()
        await pilot.pause()
        release.set()
        ok = await _wait(pilot, app, lambda: dirs and not Path(dirs[0]).exists())
        assert ok, "temp dir was not removed after leaving"
        assert not any("too late" in line for line in getattr(app.screen, "_shown_lines", []))
        await _settle(app, pilot)


def test_leave_during_download_removes_temp_dir():
    release = threading.Event()
    try:
        asyncio.run(_run_leave_during_download(release, threading.Event(), []))
    finally:
        release.set()
