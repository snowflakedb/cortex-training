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
import time
from types import SimpleNamespace
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
        # The live tail is not called. Cache stays on screen when the stage
        # download does not return a console file.
        shown = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and "7:training:0" in app.screen._stage_attempted
            and any(ln == "cached line" for ln in app.screen._shown_lines),
        )
        await pilot.pause(0.05)
        assert "cached line" in app.screen._shown_lines
        assert shown, "cached line was not kept for a terminal job"
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


def _stage_download(calls):
    from pathlib import Path

    def download(job_id, output_dir, *, resume=False):
        assert resume is True
        calls.append(output_dir)
        path = Path(output_dir) / f"{job_id}:training:0" / "stdout.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("persisted line\n", encoding="utf-8")
        return [
            {
                "sub_job_id": f"{job_id}:training:0",
                "filename": "stdout.log",
                "saved_path": str(path),
                "chunk_count": 1,
            }
        ]

    return download


async def _run_cancelled_job_shows_stage():
    calls = []
    c = _client()
    c.get_job.return_value = {
        "status": "CANCELLED",
        "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
    }
    c.download_stdout_logs.side_effect = _stage_download(calls)
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and any(ln == "persisted line" for ln in app.screen._shown_lines)
            and not any("sealed" in ln.lower() for ln in app.screen._shown_lines),
        )
        assert ok, "cancelled job did not show the persisted console"
        assert "sealed" not in (app.sub_title or "").lower()
        c.tail_logs.assert_not_called()
        assert len(calls) == 1
        await _settle(app, pilot)


def test_cancelled_job_shows_stage_log(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_cancelled_job_shows_stage())


async def _run_tail_failure_on_cancel_switches_to_stage():
    calls = []
    c = _client()
    state = {"calls": 0}

    def get_job(_job_id):
        state["calls"] += 1
        status = "RUNNING" if state["calls"] == 1 else "CANCELLED"
        return {
            "status": status,
            "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
        }

    c.get_job.side_effect = get_job
    c.tail_logs.side_effect = RuntimeError("zone gone")
    c.download_stdout_logs.side_effect = _stage_download(calls)
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and any(ln == "persisted line" for ln in app.screen._shown_lines)
            and not any("sealed" in ln.lower() for ln in app.screen._shown_lines),
        )
        assert ok, "tail failure on a cancelled job did not show the persisted console"
        assert not any(ln.startswith("[error] tail") for ln in app.screen._shown_lines)
        assert "sealed" not in (app.sub_title or "").lower()
        assert len(calls) == 1
        await _settle(app, pilot)


def test_tail_failure_on_cancel_switches_to_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_tail_failure_on_cancel_switches_to_stage())


async def _run_tail_failure_while_running_stays_error():
    c = _client()
    c.tail_logs.side_effect = RuntimeError("stream reset")
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and any(ln.startswith("[error] tail") for ln in app.screen._shown_lines),
        )
        assert ok, "running-job tail failure was not shown"
        c.download_stdout_logs.assert_not_called()
        await _settle(app, pilot)


def test_tail_failure_while_running_stays_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    asyncio.run(_run_tail_failure_while_running_stays_error())


async def _run_keeps_visible_lines_when_stage_is_empty():
    c = _client()
    c.get_job.return_value = {
        "status": "CANCELLED",
        "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
    }
    c.download_stdout_logs.return_value = []
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and "7:training:0" in app.screen._stage_attempted
            and any(ln == "cached line" for ln in app.screen._shown_lines),
        )
        await pilot.pause(0.05)
        assert "cached line" in app.screen._shown_lines
        assert ok, "cached line disappeared while the persisted console was empty"
        assert not any("sealed" in ln.lower() for ln in app.screen._shown_lines)
        assert not any(ln.startswith("[error]") for ln in app.screen._shown_lines)
        await _settle(app, pilot)


def test_keeps_visible_lines_when_stage_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    asyncio.run(_run_keeps_visible_lines_when_stage_is_empty())


async def _run_keeps_visible_lines_when_stage_download_fails():
    c = _client()
    state = {"calls": 0}

    def get_job(_job_id):
        state["calls"] += 1
        status = "RUNNING" if state["calls"] == 1 else "CANCELLED"
        return {
            "status": status,
            "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
        }

    c.get_job.side_effect = get_job
    c.tail_logs.side_effect = RuntimeError("zone gone")
    c.download_stdout_logs.side_effect = RuntimeError("stage down")
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and any(ln == "cached line" for ln in app.screen._shown_lines),
        )
        assert ok, "cached line disappeared when the persisted console failed"
        assert not any("sealed" in ln.lower() for ln in app.screen._shown_lines)
        assert not any(ln.startswith("[error] tail") for ln in app.screen._shown_lines)
        await _settle(app, pilot)


async def _run_open_panel_picks_up_later_console(monkeypatch):
    monkeypatch.setattr(LogScreen, "_STAGE_REFRESH_SECONDS", 0.05)
    calls = {"n": 0}

    def download(job_id, output_dir, *, resume=False):
        from pathlib import Path

        assert resume is True
        calls["n"] += 1
        path = Path(output_dir) / f"{job_id}:training:0" / "stdout.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "line one\n" if calls["n"] < 3 else "line one\nline two\n"
        path.write_text(text, encoding="utf-8")
        return [
            {
                "sub_job_id": f"{job_id}:training:0",
                "filename": "stdout.log",
                "saved_path": str(path),
                "chunk_count": 1,
            }
        ]

    c = _client()
    c.get_job.return_value = {
        "status": "CANCELLED",
        "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
    }
    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and any(ln == "line two" for ln in app.screen._shown_lines),
            tries=200,
        )
        assert ok, "open panel did not pick up later console output"
        assert any(ln == "line one" for ln in app.screen._shown_lines)
        await _settle(app, pilot)


def test_open_panel_picks_up_later_console(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_open_panel_picks_up_later_console(monkeypatch))


async def _run_healthy_empty_tail_hands_off(tmp_path, monkeypatch):
    monkeypatch.setattr(LogScreen, "_STAGE_REFRESH_SECONDS", 0.05)
    calls = {"jobs": 0}

    def get_job(_job_id):
        calls["jobs"] += 1
        return {
            "status": "RUNNING" if calls["jobs"] == 1 else "COMPLETED",
            "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
        }

    def download(job_id, output_dir, *, resume=False):
        from pathlib import Path

        assert resume is True
        path = Path(output_dir) / f"{job_id}:training:0" / "stdout.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("saved after quiet tail\n", encoding="utf-8")
        return [{"sub_job_id": f"{job_id}:training:0", "saved_path": str(path)}]

    c = _client()
    c.get_job.side_effect = get_job
    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and "saved after quiet tail" in app.screen._shown_lines,
            tries=200,
        )
        assert ok, "quiet live tail did not hand off after the job completed"
        assert c.tail_logs.call_count > 0
        assert calls["jobs"] == 2
        assert not any(line.startswith("[error] tail") for line in app.screen._shown_lines)
        await _settle(app, pilot)


def test_healthy_empty_tail_hands_off(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_healthy_empty_tail_hands_off(tmp_path, monkeypatch))


async def _run_quiet_tail_stops_when_status_post_fails(monkeypatch):
    monkeypatch.setattr(LogScreen, "_STAGE_REFRESH_SECONDS", 0.2)
    calls = {"jobs": 0}

    def get_job(_job_id):
        calls["jobs"] += 1
        return {
            "status": "RUNNING" if calls["jobs"] == 1 else "COMPLETED",
            "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
        }

    c = _client()
    c.get_job.side_effect = get_job
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ready = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen) and c.tail_logs.call_count > 0,
        )
        assert ready, "live tail did not start"
        screen = app.screen
        original_post = screen._post

        def post(fn, *args, **kwargs):
            if fn == screen._update_subtitle:
                return False
            return original_post(fn, *args, **kwargs)

        monkeypatch.setattr(screen, "_post", post)
        checked = await _wait(pilot, app, lambda: calls["jobs"] >= 2, tries=200)
        assert checked, "job status was not refreshed"
        await pilot.pause(0.05)
        c.download_stdout_logs.assert_not_called()
        await _settle(app, pilot)


def test_quiet_tail_stops_when_status_post_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_quiet_tail_stops_when_status_post_fails(monkeypatch))


def test_keeps_visible_lines_when_stage_download_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    asyncio.run(_run_keeps_visible_lines_when_stage_download_fails())


async def _run_stale_worker_does_not_paint():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen) and app.screen._logview is not None,
        )
        assert ok, "log screen did not open"
        screen = app.screen
        screen._current_source = "7:training:0"
        screen._tail_gen = 2
        screen._filter_gen = 2
        screen._shown_lines = ["kept"]
        screen._replace_lines(["stale console"], "7:training:0", 1)
        screen._replace_lines(
            ["stale filter snapshot"],
            "7:training:0",
            2,
            filter_gen=1,
        )
        screen._replace_lines_if_changed(
            ["stale refilter"],
            "7:training:0",
            1,
            filter_gen=2,
        )
        screen._note_if_pane_empty("7:training:0", 1)
        screen._append_if_current(["other cache"], "7:other", 2)
        screen._append_live_if_current(["stale live"], "7:training:0", 1, True)
        screen._write_error_if_current("[error] stale", "7:training:0", 1)
        assert screen._shown_lines == ["kept"]
        assert screen._pending_stage is None
        screen._apply_stage_lines(["same console"])
        screen._write_line("[saved] status")
        screen._replace_lines_if_changed(
            ["same console"], "7:training:0", 2, filter_gen=2
        )
        assert screen._shown_lines == ["same console", "[saved] status"]
        screen._replace_lines_if_changed(
            ["changed console"], "7:training:0", 2, filter_gen=2
        )
        assert screen._shown_lines == ["changed console"]
        await _settle(app, pilot)


def test_stale_worker_does_not_paint(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_stale_worker_does_not_paint())


async def _run_pause_holds_console_until_resume():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen) and app.screen._logview is not None,
        )
        assert ok, "log screen did not open"
        screen = app.screen
        screen._current_source = "7:training:0"
        screen._tail_gen = 1
        screen._paused = True
        screen._shown_lines = []
        screen._logview.clear()
        screen._write_lines(["cached line"])
        screen._replace_lines(["persisted line"], "7:training:0", 1)
        assert screen._shown_lines == ["cached line"]
        assert screen._pending_stage == ["persisted line"]
        screen.action_toggle_pause()
        assert screen._shown_lines == ["persisted line"]
        assert screen._pending_stage is None
        screen._paused = True
        screen._filter_gen = 1
        screen._shown_lines = ["persisted line"]
        screen._replace_lines_if_changed([], "7:training:0", 1, filter_gen=1)
        assert screen._shown_lines == ["persisted line"]
        screen.action_toggle_pause()
        assert screen._shown_lines == []
        await _settle(app, pilot)


def test_pause_holds_console_until_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_pause_holds_console_until_resume())


async def _run_refilter_keeps_saved_console(tmp_path):
    c = _client()
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen) and app.screen._logview is not None,
        )
        assert ok, "log screen did not open"
        screen = app.screen
        screen.workers.cancel_all()
        screen._current_source = "7:training:0"
        screen._job_status = "CANCELLED"
        path = tmp_path / "stdout.log"
        path.write_text("persisted line\nother line\n", encoding="utf-8")
        screen._stage_paths["7:training:0"] = str(path)
        screen._apply_stage_lines(["persisted line", "other line"])
        c.download_stdout_logs.reset_mock()
        tail_gen = screen._tail_gen

        screen._filter = "persisted"
        screen._refilter()
        filtered = await _wait(
            pilot, app, lambda: screen._shown_lines == ["persisted line"]
        )
        assert filtered, "saved console was not filtered from the local file"
        assert screen._tail_gen == tail_gen

        screen._filter = "no match"
        screen._refilter()
        empty_stage = await _wait(pilot, app, lambda: screen._shown_lines == [])
        assert empty_stage, "a no-match saved-console filter did not clear the pane"
        screen._note_if_pane_empty("7:training:0", screen._tail_gen)
        assert screen._shown_lines == []
        c.download_stdout_logs.assert_not_called()

        screen.workers.cancel_all()
        screen._stage_paths.clear()
        screen._stage_fetching["7:training:0"] = 1
        screen._filter = "typed during first download"
        screen._refilter()
        await pilot.pause(0.05)
        c.download_stdout_logs.assert_not_called()

        screen._stage_fetching.clear()
        screen._stage_attempted.add("7:training:0")
        screen._cache.append_entries(
            "7:training:0",
            [{"_raw": "cached keep"}, {"_raw": "cached drop"}],
        )
        screen._filter = "keep"
        screen._refilter()
        filtered_cache = await _wait(
            pilot, app, lambda: screen._shown_lines == ["cached keep"]
        )
        assert filtered_cache, "terminal cache was not filtered after a missing stage file"

        screen._filter = "no cached match"
        screen._refilter()
        empty_cache = await _wait(pilot, app, lambda: screen._shown_lines == [])
        assert empty_cache, "a no-match terminal-cache filter did not clear the pane"
        screen._note_if_pane_empty("7:training:0", screen._tail_gen)
        assert screen._shown_lines == []

        screen._stage_paths["7:training:0"] = str(tmp_path / "missing.log")
        screen._shown_lines = ["No log output is available."]
        screen._filter = "anything"
        screen._refilter()
        await pilot.pause(0.05)
        assert screen._shown_lines == ["No log output is available."]

        empty_path = tmp_path / "empty.log"
        empty_path.write_text("", encoding="utf-8")
        screen._stage_paths["7:training:0"] = str(empty_path)
        screen._shown_lines = ["cached line"]
        screen._filter = "anything"
        screen._refilter()
        await pilot.pause(0.05)
        assert screen._shown_lines == ["cached line"]
        await _settle(app, pilot)


def test_refilter_keeps_saved_console(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_refilter_keeps_saved_console(tmp_path))


async def _run_filter_during_first_missing_stage():
    started = threading.Event()
    release = threading.Event()
    calls = []
    c = _client()
    c.get_job.return_value = {
        "status": "CANCELLED",
        "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
    }

    def download(*_args, **_kwargs):
        calls.append(1)
        started.set()
        assert release.wait(2)
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ready = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and started.is_set()
            and "cached keep" in app.screen._shown_lines,
        )
        assert ready, "first saved-console download did not start"
        screen = app.screen
        screen._filter = "keep"
        screen._update_subtitle()
        screen._refilter()
        release.set()
        filtered = await _wait(
            pilot, app, lambda: screen._shown_lines == ["cached keep"]
        )
        assert filtered, "latest filter was not applied after the missing stage result"
        assert calls == [1]
        await _settle(app, pilot)


def test_filter_during_first_missing_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries(
        "7:training:0",
        [{"_raw": "cached keep"}, {"_raw": "cached drop"}],
    )
    asyncio.run(_run_filter_during_first_missing_stage())


async def _run_filter_during_first_saved_stage(tmp_path):
    started = threading.Event()
    release = threading.Event()
    calls = []
    c = _client()
    c.get_job.return_value = {
        "status": "CANCELLED",
        "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
    }

    def download(job_id, output_dir, *, resume=False):
        from pathlib import Path

        calls.append(1)
        started.set()
        assert release.wait(2)
        path = Path(output_dir) / f"{job_id}:training:0" / "stdout.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("keep this\ndrop this\n", encoding="utf-8")
        return [{"sub_job_id": f"{job_id}:training:0", "saved_path": str(path)}]

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ready = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen) and started.is_set(),
        )
        assert ready, "first saved-console download did not start"
        screen = app.screen
        screen._filter = "keep"
        screen._refilter()
        release.set()
        filtered = await _wait(
            pilot, app, lambda: screen._shown_lines == ["keep this"]
        )
        assert filtered, "latest filter was not applied to the downloaded console"
        assert calls == [1]
        await _settle(app, pilot)


def test_filter_during_first_saved_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    asyncio.run(_run_filter_during_first_saved_stage(tmp_path))


async def _run_exit_does_not_wait_for_stage_download(started, release):
    c = _client()
    c.get_job.return_value = {
        "status": "CANCELLED",
        "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
    }

    def download(*_args, **_kwargs):
        started.set()
        release.wait(2)
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ready = await _wait(pilot, app, started.is_set)
        assert ready, "saved-console download did not start"


def test_exit_does_not_wait_for_stage_download(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    started = threading.Event()
    release = threading.Event()
    before = time.monotonic()
    asyncio.run(_run_exit_does_not_wait_for_stage_download(started, release))
    elapsed = time.monotonic() - before
    release.set()
    assert elapsed < 1.5


async def _run_save_exports_console_on_screen():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen) and app.screen._logview is not None,
        )
        assert ok, "log screen did not open"
        screen = app.screen
        screen._current_source = "7:training:0"
        screen._append_if_current(["filtered cache"], "7:training:0", screen._tail_gen)
        cache_text, cache_count = screen._export_text(screen._snapshot_export())
        assert (cache_text, cache_count) == ("filtered cache", 1)
        screen._apply_stage_lines(["persisted line"])
        screen._write_line("[saved] status line")
        stage_text, stage_count = screen._export_text(screen._snapshot_export())
        assert (stage_text, stage_count) == ("persisted line", 1)
        screen.action_save_log()
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
        assert content == "persisted line\n"
        await _settle(app, pilot)


def test_save_exports_console_on_screen(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "alpha"}])
    asyncio.run(_run_save_exports_console_on_screen())


async def _run_copy_exports_console_on_screen():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    copied = []
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen) and app.screen._logview is not None,
        )
        assert ok, "log screen did not open"
        screen = app.screen
        screen._current_source = "7:training:0"
        screen._cache.append_entries("7:training:0", [{"_raw": "live line"}])
        screen._apply_stage_lines(["stale snapshot"])
        screen._append_live_if_current(["live line"], "7:training:0", screen._tail_gen, True)
        live_text, _ = screen._export_text(screen._snapshot_export())
        assert "live line" in live_text
        assert "stale snapshot" not in live_text
        screen._apply_stage_lines(["persisted line"])
        app.copy_to_clipboard = copied.append
        screen.action_copy_log()
        ok = await _wait(pilot, app, lambda: copied == ["persisted line"])
        assert ok, "copy did not use the console on screen"
        await _settle(app, pilot)


def test_copy_exports_console_on_screen(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_copy_exports_console_on_screen())


def test_stage_watcher_stops_when_ui_post_fails(monkeypatch):
    screen = LogScreen(_client(), "7", job_status="CANCELLED")
    screen._current_source = "7:training:0"
    screen._tail_gen = 1
    worker = SimpleNamespace(is_cancelled=False)
    calls = []
    monkeypatch.setattr(screen, "_adopt_stage", lambda *_args: True)
    monkeypatch.setattr(screen, "_sleep_refresh", lambda _worker: None)
    monkeypatch.setattr(
        screen,
        "_fetch_stage_lines",
        lambda *_args, **_kwargs: calls.append(1)
        or ((True, ["line"]), screen._filter_gen),
    )
    monkeypatch.setattr(screen, "_post", lambda *_args, **_kwargs: False)
    screen._watch_stage("7:training:0", worker, 1)
    assert calls == [1]


def test_terminal_worker_failure_is_reported_without_retry(monkeypatch):
    from cortex_training.tui import app as app_module

    screen = LogScreen(_client(), "7", job_status="CANCELLED")
    screen._current_source = "7:training:0"
    screen._tail_gen = 1
    worker = SimpleNamespace(is_cancelled=False)
    reported = []
    monkeypatch.setattr(app_module, "get_current_worker", lambda: worker)
    monkeypatch.setattr(
        screen, "_paint_cache", lambda *_args: (_ for _ in ()).throw(ValueError("bad"))
    )
    monkeypatch.setattr(
        screen,
        "_report_tail_failure",
        lambda source, current_worker, gen, exc: reported.append(
            (source, current_worker, gen, str(exc))
        ),
    )
    monkeypatch.setattr(
        screen,
        "_watch_stage",
        lambda *_args: pytest.fail("terminal watcher was retried"),
    )

    LogScreen._tail.__wrapped__(screen, "7:training:0", 1)
    assert reported == [("7:training:0", worker, 1, "bad")]
