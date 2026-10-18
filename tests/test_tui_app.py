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
import json
import os
import re
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from textual.widgets import Input  # noqa: E402
from textual.widgets import DataTable  # noqa: E402
from textual.widgets import Label  # noqa: E402
from textual.widgets import Static  # noqa: E402
from textual.widgets import TabbedContent  # noqa: E402

from cortex_training.client import DownloadCancelled  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402
from cortex_training.tui.app import LogScreen  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    # The saved console and caches live under the per-job cache dir; keep every
    # test (including those that do not set it themselves) out of ~/.cache.
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "isolated-cache"))
    import cortex_training.tui.app as app_module

    # The stall-check throttle is per process; start each test without one.
    monkeypatch.setattr(app_module, "_STALL_CHECKED_AT", {})


def _whole_job(download):
    """Side effect for the whole-job download only; the TUI first fetches the
    selected sub-job alone (``sub_job_ids``), which these tests do not model."""

    def side_effect(job_id, output_dir, **kwargs):
        if kwargs.get("sub_job_ids") is not None:
            return []
        return download(job_id, output_dir, **kwargs)

    return side_effect


def _raise(exc):
    def download(*_args, **_kwargs):
        raise exc

    return download


def _whole_job_calls(client):
    return sum(1 for call in client.download_stdout_logs.call_args_list if call.kwargs.get("sub_job_ids") is None)


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


async def _run_resource_tab(tmp_path):
    c = _client()

    def download(job_id, output_dir, *, resume=False, **_kwargs):
        assert job_id == "7"
        assert resume is True
        _write_gpu_metrics(output_dir, "7:training:0", gpu_count=2)
        _write_gpu_metrics(output_dir, "7:sampling:0", gpu_count=1)
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen) and app.screen._current_source == "7:training:0",
        )
        assert ok
        tabs = app.screen.query_one("#job-tabs", TabbedContent)
        assert tabs.active == "logs"
        assert {pane.id for pane in app.screen.query("TabPane")} == {"logs", "resource"}
        c.download_metrics.assert_not_called()

        tabs.active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: c.download_metrics.call_count == 1
            and app.screen.query_one("#resource-table", DataTable).row_count == 2
            and not app.screen._resource_loading,
        )
        assert ok, "Resource did not load two GPU rows"
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "2 GPU(s)" in status
        assert "7:training:0" in status
        calls = c.download_metrics.call_count

        await pilot.press("5")
        ok = await _wait(
            pilot,
            app,
            lambda: app.screen._resource_range == "all"
            and "7:training:0  ·  all  ·" in str(app.screen.query_one("#resource-status", Static).render()),
        )
        assert ok
        await pilot.press("1")
        ok = await _wait(pilot, app, lambda: app.screen._resource_range == "5m")
        assert ok
        assert c.download_metrics.call_count == calls
        app.screen._content_width = lambda: 10
        app.screen._write_line("x" * 20)
        assert len(app.screen._logview._lines) > len(app.screen._shown_lines)
        app.screen.action_refresh_sources()
        ok = await _wait(
            pilot,
            app,
            lambda: c.download_metrics.call_count == calls + 1 and not app.screen._resource_loading,
        )
        assert ok, "r did not refresh metrics on Resource"
        tabs.active = "logs"
        app.screen._content_width = lambda: 200
        ok = await _wait(
            pilot,
            app,
            lambda: app.screen._active_tab == "logs" and app.screen._wrap_width == 200,
        )
        assert ok
        assert len(app.screen._logview._lines) == len(app.screen._shown_lines)
        downloads = c.download_metrics.call_count
        app.screen.action_refresh_sources()
        await pilot.pause()
        assert c.download_metrics.call_count == downloads
        app.screen._start_tail("7:sampling:0")
        tabs.active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: app.screen.query_one("#resource-table", DataTable).row_count == 1
            and "7:sampling:0" in str(app.screen.query_one("#resource-status", Static).render()),
        )
        assert ok, "Resource did not follow the Logs-selected sub-job"
        assert c.download_metrics.call_count == downloads
        tabs.active = "logs"
        await _wait(pilot, app, lambda: app.screen._active_tab == "logs")
        previous_range = app.screen._resource_range
        app.screen.action_resource_range("all")
        assert app.screen._resource_range == previous_range
        await pilot.press("t")
        ok = await _wait(pilot, app, lambda: tabs.active == "resource")
        assert ok
        await pilot.press("t")
        ok = await _wait(pilot, app, lambda: tabs.active == "logs")
        assert ok
        await _settle(app, pilot)


def test_resource_tab_is_lazy_and_renders_multiple_gpus(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_tab(tmp_path))


async def _run_resource_cold_cache_failure():
    c = _client()
    c.download_metrics.side_effect = RuntimeError("metrics [/]")
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: c.download_metrics.called and not app.screen._resource_loading,
        )
        assert ok
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "GPU metrics unavailable: RuntimeError: metrics [/]" in status
        assert app.screen.query_one("#resource-table", DataTable).row_count == 0
        await _settle(app, pilot)


def test_resource_cold_cache_download_failure_is_visible(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_cold_cache_failure())


async def _run_resource_parser_failure_resets_loading():
    c = _client()
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: not app.screen._resource_loading
            and "OverflowError: bad metric"
            in str(app.screen.query_one("#resource-status", Static).render()),
        )
        assert ok
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "GPU metrics unavailable: OverflowError: bad metric" in status
        await _settle(app, pilot)


def test_resource_parser_failure_does_not_leave_loading_stuck(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    import cortex_training.tui.app as app_module

    monkeypatch.setattr(
        app_module,
        "load_resource_data",
        MagicMock(side_effect=OverflowError("bad metric")),
    )
    asyncio.run(_run_resource_parser_failure_resets_loading())


async def _run_resource_partial_download_failure():
    c = _client()

    def download(job_id, output_dir, *, resume, **_kwargs):
        _write_gpu_metrics(output_dir, "7:training:0", gpu_count=2)
        raise RuntimeError("later sub-job failed")

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: not app.screen._resource_loading
            and app.screen.query_one("#resource-table", DataTable).row_count == 2,
        )
        assert ok
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "cached · refresh failed: RuntimeError: later sub-job failed" in status
        await _settle(app, pilot)


def test_resource_partial_download_is_reparsed_after_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_partial_download_failure())


async def _run_resource_partial_other_sub_job_retries():
    c = _client()
    calls = 0

    def download(job_id, output_dir, *, resume, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            _write_gpu_metrics(output_dir, "7:sampling:0", gpu_count=1)
            raise RuntimeError("training metrics failed")
        _write_gpu_metrics(output_dir, "7:training:0", gpu_count=2)
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source == "7:training:0")
        assert ok
        tabs = app.screen.query_one("#job-tabs", TabbedContent)
        tabs.active = "resource"
        ok = await _wait(pilot, app, lambda: calls == 1 and not app.screen._resource_loading)
        assert ok
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "GPU metrics unavailable: RuntimeError: training metrics failed" in status
        assert "no GPU metrics" not in status
        tabs.active = "logs"
        await _wait(pilot, app, lambda: app.screen._logs_active())
        tabs.active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: calls == 2
            and not app.screen._resource_loading
            and app.screen.query_one("#resource-table", DataTable).row_count == 2,
        )
        assert ok, "re-entering Resource did not retry after a partial failure for another sub-job"
        await _settle(app, pilot)


def test_resource_partial_other_sub_job_failure_retries_on_reentry(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_partial_other_sub_job_retries())


async def _run_resource_cached_failure():
    c = _client()
    c.download_metrics.side_effect = RuntimeError("refresh unavailable")
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: not app.screen._resource_loading
            and app.screen.query_one("#resource-table", DataTable).row_count == 2,
        )
        assert ok
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "cached · refresh failed: RuntimeError: refresh unavailable" in status
        app.screen.action_resource_range("all")
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "cached · refresh failed: RuntimeError: refresh unavailable" in status
        assert app.screen.query_one("#resource-table", DataTable).row_count == 2
        await _settle(app, pilot)


def test_resource_failed_refresh_keeps_cached_rows_and_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import job_cache_dir

    _write_gpu_metrics(job_cache_dir("7") / "metrics", "7:training:0", gpu_count=2)
    asyncio.run(_run_resource_cached_failure())


async def _run_resource_unsupported_warning():
    c = _client()

    def download(job_id, output_dir, *, resume, **_kwargs):
        path = _write_gpu_metrics(output_dir, "7:training:0", gpu_count=1)
        row = json.loads(path.read_text(encoding="utf-8"))
        row["record_version"] = 1
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        path.with_name(".gpu.jsonl.manifest.json").write_text(
            json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size}),
            encoding="utf-8",
        )
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(pilot, app, lambda: not app.screen._resource_loading)
        assert ok
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "7:training:0" in status
        assert "no usable GPU metrics" in status
        assert "1 unsupported row(s)" in status
        await _settle(app, pilot)


def test_resource_all_unsupported_rows_show_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_unsupported_warning())


async def _run_resource_narrow_no_samples():
    c = _client()

    def download(job_id, output_dir, *, resume, **_kwargs):
        path = _write_gpu_metrics(output_dir, "7:training:0", gpu_count=1)
        row = json.loads(path.read_text(encoding="utf-8"))
        row["gpu_util_pct"] = None
        payload = json.dumps(row) + "\n"
        path.write_text(payload, encoding="utf-8")
        path.with_name(".gpu.jsonl.manifest.json").write_text(
            json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size}),
            encoding="utf-8",
        )
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(60, 30)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: not app.screen._resource_loading
            and app.screen.query_one("#resource-table", DataTable).row_count == 1,
        )
        assert ok
        table = app.screen.query_one("#resource-table", DataTable)
        assert app.screen._resource_table_has_fb is False
        assert len(table.columns) == 3
        assert str(app.screen.query_one("#resource-chart", Static).render()) == "no GPU samples in this range"
        calls = c.download_metrics.call_count
        app.screen.on_resize(None)
        await pilot.pause()
        assert c.download_metrics.call_count == calls
        await _settle(app, pilot)


def test_resource_narrow_table_no_samples_and_resize_is_local(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_narrow_no_samples())


async def _run_resource_chart_fits_content_width():
    c = _client()

    def download(job_id, output_dir, *, resume, **_kwargs):
        _write_gpu_metrics(output_dir, "7:training:0", gpu_count=1)
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(60, 30)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: not app.screen._resource_loading
            and app.screen.query_one("#resource-table", DataTable).row_count == 1,
        )
        assert ok
        chart = app.screen.query_one("#resource-chart", Static)
        lines = str(chart.render()).splitlines()
        assert lines
        assert chart.region.width - chart.content_region.width == 2
        assert chart.content_size.width == chart.content_region.width
        assert all(len(line) <= chart.content_size.width for line in lines)
        await _settle(app, pilot)


def test_resource_chart_accounts_for_widget_padding(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_chart_fits_content_width())


async def _run_resource_ignores_refresh_while_busy(started, release):
    c = _client()

    def download(job_id, output_dir, *, resume, **_kwargs):
        started.set()
        assert release.wait(timeout=5)
        _write_gpu_metrics(output_dir, "7:training:0")
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(pilot, app, started.is_set)
        assert ok
        app.screen.action_refresh_sources()
        app.screen.action_refresh_sources()
        await pilot.pause()
        assert c.download_metrics.call_count == 1
        release.set()
        ok = await _wait(pilot, app, lambda: not app.screen._resource_loading)
        assert ok
        await _settle(app, pilot)


def test_resource_ignores_refresh_while_busy(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    started = threading.Event()
    release = threading.Event()
    try:
        asyncio.run(_run_resource_ignores_refresh_while_busy(started, release))
    finally:
        release.set()


async def _run_resource_reopen_waits_for_lock(started, release):
    c = _client()
    calls = 0

    def download(job_id, output_dir, *, resume, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            assert release.wait(timeout=5)
        _write_gpu_metrics(output_dir, "7:training:0")
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(pilot, app, started.is_set)
        assert ok
        app.pop_screen()
        await pilot.pause()
        app.push_screen(LogScreen(c, "7", poll_interval=0.01))
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        for _ in range(5):
            await pilot.pause()
        assert calls == 1
        release.set()
        ok = await _wait(
            pilot,
            app,
            lambda: calls == 2
            and not app.screen._resource_loading
            and app.screen.query_one("#resource-table", DataTable).row_count == 2,
        )
        assert ok, "reopened Resource did not continue after the prior download released the lock"
        await _settle(app, pilot)


def test_resource_reopen_waits_for_inflight_download(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    started = threading.Event()
    release = threading.Event()
    try:
        asyncio.run(_run_resource_reopen_waits_for_lock(started, release))
    finally:
        release.set()


async def _run_terminal_resource_default():
    c = _client()
    c.get_job.return_value = _terminal_job()
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._sources_ready)
        assert ok
        assert app.screen._resource_range == "all"
        await _settle(app, pilot)


def test_direct_terminal_job_resource_defaults_to_all():
    asyncio.run(_run_terminal_resource_default())


async def _run_active_to_terminal_resource_default():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._sources_ready)
        assert ok
        assert app.screen._resource_range == "15m"
        app.screen._apply_terminal_status("COMPLETED", None)
        assert app.screen._resource_range == "all"
        app.screen._resource_range = "5m"
        app.screen._resource_range_selected = True
        app.screen._apply_terminal_status("FAILED", None)
        assert app.screen._resource_range == "5m"
        await _settle(app, pilot)


def test_active_job_becoming_terminal_updates_only_default_resource_range():
    asyncio.run(_run_active_to_terminal_resource_default())


async def _run_resource_escape_returns_to_parent(*, direct):
    c = _client()
    app = CortexTrainingLogTUI(c, "7" if direct else None, poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        if direct:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen))
        else:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen))
            assert ok
            app.push_screen(LogScreen(c, "7", poll_interval=0.01))
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen))
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(pilot, app, lambda: app.screen._resource_active())
        assert ok
        await pilot.press("escape")
        ok = await _wait(
            pilot,
            app,
            lambda: not isinstance(app.screen, LogScreen),
        )
        assert ok
        if not direct:
            assert isinstance(app.screen, JobListScreen)
        await _settle(app, pilot)


def test_resource_escape_returns_from_direct_job():
    asyncio.run(_run_resource_escape_returns_to_parent(direct=True))


def test_resource_escape_returns_to_picker():
    asyncio.run(_run_resource_escape_returns_to_parent(direct=False))


async def _run_resource_guards(tmp_path):
    c = _client()
    c.download_metrics.side_effect = lambda job_id, output_dir, *, resume: (
        _write_gpu_metrics(output_dir, "7:training:0"),
        [],
    )[1]
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(pilot, app, lambda: not app.screen._resource_loading)
        assert ok
        screen = app.screen
        table = screen.query_one("#resource-table", DataTable)
        screen.set_focus(table)
        width = screen._sources_width
        level = screen._min_level
        paused = screen._paused
        copied = []
        app.copy_to_clipboard = copied.append
        screen.action_filter()
        screen.action_cycle_level()
        screen.action_toggle_pause()
        screen.action_shrink_sources()
        screen.action_grow_sources()
        screen.action_save_log()
        screen.action_copy_log()
        screen.action_copy_text()
        for _ in range(5):
            await pilot.pause()
        assert screen.focused is table
        assert screen._sources_width == width
        assert screen._min_level == level
        assert screen._paused == paused
        assert copied == []
        assert not list(tmp_path.glob("cortex-training-*.log"))
        await _settle(app, pilot)


def test_log_only_actions_are_guarded_on_resource(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path))
    asyncio.run(_run_resource_guards(tmp_path))


async def _run_sub_job_scope(sub_job_id):
    c = _client()
    app = CortexTrainingLogTUI(c, "7", sub_job_id=sub_job_id, poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._sources_ready)
        assert ok
        if sub_job_id == "7:sampling:0":
            assert list(app.screen._source_by_item.values()) == ["7:sampling:0"]
            assert app.screen._current_source == "7:sampling:0"
        else:
            assert app.screen._source_by_item == {}
            assert app.screen._current_source is None
            app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
            for _ in range(5):
                await pilot.pause()
            c.download_metrics.assert_not_called()
            labels = [str(item.query_one(Label).render()) for item in app.screen.query("#sources ListItem")]
            assert any("sub-job not found" in label for label in labels)
        await _settle(app, pilot)


def test_sub_job_id_scopes_source_list():
    asyncio.run(_run_sub_job_scope("7:sampling:0"))


def test_unmatched_sub_job_does_not_download_metrics():
    asyncio.run(_run_sub_job_scope("7:missing:0"))


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


async def _run_empty_source_refresh_clears_selection():
    c = _client()
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        c.get_job.return_value = {}
        app.screen.action_refresh_sources()
        ok = await _wait(
            pilot,
            app,
            lambda: app.screen._current_source is None
            and len(app.screen.query("#sources ListItem")) == 0,
        )
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        await pilot.pause()
        c.download_metrics.assert_not_called()
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "select a log source first" in status
        await _settle(app, pilot)


def test_empty_source_refresh_clears_resource_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_empty_source_refresh_clears_selection())


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


def test_offline_shows_metrics_only_cached_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import job_cache_dir

    _write_gpu_metrics(job_cache_dir("7") / "metrics", "7:training:0", gpu_count=2)
    asyncio.run(_run_offline_fallback())


def _commit_resource_row(root, sub_job_id, util, filename):
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).replace(microsecond=0)
    path = Path(root) / sub_job_id / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "record_version": 2,
        "sub_job_id": sub_job_id,
        "sample_ts": now.isoformat().replace("+00:00", "Z"),
        "worker_num": 0,
        "local_gpu_index": 0,
        "global_gpu_index": 0,
        "gpu_util_pct": util,
        "fb_used_bytes": float(1024**3),
        "fb_free_bytes": float(8 * 1024**3),
        "sample_interval_s": 5.0,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    path.with_name("." + filename + ".manifest.json").write_text(
        json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size}),
        encoding="utf-8",
    )
    return path


async def _run_offline_legacy_metrics_source():
    c = _client()
    c.get_job.side_effect = RuntimeError("network down")
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and list(app.screen._source_by_item.values()) == ["7:training:0"],
        )
        assert ok, "committed legacy gpu.jsonl was not the offline source"
        await _settle(app, pilot)


def test_offline_legacy_gpu_jsonl_is_the_source_without_an_uncommitted_file(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import job_cache_dir

    root = job_cache_dir("7") / "metrics"
    _commit_resource_row(root, "7:training:0", 40, "gpu.jsonl")
    bare = root / "7:sampling:0" / "gpu.jsonl"
    bare.parent.mkdir(parents=True)
    bare.write_text("{}\n", encoding="utf-8")
    asyncio.run(_run_offline_legacy_metrics_source())


async def _run_legacy_first_paint_then_refresh_failure():
    c = _client()
    release = threading.Event()

    def download(*_args, **_kwargs):
        assert release.wait(5)
        raise RuntimeError("refresh unavailable")

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source == "7:training:0")
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: "cached · refreshing…" in str(app.screen.query_one("#resource-status", Static).render())
            and "avg 90%" in str(app.screen.query_one("#resource-status", Static).render())
            and app.screen.query_one("#resource-table", DataTable).row_count == 1,
        )
        assert ok, "legacy cache was not painted before the refresh finished"
        release.set()
        ok = await _wait(
            pilot,
            app,
            lambda: "cached · refresh failed: RuntimeError: refresh unavailable"
            in str(app.screen.query_one("#resource-status", Static).render())
            and app.screen.query_one("#resource-table", DataTable).row_count == 1,
        )
        assert ok
        await _settle(app, pilot)


def test_resource_first_paint_and_refresh_failure_keep_legacy_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import job_cache_dir

    root = job_cache_dir("7") / "metrics"
    legacy = _commit_resource_row(root, "7:training:0", 90, "gpu.jsonl")
    before = legacy.read_bytes()
    asyncio.run(_run_legacy_first_paint_then_refresh_failure())
    assert legacy.read_bytes() == before
    assert not (legacy.parent / "metrics.jsonl").exists()


async def _run_first_paint_prefers_metrics_jsonl():
    c = _client()
    release = threading.Event()

    def download(*_args, **_kwargs):
        assert release.wait(5)
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source == "7:training:0")
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: "avg 10%" in str(app.screen.query_one("#resource-status", Static).render())
            and "cached · refreshing…" in str(app.screen.query_one("#resource-status", Static).render()),
        )
        assert ok, "first paint did not prefer metrics.jsonl"
        release.set()
        ok = await _wait(pilot, app, lambda: not app.screen._resource_loading)
        assert ok
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "avg 10%" in status
        assert "avg 90%" not in status
        await _settle(app, pilot)


def test_resource_first_paint_prefers_committed_metrics_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import job_cache_dir

    root = job_cache_dir("7") / "metrics"
    legacy = _commit_resource_row(root, "7:training:0", 90, "gpu.jsonl")
    current = _commit_resource_row(root, "7:training:0", 10, "metrics.jsonl")
    legacy_bytes = legacy.read_bytes()
    current_bytes = current.read_bytes()
    asyncio.run(_run_first_paint_prefers_metrics_jsonl())
    assert legacy.read_bytes() == legacy_bytes
    assert current.read_bytes() == current_bytes


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


def _write_gpu_metrics(output_dir, sub_job_id, *, gpu_count=2):
    path = Path(output_dir) / sub_job_id / "gpu.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for gpu in range(gpu_count):
        rows.append(
            {
                "record_version": 2,
                "sub_job_id": sub_job_id,
                "sample_ts": f"2026-01-01T00:00:0{gpu}.123456789Z",
                "worker_num": gpu,
                "local_gpu_index": 0,
                "global_gpu_index": gpu,
                "gpu_util_pct": float(gpu * 80),
                "fb_used_bytes": float((gpu + 1) * 1024**3),
                "fb_free_bytes": float(8 * 1024**3),
                "sample_interval_s": 5.0,
            }
        )
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size}),
        encoding="utf-8",
    )
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

    def download(job_id, output_dir, **kwargs):
        _write_stdout(output_dir, "7:training:0", "saved line\n")
        return []

    c.download_stdout_logs.side_effect = _whole_job(download)
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
    c.download_stdout_logs.side_effect = _whole_job(_raise(RuntimeError("unavailable")))
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: "cached line" in app.screen._shown_lines and not app.screen._stdout_inflight,
        )
        assert ok, "cache line missing after a failed download"
        assert "No log output is available." not in app.screen._shown_lines
        downloads = _whole_job_calls(c)
        app.screen._refilter()
        for _ in range(5):
            await pilot.pause()
        assert _whole_job_calls(c) == downloads
        await _settle(app, pilot)


def test_failed_download_keeps_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    asyncio.run(_run_keeps_cache_when_download_fails())


async def _run_empty_file_keeps_cache():
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir, **kwargs):
        _write_stdout(output_dir, "7:training:0", "")
        return []

    c.download_stdout_logs.side_effect = _whole_job(download)
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

    def download(job_id, output_dir, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        _write_stdout(output_dir, "7:training:0", "")
        return []

    c.download_stdout_logs.side_effect = _whole_job(download)
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

    def download(job_id, output_dir, **kwargs):
        _write_stdout(output_dir, "7:training:0", "after finish\n")
        return []

    c.get_job.side_effect = get_job
    c.tail_logs.side_effect = tail_logs
    c.download_stdout_logs.side_effect = _whole_job(download)
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
    c.download_stdout_logs.side_effect = _whole_job(_raise(RuntimeError("unavailable")))
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: _whole_job_calls(c) > 0 and "cached line" in app.screen._shown_lines,
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

    def download(job_id, output_dir, **kwargs):
        calls.append(output_dir)
        started.set()
        assert release.wait(timeout=5)
        _write_stdout(output_dir, "7:training:0", "from file\n")
        return []

    c.download_stdout_logs.side_effect = _whole_job(download)
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

    def download(job_id, output_dir, **kwargs):
        calls.append(1)
        _write_stdout(output_dir, "7:training:0", "training line\n")
        return []

    c.download_stdout_logs.side_effect = _whole_job(download)
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

    def download(job_id, output_dir, **kwargs):
        _write_stdout(output_dir, "7:training:0", "training line\n")
        return []

    c.download_stdout_logs.side_effect = _whole_job(download)
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._stdout_ready)
        assert ok, "first download did not finish"
        app.screen._start_tail("7:sampling:0")
        ok = await _wait(pilot, app, lambda: "No log output is available." in app.screen._shown_lines)
        assert ok, "empty sibling did not show the finished-view note"
        assert "training line" not in app.screen._shown_lines
        assert _whole_job_calls(c) == 1
        await _settle(app, pilot)


def test_missing_sibling_after_download_shows_empty_note():
    asyncio.run(_run_missing_sibling_shows_note())


async def _run_save_exports_file(tmp_path):
    c = _client()
    c.get_job.return_value = _terminal_job()

    payload = b"file body\r\nsecond\r"

    def download(job_id, output_dir, **kwargs):
        path = _write_stdout(output_dir, "7:training:0", "")
        path.write_bytes(payload)
        return []

    c.download_stdout_logs.side_effect = _whole_job(download)
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

    def download(job_id, output_dir, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        _write_stdout(output_dir, "7:training:0", "held line\n")
        return []

    c.download_stdout_logs.side_effect = _whole_job(download)
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
        ok = await _wait(pilot, app, lambda: "held line" in app.screen._shown_lines)
        assert ok, "re-tail did not repaint the saved console"
        assert "stale line" not in app.screen._shown_lines
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


def _stdout_root():
    from cortex_training.tui.log_cache import job_cache_dir

    return job_cache_dir("7") / "stdout"


async def _run_reopen_reuses_cache(release, calls):
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir, **kwargs):
        calls.append((Path(output_dir), kwargs))
        if len(calls) == 1:
            _write_stdout(output_dir, "7:training:0", "saved line\n")
            return []
        # Second open: nothing new to fetch; hold so the test can observe the
        # cached file painted while the incremental refresh is still running.
        assert release.wait(timeout=5)
        return []

    c.download_stdout_logs.side_effect = _whole_job(download)
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "saved line" in app.screen._shown_lines)
        assert ok, "first open did not show the saved console"
        app.pop_screen()
        await pilot.pause()
        app.push_screen(LogScreen(c, "7", poll_interval=0.01))
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and "saved line" in app.screen._shown_lines
            and len(calls) == 2,
        )
        assert ok, "reopen did not paint the cached console during the refresh"
        assert "downloading" in (app.sub_title or "")
        release.set()
        ok = await _wait(pilot, app, lambda: app.screen._stdout_done)
        assert ok, "incremental refresh did not finish"
        assert "saved line" in app.screen._shown_lines
        assert "saved console" not in (app.sub_title or "")
        await _settle(app, pilot)
    assert [path for path, _ in calls] == [_stdout_root(), _stdout_root()]
    assert all(kwargs.get("resume") is True for _, kwargs in calls)


def test_reopen_reuses_persistent_stdout_cache_with_resume():
    release = threading.Event()
    calls = []
    try:
        asyncio.run(_run_reopen_reuses_cache(release, calls))
    finally:
        release.set()
    assert len(calls) == 2


async def _run_reopen_during_download(release, state):
    c = _client()
    c.get_job.return_value = _terminal_job()
    guard = threading.Lock()

    def download(job_id, output_dir, **kwargs):
        with guard:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
            state["calls"] += 1
            first = state["calls"] == 1
        try:
            if first:
                assert release.wait(timeout=5)
                _write_stdout(output_dir, "7:training:0", "first download\n")
        finally:
            with guard:
                state["active"] -= 1
        return []

    c.download_stdout_logs.side_effect = _whole_job(download)
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: state["calls"] == 1)
        assert ok, "first download did not start"
        app.pop_screen()
        await pilot.pause()
        app.push_screen(LogScreen(c, "7", poll_interval=0.01))
        ok = await _wait(pilot, app, lambda: "waiting for another download" in (app.sub_title or ""))
        assert ok, "reopened view did not report waiting on the running download"
        for _ in range(10):
            await pilot.pause()
        assert state["calls"] == 1, "reopen started a second concurrent download"
        release.set()
        ok = await _wait(pilot, app, lambda: "first download" in app.screen._shown_lines)
        assert ok, "reopened view did not show the console the first download saved"
        ok = await _wait(pilot, app, lambda: app.screen._stdout_done)
        assert ok, "reopened view never ran its own refresh"
        await _settle(app, pilot)


def test_reopen_during_download_waits_instead_of_downloading_twice():
    release = threading.Event()
    state = {"active": 0, "max": 0, "calls": 0}
    try:
        asyncio.run(_run_reopen_during_download(release, state))
    finally:
        release.set()
    assert state["max"] == 1
    assert state["calls"] == 2


async def _run_failure_then_retry():
    c = _client()
    c.get_job.return_value = _terminal_job()
    calls = []

    def download(job_id, output_dir, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            exc = RuntimeError("stage unavailable")
            exc.download_progress_saved = True
            raise exc
        _write_stdout(output_dir, "7:training:0", "after retry\n")
        return []

    c.download_stdout_logs.side_effect = _whole_job(download)
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: any("saved console download failed" in line for line in app.screen._shown_lines),
        )
        assert ok, "download failure was not shown in the pane"
        assert "RuntimeError: stage unavailable; progress saved, r resumes" in "\n".join(app.screen._shown_lines)
        assert "r retries" in (app.sub_title or "")
        app.screen._refilter()
        for _ in range(5):
            await pilot.pause()
        assert len(calls) == 1, "a filter change retried the failed download"
        app.screen.action_refresh_sources()
        ok = await _wait(pilot, app, lambda: "after retry" in app.screen._shown_lines)
        assert ok, "r did not retry the saved-console download"
        assert len(calls) == 2
        assert app.screen._stdout_error is None
        await _settle(app, pilot)


def test_download_failure_shows_error_and_r_retries(monkeypatch, tmp_path):
    import cortex_training.tui.app as app_module

    monkeypatch.setattr(app_module, "_ERROR_LOG", str(tmp_path / "errors.log"))
    asyncio.run(_run_failure_then_retry())


async def _run_saved_console_read_off_ui_thread(seen):
    import cortex_training.tui.app as app_module

    real = app_module.read_saved_tail

    def spy(path, query=""):
        seen.append((threading.current_thread() is threading.main_thread(), query))
        return real(path, query)

    app_module.read_saved_tail = spy
    try:
        c = _client()
        c.get_job.return_value = _terminal_job()

        def download(job_id, output_dir, **kwargs):
            _write_stdout(output_dir, "7:training:0", "alpha\nbeta\n")
            return []

        c.download_stdout_logs.side_effect = _whole_job(download)
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test() as pilot:
            ok = await _wait(pilot, app, lambda: "beta" in app.screen._shown_lines)
            assert ok, "saved console was not shown"
            app.screen._filter = "alp"
            app.screen._refilter()
            ok = await _wait(
                pilot,
                app,
                lambda: "alpha" in app.screen._shown_lines and "beta" not in app.screen._shown_lines,
            )
            assert ok, "filter was not applied to the saved console"
            await _settle(app, pilot)
    finally:
        app_module.read_saved_tail = real


def test_saved_console_is_read_off_the_ui_thread():
    seen = []
    asyncio.run(_run_saved_console_read_off_ui_thread(seen))
    assert seen and not any(on_main for on_main, _ in seen)
    assert "alp" in [query for _, query in seen]


async def _run_progress_status_and_cancel(state):
    c = _client()
    c.get_job.return_value = _terminal_job()

    class _Downloads:
        # A real signature: unknown keyword arguments would fail the call.
        def download_stdout_logs(self, job_id, output_dir, *, resume=False, on_chunk=None, **kwargs):
            if kwargs.get("sub_job_ids") is not None:
                return []
            state["resume"] = resume
            try:
                for i in range(1, 10000):
                    on_chunk("7:training:0", i, 10000, i * 1024)
                    if i == 1:
                        state["started"].set()
                    time.sleep(0.005)
            except BaseException as exc:  # noqa: BLE001 - the TUI's cancel signal
                state["raised"] = type(exc)
                raise
            return []

    c.download_stdout_logs = _Downloads().download_stdout_logs
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: re.search(r"\d+/10000 chunks", app.sub_title or ""))
        assert ok, "download progress did not reach the status line"
        assert re.search(r"\d+/10000 chunks, \d+\.\d (KiB|MiB)", app.sub_title), app.sub_title
        app.pop_screen()
        for _ in range(5):
            await pilot.pause()
        await _settle(app, pilot)
    for _ in range(200):
        if "raised" in state:
            break
        time.sleep(0.01)


def test_progress_updates_status_and_leaving_cancels_download(monkeypatch, tmp_path):
    import cortex_training.tui.app as app_module

    monkeypatch.setattr(app_module, "_ERROR_LOG", str(tmp_path / "errors.log"))
    state = {"started": threading.Event()}
    asyncio.run(_run_progress_status_and_cancel(state))
    assert state["resume"] is True
    assert issubclass(state.get("raised"), DownloadCancelled), "download was not cancelled"
    assert not (tmp_path / "errors.log").exists(), "a cancel was logged as a failure"


async def _run_selected_sub_job_first(release, calls):
    c = _client()
    c.get_job.return_value = _terminal_job()

    class _Downloads:
        def download_stdout_logs(
            self, job_id, output_dir, *, resume=False, on_chunk=None, sub_job_ids=None, **_kwargs
        ):
            calls.append(None if sub_job_ids is None else list(sub_job_ids))
            if sub_job_ids is not None:
                for sid in sub_job_ids:
                    _write_stdout(output_dir, sid, f"{sid} saved\n")
                return []
            assert release.wait(timeout=5)
            _write_stdout(output_dir, "7:sampling:0", "7:sampling:0 saved\n")
            return []

    c.download_stdout_logs = _Downloads().download_stdout_logs
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "7:training:0 saved" in app.screen._shown_lines)
        assert ok, "selected sub-job was not shown before the rest finished"
        ok = await _wait(pilot, app, lambda: len(calls) == 2)
        assert ok, "the rest of the job was not downloaded"
        assert calls == [["7:training:0"], None]
        assert app.screen._stdout_inflight
        release.set()
        ok = await _wait(pilot, app, lambda: app.screen._stdout_done)
        assert ok, "whole-job download did not finish"
        app.screen._start_tail("7:sampling:0")
        ok = await _wait(pilot, app, lambda: "7:sampling:0 saved" in app.screen._shown_lines)
        assert ok, "sibling sub-job was not shown after the full download"
        await _settle(app, pilot)


def test_selected_sub_job_downloads_and_paints_first():
    release = threading.Event()
    calls = []
    try:
        asyncio.run(_run_selected_sub_job_first(release, calls))
    finally:
        release.set()


async def _run_reopen_while_download_blocks(state):
    c = _client()
    c.get_job.return_value = _terminal_job()

    class _Downloads:
        # The first download blocks (like a retry backoff that still polls the
        # callback) until the TUI cancels it; later ones finish at once.
        def download_stdout_logs(self, job_id, output_dir, *, resume=False, on_chunk=None, **kwargs):
            if kwargs.get("sub_job_ids") is not None:
                return []
            state["calls"] += 1
            if state["calls"] > 1:
                _write_stdout(output_dir, "7:training:0", "second view\n")
                return []
            state["started"].set()
            try:
                while True:
                    on_chunk("7:training:0", 0, 10, 0)
                    time.sleep(0.02)
            except BaseException as exc:  # noqa: BLE001 - the TUI's cancel signal
                state["raised"] = type(exc)
                raise

    c.download_stdout_logs = _Downloads().download_stdout_logs
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, state["started"].is_set)
        assert ok, "first download did not start"
        app.pop_screen()
        await pilot.pause()
        app.push_screen(LogScreen(c, "7", poll_interval=0.01))
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen) and "second view" in app.screen._shown_lines,
            tries=300,
        )
        assert ok, "reopened view stayed waiting on the cancelled download"
        assert "waiting" not in (app.sub_title or "")
        await _settle(app, pilot)


def test_reopen_does_not_wait_forever_on_a_blocked_download(monkeypatch, tmp_path):
    import cortex_training.tui.app as app_module

    monkeypatch.setattr(app_module, "_ERROR_LOG", str(tmp_path / "errors.log"))
    state = {"calls": 0, "started": threading.Event()}
    asyncio.run(_run_reopen_while_download_blocks(state))
    assert issubclass(state["raised"], DownloadCancelled), "download was not cancelled"
    assert state["calls"] == 2
    assert not (tmp_path / "errors.log").exists()


async def _run_quit_during_download(state):
    c = _client()
    c.get_job.return_value = _terminal_job()

    class _Downloads:
        def download_stdout_logs(self, job_id, output_dir, *, resume=False, on_chunk=None, **kwargs):
            if kwargs.get("sub_job_ids") is not None:
                return []
            state["started"].set()
            try:
                while True:
                    on_chunk("7:training:0", 0, 10, 0)
                    time.sleep(0.02)
            finally:
                state["returned"].set()

    c.download_stdout_logs = _Downloads().download_stdout_logs
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, state["started"].is_set)
        assert ok, "download did not start"
        await app.action_quit()
        await pilot.pause()


def test_quit_during_download_stops_it_and_frees_the_lock():
    from cortex_training._download_lock import DownloadLock

    state = {"started": threading.Event(), "returned": threading.Event()}
    asyncio.run(_run_quit_during_download(state))
    assert state["returned"].wait(timeout=5), "download kept running after quit"
    lock = DownloadLock(_stdout_root())
    deadline = time.monotonic() + 5
    assert lock.acquire(lambda: time.monotonic() < deadline), "lock still held after quit"
    lock.release()


async def _run_stdout_warnings(calls):
    c = _client()
    c.get_job.return_value = _terminal_job()

    class _Downloads:
        def download_stdout_logs(
            self, job_id, output_dir, *, resume=False, on_chunk=None, sub_job_ids=None, stale_after_seconds=600
        ):
            calls.append(stale_after_seconds)
            _write_stdout(output_dir, "7:training:0", "saved\n")
            return [
                {"sub_job_id": "7:training:0", "warnings": ["1 damaged chunk was skipped"]},
                {"sub_job_id": "7:sampling:0", "warnings": ["sampling warning"]},
            ]

    c.download_stdout_logs = _Downloads().download_stdout_logs
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "warning: 1 damaged chunk was skipped" in (app.sub_title or ""))
        assert ok, "download warning for the selected sub-job was not shown"
        assert "saved" in app.screen._shown_lines
        assert "sampling warning" not in (app.sub_title or "")
        app.screen._start_tail("7:sampling:0")
        ok = await _wait(pilot, app, lambda: "warning: sampling warning" in (app.sub_title or ""))
        assert ok, "warnings did not follow the selected sub-job"
        await _settle(app, pilot)


def test_stdout_download_warnings_show_for_the_selected_sub_job():
    calls = []
    asyncio.run(_run_stdout_warnings(calls))
    # Finished jobs skip the client's stall check (its status lookup is wasted).
    assert calls and all(value is None for value in calls)


async def _run_metrics_stall_check_throttled(calls, monkeypatch):
    import cortex_training.tui.app as app_module

    c = _client()

    class _Downloads:
        def download_metrics(self, job_id, output_dir, *, resume=False, stale_after_seconds=600):
            calls.append(stale_after_seconds)
            _write_gpu_metrics(output_dir, "7:training:0", gpu_count=1)
            if stale_after_seconds is None:
                return [{"sub_job_id": "7:training:0"}]
            return [{"sub_job_id": "7:training:0", "warnings": ["uploads may be stalled"]}]

    c.download_metrics = _Downloads().download_metrics

    def status():
        return str(app.screen.query_one("#resource-status", Static).render())

    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source == "7:training:0")
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(pilot, app, lambda: len(calls) == 1 and "uploads may be stalled" in status())
        assert ok, "stall warning was not shown on Resource"
        assert "uploads may be stalled" in (app.sub_title or "")
        app.screen.action_refresh_sources()
        ok = await _wait(pilot, app, lambda: len(calls) == 2 and not app.screen._resource_loading)
        assert ok, "r did not refresh metrics"
        # The throttled refresh skipped the check but keeps its last answer.
        assert "uploads may be stalled" in status()
        monkeypatch.setattr(app_module, "_STALL_CHECK_SECONDS", 0.0)
        app.screen.action_refresh_sources()
        ok = await _wait(pilot, app, lambda: len(calls) == 3 and not app.screen._resource_loading)
        assert ok
        await _settle(app, pilot)


def test_metrics_stall_check_runs_at_most_once_per_window(monkeypatch):
    calls = []
    asyncio.run(_run_metrics_stall_check_throttled(calls, monkeypatch))
    assert calls == [600, None, 600]


async def _run_repaint_on_sub_job_done(release):
    c = _client()
    c.get_job.return_value = _terminal_job()

    class _Downloads:
        # No sub_job_ids support: the whole job downloads in one call.
        def download_stdout_logs(self, job_id, output_dir, *, resume=False, on_chunk=None, **kwargs):
            if kwargs.get("sub_job_ids") is not None:
                return []
            on_chunk("7:sampling:0", 0, 2, 0)
            _write_stdout(output_dir, "7:sampling:0", "sampling saved\n")
            on_chunk("7:sampling:0", 2, 2, 15)
            on_chunk("7:training:0", 0, 3, 0)
            _write_stdout(output_dir, "7:training:0", "training saved\n")
            on_chunk("7:training:0", 3, 3, 15)
            assert release.wait(timeout=5)
            return []

    c.download_stdout_logs = _Downloads().download_stdout_logs
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "training saved" in app.screen._shown_lines)
        assert ok, "selected sub-job was not shown when its file was committed"
        assert app.screen._stdout_inflight
        release.set()
        ok = await _wait(pilot, app, lambda: app.screen._stdout_done)
        assert ok
        await _settle(app, pilot)


def test_selected_sub_job_repaints_when_progress_reports_it_done():
    release = threading.Event()
    try:
        asyncio.run(_run_repaint_on_sub_job_done(release))
    finally:
        release.set()


@pytest.mark.parametrize(
    "kind, command",
    [
        ("stdout", "cortex-training download-log job-1 --log-type stdout"),
        ("metrics", "cortex-training download-metrics job-1"),
    ],
)
def test_skipped_chunk_hint_names_the_cli_command_for_the_tui_cache(kind, command, tmp_path):
    import shlex

    from cortex_training.tui.app import _result_warnings

    cache = tmp_path / "cache dir" / kind
    rows = [
        {
            "sub_job_id": "job-1:training:0",
            "skipped_chunks": ["chunk"],
            "warnings": [
                "1 damaged chunk(s) skipped by an earlier download are still missing from the saved file; "
                "download again without resume to retry them"
            ],
        }
    ]
    (message,) = _result_warnings(rows, kind=kind, job_id="job-1", output_dir=cache)["job-1:training:0"]
    assert "download again without resume" not in message
    retry = message.split("`")[1]
    assert retry == f"{command} --sub-job job-1:training:0 --output-dir {shlex.quote(str(cache))}"
    # The quoted path survives a shell round trip, so the retry repairs the cached file.
    assert shlex.split(retry)[-1] == str(cache)


async def _run_failed_refresh_keeps_stall_check_due(calls):
    c = _client()

    class _Downloads:
        def download_metrics(self, job_id, output_dir, *, resume=False, stale_after_seconds=600, **_kwargs):
            calls.append(stale_after_seconds)
            if len(calls) == 1:
                raise RuntimeError("listing failed")
            _write_gpu_metrics(output_dir, "7:training:0", gpu_count=1)
            return []

    c.download_metrics = _Downloads().download_metrics
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source == "7:training:0")
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(pilot, app, lambda: len(calls) == 1 and not app.screen._resource_loading)
        assert ok
        app.screen.action_refresh_sources()
        ok = await _wait(pilot, app, lambda: len(calls) == 2 and not app.screen._resource_loading)
        assert ok
        app.screen.action_refresh_sources()
        ok = await _wait(pilot, app, lambda: len(calls) == 3 and not app.screen._resource_loading)
        assert ok
        await _settle(app, pilot)


def test_failed_metrics_refresh_does_not_use_up_the_stall_check():
    calls = []
    asyncio.run(_run_failed_refresh_keeps_stall_check_due(calls))
    # The failed first refresh leaves the check due; the next one runs it and
    # starts the quiet window.
    assert calls == [600, 600, None]


def _real_download_client(monkeypatch, *, subs, count=3):
    """A TUI client whose downloads go through the real SDK code over a fake
    Snowflake connection (job metadata and tails stay mocked)."""
    from tests.test_client import TestChunkedArtifactDownload

    real, connection, _paths, _sleeps, _opened = TestChunkedArtifactDownload()._setup(
        monkeypatch, subs=subs, count=count
    )
    c = _client()
    c.download_stdout_logs = real.download_stdout_logs
    return c, connection


async def _run_selected_sub_job_without_chunks(c):
    app = CortexTrainingLogTUI(c, "job-1", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._stdout_done, tries=300)
        assert ok, "saved-console download did not finish"
        screen = app.screen
        assert screen._current_source == "job-1:sampling:0"
        assert screen._stdout_error is None
        assert not any("download failed" in line for line in screen._shown_lines)
        ok = await _wait(pilot, app, lambda: "No log output is available." in screen._shown_lines)
        assert ok, "a sub-job without saved output did not show the empty note"
        screen._start_tail("job-1:training:0")
        ok = await _wait(pilot, app, lambda: "job-1:training:0 line 2" in screen._shown_lines)
        assert ok, "the other sub-job's console was not fetched"
        await _settle(app, pilot)


def test_selected_sub_job_without_chunks_falls_through_to_the_whole_job(monkeypatch):
    c, connection = _real_download_client(monkeypatch, subs=["job-1:training:0"])
    c.get_job.return_value = {
        "status": "FAILED",
        "sub_jobs": [
            {"sub_job_id": "job-1:sampling:0", "job_type": "sampling"},
            {"sub_job_id": "job-1:training:0", "job_type": "training"},
        ],
    }
    asyncio.run(_run_selected_sub_job_without_chunks(c))
    gets = [cmd for cmd in connection.commands if cmd.startswith("GET ")]
    assert len(gets) == 3


async def _run_live_raw_lines_render_and_filter():
    c = _client()
    stamp = "2026-01-01T00:00:00.123456789Z "
    c.tail_logs.return_value = {
        "entries": [
            {"_raw": stamp + '{"level": "INFO", "msg": "drain_metrics"}'},
            {"_raw": stamp + "(EngineCore pid=7) WARNING 01-01 00:00:00 [a.py:1] slow"},
            {"_raw": stamp + '{"level": "ERROR", "msg": "boom"}'},
            {"_raw": stamp + "plain line"},
        ],
        "next_cursor": "1",
        "eof": True,
    }
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "plain line" in getattr(app.screen, "_shown_lines", []))
        assert ok, "live lines were not shown"
        screen = app.screen
        assert not any(line.startswith("2026-01-01T00:00:00") for line in screen._shown_lines)
        assert '{"level": "ERROR", "msg": "boom"}' in screen._shown_lines
        screen.action_cycle_level()  # INFO
        screen.action_cycle_level()  # WARNING
        screen.action_cycle_level()  # ERROR
        ok = await _wait(
            pilot,
            app,
            lambda: "plain line" in screen._shown_lines and '{"level": "INFO", "msg": "drain_metrics"}' not in screen._shown_lines,
        )
        assert ok, "the level filter did not apply to live lines"
        assert '{"level": "ERROR", "msg": "boom"}' in screen._shown_lines
        assert not any("WARNING 01-01" in line for line in screen._shown_lines)
        await _settle(app, pilot)


def test_live_raw_lines_drop_the_runtime_time_and_honor_the_level_filter():
    asyncio.run(_run_live_raw_lines_render_and_filter())


async def _run_recently_ended_hint(state_changed_at, expect_hint):
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    c = _client()
    c.get_job.return_value = {
        **_terminal_job(),
        "state_changed_at": _dt.fromtimestamp(state_changed_at, tz=_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    c.download_stdout_logs.side_effect = _whole_job(lambda job_id, output_dir, **kwargs: [])
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._ended_at is not None)
        assert ok
        await _wait(pilot, app, lambda: "FAILED" in (app.sub_title or ""))
        hint = "its last output may still be uploading (r refreshes)"
        assert (hint in (app.sub_title or "")) is expect_hint, app.sub_title
        if expect_hint:
            assert re.search(r"ended \d+s ago", app.sub_title)
        await _settle(app, pilot)


def test_a_just_ended_job_says_its_last_output_may_still_be_uploading():
    asyncio.run(_run_recently_ended_hint(time.time() - 30, True))


def test_a_long_ended_job_has_no_upload_hint():
    asyncio.run(_run_recently_ended_hint(time.time() - 3 * 86400, False))


async def _run_resource_age_is_humanized(tmp_path):
    c = _client()

    def download(job_id, output_dir, *, resume=False, **_kwargs):
        _write_gpu_metrics(output_dir, "7:training:0", gpu_count=1)
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(140, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source == "7:training:0")
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"

        def status():
            return str(app.screen.query_one("#resource-status", Static).render())

        ok = await _wait(pilot, app, lambda: "last sample" in status())
        assert ok, status()
        assert re.search(r"last sample \d+d \d+h ago", status()), status()
        await _settle(app, pilot)


def test_resource_last_sample_age_is_humanized(tmp_path):
    asyncio.run(_run_resource_age_is_humanized(tmp_path))



async def _run_no_end_time_no_upload_hint():
    c = _client()
    c.get_job.return_value = _terminal_job()  # no state_changed_at
    c.download_stdout_logs.side_effect = _whole_job(lambda job_id, output_dir, **kwargs: [])
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and "FAILED" in (app.sub_title or ""))
        assert ok
        for _ in range(5):
            await pilot.pause()
        assert app.screen._ended_at is None
        assert "may still be uploading" not in (app.sub_title or "")
        await _settle(app, pilot)


def test_an_ended_job_without_an_end_time_gets_no_upload_hint():
    asyncio.run(_run_no_end_time_no_upload_hint())
