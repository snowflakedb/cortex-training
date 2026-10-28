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

"""A live job's periodic telemetry refreshes on the job details screen.

Once a whole-job download is in, the Resources tab's refresh of a live job downloads the
sub-job on screen alone, so its LIST returns that sub-job's chunks rather than the whole
run's; so does the Training metrics tab's, while it has one sub-job to show. Such a download
leaves the other sub-jobs' copies behind: showing one, by a switch or on the tab's return,
refreshes it once it is a refresh interval old or the job has ended (until then, the next
tick refreshes the one shown). r downloads the whole job, after the download of one sub-job
alone it finds running. A client whose download takes no ``sub_job_ids`` (a MagicMock, a
test's fake, an older client) keeps downloading the whole job, and leaves no copy behind.
"""

import asyncio
import functools
import gzip
import json
import tempfile
import threading
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from textual.widgets import Static  # noqa: E402
from textual.widgets import TabbedContent  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training import NoChunksError  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.logs import LogScreen  # noqa: E402
from cortex_training.tui.resource import GPU  # noqa: E402

TRAINING = "7:training:0"
SAMPLING = "7:sampling:0"
# Older than a refresh interval: a copy left behind that was downloaded this long ago is due.
_DUE = LogScreen._DATA_REFRESH_SECONDS + 1


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    # As in test_tui_app: the job's cache, the error log and the staging directories
    # all live in tmp_path.
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "isolated-cache"))
    import cortex_training.tui.widgets as widgets

    monkeypatch.setattr(widgets, "_ERROR_LOG", str(tmp_path / "errors.log"))
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    yield
    from cortex_training.tui.cache_cleanup import _release_pending

    _release_pending()  # a closed screen's lease, still in its grace, holds a lock file open


def _client(job_id="7", sub_job_ids=(TRAINING, SAMPLING)):
    """A RUNNING job with ``sub_job_ids`` (a training and a sampling sub-job). A MagicMock, as
    test_tui_app's client is: its download methods take any keyword, and their signature names none."""
    c = MagicMock()
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = []
    c.get_job.return_value = {"status": "RUNNING", "sub_jobs": [
        {"sub_job_id": sub_job_id, "job_type": sub_job_id.split(":")[1]} for sub_job_id in sub_job_ids]}
    c.tail_logs.return_value = {"entries": [], "next_cursor": "", "eof": True}
    return c


def _gpu_rows(sub_job_id, *, ago=30.0) -> str:
    """A sample of one GPU, as gpu.jsonl holds it, ``ago`` seconds old."""
    when = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=ago)
    row = {"record_version": 2, "sub_job_id": sub_job_id,
           "sample_ts": when.strftime("%Y-%m-%dT%H:%M:%S") + ".123456789Z",
           "worker_num": 0, "local_gpu_index": 0, "global_gpu_index": 0, "gpu_util_pct": 50.0,
           "fb_used_bytes": float(1024**3), "fb_free_bytes": float(8 * 1024**3), "sample_interval_s": 5.0}
    return json.dumps(row) + "\n"


def _write_gpu_rows(output_dir, sub_job_id, *, ago=30.0) -> Path:
    """``sub_job_id``'s gpu.jsonl and its manifest, as a download commits them (see
    test_tui_app's _write_gpu_metrics): two samples 5 s apart, as their sample_interval_s
    says (so a running sub-job's copy is stale once its newest is 180 s old), the newest
    ``ago`` seconds old."""
    path = Path(output_dir) / sub_job_id / "gpu.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_gpu_rows(sub_job_id, ago=ago + 5) + _gpu_rows(sub_job_id, ago=ago), encoding="utf-8")
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size}), encoding="utf-8")
    return path


class _Stage:
    """A job's GPU metrics as its stage holds them, and a ``download_metrics`` over them that
    takes ``sub_job_ids``, as the client's does.

    Each call's ``sub_job_ids`` is recorded (None: the whole job). A call writes the rows of
    the sub-jobs it covers, committing each, and returns their result entries; a sub-job it
    is asked for alone that has no chunks raises :class:`NoChunksError`, as the client does.
    """

    def __init__(self, ages=None):
        # How old each sub-job's newest sample is, in seconds; a sub-job left out has no chunks.
        self.ages = dict(ages if ages is not None else {TRAINING: 30.0, SAMPLING: 30.0})
        self.warnings: dict[str, list[str]] = {}  # each sub-job's result entry's warnings
        self.calls: list[list[str] | None] = []
        # Each download raises the first of these, taking it, as a failing stage would.
        self.errors: list[Exception] = []
        # A download of one of these sub-jobs alone (None: of the whole job) waits for its
        # event (once); ``waiting`` is set meanwhile. An error waits with it.
        self.held: dict[str | None, threading.Event] = {}
        self.waiting = threading.Event()

    def hold(self, sub_job_id: str | None) -> threading.Event:
        release = self.held[sub_job_id] = threading.Event()
        return release

    def release_all(self) -> None:
        for release in self.held.values():
            release.set()

    def download_metrics(self, job_id, output_dir, *, resume=False, priority=(), on_commit=None, sub_job_ids=None,
                         stale_after_seconds=600, on_chunk=None, preview_seconds=None, on_preview=None):
        named = None if sub_job_ids is None else list(sub_job_ids)
        self.calls.append(named)
        if named is None:
            release = self.held.pop(None, None)
        else:
            release = self.held.pop(named[0], None) if len(named) == 1 else None
        if release is not None:
            self.waiting.set()
            release.wait(10)
            self.waiting.clear()
        if self.errors:
            raise self.errors.pop(0)
        missing = [sub_job_id for sub_job_id in named or () if sub_job_id not in self.ages]
        if missing:
            raise NoChunksError(f"no metrics chunks for sub-job(s) {', '.join(missing)}; sub-jobs with metrics: none",
                                sub_job_ids=missing, available=[])
        first = list(priority)
        covered = sorted(self.ages if named is None else named,
                         key=lambda sub_job_id: first.index(sub_job_id) if sub_job_id in first else len(first))
        results = []
        for sub_job_id in covered:  # the ones in priority first, as the client commits them
            entry = {"sub_job_id": sub_job_id,
                     "saved_path": str(_write_gpu_rows(output_dir, sub_job_id, ago=self.ages[sub_job_id]))}
            if self.warnings.get(sub_job_id):
                entry["warnings"] = list(self.warnings[sub_job_id])
            if on_commit is not None:
                on_commit(dict(entry))
            results.append(entry)
        return sorted(results, key=lambda entry: entry["sub_job_id"])


async def _wait(pilot, predicate, seconds=10.0) -> bool:
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() > deadline:
            return False
        await pilot.pause(0.02)
    return True


async def _settle(app, pilot):
    """Stop background workers and drain before ``run_test`` tears down (see test_tui_app)."""
    try:
        app.workers.cancel_all()
    except Exception:  # noqa: BLE001
        pass
    for _ in range(5):
        await pilot.pause()


def _header(screen) -> str:
    return str(screen.query_one("#resource-status", Static).render())


async def _resources(app, pilot, first=TRAINING) -> LogScreen:
    """The job's screen on its Resources tab, once the tab's first download is in."""
    assert await _wait(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source == first)
    screen = app.screen
    screen.query_one("#job-tabs", TabbedContent).active = "resource"
    assert await _wait(pilot, lambda: screen._resource_full_at is not None and not screen._resource_loading), \
        screen._resource_error
    return screen


async def _show_again(pilot, screen):
    """Away to the Logs tab (which doesn't poll the telemetry) and back to Resources."""
    tabs = screen.query_one("#job-tabs", TabbedContent)
    tabs.active = "logs"
    assert await _wait(pilot, lambda: screen._active_tab == "logs")
    tabs.active = "resource"
    assert await _wait(pilot, lambda: screen._active_tab == "resource")


async def _landed(pilot, screen, calls, count) -> bool:
    """Whether the ``count``-th download was asked for and the refresh it was part of ended."""
    return await _wait(pilot, lambda: len(calls) >= count and not screen._resource_loading)


async def _no_download(pilot, screen, calls):
    """No refresh started (a refresh marks the tab loading as it starts), nor does one after."""
    count = len(calls)
    assert not screen._resource_loading, "a refresh started"
    await pilot.pause(0.2)
    assert len(calls) == count and not screen._resource_loading, calls


def _age(screen, seconds):
    """Every download the screen made, moved ``seconds`` into the past rather than waited for."""
    if screen._resource_full_at is not None:
        screen._resource_full_at -= seconds
    for sub_job_id in screen._resource_alone_at:
        screen._resource_alone_at[sub_job_id] -= seconds


def _end(c, screen):
    """The job ends, as the status check would find it."""
    c.get_job.return_value = {**c.get_job.return_value, "status": "CANCELLED"}
    screen._apply_terminal_status("CANCELLED", None)
    assert not screen._live()


# ─── Resources ───────────────────────────────────────────────────────────────


def test_a_live_job_s_refreshes_download_the_sub_job_shown_alone():
    stage = _Stage()
    c = _client()
    c.download_metrics = stage.download_metrics

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _resources(app, pilot)
            # The first download is the whole job's, so every sub-job is on disk for a switch.
            assert stage.calls == [None]
            assert screen._resource_with_chunks == {TRAINING, SAMPLING}
            # A tick (what the 30-second timer calls) downloads the sub-job shown, alone.
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 2)
            assert stage.calls[1] == [TRAINING]
            # That left the other sub-job's copy behind, as the whole download brought it moments
            # ago: newer than a refresh interval, it is shown as it is while the job is live (the
            # next tick refreshes the sub-job shown), so switching to it, and back, downloads nothing.
            screen.action_resource_sub_job(1)
            assert "sampling #0" in _header(screen)
            await _no_download(pilot, screen, stage.calls)
            screen.action_resource_sub_job(1)
            await _no_download(pilot, screen, stage.calls)
            # A refresh interval later, a switch refreshes the copy it shows: that sub-job alone.
            _age(screen, _DUE)
            screen.action_resource_sub_job(1)
            assert await _landed(pilot, screen, stage.calls, 3)
            assert stage.calls[2] == [SAMPLING]
            # r downloads the whole job, as a first download does; the next tick, the one shown.
            await pilot.press("r")
            assert await _landed(pilot, screen, stage.calls, 4)
            assert stage.calls[3] is None
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 5)
            assert stage.calls[4] == [SAMPLING]
            # The job ends: a tick downloads nothing, and every download is the whole job's.
            _end(c, screen)
            screen._auto_refresh_data()
            await _no_download(pilot, screen, stage.calls)
            # training's newest download (r's) is older than sampling's alone since: a switch to it
            # refreshes it, with every sub-job ...
            screen.action_resource_sub_job(1)
            assert await _landed(pilot, screen, stage.calls, 6)
            assert stage.calls[5] is None
            # ... so sampling's copy is as new as that left it, and showing it again downloads nothing.
            screen.action_resource_sub_job(1)
            await _no_download(pilot, screen, stage.calls)
            await pilot.press("r")
            assert await _landed(pilot, screen, stage.calls, 7)
            screen._auto_refresh_data()
            await _no_download(pilot, screen, stage.calls)
            assert stage.calls == [None, [TRAINING], [SAMPLING], None, [SAMPLING], None, None]
            assert screen._resource_error is None
            await _settle(app, pilot)

    asyncio.run(run())


def test_showing_the_resources_tab_again_refreshes_a_copy_that_is_due():
    stage = _Stage()
    c = _client()
    c.download_metrics = stage.download_metrics

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _resources(app, pilot)
            # Back moments after the tab's download: nothing is downloaded.
            await _show_again(pilot, screen)
            await _no_download(pilot, screen, stage.calls)
            # Nor back a refresh interval later, while no download of another sub-job alone has
            # left the copy behind: as when every refresh was the whole job's, a tick refreshes it.
            _age(screen, _DUE)
            await _show_again(pilot, screen)
            await _no_download(pilot, screen, stage.calls)
            # A tick downloads training alone, and a switch to sampling, which that left behind a
            # refresh interval old, sampling alone ...
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 2)
            screen.action_resource_sub_job(1)
            assert await _landed(pilot, screen, stage.calls, 3)
            # ... which leaves training's copy behind, newer than a refresh interval: back on it, and
            # back on the tab, nothing is downloaded while the job is live.
            screen.action_resource_sub_job(1)
            await _no_download(pilot, screen, stage.calls)
            await _show_again(pilot, screen)
            await _no_download(pilot, screen, stage.calls)
            # Back on the tab a refresh interval later, the copy is due: the refresh downloads the
            # sub-job shown, alone.
            _age(screen, _DUE)
            await _show_again(pilot, screen)
            assert await _landed(pilot, screen, stage.calls, 4)
            assert stage.calls == [None, [TRAINING], [SAMPLING], [TRAINING]]
            assert screen._resource_error is None
            await _settle(app, pilot)

    asyncio.run(run())


@pytest.mark.parametrize("show", ["switch", "tab"])
def test_a_copy_left_behind_newer_than_an_interval_is_refreshed_when_shown_once_the_job_ended(show):
    stage = _Stage()
    c = _client()
    c.download_metrics = stage.download_metrics

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _resources(app, pilot)
            # A tick downloads training alone: sampling's copy, from the whole download moments
            # before, is left behind ...
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 2)
            assert screen._resource_behind(SAMPLING) and not screen._resource_behind(TRAINING)
            # ... newer than a refresh interval: while the job is live, showing it downloads nothing,
            # by a switch or on the tab's return (the next tick refreshes the sub-job shown).
            screen.action_resource_sub_job(1)
            await _no_download(pilot, screen, stage.calls)
            await _show_again(pilot, screen)
            await _no_download(pilot, screen, stage.calls)
            if show == "switch":
                screen.action_resource_sub_job(1)  # back on training, which is not behind
                await _no_download(pilot, screen, stage.calls)
            # Once the job has ended no tick will: showing the copy refreshes it, with every
            # sub-job, as an ended job's downloads are.
            _end(c, screen)
            if show == "switch":
                screen.action_resource_sub_job(1)
            else:
                await _show_again(pilot, screen)
            assert screen._resource_source() == SAMPLING
            assert await _landed(pilot, screen, stage.calls, 3)
            assert stage.calls == [None, [TRAINING], None]
            # That brought every copy up to date: showing either again downloads nothing.
            for _ in range(2):
                screen.action_resource_sub_job(1)
                await _no_download(pilot, screen, stage.calls)
            await _show_again(pilot, screen)
            await _no_download(pilot, screen, stage.calls)
            assert not screen._resource_behind(TRAINING) and not screen._resource_behind(SAMPLING)
            assert screen._resource_error is None
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_sub_job_alone_without_chunks_is_nothing_new_and_the_next_tick_is_whole(tmp_path):
    stage = _Stage()
    c = _client()
    c.download_metrics = stage.download_metrics

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _resources(app, pilot)
            # The stage has none of training's chunks now: its download alone raises NoChunksError.
            del stage.ages[TRAINING]
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 2)
            assert stage.calls == [None, [TRAINING]]
            # That is nothing new, not a failure: the copy on screen stays, and nothing is said.
            header = _header(screen)
            assert "refresh failed" not in header and "unavailable" not in header, header
            assert screen._resource_error is None and screen._resource_refresh_succeeded
            assert screen._resource_data.kind(TRAINING, GPU) is not None
            # The next tick downloads the whole job, which finds which sub-jobs have chunks ...
            assert TRAINING not in screen._resource_with_chunks
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 3)
            assert stage.calls[2] is None
            # ... and while training has none, every tick does.
            stage.ages[TRAINING] = 30.0
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 4)
            assert stage.calls[3] is None
            # Once a whole download has found them again, a tick downloads training alone again.
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 5)
            assert stage.calls[4] == [TRAINING]
            assert screen._resource_error is None
            await _settle(app, pilot)

    asyncio.run(run())
    log = tmp_path / "errors.log"
    assert not log.exists() or "NoChunksError" not in log.read_text(encoding="utf-8")  # nor logged as an error


def test_a_sub_job_the_whole_download_found_no_chunks_for_is_refreshed_with_the_whole_job():
    stage = _Stage(ages={TRAINING: 30.0})  # sampling uploads no GPU metrics
    c = _client()
    c.download_metrics = stage.download_metrics

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _resources(app, pilot)
            assert screen._resource_with_chunks == {TRAINING}
            # Shown, sampling has nothing on disk: the switch downloads, and so does each tick,
            # but never sampling alone (which has no chunks to list).
            screen.action_resource_sub_job(1)
            assert await _landed(pilot, screen, stage.calls, 2)
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 3)
            assert stage.calls == [None, None, None]
            # Back on training, whose copy that tick refreshed, a tick downloads it alone.
            screen.action_resource_sub_job(1)
            await _no_download(pilot, screen, stage.calls)
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 4)
            assert stage.calls[3] == [TRAINING]
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_sub_job_s_download_alone_keeps_the_other_sub_jobs_warnings():
    stage = _Stage()
    stage.warnings[SAMPLING] = ["1 damaged chunk was skipped"]
    c = _client()
    c.download_metrics = stage.download_metrics

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            screen = await _resources(app, pilot)
            # Training alone: its own warnings are this download's ...
            stage.warnings[TRAINING] = ["2 chunk(s) could not be downloaded"]
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 2)
            assert "warning: 2 chunk(s) could not be downloaded" in _header(screen)
            del stage.warnings[TRAINING]
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 3)
            assert stage.calls == [None, [TRAINING], [TRAINING]]
            assert "warning:" not in _header(screen)
            # ... and sampling's, which those downloads didn't look at, stay as the whole download
            # said (its copy, left behind newer than a refresh interval, isn't refreshed on showing).
            screen.action_resource_sub_job(1)
            await _no_download(pilot, screen, stage.calls)
            assert "warning: 1 damaged chunk was skipped" in _header(screen)
            await _show_again(pilot, screen)  # the subtitle shows the Resources tab's sub-job's warnings
            assert await _wait(pilot, lambda: "warning: 1 damaged chunk was skipped" in (app.sub_title or ""))
            await _no_download(pilot, screen, stage.calls)
            # A whole download says what every sub-job's chunks are now: nothing skipped.
            stage.warnings.clear()
            await pilot.press("r")
            assert await _landed(pilot, screen, stage.calls, 4)
            assert stage.calls[3] is None
            assert "warning:" not in _header(screen) and "warning:" not in (app.sub_title or "")
            await _settle(app, pilot)

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["magicmock", "older-client"])
def test_a_client_whose_download_takes_no_sub_job_ids_always_downloads_the_whole_job(kind):
    stage = _Stage()
    c = _client()
    if kind == "magicmock":
        c.download_metrics.side_effect = stage.download_metrics  # the signature is MagicMock's: (*args, **kwargs)
    else:
        def download_metrics(job_id, output_dir, *, resume=False, priority=(), on_commit=None):
            return stage.download_metrics(job_id, output_dir, resume=resume, priority=priority, on_commit=on_commit)

        c.download_metrics = download_metrics

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _resources(app, pilot)
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 2)
            assert screen._resource_scope() is None
            # Every download covers every sub-job, so no copy is ever left behind: however old,
            # showing one downloads nothing, by a switch or on the tab's return, as when every
            # refresh was the whole job's ...
            assert not screen._resource_alone_at
            assert not screen._resource_behind(TRAINING) and not screen._resource_behind(SAMPLING)
            _age(screen, _DUE)
            screen.action_resource_sub_job(1)
            await _no_download(pilot, screen, stage.calls)
            await _show_again(pilot, screen)
            await _no_download(pilot, screen, stage.calls)
            screen.action_resource_sub_job(1)
            await _no_download(pilot, screen, stage.calls)
            # ... and a tick, or r, downloads the whole job.
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 3)
            # Nor does showing one download once the job has ended.
            _end(c, screen)
            screen.action_resource_sub_job(1)
            await _no_download(pilot, screen, stage.calls)
            await _show_again(pilot, screen)
            await _no_download(pilot, screen, stage.calls)
            await pilot.press("r")
            assert await _landed(pilot, screen, stage.calls, 4)
            assert stage.calls == [None, None, None, None]
            assert not screen._resource_alone_at
            assert screen._resource_error is None  # nothing the client doesn't take was passed
            await _settle(app, pilot)

    asyncio.run(run())
    if kind == "magicmock":
        assert c.download_metrics.call_count == 4
        assert not any("sub_job_ids" in call.kwargs for call in c.download_metrics.call_args_list)


def test_a_sub_job_shown_while_another_downloads_alone_is_refreshed_once_that_ends():
    stage = _Stage()
    c = _client()
    c.download_metrics = stage.download_metrics

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _resources(app, pilot)
            release = stage.hold(TRAINING)
            screen._auto_refresh_data()
            assert await _wait(pilot, stage.waiting.is_set)
            # While training downloads alone, sampling, a refresh interval old, is shown: a
            # refresh is already running, so the switch starts none ...
            _age(screen, _DUE)
            screen.action_resource_sub_job(1)
            await pilot.pause(0.2)
            assert stage.calls == [None, [TRAINING]] and screen._resource_loading
            # ... and once training's is in, which leaves sampling's copy behind, sampling is
            # downloaded, alone, without waiting for a tick.
            release.set()
            assert await _wait(pilot, lambda: len(stage.calls) == 3 and not screen._resource_loading)
            assert stage.calls == [None, [TRAINING], [SAMPLING]]
            assert set(screen._resource_alone_at) == {TRAINING, SAMPLING}
            # Back to training during sampling's next download alone: training's copy, left behind
            # by sampling's downloads, is newer than a refresh interval, so nothing follows it while
            # the job is live (the next tick refreshes it).
            release = stage.hold(SAMPLING)
            screen._auto_refresh_data()
            assert await _wait(pilot, stage.waiting.is_set)
            screen.action_resource_sub_job(1)
            release.set()
            assert await _wait(pilot, lambda: len(stage.calls) == 4 and not screen._resource_loading)
            await _no_download(pilot, screen, stage.calls)
            assert stage.calls == [None, [TRAINING], [SAMPLING], [SAMPLING]]
            assert screen._resource_behind(TRAINING) and not screen._resource_due(TRAINING)
            await _settle(app, pilot)

    try:
        asyncio.run(run())
    finally:
        stage.release_all()


@pytest.mark.parametrize("alone", ["lands", "fails"])
def test_r_pressed_while_a_sub_job_downloads_alone_still_downloads_the_whole_job(alone):
    stage = _Stage()
    c = _client()
    c.download_metrics = stage.download_metrics

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _resources(app, pilot)
            release = stage.hold(TRAINING)
            if alone == "fails":
                stage.errors.append(RuntimeError("stage unavailable"))
            screen._auto_refresh_data()
            assert await _wait(pilot, stage.waiting.is_set)
            # r asks for every sub-job, as a first download does, while training's own download runs:
            # the header says refreshing, and r is kept for when that download ends (pressed twice,
            # still once).
            await pilot.press("r")
            assert "refreshing…" in _header(screen)
            await pilot.press("r")
            assert screen._resource_whole_pending
            # Once training's download ends, landed or failed, the whole job is downloaded, once.
            release.set()
            assert await _landed(pilot, screen, stage.calls, 3)
            await _no_download(pilot, screen, stage.calls)
            assert stage.calls == [None, [TRAINING], None]
            assert not screen._resource_whole_pending
            assert screen._resource_error is None and screen._resource_refresh_succeeded
            await _settle(app, pilot)

    try:
        asyncio.run(run())
    finally:
        stage.release_all()


def test_r_pressed_while_the_whole_job_downloads_is_that_download():
    # As before any download of one sub-job alone: the download running is the one r asks for,
    # so r adds nothing to it, nor does a tick.
    stage = _Stage()
    c = _client()
    c.download_metrics = stage.download_metrics

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _resources(app, pilot)
            release = stage.hold(None)
            await pilot.press("r")
            assert await _wait(pilot, stage.waiting.is_set)
            assert screen._resource_loading_scope is None
            await pilot.press("r")
            screen._auto_refresh_data()
            assert not screen._resource_whole_pending
            release.set()
            assert await _landed(pilot, screen, stage.calls, 2)
            await _no_download(pilot, screen, stage.calls)
            assert stage.calls == [None, None]
            assert screen._resource_error is None
            await _settle(app, pilot)

    try:
        asyncio.run(run())
    finally:
        stage.release_all()


def test_a_switched_to_copy_is_not_stale_before_its_refresh_lands():
    # Both sub-jobs' newest samples are ten minutes old.
    stage = _Stage(ages={TRAINING: 600.0, SAMPLING: 600.0})
    c = _client()
    c.download_metrics = stage.download_metrics

    def stale(screen) -> bool:
        return screen._resource_stale(screen._resource_kind_data(), datetime.now(timezone.utc))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            screen = await _resources(app, pilot)
            _age(screen, 600)  # ... and the whole download that brought them was ten minutes ago
            # Training, polled every tick, is judged as ever: its newest sample is old, so it's stale.
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 2)
            assert stage.calls[1] == [TRAINING]
            assert stale(screen) and screen._resource_stale_shown
            assert "no new GPU samples for" in _header(screen)  # (9m 59s or 10m, by the clock's second)
            # Sampling's copy, left behind by that tick, is as old as the download that brought it:
            # shown by a switch, it isn't called stale while the refresh that brings it up to date
            # runs (the last refresh having succeeded) ...
            release = stage.hold(SAMPLING)
            screen.action_resource_sub_job(1)
            assert await _wait(pilot, stage.waiting.is_set)
            assert screen._resource_source() == SAMPLING and screen._resource_loading
            assert not stale(screen) and not screen._resource_stale_shown
            assert "no new GPU samples" not in _header(screen)
            # ... and once it lands with nothing newer, it is.
            release.set()
            assert await _landed(pilot, screen, stage.calls, 3)
            assert stage.calls[2] == [SAMPLING]
            assert stale(screen) and screen._resource_stale_shown
            assert "no new GPU samples for" in _header(screen)
            await _settle(app, pilot)

    try:
        asyncio.run(run())
    finally:
        stage.release_all()


def test_with_refreshes_failing_a_copy_left_behind_ages_from_the_newest_download_that_succeeded():
    # Sampling's newest sample is older than the 180 s after which a running sub-job's is stale.
    stage = _Stage(ages={TRAINING: 30.0, SAMPLING: 200.0})
    c = _client()
    c.download_metrics = stage.download_metrics
    stale_after = timedelta(seconds=LogScreen._RESOURCE_STALE_SECONDS)  # 180 s
    second = timedelta(seconds=1)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            screen = await _resources(app, pilot)
            _age(screen, 600)  # the whole download that brought both copies was ten minutes ago
            screen._auto_refresh_data()  # a tick downloads training alone: sampling's copy is left behind
            assert await _landed(pilot, screen, stage.calls, 2)
            # Sampling's copy is judged as if that tick had kept it current, as a tick of the whole
            # job would have: as of the ten minutes since its own download, it is not stale.
            behind = timedelta(seconds=screen._resource_newest_success() - screen._resource_polled_at(SAMPLING))
            assert behind >= timedelta(seconds=600)
            # Shown by a switch, it is due; while its refresh runs, that holds ...
            release = stage.hold(SAMPLING)
            stage.errors.append(RuntimeError("stage unavailable"))
            screen.action_resource_sub_job(1)
            assert await _wait(pilot, stage.waiting.is_set)
            resource = screen._resource_kind_data()
            latest = resource.latest_sample_ts
            assert 6 * resource.sample_interval_s <= stale_after.total_seconds()  # sampled every 5 s
            assert not screen._resource_stale(resource, datetime.now(timezone.utc))
            assert not screen._resource_stale(resource, latest + stale_after + second)
            assert not screen._resource_stale_shown and "no new GPU samples" not in _header(screen)
            # ... and goes on holding once that refresh fails: one failure says nothing of the job's
            # uploads. With refreshes failing, the copy ages from the newest download that succeeded,
            # as one refreshed every tick does: stale once that is 180 s past its newest sample.
            release.set()
            assert await _landed(pilot, screen, stage.calls, 3)
            assert stage.calls[2] == [SAMPLING]
            assert not screen._resource_refresh_succeeded and screen._resource_behind(SAMPLING)
            resource = screen._resource_kind_data()
            assert resource.latest_sample_ts == latest  # the copy on screen stays
            assert not screen._resource_stale(resource, latest + stale_after + behind - second)
            assert screen._resource_stale(resource, latest + stale_after + behind + second)
            assert not screen._resource_stale_shown
            header = _header(screen)
            assert "refresh failed" in header and "no new GPU samples" not in header, header
            # It stays so while the ticks' refreshes fail, until that time comes.
            stage.errors.append(RuntimeError("stage unavailable"))
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 4)
            assert stage.calls[3] == [SAMPLING]
            assert screen._resource_error is not None and not screen._resource_stale_shown
            await _settle(app, pilot)

    try:
        asyncio.run(run())
    finally:
        stage.release_all()


def test_a_copy_the_running_download_leaves_out_is_not_said_to_be_refreshing():
    stage = _Stage()
    c = _client()
    c.download_metrics = stage.download_metrics

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            screen = await _resources(app, pilot)
            release = stage.hold(TRAINING)
            screen._auto_refresh_data()  # a tick downloads training alone
            assert await _wait(pilot, stage.waiting.is_set)
            assert "refreshing…" in _header(screen)
            # Sampling's copy, from the whole download just now, isn't due: nothing refreshes it.
            screen.action_resource_sub_job(1)
            assert screen._resource_source() == SAMPLING and screen._resource_loading
            assert "refreshing" not in _header(screen), _header(screen)
            release.set()
            assert await _landed(pilot, screen, stage.calls, 2)
            await _no_download(pilot, screen, stage.calls)
            assert stage.calls == [None, [TRAINING]]
            assert "refreshing" not in _header(screen)
            await _settle(app, pilot)

    try:
        asyncio.run(run())
    finally:
        stage.release_all()


# ─── Training metrics ────────────────────────────────────────────────────────


def _training_rows(sub_job_id, steps) -> str:
    """A ``/step`` row for each of ``steps``, as training.jsonl holds them."""
    return "".join(json.dumps({"kind": "training", "record_version": 2, "sub_job_id": sub_job_id,
                               "sample_ts": "2026-10-19T10:00:00Z", "seq": step, "seq_epoch": "e", "op": "step",
                               "step": step, "call": None, "metrics": {"grad_norm": 1.0 / step}}) + "\n"
                   for step in steps)


def _write_training_rows(output_dir, on_commit=None, sub_job_ids=(TRAINING,)) -> list:
    """Each of ``sub_job_ids``' training.jsonl, as download_training_metrics commits them."""
    entries = []
    for sub_job_id in sub_job_ids:
        path = Path(output_dir) / sub_job_id / "training.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _training_rows(sub_job_id, (1, 2)).encode()
        path.write_bytes(data)
        path.with_name(".training.jsonl.manifest.json").write_text(
            json.dumps({"version": 1, "names": ["_metrics/a.gz"], "bytes": len(data)}))
        entry = {"sub_job_id": sub_job_id, "saved_path": str(path)}
        if on_commit is not None:
            on_commit(dict(entry))
        entries.append(entry)
    return entries


async def _run_training_refreshes(c, calls, shown=(TRAINING,)):
    """First show, a live refresh, r, then a refresh after the job ended; ``calls()`` is the
    number of downloads so far, and ``shown`` the sub-jobs the tab has to show."""
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(120, 40)) as pilot:
        assert await _wait(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen._sources_ready)
        screen = app.screen
        await pilot.press("m")
        pane = screen.query_one("#training-tab")
        assert await _wait(pilot, lambda: pane._downloaded and not pane._loading and calls() == 1)
        assert pane._sub_job_ids() == list(shown)
        screen._auto_refresh_data()  # what the 30-second timer calls
        assert await _wait(pilot, lambda: calls() == 2 and not pane._loading)
        await pilot.press("r")
        assert await _wait(pilot, lambda: calls() == 3 and not pane._loading)
        _end(c, screen)
        screen._auto_refresh_data()
        pane.refresh_live()
        assert not pane._loading
        await pilot.pause(0.2)
        assert calls() == 3 and pane._error is None
        await _settle(app, pilot)


def test_the_training_tab_s_live_refresh_downloads_the_sub_job_shown_alone():
    calls = []

    def download_training_metrics(job_id, output_dir, *, resume=False, priority=(), on_commit=None,
                                  sub_job_ids=None, stale_after_seconds=600):
        calls.append(None if sub_job_ids is None else list(sub_job_ids))
        return _write_training_rows(output_dir, on_commit)

    c = _client()
    c.download_training_metrics = download_training_metrics
    asyncio.run(_run_training_refreshes(c, lambda: len(calls)))
    # The first show and r download the whole job; the live refresh, the tab's only sub-job.
    assert calls == [None, [TRAINING], None]


@pytest.mark.parametrize("second", ["7:training:1", SAMPLING])
def test_the_training_tab_s_live_refresh_downloads_every_sub_job_while_it_has_two_to_show(second):
    # A second training sub-job, or another sub-job with training rows: Space shows each as the
    # newest refresh left it, so each refresh downloads every sub-job.
    calls = []

    def download_training_metrics(job_id, output_dir, *, resume=False, priority=(), on_commit=None,
                                  sub_job_ids=None, stale_after_seconds=600):
        calls.append(None if sub_job_ids is None else list(sub_job_ids))
        return _write_training_rows(output_dir, on_commit, (TRAINING, second))

    c = _client(sub_job_ids=tuple(dict.fromkeys((TRAINING, second, SAMPLING))))
    c.download_training_metrics = download_training_metrics
    asyncio.run(_run_training_refreshes(c, lambda: len(calls), shown=(TRAINING, second)))
    assert calls == [None, None, None]


@pytest.mark.parametrize("kind", ["magicmock", "older-client"])
def test_the_training_tab_passes_no_sub_job_ids_to_a_client_that_takes_none(kind):
    c = _client()
    if kind == "magicmock":
        c.download_training_metrics.side_effect = lambda job_id, output_dir, **kwargs: _write_training_rows(
            output_dir, kwargs.get("on_commit"))
        asyncio.run(_run_training_refreshes(c, lambda: c.download_training_metrics.call_count))
        assert not any("sub_job_ids" in call.kwargs for call in c.download_training_metrics.call_args_list)
    else:
        calls = []

        def download_training_metrics(job_id, output_dir, *, resume=False, priority=(), on_commit=None):
            calls.append(job_id)  # a sub_job_ids it doesn't take would fail the refresh (pane._error)
            return _write_training_rows(output_dir, on_commit)

        c.download_training_metrics = download_training_metrics
        asyncio.run(_run_training_refreshes(c, lambda: len(calls)))


# ─── through the client ──────────────────────────────────────────────────────

JOB_TRAINING = "job-1:training:0"
JOB_SAMPLING = "job-1:sampling:0"


class _ClientStage:
    """The client's own downloads, over tests/test_client.py's fake stage connection.

    ``roots`` names the root each of a sub-job's chunks was saved under, oldest first (by
    default, three chunks under ``_metrics/``); they are a minute apart, of the ``stream``
    (``gpu`` or ``training``). Each download's ``sub_job_ids`` is recorded, and the rows each
    LIST returned; ``connection.commands`` has every statement run. A LIST of a location in
    ``refused`` (from the run's path) fails.
    """

    RUN_PATH = "/versions/RUN_ABC/"  # the start of a LIST row's name: the run's own path
    ROOTS = ("_metrics/", "checkpoints/_metrics/")

    def __init__(self, monkeypatch, sub_job_ids=(JOB_TRAINING, JOB_SAMPLING), roots=None, *, stream="gpu"):
        from tests.test_client import TestChunkedArtifactDownload

        fixture = TestChunkedArtifactDownload()
        self.client, self.connection, _paths, _sleeps, _opened = fixture._setup(
            monkeypatch, subs=list(sub_job_ids), count=1, artifact_name="metrics")
        self.run_uri, self.stream = fixture.RUN_URI, stream
        self.connection.list_rows = {f"{self.run_uri}{root}": [] for root in self.ROOTS}
        self.connection.content_by_uri = {}
        for sub_job_id in sub_job_ids:
            for index, root in enumerate((roots or {}).get(sub_job_id, ["_metrics"] * 3)):
                self.add(sub_job_id, index, root=root)
        self.calls: list = []
        self.listed: list = []
        self.refused: set[str] = set()
        listed = self.connection.listed

        def recording(uri):
            location = uri.removeprefix(self.run_uri)
            if location in self.refused:
                raise RuntimeError(f"LIST {location} denied")
            rows = listed(uri)
            self.listed.append(rows)
            return rows

        self.connection.listed = recording
        self.download_metrics = self.recorded(self.client.download_metrics)
        self.download_training_metrics = self.recorded(self.client.download_training_metrics)

    def recorded(self, method):
        """``method``, recording each call's ``sub_job_ids``."""
        @functools.wraps(method)  # so its signature, sub_job_ids and all, is the client's
        def download(*args, **kwargs):
            self.calls.append(kwargs.get("sub_job_ids"))
            return method(*args, **kwargs)

        return download

    def add(self, sub_job_id, index, *, root="_metrics") -> str:
        """Chunk ``index`` of ``sub_job_id``'s, saved under ``root``: the path it is fetched by."""
        path = f"{root}/{sub_job_id}/{self.stream}.20260904-12{index:02d}00.c{index:04d}.gz"
        rows = (_gpu_rows(sub_job_id, ago=30.0 + 60 * max(0, 3 - index)) if self.stream == "gpu"
                else _training_rows(sub_job_id, (index + 1,)))
        self.connection.list_rows[f"{self.run_uri}{root}/"].append((self.RUN_PATH + path,))
        self.connection.content_by_uri[self.run_uri + path] = gzip.compress(rows.encode())
        return path

    def remove(self, sub_job_id):
        """The stage keeps none of ``sub_job_id``'s chunks any more."""
        for rows in self.connection.list_rows.values():
            rows[:] = [row for row in rows if f"/{sub_job_id}/" not in row[0]]

    def mark(self) -> tuple[int, int]:
        return len(self.connection.commands), len(self.listed)

    def lists(self, mark=(0, 0)) -> list[str]:
        """The LIST statements run since ``mark``."""
        return [command for command in self.connection.commands[mark[0]:] if command.startswith("LIST ")]

    def gets(self, mark=(0, 0)) -> list[str]:
        return [command for command in self.connection.commands[mark[0]:] if command.startswith("GET ")]

    def rows(self, mark=(0, 0)) -> list[str]:
        """The names of the files the LISTs since ``mark`` returned."""
        return [row[0] for rows in self.listed[mark[1]:] for row in rows]

    def list_of(self, location) -> str:
        return f"LIST '{self.run_uri}{location}'"

    def both_roots(self, under="") -> list[str]:
        """The LIST of each root (of ``under`` in each: a sub-job's directory), sorted. A listing of
        both runs the two at once, so they reach the stage in either order: compare them sorted."""
        return sorted(self.list_of(root + under) for root in self.ROOTS)


def test_a_tick_lists_only_the_shown_sub_job_s_chunks(monkeypatch):
    # The acceptance check, through the client's own download: once the first, whole download
    # is in, a live job's tick LISTs the shown sub-job's directory alone (the checkpoints root is
    # skipped, the sub-job's root being known), and every row it returns is that sub-job's.
    stage = _ClientStage(monkeypatch)
    c = _client("job-1", (JOB_TRAINING, JOB_SAMPLING))
    c.download_metrics = stage.download_metrics

    async def run():
        app = CortexTrainingLogTUI(c, "job-1", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _resources(app, pilot, first=JOB_TRAINING)
            # The whole job: each root listed whole, which shows the client where each sub-job's files are.
            assert stage.calls == [None]
            assert sorted(stage.lists()) == stage.both_roots()
            assert len(stage.gets()) == 6
            assert screen._resource_with_chunks == {JOB_TRAINING, JOB_SAMPLING}
            # A minute later each sub-job has a new chunk. The tick lists training's directory alone ...
            new = {sub_job_id: stage.add(sub_job_id, 3) for sub_job_id in (JOB_TRAINING, JOB_SAMPLING)}
            newest = {sub_job_id: screen._resource_data.kind(sub_job_id, GPU).latest_sample_ts
                      for sub_job_id in (JOB_TRAINING, JOB_SAMPLING)}
            mark = stage.mark()
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 2)
            assert stage.calls[1] == [JOB_TRAINING]
            assert stage.lists(mark) == [stage.list_of(f"_metrics/{JOB_TRAINING}/")]
            rows = stage.rows(mark)
            assert len(rows) == 4 and all(row.startswith(f"{stage.RUN_PATH}_metrics/{JOB_TRAINING}/") for row in rows)
            # ... and fetches its new chunk, and nothing of sampling's.
            gets = stage.gets(mark)
            assert len(gets) == 1 and f"'{stage.run_uri}{new[JOB_TRAINING]}'" in gets[0], gets
            assert screen._resource_data.kind(JOB_TRAINING, GPU).latest_sample_ts > newest[JOB_TRAINING]
            assert screen._resource_data.kind(JOB_SAMPLING, GPU).latest_sample_ts == newest[JOB_SAMPLING]
            # Shown a refresh interval later, sampling (left behind by that tick) is listed and fetched alone too.
            _age(screen, _DUE)
            mark = stage.mark()
            screen.action_resource_sub_job(1)
            assert await _landed(pilot, screen, stage.calls, 3)
            assert stage.calls[2] == [JOB_SAMPLING]
            assert stage.lists(mark) == [stage.list_of(f"_metrics/{JOB_SAMPLING}/")]
            rows = stage.rows(mark)
            assert len(rows) == 4 and all(row.startswith(f"{stage.RUN_PATH}_metrics/{JOB_SAMPLING}/") for row in rows)
            gets = stage.gets(mark)
            assert len(gets) == 1 and f"'{stage.run_uri}{new[JOB_SAMPLING]}'" in gets[0], gets
            assert screen._resource_data.kind(JOB_SAMPLING, GPU).latest_sample_ts > newest[JOB_SAMPLING]
            assert screen._resource_error is None
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_sub_job_whose_chunks_are_gone_is_nothing_new_through_the_client(monkeypatch):
    # The stage keeps none of training's chunks after the first download: listed alone, under
    # its root and then the other, it has none, and the client raises NoChunksError, naming
    # (from one listing of the whole run) the sub-jobs that have chunks. The tab takes that for
    # nothing new, and its next tick is the whole job's.
    stage = _ClientStage(monkeypatch)
    raised = []

    @functools.wraps(stage.client.download_metrics)
    def download_metrics(*args, **kwargs):
        try:
            return stage.client.download_metrics(*args, **kwargs)
        except NoChunksError as exc:
            raised.append(exc)
            raise

    c = _client("job-1", (JOB_TRAINING, JOB_SAMPLING))
    c.download_metrics = stage.recorded(download_metrics)

    async def run():
        app = CortexTrainingLogTUI(c, "job-1", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _resources(app, pilot, first=JOB_TRAINING)
            stage.remove(JOB_TRAINING)
            mark = stage.mark()
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 2)
            assert stage.calls == [None, [JOB_TRAINING]]
            # Its root, then the other, one after the other; then both roots whole, at once.
            lists = stage.lists(mark)
            assert lists[:2] == [stage.list_of(f"_metrics/{JOB_TRAINING}/"),
                                 stage.list_of(f"checkpoints/_metrics/{JOB_TRAINING}/")]
            assert sorted(lists[2:]) == stage.both_roots()
            (error,) = raised
            assert (error.sub_job_ids, error.available) == ([JOB_TRAINING], [JOB_SAMPLING])
            header = _header(screen)
            assert "refresh failed" not in header and "unavailable" not in header, header
            assert screen._resource_error is None
            assert screen._resource_data.kind(JOB_TRAINING, GPU) is not None  # the copy on disk stays
            mark = stage.mark()
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 3)
            assert stage.calls[2] is None
            assert sorted(stage.lists(mark)) == stage.both_roots()
            assert screen._resource_with_chunks == {JOB_SAMPLING} and screen._resource_error is None
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_sub_job_whose_chunks_are_gone_fails_the_refresh_when_the_whole_listing_fails(monkeypatch):
    # As above, but the listing of the whole run that would name the sub-jobs with chunks fails:
    # the download raises its error, so the tab says the refresh failed rather than that nothing
    # is new, and learns nothing from it. Once the stage answers again, a tick finds nothing new.
    stage = _ClientStage(monkeypatch)
    c = _client("job-1", (JOB_TRAINING, JOB_SAMPLING))
    c.download_metrics = stage.download_metrics

    async def run():
        app = CortexTrainingLogTUI(c, "job-1", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            screen = await _resources(app, pilot, first=JOB_TRAINING)
            stage.remove(JOB_TRAINING)
            stage.refused.add("_metrics/")
            mark = stage.mark()
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 2)
            assert stage.calls == [None, [JOB_TRAINING]]
            lists = stage.lists(mark)
            assert lists[:2] == [stage.list_of(f"_metrics/{JOB_TRAINING}/"),
                                 stage.list_of(f"checkpoints/_metrics/{JOB_TRAINING}/")]
            assert sorted(lists[2:]) == stage.both_roots()
            assert "LIST _metrics/ denied" in (screen._resource_error or ""), screen._resource_error
            assert "refresh failed" in _header(screen)
            assert not screen._resource_refresh_succeeded
            assert screen._resource_data.kind(JOB_TRAINING, GPU) is not None  # the copy on disk stays
            assert screen._resource_with_chunks == {JOB_TRAINING, JOB_SAMPLING}
            assert JOB_TRAINING not in screen._resource_alone_at
            stage.refused.clear()
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 3)
            assert stage.calls[2] == [JOB_TRAINING]
            assert screen._resource_error is None and screen._resource_with_chunks == {JOB_SAMPLING}
            await _settle(app, pilot)

    asyncio.run(run())


def test_each_sub_job_alone_is_listed_under_the_roots_its_files_are_under(monkeypatch):
    # Sampling saved its chunks under checkpoints/_metrics/ alone: listed alone, only there.
    # Training saved one under each root: listed alone, under both, its chunks merged in time
    # order, as the whole job's download merges them, and the warning that says so stays.
    stage = _ClientStage(monkeypatch, roots={
        JOB_TRAINING: ["checkpoints/_metrics", "_metrics", "_metrics"],
        JOB_SAMPLING: ["checkpoints/_metrics"] * 3,
    })
    c = _client("job-1", (JOB_TRAINING, JOB_SAMPLING))
    c.download_metrics = stage.download_metrics
    merged = f"metrics chunks for {JOB_TRAINING} were saved under both _metrics/ and checkpoints/_metrics/"

    async def run():
        app = CortexTrainingLogTUI(c, "job-1", poll_interval=0.01)
        async with app.run_test(size=(220, 40)) as pilot:
            screen = await _resources(app, pilot, first=JOB_TRAINING)
            assert len(stage.gets()) == 6
            assert merged in " ".join(screen._metrics_warnings.get(JOB_TRAINING, []))
            mark = stage.mark()
            screen._auto_refresh_data()
            assert await _landed(pilot, screen, stage.calls, 2)
            assert stage.calls[1] == [JOB_TRAINING]
            assert sorted(stage.lists(mark)) == stage.both_roots(f"{JOB_TRAINING}/")  # both at once
            assert merged in " ".join(screen._metrics_warnings.get(JOB_TRAINING, []))
            assert "warning: " + merged in _header(screen)
            _age(screen, _DUE)
            mark = stage.mark()
            screen.action_resource_sub_job(1)
            assert await _landed(pilot, screen, stage.calls, 3)
            assert stage.calls[2] == [JOB_SAMPLING]
            assert stage.lists(mark) == [stage.list_of(f"checkpoints/_metrics/{JOB_SAMPLING}/")]
            assert stage.rows(mark) and all(
                row.startswith(f"{stage.RUN_PATH}checkpoints/_metrics/{JOB_SAMPLING}/") for row in stage.rows(mark))
            # Each sub-job's file has its three chunks, in time order.
            for sub_job_id in (JOB_TRAINING, JOB_SAMPLING):
                path = screen._metrics_root / sub_job_id / "gpu.jsonl"
                stamps = [json.loads(line)["sample_ts"] for line in path.read_text(encoding="utf-8").splitlines()]
                assert len(stamps) == 3 and stamps == sorted(stamps)
            assert screen._resource_error is None
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_training_tab_s_live_refresh_lists_only_its_sub_job_through_the_client(monkeypatch):
    # The training rows' chunks share _metrics/ with the GPU chunks: once the tab's first download
    # has listed both roots whole, its live refresh lists the training sub-job's directory alone.
    stage = _ClientStage(monkeypatch, stream="training", roots={JOB_TRAINING: ["_metrics"] * 2, JOB_SAMPLING: []})
    c = _client("job-1", (JOB_TRAINING, JOB_SAMPLING))
    c.download_training_metrics = stage.download_training_metrics

    async def run():
        app = CortexTrainingLogTUI(c, "job-1", poll_interval=0.01)
        async with app.run_test(size=(120, 40)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen._sources_ready)
            screen = app.screen
            await pilot.press("m")
            pane = screen.query_one("#training-tab")
            assert await _wait(pilot, lambda: pane._downloaded and not pane._loading)
            assert stage.calls == [None]
            assert sorted(stage.lists()) == stage.both_roots()
            new = stage.add(JOB_TRAINING, 2)
            mark = stage.mark()
            screen._auto_refresh_data()
            assert await _wait(pilot, lambda: len(stage.calls) == 2 and not pane._loading)
            assert stage.calls[1] == [JOB_TRAINING]
            assert stage.lists(mark) == [stage.list_of(f"_metrics/{JOB_TRAINING}/")]
            gets = stage.gets(mark)
            assert len(gets) == 1 and f"'{stage.run_uri}{new}'" in gets[0], gets
            assert await _wait(pilot, lambda: "step 3" in str(screen.query_one("#training-status", Static).render()))
            assert pane._error is None
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_training_tab_s_live_refresh_lists_each_root_whole_with_two_sub_jobs_through_the_client(monkeypatch):
    # With two training sub-jobs to show, a live refresh downloads both: it lists each root whole,
    # as the tab's first download did, and fetches each one's new chunk.
    trainer = "job-1:training:1"
    sub_job_ids = (JOB_TRAINING, trainer, JOB_SAMPLING)
    stage = _ClientStage(monkeypatch, sub_job_ids, stream="training",
                         roots={JOB_TRAINING: ["_metrics"] * 2, trainer: ["_metrics"] * 2, JOB_SAMPLING: []})
    c = _client("job-1", sub_job_ids)
    c.download_training_metrics = stage.download_training_metrics

    async def run():
        app = CortexTrainingLogTUI(c, "job-1", poll_interval=0.01)
        async with app.run_test(size=(120, 40)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen._sources_ready)
            screen = app.screen
            await pilot.press("m")
            pane = screen.query_one("#training-tab")
            assert await _wait(pilot, lambda: pane._downloaded and not pane._loading)
            assert pane._sub_job_ids() == [JOB_TRAINING, trainer]
            assert stage.calls == [None]
            new = [stage.add(sub_job_id, 2) for sub_job_id in (JOB_TRAINING, trainer)]
            mark = stage.mark()
            screen._auto_refresh_data()
            assert await _wait(pilot, lambda: len(stage.calls) == 2 and not pane._loading)
            assert stage.calls == [None, None]
            assert sorted(stage.lists(mark)) == stage.both_roots()
            gets = stage.gets(mark)
            assert len(gets) == 2 and all(any(f"'{stage.run_uri}{path}'" in get for get in gets) for path in new), gets
            assert pane._error is None
            await _settle(app, pilot)

    asyncio.run(run())


def test_r_during_the_training_tab_s_live_refresh_of_its_sub_job_fetches_every_sub_job_next(monkeypatch):
    # r while a live refresh downloads the tab's one sub-job alone: once that ends, every
    # sub-job's chunks are fetched, as r promises, rather than the r being dropped.
    stage = _ClientStage(monkeypatch, stream="training", roots={JOB_TRAINING: ["_metrics"] * 2, JOB_SAMPLING: []})
    c = _client("job-1", (JOB_TRAINING, JOB_SAMPLING))
    c.download_training_metrics = stage.download_training_metrics
    held, release = threading.Event(), threading.Event()
    listed = stage.connection.listed

    def holding(uri):
        if uri.endswith(f"/{JOB_TRAINING}/"):  # the live refresh's LIST of the sub-job alone
            held.set()
            release.wait(10)
        return listed(uri)

    stage.connection.listed = holding

    async def run():
        app = CortexTrainingLogTUI(c, "job-1", poll_interval=0.01)
        async with app.run_test(size=(120, 40)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen._sources_ready)
            screen = app.screen
            await pilot.press("m")
            pane = screen.query_one("#training-tab")
            assert await _wait(pilot, lambda: pane._downloaded and not pane._loading)
            screen._auto_refresh_data()
            assert await _wait(pilot, held.is_set)
            assert pane._loading and pane._loading_scope == JOB_TRAINING
            await pilot.press("r")
            assert pane._whole_pending
            release.set()
            assert await _wait(pilot, lambda: len(stage.calls) == 3 and not pane._loading)
            assert stage.calls == [None, [JOB_TRAINING], None]
            assert not pane._whole_pending and pane._error is None
            await _settle(app, pilot)

    try:
        asyncio.run(run())
    finally:
        release.set()


def test_a_training_refresh_started_as_the_last_one_s_worker_ends_keeps_the_tab_loading(monkeypatch):
    # The last refresh's worker, slow to return once it has posted _finish, is cancelled as the
    # next refresh starts: the tab stays loading until that one's download is in.
    from cortex_training.tui import detail_tabs

    stage = _ClientStage(monkeypatch, stream="training", roots={JOB_TRAINING: ["_metrics"] * 2, JOB_SAMPLING: []})
    c = _client("job-1", (JOB_TRAINING, JOB_SAMPLING))
    c.download_training_metrics = stage.download_training_metrics
    post, lingered, slow = detail_tabs._Pane._post, [], threading.Event()

    def lingering(self, fn, *args):
        post(self, fn, *args)
        if getattr(fn, "__name__", "") == "_finish" and not lingered:
            lingered.append(fn)
            time.sleep(0.5)  # the first refresh's worker, just after posting _finish

    monkeypatch.setattr(detail_tabs._Pane, "_post", lingering)
    listed = stage.connection.listed

    def slowly(uri):
        if slow.is_set():
            time.sleep(1.0)  # the next refresh's LIST
        return listed(uri)

    stage.connection.listed = slowly

    async def run():
        app = CortexTrainingLogTUI(c, "job-1", poll_interval=0.01)
        async with app.run_test(size=(120, 40)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen._sources_ready)
            screen = app.screen
            await pilot.press("m")
            pane = screen.query_one("#training-tab")
            assert await _wait(pilot, lambda: pane._downloaded and not pane._loading)
            stage.add(JOB_TRAINING, 2)
            mark = stage.mark()
            slow.set()
            screen._auto_refresh_data()  # while the first refresh's worker lingers
            assert await _wait(pilot, lambda: len(stage.calls) == 2 and not pane._loading)
            assert len(stage.gets(mark)) == 1  # its download was in before the tab stopped loading
            assert pane._error is None
            await _settle(app, pilot)

    asyncio.run(run())
