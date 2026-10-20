# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The job screen's Training metrics and Checkpoints tabs (``tui/detail_tabs.py``), in the app."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from textual.widgets import DataTable  # noqa: E402
from textual.widgets import Static  # noqa: E402
from textual.widgets import TabbedContent  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.logs import LogScreen  # noqa: E402

TRAINING = "7:training:0"


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "cache"))
    import cortex_training.tui.widgets as widgets_module

    monkeypatch.setattr(widgets_module, "_ERROR_LOG", str(tmp_path / "errors.log"))


def _row(step, op="fwd_bwd", **metrics):
    return {"kind": "training", "record_version": 1, "sub_job_id": TRAINING, "sample_ts": "2026-10-19T10:00:00Z",
            "seq": step, "seq_epoch": "e", "op": op, "step": step, "metrics": metrics}


def _download(rows):
    """A ``download_training_metrics`` that commits ``rows`` for the training sub-job."""

    def download(job_id, output_dir, *, resume=False, priority=(), on_commit=None, **_kwargs):
        path = Path(output_dir) / TRAINING / "training.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = "".join(json.dumps(row) + "\n" for row in rows).encode()
        path.write_bytes(data)
        path.with_name(".training.jsonl.manifest.json").write_text(
            json.dumps({"version": 1, "names": ["_metrics/a.gz"], "bytes": len(data)}))
        if on_commit is not None:
            on_commit({"sub_job_id": TRAINING, "saved_path": str(path)})
        return [{"sub_job_id": TRAINING, "saved_path": str(path)}]

    return download


def _client(sub_jobs=None, status="RUNNING"):
    c = MagicMock()
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = []
    c.get_job.return_value = {
        "job_id": "7",
        "status": status,
        "sub_jobs": sub_jobs if sub_jobs is not None else [
            {"sub_job_id": TRAINING, "job_type": "training", "status": status},
            {"sub_job_id": "7:sampling:0", "job_type": "sampling", "status": status},
        ],
    }
    c.tail_logs.return_value = {"entries": [], "next_cursor": "", "eof": True}
    c.download_training_metrics.side_effect = _download([
        _row(1, sum_loss=4.0, num_calls=2.0), _row(1, op="step", grad_norm=0.9, last_lr=1e-5),
        _row(2, sum_loss=2.0, num_calls=2.0), _row(2, op="step", grad_norm=0.5, last_lr=1e-5),
    ])
    c.list_checkpoints.return_value = [
        {"checkpoint_id": "global_step1", "global_steps": 1, "avg_loss": 2.0, "created_at": "2026-10-19T10:00:00Z",
         "checkpoint_type": "resumable"},
        {"checkpoint_id": "global_step2", "global_steps": 2, "avg_loss": 1.0, "created_at": "2026-10-19T10:05:00Z",
         "checkpoint_type": "weights-only"},
    ]
    return c


async def _wait(pilot, predicate, tries=200):
    for _ in range(tries):
        await pilot.pause(0.01)
        if predicate():
            return True
    return False


async def _settle(app, pilot):
    app.workers.cancel_all()
    for _ in range(5):
        await pilot.pause()


def _text(screen, selector) -> str:
    return str(screen.query_one(selector, Static).render())


async def _open(app, pilot) -> LogScreen:
    assert await _wait(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen._sources_ready)
    return app.screen


def test_training_tab_downloads_and_draws_each_metric():
    async def run():
        c = _client()
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(app, pilot)
            assert not c.download_training_metrics.called  # lazy: only once the tab is shown
            await pilot.press("m")
            pane = screen.query_one("#training-tab")
            assert await _wait(pilot, lambda: pane._downloaded and not pane._loading)
            assert await _wait(pilot, lambda: screen.query_one("#job-tabs", TabbedContent).active == "training")
            assert c.download_training_metrics.call_args.kwargs["priority"] == [TRAINING]
            status = _text(screen, "#training-status")
            assert "training #0" in status and "step 2" in status and "all steps (w)" in status
            stats = _text(screen, "#training-stats")
            assert [line.split()[0] for line in stats.splitlines()[1:]] == \
                ["mean_loss", "sum_loss", "grad_norm", "last_lr"]
            assert "1 @ 2" in stats  # the derived mean loss at step 2: 2.0 / 2 calls
            charts = _text(screen, "#training-charts")
            # num_calls is the mean's divisor, not a chart of its own.
            assert "mean_loss · all steps" in charts and "num_calls" not in charts
            await pilot.press("w")
            assert "last 100 steps (w)" in _text(screen, "#training-status")
            await _settle(app, pilot)

    asyncio.run(run())


def test_training_tab_download_failure_is_said_in_its_header():
    async def run():
        c = _client()
        c.download_training_metrics.side_effect = RuntimeError("stage [/] gone")
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(app, pilot)
            await pilot.press("m")
            pane = screen.query_one("#training-tab")
            assert await _wait(pilot, lambda: pane._error is not None and not pane._loading)
            assert "training metrics unavailable: RuntimeError: stage [/] gone" in _text(screen, "#training-status")
            await _settle(app, pilot)

    asyncio.run(run())


def test_training_tab_of_a_job_without_training_downloads_nothing():
    async def run():
        c = _client(sub_jobs=[{"sub_job_id": "7:sampling:0", "job_type": "sampling", "status": "RUNNING"}])
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(app, pilot)
            await pilot.press("m")
            assert await _wait(pilot, lambda: "no training sub-job" in _text(screen, "#training-stats"))
            assert not c.download_training_metrics.called
            await _settle(app, pilot)

    asyncio.run(run())


def test_t_cycles_through_every_tab():
    async def run():
        app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(app, pilot)
            tabs = screen.query_one("#job-tabs", TabbedContent)
            seen = []
            for _ in range(4):
                await pilot.press("t")
                await pilot.pause()
                seen.append(tabs.active)
            assert seen == ["resource", "training", "checkpoints", "logs"]
            await _settle(app, pilot)

    asyncio.run(run())


def test_opens_on_a_new_tab_when_asked():
    async def run():
        c = _client()
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open(app, pilot)
            app.push_screen(LogScreen(c, "7", initial_tab="training", poll_interval=0.01))
            assert await _wait(pilot, lambda: app.screen.query_one("#training-tab")._downloaded)
            assert app.screen.query_one("#job-tabs", TabbedContent).active == "training"
            await _settle(app, pilot)

    asyncio.run(run())
