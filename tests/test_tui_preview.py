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

"""The job preview's pieces: its facts, elapsed time, lifecycle and callout, the
sub-job table's status badges, and the dashboard's identity line."""

from __future__ import annotations

import asyncio
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from textual.content import Content  # noqa: E402
from textual.widgets import Static  # noqa: E402

from cortex_training.tui import views as v  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.screens import JobSummaryScreen  # noqa: E402
from cortex_training.tui.screens import status_text  # noqa: E402
from cortex_training.tui.widgets import fit_line  # noqa: E402

NOW = datetime(2026, 7, 20, 18, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _job(status="RUNNING", *, created=timedelta(hours=2), changed=timedelta(minutes=30), **extra) -> dict:
    job = {"job_id": "JOB_ID", "status": status, "sub_jobs": []}
    if created is not None:
        job["created_at"] = _iso(NOW - created)
    if changed is not None:
        job["state_changed_at"] = _iso(NOW - changed)
    return {**job, **extra}


# ─── facts and elapsed time ──────────────────────────────────────────────────


def test_preview_facts_leave_the_hero_and_lifecycle_out():
    labels = [label for label, _value in v.preview_facts(_job(), NOW)]
    assert labels == ["Model", "Type", "GPUs", "Owner", "Created", "Last update"]
    located = _job(database_name="CORTEX_TRAINING_DB", schema_name="PUBLIC")
    assert [label for label, _value in v.preview_facts(located, NOW)][-1] == "Schema"


def test_subjob_config_summary_is_the_submitted_config():
    sj = {"job_type": "training", "model_name": "Qwen/Qwen3-8B", "dtype": "bfloat16",
          "training_config": {"n_gpus": 8.0, "max_seq_len": 4096.0, "save_training_metrics": False,
                              "optimizer": {"name": "adamw", "lr": 1e-5, "betas": [0.9, 0.95]}}}
    assert v.subjob_config_summary(sj) == ("max_seq_len=4096 · n_gpus=8 · "
                                           "optimizer={betas: [0.9, 0.95], lr: 1e-05, name: adamw} · "
                                           "save_training_metrics=false")
    # Nothing derived is added: no "full-param" for a training sub-job without peft_config.
    assert "full-param" not in v.subjob_config_summary(sj)
    assert v.subjob_config_summary({"job_type": "sampling",
                                    "inference_config": {"n_gpus": 2.0, "gpu_memory_utilization": 0.9}}) \
        == "gpu_memory_utilization=0.9 · n_gpus=2"
    assert v.subjob_config_summary({"job_type": "training"}) == ""


@pytest.mark.parametrize("config, shown", [
    # One line whatever the keys or values hold, nested ones too.
    ({"a\nb": 1, "n": {"x\ny": 2}}, "a b=1 · n={x y: 2}"),
    ({"u\u2028v": "p\u2029q"}, "u v=p q"),
    ({"s": "  two   words\t"}, "s=two words"),
    ({"bell": "a\x07b"}, "bell=a\ufffdb"),
    # Empty and null values as JSON writes them.
    ({"n": None, "d": {}, "l": []}, "d={} · l=[] · n=null"),
    ({"layers": [{"r": 8.0, "name": "q"}, {"r": 16.0}]}, "layers=[{name: q, r: 8}, {r: 16}]"),
    # A whole float as an int, unless that would be a 21-digit number.
    ({"big": 1e20, "neg": -2.0, "frac": 0.5}, "big=1e+20 · frac=0.5 · neg=-2"),
])
def test_subjob_config_summary_renders_each_value(config, shown):
    summary = v.subjob_config_summary({"job_type": "training", "training_config": config})
    assert summary == shown
    assert "\n" not in summary and "\u2028" not in summary


def test_multiplexed_sub_job_names_its_source_first():
    target = "JOB_ID:training:0"
    shared = {"sub_job_id": "JOB_ID:sampling:0", "job_type": "sampling", "status": "RUNNING",
              "inference_config": {"max_seq_len": 4096.0, "n_gpus": 2.0, "multiplex_job_id": target}}
    assert v.subjob_config_summary(shared) == f"multiplex_job_id={target} · max_seq_len=4096 · n_gpus=2"
    job = _job(sub_jobs=[{"sub_job_id": target, "job_type": "training", "status": "RUNNING",
                          "training_config": {"n_gpus": 4.0}}, shared])
    rows = dict(JobSummaryScreen(MagicMock(), "JOB_ID", job=job)._subjob_rows(job))
    assert rows["sj-0"][-1].plain == "n_gpus=4"
    # The table's cell is the config itself: no note added after it.
    assert rows["sj-1"][-1].plain == f"multiplex_job_id={target} · max_seq_len=4096 · n_gpus=2"


def test_elapsed_text_runs_from_creation():
    assert v.elapsed_text(_job("RUNNING"), NOW) == "2h 00m so far"
    # Ended: to when it entered its last state, and "lasted", since it may never have run.
    assert v.elapsed_text(_job("FAILED"), NOW) == "lasted 1h 30m"
    assert v.elapsed_text(_job("RUNNING", created=None), NOW) == ""


def test_short_time_drops_the_date_only_on_the_same_day():
    assert v.short_time(_iso(NOW), NOW) == v.format_created_at(_iso(NOW))[11:]
    assert v.short_time(_iso(NOW - timedelta(days=3)), NOW) == v.format_created_at(_iso(NOW - timedelta(days=3)))
    assert v.short_time(None, NOW) == "—"


# ─── lifecycle ───────────────────────────────────────────────────────────────


def test_lifecycle_of_a_live_job_runs_to_now():
    steps = v.lifecycle(_job("RUNNING"), NOW)
    assert [(s.status, s.label) for s in steps] == [("created", "Created"), ("running", "Running since"),
                                                    ("now", "now")]
    assert steps[0].then == "1h 30m"  # its time pending and placing, which has no step of its own
    assert steps[1].then == "30m 00s so far"


def test_lifecycle_of_an_ended_job_says_how_long_ago_it_ended():
    steps = v.lifecycle(_job("FAILED"), NOW)
    assert [s.label for s in steps] == ["Created", "Failed"]
    assert steps[-1].at.endswith("(30m 00s ago)") and not steps[-1].then


def test_lifecycle_invents_no_time_the_record_lacks():
    # No state_changed_at: the current state has no time, and no duration leads to it.
    steps = v.lifecycle(_job("PENDING", changed=None), NOW)
    assert [(s.label, s.at, s.then) for s in steps] == [("Created", v.short_time(_iso(NOW - timedelta(hours=2)), NOW),
                                                         ""), ("Pending", "", "")]
    # UNKNOWN: the service couldn't read the state, so it has no time either.
    steps = v.lifecycle(_job("UNKNOWN"), NOW)
    assert [s.label for s in steps] == ["Created", "status unavailable"] and not steps[0].then
    assert [s.label for s in v.lifecycle(_job("RUNNING", created=None), NOW)] == ["Running since", "now"]
    # A live state entered at creation is one step; an ended one has no duration before it.
    assert [s.label for s in v.lifecycle(_job("PENDING", changed=timedelta(hours=2)), NOW)] == [
        "Created, pending since", "now"]
    assert v.lifecycle(_job("FAILED", changed=timedelta(hours=2)), NOW)[0].then == ""


# ─── callout ─────────────────────────────────────────────────────────────────


def test_callout_for_a_failure_always_has_a_reason():
    callout = v.health_callout(_job("FAILED"))
    assert (callout.tone, callout.title, callout.reason) == ("failed", "Why it failed", "not reported")
    assert "Logs" in callout.hint


def test_callout_for_a_job_that_never_got_gpus_points_away_from_logs():
    callout = v.health_callout(_job("FAILED", reason="placement_timeout"))
    assert callout.title == "Failed before it got GPUs" and "never got GPUs" in callout.hint
    cancelled = v.health_callout(_job("CANCELLED", reason="cancelled_by_client",
                                      sub_jobs=[{"sub_job_id": "JOB_ID:training:0", "status": "PENDING"}]))
    assert cancelled.tone == "ended" and "never got GPUs" in cancelled.hint


def test_callout_for_a_waiting_job_says_why_or_what_the_state_means():
    callout = v.health_callout(_job("PENDING", reason="quota_filled", reason_details="needs 16 GPUs"))
    assert (callout.tone, callout.reason, callout.details) == ("waiting", "quota_filled", "needs 16 GPUs")
    assert v.health_callout(_job("PLACING")).details == v.WAITING_TEXT["placing"]
    assert v.health_callout(_job("INITIALIZING")).tone == "starting"


def test_callout_for_running_and_unknown_jobs():
    assert v.health_callout(_job("RUNNING")) is None  # the status pill says it all
    assert v.health_callout(_job("RUNNING", reason_details="slow node")).details == "slow node"
    unknown = v.health_callout(_job("UNKNOWN", reason="Job status unavailable: Control Plane is unavailable"))
    assert unknown.tone == "unknown" and unknown.details.startswith("Job status unavailable")
    assert unknown.hint == "The job may still be running."
    gone = v.health_callout(_job("UNKNOWN", reason="Job status unavailable: job not found on Control Plane"))
    assert gone.hint == ""


# ─── the sub-job table and the dashboard's identity line ─────────────────────


def test_status_text_is_a_badge():
    assert status_text("JOB_STATE_RUNNING").plain == "● RUNNING"
    assert status_text("FAILED").plain == "✕ FAILED"
    assert status_text("").plain == "?"


def test_fit_line_draws_chips_and_keeps_content_whole():
    badge = Content("● RUNNING")
    line = fit_line([("name", "bold", None, False), badge, ("train", "", None, True)], 80)
    assert line.plain == "name   ● RUNNING    train "


def test_fit_line_cuts_what_may_be_cut_to_stay_on_one_row():
    segments = [("a long job name", "", 4, False), ("model", "", None, True)]
    line = fit_line(segments, 16)
    assert "\n" not in line.plain and line.plain.endswith(" model ") and "…" in line.plain
    # Too narrow even cut: nothing is cut, and the line wraps between segments.
    assert fit_line(segments, 6).plain.split("\n") == ["a long job name", " model "]


# ─── the preview on screen ───────────────────────────────────────────────────


def _rows(app, selector: str) -> list:
    widget = app.screen.query_one(selector, Static)
    return [strip.text for strip in widget.render_lines(widget.content_region.reset_offset)]


def test_preview_shows_cards_with_the_id_facts_lifecycle_and_callout():
    job = _job("PENDING", reason="quota_filled", reason_details="needs 16 GPUs", hardware="H200",
               sub_jobs=[{"sub_job_id": "JOB_ID:training:0", "job_type": "training", "status": "PENDING",
                          "model_name": "org/model", "training_config": {"n_gpus": 16}}])
    client = MagicMock()
    client.list_jobs.return_value = [job]
    client.get_job.return_value = job

    async def run():
        app = CortexTrainingLogTUI(client, poll_interval=0.01)
        async with app.run_test(size=(150, 42)) as pilot:
            app.push_screen(JobSummaryScreen(client, "JOB_ID", job=job))
            for _ in range(50):
                await pilot.pause(0.02)
                if isinstance(app.screen, JobSummaryScreen) and "Lifecycle" in "".join(_rows(app, "#preview")):
                    break
            rows = [row.rstrip() for row in _rows(app, "#preview")]
            # Two columns of facts at this width, and the lifecycle under them.
            assert any(row.startswith("Model") and "Owner" in row for row in rows)
            assert any(row.startswith("Lifecycle") and "Pending since" in row for row in rows)
            hero = _rows(app, "#hero")
            assert hero[0].startswith("JOB_ID") and "PENDING" in hero[0] and "so far" in hero[0]
            assert "l Logs" in hero[-1] and "g Resources" in hero[-1]
            health = app.screen.query_one("#health", Static)
            assert "quota_filled" in "".join(_rows(app, "#health")) and "waiting" in str(health.border_title)
            assert "opens its Logs" in str(app.screen.query_one("#subjobs-card").border_subtitle)
            app.workers.cancel_all()
            await pilot.pause(0.1)

    asyncio.run(run())


def test_preview_of_a_running_job_has_no_callout_and_one_column_when_narrow():
    job = _job("RUNNING")
    client = MagicMock()
    client.list_jobs.return_value = [job]
    client.get_job.return_value = job

    async def run():
        app = CortexTrainingLogTUI(client, poll_interval=0.01)
        async with app.run_test(size=(80, 30)) as pilot:
            app.push_screen(JobSummaryScreen(client, "JOB_ID", job=job))
            for _ in range(50):
                await pilot.pause(0.02)
                if isinstance(app.screen, JobSummaryScreen) and "Lifecycle" in "".join(_rows(app, "#preview")):
                    break
            rows = [row.rstrip() for row in _rows(app, "#preview")]
            assert not any(row.startswith("Model") and "Owner" in row for row in rows)
            assert not app.screen.query_one("#health", Static).display
            app.workers.cancel_all()
            await pilot.pause(0.1)

    asyncio.run(run())
