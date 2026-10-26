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
import json
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


def test_subjob_config_json_is_what_ct_get_prints():
    sj = {"sub_job_id": "JOB_ID:training:0", "status": "RUNNING", "job_type": "TRAINING",
          "model_name": "Qwen/Qwen3-8B", "dtype": "bfloat16", "seed": 42.0,
          "training_config": {"n_gpus": 8.0, "max_seq_len": 4096.0, "gradient_checkpointing": True,
                              "optimizer": {"type": "AdamW", "params": {"lr": 1e-5, "beta2": 0.95}}}}
    # The entry as `cortex-training get` prints it (indent 2, by key, values as
    # returned, aliases as submitted), without the fields the service adds.
    entry = {key: value for key, value in sj.items() if key not in ("sub_job_id", "status")}
    assert v.subjob_config_json(sj) == json.dumps(entry, indent=2, sort_keys=True).splitlines()
    shown = "\n".join(v.subjob_config_json(sj))
    assert '"n_gpus": 8.0' in shown and '"gradient_checkpointing": true' in shown and '"type": "AdamW"' in shown
    assert "sub_job_id" not in shown and "RUNNING" not in shown
    assert v.subjob_config_json({"sub_job_id": "JOB_ID:training:0", "status": "RUNNING"}) == ["{}"]


def test_subjob_config_json_escapes_what_a_terminal_would_act_on():
    lines = v.subjob_config_json({"job_type": "training", "training_config": {"a\nb": "x\x1b[2Jy\u2028z"}})
    assert len(lines) == 6 and not any(ch in "".join(lines) for ch in "\n\x1b\u2028")


def test_multiplexed_sub_job_shows_its_source():
    target = "JOB_ID:training:0"
    shared = {"sub_job_id": "JOB_ID:sampling:0", "job_type": "sampling", "status": "RUNNING",
              "inference_config": {"max_seq_len": 4096.0, "n_gpus": 2.0, "multiplex_job_id": target}}
    assert f'    "multiplex_job_id": "{target}",' in v.subjob_config_json(shared)
    job = _job(sub_jobs=[{"sub_job_id": target, "job_type": "training", "status": "RUNNING",
                          "training_config": {"n_gpus": 4.0}}, shared])
    rows = dict(JobSummaryScreen(MagicMock(), "JOB_ID", job=job)._subjob_rows(job))
    # The table has no Config column: the config card shows it.
    assert len(rows["sj-1"]) == 5 and "multiplex_job_id" not in "".join(c.plain for c in rows["sj-1"])


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


def test_config_card_follows_the_highlighted_row():
    job = _job("RUNNING", sub_jobs=[
        {"sub_job_id": "JOB_ID:training:0", "job_type": "training", "status": "RUNNING", "model_name": "org/model",
         "training_config": {"n_gpus": 8.0, "optimizer": {"name": "adamw", "lr": 1e-5}}},
        {"sub_job_id": "JOB_ID:sampling:0", "job_type": "sampling", "status": "RUNNING", "model_name": "org/model",
         "inference_config": {"n_gpus": 2.0, "max_seq_len": 4096.0}}])
    client = MagicMock()
    client.list_jobs.return_value = [job]
    client.get_job.return_value = job

    async def run():
        app = CortexTrainingLogTUI(client, poll_interval=0.01)
        async with app.run_test(size=(150, 42)) as pilot:
            app.push_screen(JobSummaryScreen(client, "JOB_ID", job=job))
            for _ in range(50):
                await pilot.pause(0.02)
                if isinstance(app.screen, JobSummaryScreen) and '"lr": 1e-05' in "".join(_rows(app, "#config")):
                    break
            card = app.screen.query_one("#config-card")
            # The job's row: each sub-job's config as JSON under its name.
            text = "\n".join(_rows(app, "#config"))
            assert "every sub-job" in str(card.border_title) and "as cortex-training get shows it" in str(card.border_subtitle)
            # Side by side: both sub-jobs' names on the first row.
            assert "training #0" in text.splitlines()[0] and "sampling #0" in text.splitlines()[0]
            assert '"max_seq_len": 4096.0' in text and '"lr": 1e-05' in text
            await pilot.press("down")
            await pilot.pause(0.1)
            text = "\n".join(_rows(app, "#config"))
            assert "training #0" in str(card.border_title)
            assert '"name": "adamw"' in text and "max_seq_len" not in text
            await pilot.press("down")
            await pilot.pause(0.1)
            assert "sampling #0" in str(card.border_title) and "optimizer" not in "\n".join(_rows(app, "#config"))
            app.workers.cancel_all()
            await pilot.pause(0.1)

    asyncio.run(run())

