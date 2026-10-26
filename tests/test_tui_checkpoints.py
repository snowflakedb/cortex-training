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

"""The Checkpoints tab: its data helpers and the tab on the job details screen."""

import asyncio
import contextlib
import errno
import json
import shlex
import threading
import time
from unittest.mock import MagicMock

import pytest
import requests

pytest.importorskip("textual")

from textual.content import Content  # noqa: E402
from textual.widgets import DataTable  # noqa: E402
from textual.widgets import Static  # noqa: E402
from textual.widgets import TabbedContent  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.checkpoints import RESUMABLE  # noqa: E402
from cortex_training.tui.checkpoints import WEIGHTS_ONLY  # noqa: E402
from cortex_training.tui.checkpoints import DownloadCancelled  # noqa: E402
from cortex_training.tui.checkpoints import compact_numbers  # noqa: E402
from cortex_training.tui.checkpoints import download_checkpoint  # noqa: E402
from cortex_training.tui.checkpoints import download_export  # noqa: E402
from cortex_training.tui.checkpoints import new_job_body  # noqa: E402
from cortex_training.tui.checkpoints import use_plan  # noqa: E402
from cortex_training.tui.checkpoints import files_info  # noqa: E402
from cortex_training.tui.checkpoints import fmt_bytes  # noqa: E402
from cortex_training.tui.checkpoints import parse_checkpoints  # noqa: E402
from cortex_training.tui.checkpoints import saves_no_checkpoints  # noqa: E402
from cortex_training.tui.checkpoints import sort_checkpoints  # noqa: E402
from tests.tui_fake import FakeClient  # noqa: E402
from cortex_training.tui.logs import LogScreen  # noqa: E402
from cortex_training.tui.screens import JobSummaryScreen  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "isolated-cache"))
    import cortex_training.tui.widgets as widgets

    monkeypatch.setattr(widgets, "_ERROR_LOG", str(tmp_path / "errors.log"))


_LISTED = [
    {"checkpoint_id": "cp_a", "created_at": "2026-07-20T18:05:00.123456789Z"},
    {"checkpoint_id": "cp_b", "created_at": "2026-07-20T19:05:00Z", "checkpoint_type": "weights-only"},
    {"checkpoint_id": "global_step12", "global_steps": 12, "avg_loss": 0.83,
     "created_at": "2026-07-20T17:05:00Z", "checkpoint_type": "RESUMABLE", "note": {"x": 1}},
    {"created_at": "2026-07-20T17:00:00Z"},  # no id: left out
    "not a dict",
]

_RESUMABLE_FILES = {"checkpoint_id": "cp_a", "expires_at": "2026-07-20T20:00:00Z", "files": [
    {"filename": "global_step10/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt", "size_bytes": "5161740657",
     "url": "https://links.example/secret-a"},
    {"filename": "global_step10/mp_rank_00_model_states.pt", "size_bytes": "3441239525",
     "url": "https://links.example/secret-b"},
    {"filename": "metadata.json", "size_bytes": "298", "url": "https://links.example/secret-c"},
]}
_WEIGHTS_FILES = {"checkpoint_id": "cp_b", "files": [
    {"filename": "global_step8/config.json", "size_bytes": "1418", "url": "https://links.example/secret-d"},
    {"filename": "global_step8/model.safetensors", "size_bytes": "1192135096", "url": "https://links.example/x"},
]}


def test_parse_checkpoints_reads_fields_and_skips_bad_entries():
    rows = parse_checkpoints(_LISTED)
    assert [r.checkpoint_id for r in rows] == ["cp_a", "cp_b", "global_step12"]
    a, b, c = rows
    assert a.created is not None and a.step is None and a.kind is None and a.loss is None
    assert b.kind == WEIGHTS_ONLY
    assert c.step == 12 and c.kind == RESUMABLE and c.loss == pytest.approx(0.83)
    assert c.extra == {"note": {"x": 1}}
    assert parse_checkpoints(None) == [] and parse_checkpoints({"checkpoints": []}) == []


def test_parse_checkpoints_tolerates_odd_values_and_hides_link_fields():
    rows = parse_checkpoints([{"checkpoint_id": "x", "global_steps": "²", "avg_loss": "n/a",
                               "download_url": "https://links.example/secret", "stage_info": {"k": "v"},
                               "epoch": 3}])
    assert rows[0].step is None and rows[0].loss is None
    assert rows[0].extra == {"epoch": 3}


def test_files_info_infers_step_and_type_and_drops_links():
    info = files_info(_RESUMABLE_FILES)
    assert info.step == 10 and info.kind == RESUMABLE
    assert info.total_bytes == 5161740657 + 3441239525 + 298
    assert "secret" not in repr(info)
    weights = files_info(_WEIGHTS_FILES)
    assert weights.step == 8 and weights.kind == WEIGHTS_ONLY
    assert files_info({"files": [{"filename": "metadata.json"}]}).kind is None
    assert files_info("nonsense").files == []


def test_sort_checkpoints_by_created_step_and_type():
    rows = parse_checkpoints(_LISTED)
    assert [r.checkpoint_id for r in sort_checkpoints(rows)] == ["cp_b", "cp_a", "global_step12"]
    assert [r.checkpoint_id for r in sort_checkpoints(rows, descending=False)] == ["global_step12", "cp_a", "cp_b"]
    # Without the files, only global_step12 has a step: the others come after it.
    assert [r.checkpoint_id for r in sort_checkpoints(rows, "step")] == ["global_step12", "cp_b", "cp_a"]
    files = {"cp_a": files_info(_RESUMABLE_FILES), "cp_b": files_info(_WEIGHTS_FILES)}
    assert [r.checkpoint_id for r in sort_checkpoints(rows, "step", files=files)] == ["global_step12", "cp_a", "cp_b"]
    assert [r.checkpoint_id for r in sort_checkpoints(rows, "step", False, files)] == ["cp_b", "cp_a", "global_step12"]
    by_type = [r.checkpoint_id for r in sort_checkpoints(rows, "type", files=files)]
    assert by_type == ["cp_b", "cp_a", "global_step12"]  # weights-only, then resumable newest first


def test_fmt_bytes_and_training_detection():
    assert fmt_bytes(None) == "—" and fmt_bytes(12) == "12 B" and fmt_bytes(1192135096) == "1.2 GB"
    assert saves_no_checkpoints({"sub_jobs": [{"job_type": "SAMPLING"}]})
    assert not saves_no_checkpoints({"sub_jobs": [{"job_type": "TRAINING"}, {"job_type": "SAMPLING"}]})
    assert not saves_no_checkpoints({"job_id": "7"})  # sub-jobs not known yet


def test_compact_numbers():
    assert compact_numbers({"n": 1.0, "lr": 1e-05, "xs": [2.0, 2.5], "s": "1.0"}) == {
        "n": 1, "lr": 1e-05, "xs": [2, 2.5], "s": "1.0"}


def test_listed_checkpoints_have_types_from_the_list_or_the_files():
    fake = FakeClient()
    listed = {job["job_id"]: fake.list_checkpoints(job["job_id"]) for job in fake.list_jobs()}
    job_id, entries = next((jid, cps) for jid, cps in listed.items() if len(cps) >= 3)
    rows = parse_checkpoints(entries)
    assert any(r.kind is None for r in rows) and any(r.kind for r in rows)
    for row in rows:
        info = files_info(fake.export_checkpoint(job_id, row.checkpoint_id))
        assert info.kind in (RESUMABLE, WEIGHTS_ONLY) and info.step is not None
        assert row.kind in (None, info.kind)


def _client(*, checkpoints=_LISTED, sub_jobs=None, status="CANCELLED", job_id="7"):
    c = MagicMock()
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = []
    c.get_job.return_value = {
        "status": status,
        "sub_jobs": sub_jobs if sub_jobs is not None else [
            {"sub_job_id": f"{job_id}:training:0", "job_type": "TRAINING", "model_name": "Qwen/Qwen3-0.6B",
             "training_config": {"n_gpus": 1.0, "optimizer": {"name": "AdamW", "lr": 0.0001}}},
            {"sub_job_id": f"{job_id}:sampling:0", "job_type": "SAMPLING"},
        ],
    }
    c.tail_logs.return_value = {"entries": [], "next_cursor": "", "eof": True}
    c.download_stdout_logs.return_value = []
    c.fetch_execution_logs.return_value = []
    c.list_checkpoints.return_value = checkpoints
    exports = {"cp_a": _RESUMABLE_FILES, "cp_b": _WEIGHTS_FILES}
    c.export_checkpoint.side_effect = lambda job_id, cid: exports.get(cid, {"files": []})
    return c


async def _wait(pilot, predicate, tries=200):
    for _ in range(tries):
        await pilot.pause()
        if predicate():
            return True
    return False


async def _settle(app, pilot):
    try:
        app.workers.cancel_all()
    except Exception:  # noqa: BLE001
        pass
    for _ in range(5):
        await pilot.pause()


def _text(screen, selector) -> str:
    return str(screen.query_one(selector, Static).render())


def _detail(screen) -> str:
    """Everything shown for the highlighted checkpoint: its card, the copied-command box and the job's card."""
    shown = [_text(screen, f"#checkpoint-{part}") for part in ("hero", "detail", "action-status", "origin")]
    copied = screen.query_one("#checkpoint-copied", Static)
    if copied.display and copied.border_title:
        shown += [Content.from_markup(copied.border_title).plain, _text(screen, "#checkpoint-copied")]
    return "\n".join(shown)


async def _run_tab_lists_and_details(monkeypatch):
    c = _client()
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(140, 50)) as pilot:
        assert await _wait(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen._sources_ready)
        screen = app.screen
        assert not c.list_checkpoints.called  # read only when the tab opens
        await pilot.press("c")
        tabs = screen.query_one("#job-tabs", TabbedContent)
        assert await _wait(pilot, lambda: tabs.active == "checkpoints")
        assert await _wait(pilot, lambda: screen._ckpt_loaded and len(screen._ckpt_files) == 3)
        table = screen.query_one("#checkpoint-table", DataTable)
        assert table.row_count == 3
        assert [k.value for k in table.rows] == ["cp_b", "cp_a", "global_step12"]
        assert "Checkpoints · 3" in Content.from_markup(screen.query_one("#checkpoint-list-card").border_title).plain
        # Highlighting a row shows its details, with the type and step its files show.
        detail = _detail(screen)
        assert "cp_b" in detail and "WEIGHTS-ONLY" in detail and "from its files" in detail
        assert "links.example" not in detail  # the export's links are never shown
        assert "export_checkpoint" not in detail  # a button downloads it, not a script

        def clickable():
            markup = screen.query_one("#checkpoint-actions", Static).content.markup
            return [f"screen.checkpoint_use('{use}')" in markup for use in ("training", "sampling", "both")]

        # Weights-only: sampling only (training resumes from a resumable one, and no
        # resumable checkpoint shares its step for training + sampling).
        assert clickable() == [False, True, False]
        assert "Training resumes from a resumable checkpoint" in _text(screen, "#checkpoint-action-status")
        # A use button copies one command that starts a job from the checkpoint.
        copied = []
        monkeypatch.setattr(app, "copy_out", lambda text, what: copied.append(text))
        await app.run_action("screen.checkpoint_use('sampling')")  # what a click on the segment runs
        assert await _wait(pilot, lambda: copied)
        body = _submitted_body(copied[-1])
        assert [sj["job_type"] for sj in body["sub_job_configs"]] == ["sampling"]
        assert body["sub_job_configs"][0]["source_checkpoint_info"] == {"checkpoint_id": "cp_b", "source_job_id": "7"}
        assert "Copied: a new" in _detail(screen)
        await pilot.press("down")
        assert await _wait(pilot, lambda: screen._ckpt_highlighted == "cp_a")
        detail = _detail(screen)
        assert "RESUMABLE" in detail and "Copied: a new" not in detail
        # Sampling loads only weights: a resumable checkpoint starts a training job only.
        assert await _wait(pilot, lambda: clickable() == [True, False, False])
        await app.run_action("screen.checkpoint_use('training')")
        assert await _wait(pilot, lambda: len(copied) == 2)
        body = _submitted_body(copied[-1])
        assert [sj["job_type"] for sj in body["sub_job_configs"]] == ["training"]
        assert body["sub_job_configs"][0]["source_checkpoint_info"]["checkpoint_id"] == "cp_a"
        assert body["sub_job_configs"][0]["training_config"]["n_gpus"] == 1
        assert "7:training:0" in detail and '"lr": 0.0001' in detail and '"n_gpus": 1,' in detail
        # s sorts by step (the highlight stays on its checkpoint), o reverses it.
        await pilot.press("s")
        assert screen._ckpt_sort == "step"
        assert [k.value for k in table.rows] == ["global_step12", "cp_a", "cp_b"]
        assert screen._ckpt_highlighted == "cp_a"
        await pilot.press("o")
        assert [k.value for k in table.rows] == ["cp_b", "cp_a", "global_step12"]
        # r lists them again; the files already read are not read again.
        exports = c.export_checkpoint.call_count
        await pilot.press("r")
        assert await _wait(pilot, lambda: c.list_checkpoints.call_count == 2 and not screen._ckpt_loading)
        assert c.export_checkpoint.call_count == exports
        await _settle(app, pilot)


def test_checkpoints_tab_lists_sorts_and_shows_details(monkeypatch):
    asyncio.run(_run_tab_lists_and_details(monkeypatch))


def _submitted_body(command: str) -> dict:
    """The CreateJob body a copied ``printf '%s' BODY | cortex-training ... submit -`` sends."""
    words = shlex.split(command)
    assert words[:2] == ["printf", "%s"] and words[3] == "|" and words[-2:] == ["submit", "-"]
    return json.loads(words[2])


def test_new_job_body_copies_the_job_s_sub_jobs():
    job = _client().get_job.return_value
    job["hardware"] = "H200"
    body = new_job_body(job, "7", sampling_checkpoint="cp_b")
    assert body["hardware"] == "H200"
    (sampler,) = body["sub_job_configs"]
    # The job had a sampling sub-job without a config: one is made for the trainer's model.
    assert sampler == {"job_type": "sampling", "model_name": "Qwen/Qwen3-0.6B", "inference_config": {"n_gpus": 1},
                       "source_checkpoint_info": {"checkpoint_id": "cp_b", "source_job_id": "7"}}
    assert new_job_body({"sub_jobs": [{"job_type": "SAMPLING"}]}, "7", training_checkpoint="cp_b") is None
    both = new_job_body(job, "7", "cp_r", "cp_w")["sub_job_configs"]
    assert [(sj["job_type"], sj["source_checkpoint_info"]["checkpoint_id"]) for sj in both] == [
        ("training", "cp_r"), ("sampling", "cp_w")]


def test_use_plan_pairs_a_resumable_and_a_weights_only_checkpoint_of_one_step():
    rows = parse_checkpoints([
        {"checkpoint_id": "r5", "global_steps": 5, "checkpoint_type": "resumable",
         "created_at": "2026-07-20T10:00:00Z"},
        {"checkpoint_id": "w5", "global_steps": 5, "checkpoint_type": "weights-only",
         "created_at": "2026-07-20T10:01:00Z"},
        {"checkpoint_id": "w9", "global_steps": 9, "checkpoint_type": "weights-only",
         "created_at": "2026-07-20T11:00:00Z"},
        {"checkpoint_id": "u", "created_at": "2026-07-20T12:00:00Z"},
    ])
    r5, w5, w9, u = rows
    assert use_plan(r5, rows) == {"training": ("r5", None), "sampling": "sampling needs a weights-only checkpoint",
                                  "both": ("r5", "w5")}
    assert use_plan(w5, rows)["both"] == ("r5", "w5") and use_plan(w5, rows)["training"].startswith("training")
    assert use_plan(w9, rows)["both"] == "training + sampling needs a resumable checkpoint saved at the same step"
    assert use_plan(u, rows) == {"training": ("u", None), "sampling": (None, "u"), "both": ("u", "u")}


class _Response:
    def __init__(self, data):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        for i in range(0, len(self._data), size):
            yield self._data[i:i + size]


def test_download_export_writes_files_and_skips_complete_ones(tmp_path):
    export = {"files": [{"filename": "global_step8/model.safetensors", "size_bytes": "6", "url": "u1"},
                        {"filename": "metadata.json", "size_bytes": "2", "url": "u2"}]}
    data = {"u1": b"abcdef", "u2": b"{}"}
    gets = []

    def get(url, **kwargs):
        gets.append(url)
        return _Response(data[url])

    progress = []
    paths = download_export(export, tmp_path, get=get, on_progress=lambda *a: progress.append(a), chunk_bytes=4)
    assert [p.read_bytes() for p in paths] == [b"abcdef", b"{}"]
    assert progress[-1] == (8, 8, 2, 2) and not list(tmp_path.rglob("*.partial"))
    download_export(export, tmp_path, get=get)  # both whole already: nothing is fetched
    assert gets == ["u1", "u2"]
    with pytest.raises(DownloadCancelled):
        download_export(export, tmp_path / "again", get=get, cancelled=lambda: True)
    with pytest.raises(ValueError):
        download_export({"files": [{"filename": "../x", "url": "u2"}]}, tmp_path, get=get)


def _telemetry_client(monkeypatch):
    """A real client whose metrics are recorded instead of sent, as the TUI builds it."""
    from cortex_training.client import CortexTrainingClient
    from cortex_training.client import TUI_SURFACE

    monkeypatch.delenv("CORTEX_TRAINING_ENABLE_SUCCESS_TELEMETRY", raising=False)
    c = CortexTrainingClient(base_url="http://test.local", database="DB", schema="SCH")
    c._metric_emitter = MagicMock()
    c._telemetry_surface = TUI_SURFACE
    return c


_LINK = "https://stage.test/checkpoints/global_step8/model.safetensors?X-Amz-Signature=deadbeef"


def _checkpoint_export():
    return {"files": [{"filename": "global_step8/model.safetensors", "size_bytes": "6", "url": _LINK},
                      {"filename": "metadata.json", "size_bytes": "2", "url": _LINK + "&part=2"}]}


def _get(data):
    return lambda url, **kwargs: _Response(data)


class _Refused(_Response):
    """The stage's answer with the error ``status``, such as 403 for a link that expired."""

    def __init__(self, status):
        super().__init__(b"")
        self._status = status

    def raise_for_status(self):
        response = requests.Response()
        response.status_code, response.url = self._status, _LINK
        raise requests.HTTPError(f"{self._status} Client Error: for url: {_LINK}", response=response)


def test_a_checkpoint_download_is_one_metric_with_its_files_bytes_and_time(tmp_path, monkeypatch):
    c = _telemetry_client(monkeypatch)
    (tmp_path / "metadata.json").write_bytes(b"{}")  # already whole: kept, not fetched

    paths = download_checkpoint(c, "job-1", "cp-1", _checkpoint_export(), tmp_path, get=_get(b"abcdef"),
                                chunk_bytes=4)

    assert [p.name for p in paths] == ["model.safetensors", "metadata.json"]
    c._metric_emitter.emit.assert_called_once()
    operation, value = c._metric_emitter.emit.call_args.args
    attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
    assert operation == "download_checkpoint"
    assert value["success"] is True and value["cancelled"] is False and value["duration_ms"] >= 0
    assert {key: value[key] for key in ("artifact", "file_count", "files_written", "files_skipped",
                                        "bytes_written", "total_bytes")} == {
        "artifact": "checkpoint", "file_count": 2, "files_written": 1, "files_skipped": 1,
        "bytes_written": 6, "total_bytes": 8}
    assert attributes == {"client.surface": "tui", "job_id": "job-1", "checkpoint_id": "cp-1"}
    # No link (a credential), file name or local path leaves the machine.
    sent = repr(c._metric_emitter.emit.call_args)
    assert "deadbeef" not in sent and "stage.test" not in sent and "metadata" not in sent
    assert str(tmp_path) not in sent


@pytest.mark.parametrize(("ending", "raised"), [
    ("finished", None), ("stopped", DownloadCancelled), ("failed", requests.ConnectionError)])
def test_a_checkpoint_download_s_duration_is_the_time_it_took(tmp_path, monkeypatch, ending, raised):
    import cortex_training.tui.checkpoints as checkpoints

    c = _telemetry_client(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(checkpoints.time, "monotonic", lambda: now[0])
    gets = []

    def get(url, **kwargs):  # each file takes 2.5 s; a failed download fails on the second
        gets.append(url)
        now[0] += 2.5
        if ending == "failed" and len(gets) == 2:
            raise requests.ConnectionError("connection reset")
        return _Response(b"abcdef")

    stop = (lambda: now[0] > 1003) if ending == "stopped" else (lambda: False)  # true once the second has started
    with pytest.raises(raised) if raised else contextlib.nullcontext():
        download_checkpoint(c, "job-1", "cp-1", _checkpoint_export(), tmp_path, get=get, cancelled=stop)

    c._metric_emitter.emit.assert_called_once()
    value = c._metric_emitter.emit.call_args.args[1]
    assert (value["success"], value["cancelled"]) == (ending == "finished", ending == "stopped")
    assert value["duration_ms"] == 5000.0  # both files' 2.5 s, however it ended


def test_a_checkpoint_download_sends_total_bytes_only_when_the_export_gives_every_size(tmp_path, monkeypatch):
    c = _telemetry_client(monkeypatch)
    export = {"files": [{"filename": "model.safetensors", "size_bytes": "6", "url": _LINK},
                        {"filename": "metadata.json", "url": _LINK + "&part=2"}]}  # its size not given

    download_checkpoint(c, "job-1", "cp-1", export, tmp_path, get=_get(b"abcdef"))

    value = c._metric_emitter.emit.call_args.args[1]
    assert (value["success"], value["file_count"], value["files_written"], value["bytes_written"]) == (
        True, 2, 2, 12)
    assert "total_bytes" not in value  # the known sizes alone (6) would understate it


def test_a_checkpoint_download_with_an_empty_file_still_sends_total_bytes(tmp_path, monkeypatch):
    c = _telemetry_client(monkeypatch)
    data = {"model.safetensors": b"abcdef", "done.marker": b""}  # 0 is a size the export gives
    export = {"files": [{"filename": name, "size_bytes": str(len(body)), "url": f"{_LINK}&file={name}"}
                        for name, body in data.items()]}

    paths = download_checkpoint(c, "job-1", "cp-1", export, tmp_path,
                                get=lambda url, **kwargs: _Response(data[url.rsplit("=", 1)[1]]))

    assert [p.read_bytes() for p in paths] == list(data.values())
    value = c._metric_emitter.emit.call_args.args[1]
    assert (value["success"], value["file_count"], value["files_written"], value["bytes_written"],
            value.get("total_bytes")) == (True, 2, 2, 6, 6)


def test_a_file_that_cannot_be_put_in_place_is_not_counted_as_written(tmp_path, monkeypatch):
    c = _telemetry_client(monkeypatch)
    (tmp_path / "metadata.json").mkdir()  # the second file's name is taken by a folder

    with pytest.raises(OSError):
        download_checkpoint(c, "job-1", "cp-1", _checkpoint_export(), tmp_path, get=_get(b"ab"))

    value = c._metric_emitter.emit.call_args.args[1]
    assert (value["success"], value["file_count"], value["files_written"], value["bytes_written"]) == (
        False, 2, 1, 4)
    assert str(tmp_path) not in repr(c._metric_emitter.emit.call_args)


def test_a_chunk_a_full_disk_refuses_is_not_counted_as_written(tmp_path, monkeypatch):
    import cortex_training.tui.checkpoints as checkpoints

    c = _telemetry_client(monkeypatch)

    class FullDisk:  # room for 4 bytes; then a write fails, as on a disk that filled up
        def __init__(self, path, mode):
            self._path, self._room = path, 4

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write(self, chunk):
            if len(chunk) > self._room:
                raise OSError(errno.ENOSPC, "No space left on device", str(self._path))
            self._room -= len(chunk)
            return len(chunk)

    monkeypatch.setattr(checkpoints, "open", FullDisk, raising=False)
    with pytest.raises(OSError):
        download_checkpoint(c, "job-1", "cp-1", _checkpoint_export(), tmp_path, get=_get(b"abcdef"),
                            chunk_bytes=4)

    c._metric_emitter.emit.assert_called_once()
    value = c._metric_emitter.emit.call_args.args[1]
    attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
    assert (value["success"], value["cancelled"], value["files_written"], value["bytes_written"]) == (
        False, False, 0, 4)  # the first chunk only
    assert value["error_message"] == "OSError: No space left on device"
    assert attributes["error.type"] == "OSError" and "error.http_status" not in attributes
    assert str(tmp_path) not in repr(c._metric_emitter.emit.call_args)


def test_an_export_entry_without_a_link_is_not_counted(tmp_path, monkeypatch):
    c = _telemetry_client(monkeypatch)
    export = {"files": [{"filename": "model.safetensors", "size_bytes": "6", "url": _LINK},
                        {"filename": "notes.txt", "size_bytes": "2"}]}  # no link: nothing to download

    paths = download_checkpoint(c, "job-1", "cp-1", export, tmp_path, get=_get(b"abcdef"))

    assert [p.name for p in paths] == ["model.safetensors"]
    value = c._metric_emitter.emit.call_args.args[1]
    assert (value["success"], value["file_count"], value["files_written"], value["total_bytes"]) == (
        True, 1, 1, 6)


def test_a_stopped_checkpoint_download_is_cancelled_not_failed(tmp_path, monkeypatch):
    c = _telemetry_client(monkeypatch)
    checks = []

    with pytest.raises(DownloadCancelled):  # stopped after its first chunk
        download_checkpoint(c, "job-1", "cp-1", _checkpoint_export(), tmp_path, get=_get(b"abcdef"),
                            chunk_bytes=4, cancelled=lambda: checks.append(1) or len(checks) > 2)

    c._metric_emitter.emit.assert_called_once()
    value = c._metric_emitter.emit.call_args.args[1]
    attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
    assert (value["success"], value["cancelled"], value["bytes_written"], value["files_written"]) == (
        False, True, 4, 0)
    assert "error_message" not in value and "error.type" not in attributes


def test_a_resumed_checkpoint_download_counts_only_what_it_fetched(tmp_path, monkeypatch):
    # Files are fetched in order, so a stop leaves the first ones whole: the next
    # download keeps those, then fetches the rest.
    c = _telemetry_client(monkeypatch)
    data = {"a.bin": b"aaa", "b.bin": b"bbb", "c.bin": b"cc"}
    export = {"files": [{"filename": name, "size_bytes": str(len(body)), "url": f"{_LINK}&file={name}"}
                        for name, body in data.items()]}

    def get(url, **kwargs):
        return _Response(data[url.rsplit("=", 1)[1]])

    whole = [0]  # files done so far, as on_progress says
    with pytest.raises(DownloadCancelled):  # stopped once two files are whole
        download_checkpoint(c, "job-1", "cp-1", export, tmp_path, get=get,
                            on_progress=lambda done, total, files_done, files: whole.append(files_done),
                            cancelled=lambda: whole[-1] >= 2)
    download_checkpoint(c, "job-1", "cp-1", export, tmp_path, get=get)

    fields = ("success", "cancelled", "files_skipped", "files_written", "bytes_written", "total_bytes")
    assert [tuple(call.args[1][key] for key in fields) for call in c._metric_emitter.emit.call_args_list] == [
        (False, True, 0, 2, 6, 8),  # the stop: a.bin and b.bin written
        (True, False, 2, 1, 2, 8),  # the resume: those two kept, only c.bin fetched
    ]
    assert [(tmp_path / name).read_bytes() for name in data] == list(data.values())


@pytest.mark.parametrize("status", [None, 403, 503], ids=["no-connection", "expired-link", "throttled"])
def test_a_file_that_fails_after_the_stop_counts_as_stopped(tmp_path, monkeypatch, status):
    c = _telemetry_client(monkeypatch)
    stop = []

    def get(url, **kwargs):  # Stop is pressed while the request still waits for its answer
        stop.append(True)
        if status is None:  # the connection is still being made
            raise requests.ConnectionError(f"Max retries exceeded with url: {_LINK}")
        return _Refused(status)  # the answer, once it comes, is an error status

    with pytest.raises(DownloadCancelled):
        download_checkpoint(c, "job-1", "cp-1", _checkpoint_export(), tmp_path, get=get,
                            cancelled=lambda: bool(stop))

    c._metric_emitter.emit.assert_called_once()
    value = c._metric_emitter.emit.call_args.args[1]
    assert (value["success"], value["cancelled"]) == (False, True)
    assert "error_message" not in value
    assert "error.type" not in c._metric_emitter.emit.call_args.kwargs["attributes"]


def test_a_failed_checkpoint_download_reports_its_status_but_never_the_link(tmp_path, monkeypatch):
    c = _telemetry_client(monkeypatch)

    with pytest.raises(requests.HTTPError):
        download_checkpoint(c, "job-1", "cp-1", _checkpoint_export(), tmp_path,
                            get=lambda url, **kwargs: _Refused(403))

    c._metric_emitter.emit.assert_called_once()
    value = c._metric_emitter.emit.call_args.args[1]
    attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
    assert (value["success"], value["cancelled"]) == (False, False)
    assert value["error_message"] == "HTTPError: HTTP 403"
    assert (attributes["error.type"], attributes["error.http_status"]) == ("HTTPError", 403)
    assert "deadbeef" not in repr(c._metric_emitter.emit.call_args)


@pytest.mark.parametrize(("error", "text"), [
    (OSError(28, "No space left on device", "/home/someone/cortex-training-checkpoints/x"),
     "OSError: No space left on device"),
    (ValueError("refusing to write '../x' outside /home/someone"), "ValueError"),
    # A requests error is an OSError too, and can carry the link as its strerror.
    (requests.exceptions.ConnectionError(111, f"Connection refused for {_LINK}"), "ConnectionError"),
    (requests.exceptions.ConnectionError(f"HTTPSConnectionPool: Max retries exceeded with url: {_LINK}"),
     "ConnectionError"),
])
def test_a_failed_download_s_metric_names_the_error_without_its_text(error, text):
    from cortex_training.tui.checkpoints import _metric_error_text

    assert _metric_error_text(error) == text


def test_ctrl_c_during_a_checkpoint_download_is_cancelled_and_re_raised(tmp_path, monkeypatch):
    c = _telemetry_client(monkeypatch)

    def get(url, **kwargs):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        download_checkpoint(c, "job-1", "cp-1", _checkpoint_export(), tmp_path, get=get)

    c._metric_emitter.emit.assert_called_once()
    value = c._metric_emitter.emit.call_args.args[1]
    assert (value["success"], value["cancelled"]) == (False, True)


def test_with_success_telemetry_off_only_failed_checkpoint_downloads_are_sent(tmp_path, monkeypatch):
    c = _telemetry_client(monkeypatch)
    monkeypatch.setenv("CORTEX_TRAINING_ENABLE_SUCCESS_TELEMETRY", "0")

    download_checkpoint(c, "job-1", "cp-1", _checkpoint_export(), tmp_path, get=_get(b"abcdef"))
    c._metric_emitter.emit.assert_not_called()

    with pytest.raises(ValueError):
        download_checkpoint(c, "job-1", "cp-1", {"files": [{"filename": "../x", "url": "u"}]}, tmp_path)
    [failed] = c._metric_emitter.emit.call_args_list
    # Refused while planning, before any file: the counts still say so.
    assert {key: failed.args[1][key] for key in ("success", "file_count", "files_written", "files_skipped",
                                                 "bytes_written")} == {
        "success": False, "file_count": 1, "files_written": 0, "files_skipped": 0, "bytes_written": 0}


def test_a_checkpoint_download_s_report_never_changes_the_download(tmp_path):
    failing = MagicMock()
    failing._emit_operation_outcome.side_effect = RuntimeError("telemetry is down")
    for client in (failing, FakeClient()):  # the fake client has no telemetry at all
        root = tmp_path / type(client).__name__
        paths = download_checkpoint(client, "job-1", "cp-1", _checkpoint_export(), root, get=_get(b"ab"))
        assert [p.read_bytes() for p in paths] == [b"ab", b"ab"]
        # A stop and a failure end as they would without telemetry, too.
        with pytest.raises(DownloadCancelled):
            download_checkpoint(client, "job-1", "cp-1", _checkpoint_export(), root / "stopped",
                                get=_get(b"ab"), cancelled=lambda: True)
        with pytest.raises(ValueError, match="refusing to write"):
            download_checkpoint(client, "job-1", "cp-1", {"files": [{"filename": "../x", "url": "u"}]}, root)


# Longer than the 8 characters of it that name the job's download folder.
_LONG_JOB_ID = "job-0123456789abcdef"


async def _run_download_button(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    import cortex_training.tui.checkpoints as checkpoints

    calls = []

    def fake_download(export, root, *, on_progress=None, cancelled=lambda: False, stats=None):
        calls.append((export["checkpoint_id"], root))
        stats.update(file_count=1, files_skipped=0, bytes_written=10)
        on_progress(10, 10, 1, 1)
        return [root / "metadata.json"]

    monkeypatch.setattr(checkpoints, "download_export", fake_download)
    c = _client(job_id=_LONG_JOB_ID)
    app = CortexTrainingLogTUI(c, _LONG_JOB_ID, poll_interval=0.01)
    async with app.run_test(size=(140, 50)) as pilot:
        assert await _wait(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen._sources_ready)
        screen = app.screen
        await pilot.press("c")
        assert await _wait(pilot, lambda: screen._ckpt_loaded and screen._ckpt_highlighted == "cp_b")
        assert "@click=screen.checkpoint_download" in screen.query_one("#checkpoint-actions", Static).content.markup
        await pilot.press("d")
        ok = await _wait(pilot, lambda: "downloaded 1 files" in _text(screen, "#checkpoint-action-status"))
        assert ok, (_text(screen, "#checkpoint-action-status"), calls, screen._ckpt_download)
        assert calls == [("cp_b", tmp_path / "cortex-training-checkpoints" / "job-0123" / "cp_b")]
        assert "~/cortex-training-checkpoints/job-0123/cp_b" in _text(screen, "#checkpoint-action-status")
        assert {call.args[0] for call in c.export_checkpoint.call_args_list} == {_LONG_JOB_ID}
        # The download is one metric, sent through the client's own telemetry, for the whole job id.
        c._emit_operation_outcome.assert_called_once()
        (operation, _started, identity), report = c._emit_operation_outcome.call_args
        assert (operation, identity) == ("download_checkpoint", {"job_id": _LONG_JOB_ID, "checkpoint_id": "cp_b"})
        assert report["values"] == {"artifact": "checkpoint", "cancelled": False, "file_count": 1,
                                    "files_skipped": 0, "bytes_written": 10}
        assert (report["error"], report["cancelled"]) == (None, False)
        await _settle(app, pilot)


def test_download_button_downloads_the_highlighted_checkpoint(tmp_path, monkeypatch):
    asyncio.run(_run_download_button(tmp_path, monkeypatch))


async def _run_stop_a_checkpoint_download(tmp_path, monkeypatch, key):
    monkeypatch.setenv("HOME", str(tmp_path))
    import cortex_training.tui.checkpoints as checkpoints

    started = threading.Event()

    def slow_download(export, root, *, on_progress=None, cancelled=lambda: False, stats=None):
        started.set()
        deadline = time.monotonic() + 5
        while not cancelled():  # a big file, still arriving
            if time.monotonic() > deadline:
                raise AssertionError("the stop never reached the download")
            time.sleep(0.01)
        raise DownloadCancelled()

    monkeypatch.setattr(checkpoints, "download_export", slow_download)
    c = _client(job_id=_LONG_JOB_ID)
    app = CortexTrainingLogTUI(c, _LONG_JOB_ID, poll_interval=0.01)
    async with app.run_test(size=(140, 50)) as pilot:
        assert await _wait(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen._sources_ready)
        screen = app.screen
        await pilot.press("c")
        assert await _wait(pilot, lambda: screen._ckpt_loaded and screen._ckpt_highlighted == "cp_b")
        await pilot.press("d")
        assert await _wait(pilot, started.is_set)
        await pilot.press(key)  # d again stops it; Esc leaves the job's screen, which stops it too
        assert await _wait(pilot, lambda: c._emit_operation_outcome.called), screen._ckpt_download
        # One metric, and a stop rather than a failure.
        c._emit_operation_outcome.assert_called_once()
        (operation, _started, identity), report = c._emit_operation_outcome.call_args
        assert (operation, identity) == ("download_checkpoint", {"job_id": _LONG_JOB_ID, "checkpoint_id": "cp_b"})
        assert (report["cancelled"], report["error"]) == (True, None)
        if key == "d":
            assert await _wait(pilot, lambda: not screen._ckpt_download["running"])
            assert screen._ckpt_download["error"] == "stopped"
            assert "download stopped" in _text(screen, "#checkpoint-action-status")
        await _settle(app, pilot)


@pytest.mark.parametrize("key", ["d", "escape"], ids=["stop", "leave-the-job"])
def test_stopping_a_checkpoint_download_ends_it_and_reports_it_cancelled(tmp_path, monkeypatch, key):
    asyncio.run(_run_stop_a_checkpoint_download(tmp_path, monkeypatch, key))


async def _run_tab_states(client, expect):
    app = CortexTrainingLogTUI(client, "7", poll_interval=0.01)
    async with app.run_test(size=(120, 40)) as pilot:
        assert await _wait(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen._sources_ready)
        screen = app.screen
        screen.query_one("#job-tabs", TabbedContent).active = "checkpoints"
        assert await _wait(pilot, lambda: expect in _text(screen, "#checkpoint-status")
                           + _detail(screen))
        await _settle(app, pilot)


async def _run_s_still_saves_on_logs(monkeypatch):
    saved = []
    monkeypatch.setattr(LogScreen, "action_save_log", lambda self: saved.append(self._active_tab))
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test(size=(120, 40)) as pilot:
        assert await _wait(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen._sources_ready)
        screen = app.screen
        screen.set_focus(screen.query_one("#sources"))
        await pilot.press("s")
        assert saved == ["logs"] and screen._ckpt_sort == "created"
        await pilot.press("c")
        assert await _wait(pilot, lambda: screen._checkpoints_active())
        await pilot.press("s")
        assert saved == ["logs"] and screen._ckpt_sort == "step"
        await _settle(app, pilot)


def test_s_saves_on_logs_and_sorts_on_checkpoints(monkeypatch):
    asyncio.run(_run_s_still_saves_on_logs(monkeypatch))


async def _run_live_refresh_picks_up_a_new_checkpoint():
    c = _client(checkpoints=[], status="RUNNING")
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(120, 40)) as pilot:
        assert await _wait(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen._sources_ready)
        screen = app.screen
        await pilot.press("c")
        assert await _wait(pilot, lambda: screen._ckpt_loaded)
        assert "No checkpoints yet" in _text(screen, "#checkpoint-status")
        c.list_checkpoints.return_value = _LISTED[:2]
        screen._auto_refresh_data()  # what the 30-second timer calls
        assert await _wait(pilot, lambda: len(screen._ckpt_rows) == 2 and not screen._ckpt_reading_files)
        assert screen.query_one("#checkpoint-table", DataTable).row_count == 2
        await _settle(app, pilot)


def test_live_refresh_picks_up_a_new_checkpoint():
    asyncio.run(_run_live_refresh_picks_up_a_new_checkpoint())


def test_checkpoints_tab_failure_is_visible():
    c = _client()
    c.list_checkpoints.side_effect = RuntimeError("boom")
    asyncio.run(_run_tab_states(c, "Checkpoints unavailable: RuntimeError: boom"))


def test_checkpoints_tab_empty_and_sampling_only():
    asyncio.run(_run_tab_states(_client(checkpoints=[]), "This job saved no checkpoints."))
    sampling = _client(sub_jobs=[{"sub_job_id": "7:sampling:0", "job_type": "SAMPLING"}])
    asyncio.run(_run_tab_states(sampling, "no training sub-job"))
    assert not sampling.list_checkpoints.called


def test_checkpoints_export_failure_shows_in_details():
    c = _client(checkpoints=_LISTED[:1])
    c.export_checkpoint.side_effect = RuntimeError("no links")
    asyncio.run(_run_tab_states(c, "unavailable: RuntimeError: no links"))


async def _run_preview_opens_checkpoints():
    app = CortexTrainingLogTUI(_client(), poll_interval=0.01)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(JobSummaryScreen(app._client, "7", job={"job_id": "7"}))
        assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen))
        await pilot.press("c")
        assert await _wait(pilot, lambda: isinstance(app.screen, LogScreen))
        assert await _wait(pilot, lambda: app.screen._checkpoints_active() and app.screen._ckpt_loaded)
        await pilot.press("t")  # Checkpoints → Logs: t goes round all three tabs
        assert await _wait(pilot, lambda: app.screen._active_tab == "logs")
        await _settle(app, pilot)


def test_preview_c_opens_the_checkpoints_tab():
    asyncio.run(_run_preview_opens_checkpoints())
