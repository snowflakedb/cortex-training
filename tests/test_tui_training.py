# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Training metrics for the TUI's Training metrics tab (``tui/training.py``).

The rows come from the zone's own ``training.jsonl``, reconstructed per sub-job
by ``download_training_metrics``, beside the ``gpu.jsonl`` the Resources tab
reads: one row per ``/fwd-bwd`` call and one per ``/step``, each with the
numbers its result has.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from cortex_training.tui import training
from cortex_training.tui.resource import load_resource_data

SUB = "job:training:0"
STEP, CALL = training.VIEW_STEP, training.VIEW_CALL
# Each row its own seq, as the zone numbers them: rows with the same (seq_epoch, seq) count once.
_SEQ = itertools.count(1)


def _call(step, call, ts="2026-10-16T18:00:00Z", **metrics):
    return {"kind": "training", "record_version": 2, "sub_job_id": SUB, "sample_ts": ts, "seq": next(_SEQ),
            "seq_epoch": "e", "op": "fwd_bwd", "step": step, "call": call, "request_id": "c",
            "metrics": metrics}


def _step(step, ts="2026-10-16T18:00:00Z", **metrics):
    return {"kind": "training", "record_version": 2, "sub_job_id": SUB, "sample_ts": ts, "seq": next(_SEQ),
            "seq_epoch": "e", "op": "step", "step": step, "call": None, "request_id": "s",
            "metrics": metrics}


def _gpu(ts="2026-10-16T18:00:00Z"):
    return {"record_version": 2, "sub_job_id": SUB, "sample_ts": ts, "worker_num": 0, "local_gpu_index": 0,
            "global_gpu_index": 0, "gpu_util_pct": 90.0, "seq": 1, "seq_epoch": "e"}


def _write(root: Path, rows, *, names=("_metrics/a.gz",), sub=SUB, name=training.FILE_NAME) -> Path:
    """A committed cache file, as ``download_training_metrics`` leaves it."""
    path = root / sub / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row) + "\n" if isinstance(row, dict) else row for row in rows).encode()
    path.write_bytes(data)
    _commit(path, names, len(data))
    return path


def _commit(path: Path, names, size: int) -> None:
    path.with_name("." + path.name + ".manifest.json").write_text(
        json.dumps({"version": 1, "names": list(names), "bytes": size}))


def _parse(rows) -> training.SubJobTraining:
    lines = [json.dumps(row).encode() if isinstance(row, dict) else row for row in rows]
    return training.parse_lines(SUB, lines).snapshot()


def _by_name(series):
    return {s.name: (s.xs, s.values) for s in series}


# A real SFT run: three steps of two /fwd-bwd calls each, every number as the results had it.
RUN = []
for _step_no, _loss, _grad in ((1, 9.8489, 156.866), (2, 5.0613, 77.127), (3, 2.0419, 84.373)):
    for _call_no in (1, 2):
        RUN.append(_call(_step_no, _call_no, avg_loss=_loss, rank=0, sft_model_calls=1, sft_packed_rows=1))
    RUN.append(_step(_step_no, global_steps=float(_step_no), grad_norm=_grad, last_lr=1e-4,
                     update_successful=1.0, weight_delta_norm_rank0_sample=0.17))


def test_per_step_shows_every_number_in_the_step_rows():
    parsed = _parse(RUN)
    # grad_norm and last_lr first, then the rest in the order the file has them; nothing derived.
    assert [s.name for s in parsed.metrics(STEP, everything=True)] == [
        "grad_norm", "last_lr", "global_steps", "update_successful", "weight_delta_norm_rank0_sample"]
    assert [s.name for s in parsed.metrics(STEP)] == ["grad_norm", "last_lr", "weight_delta_norm_rank0_sample"]
    assert _by_name(parsed.metrics(STEP))["grad_norm"] == ((1, 2, 3), (156.866, 77.127, 84.373))
    assert parsed.last_step == 3


def test_per_call_shows_every_number_in_each_call_with_its_step():
    parsed = _parse(RUN)
    assert [s.name for s in parsed.metrics(CALL, everything=True)] == [
        "avg_loss", "rank", "sft_model_calls", "sft_packed_rows"]
    # rank is 0 in every call, so it is hidden; the packing counts show.
    assert [s.name for s in parsed.metrics(CALL)] == ["avg_loss", "sft_model_calls", "sft_packed_rows"]
    # Six calls, numbered in the order they ran, each with its own avg_loss.
    assert _by_name(parsed.metrics(CALL))["avg_loss"] == (
        (1, 2, 3, 4, 5, 6), (9.8489, 9.8489, 5.0613, 5.0613, 2.0419, 2.0419))
    assert parsed.calls == 6
    assert parsed.call_steps == (1, 1, 2, 2, 3, 3)
    assert [parsed.step_of_call(call) for call in (1, 4, 6, 7)] == [1, 2, 3, None]


def test_calls_the_zone_could_not_tag_belong_to_the_next_step():
    rows = [
        _call(None, 1, avg_loss=9.8), _call(None, 2, avg_loss=9.7),  # before the zone saw any /step
        _step(1, grad_norm=1.0),
        _call(2, 1, avg_loss=5.0),
        _call(None, 1, avg_loss=4.0),  # after a zone restart, before its next /step
    ]
    assert _parse(rows).call_steps == (1, 1, 2, None)
    assert _parse([*rows, _step(2, grad_norm=0.5)]).call_steps == (1, 1, 2, 2)


def test_a_step_run_again_keeps_the_later_row_and_every_call():
    parsed = _parse([
        _call(5, 1, avg_loss=1.0), _step(5, grad_norm=2.0),
        _call(5, 1, avg_loss=0.9), _step(5, grad_norm=1.5),  # step 5 again, after a resume
    ])
    assert _by_name(parsed.metrics(STEP))["grad_norm"] == ((5,), (1.5,))
    assert _by_name(parsed.metrics(CALL))["avg_loss"] == ((1, 2), (1.0, 0.9))


def test_unreadable_rows_are_counted_and_non_numbers_ignored():
    parsed = _parse([
        {**_step(1, grad_norm=1.0), "record_version": 9},  # a version this client doesn't read yet
        {**_step(1, grad_norm=1.0), "sub_job_id": "job:training:1"},  # another sub-job's row
        {**_step(1, grad_norm=1.0), "step": True},
        {**_step(1, grad_norm=1.0), "step": -1},
        {**_step(1, grad_norm=1.0), "step": None},  # a step row must say its step
        {**_step(1, grad_norm=1.0), "op": "\x1b]52;c;x\x07"},
        {**_step(1), "metrics": [1.0]},
        {**_step(1, grad_norm=1.0), "call": 2},  # a step row is no call
        {**_call(1, 1, avg_loss=1.0), "call": "x"},
        {**_call(1, 1, avg_loss=1.0), "call": 0},
        _step(3, grad_norm=1.0, note="text", flag=True, inf=float("inf"), **{"bad name\x1b": 1.0}),
        b'{"kind": "training", not json',
    ])
    assert parsed.skipped == 10
    assert [s.name for s in parsed.metrics(STEP)] == ["grad_norm"]
    assert parsed.last_step == 3
    # A number under a name that can't be drawn as it is is counted, not dropped silently.
    assert parsed.hidden_names == 1


def test_loader_reads_only_what_was_committed_since_its_last_read(tmp_path):
    path = _write(tmp_path, [_step(1, grad_norm=2.0)])
    loader = training.TrainingLoader(tmp_path)
    assert loader.load().get(SUB).metrics(STEP)[0].values == (2.0,)

    # The download appends a chunk; bytes past the committed length are not read yet.
    committed = path.stat().st_size
    with open(path, "ab") as handle:
        handle.write((json.dumps(_step(2, grad_norm=1.0)) + "\n").encode())
        handle.write(json.dumps(_step(3, grad_norm=0.5)).encode())  # not yet whole
    _commit(path, ["_metrics/a.gz", "_metrics/b.gz"], path.stat().st_size)
    assert loader.load().get(SUB).metrics(STEP)[0].xs == (1, 2)
    with open(path, "ab") as handle:
        handle.write(b"\n")
    _commit(path, ["_metrics/a.gz", "_metrics/b.gz"], path.stat().st_size)
    assert loader.load().get(SUB).metrics(STEP)[0].xs == (1, 2, 3)
    assert committed < path.stat().st_size


def test_untagged_calls_are_tagged_by_a_step_read_in_a_later_load(tmp_path):
    path = _write(tmp_path, [_call(None, 1, avg_loss=9.8)])
    loader = training.TrainingLoader(tmp_path)
    assert loader.load().get(SUB).call_steps == (None,)
    with open(path, "ab") as handle:
        handle.write((json.dumps(_step(1, grad_norm=1.0)) + "\n").encode())
    _commit(path, ["_metrics/a.gz", "_metrics/b.gz"], path.stat().st_size)
    assert loader.load().get(SUB).call_steps == (1,)


def test_loader_reads_a_rebuilt_file_again_from_the_start(tmp_path):
    _write(tmp_path, [_step(1, grad_norm=2.0), _step(2, grad_norm=1.0)], names=["_metrics/b.gz"])
    loader = training.TrainingLoader(tmp_path)
    assert loader.load().get(SUB).metrics(STEP)[0].xs == (1, 2)
    # An older chunk landed: the download rebuilt the file with it first.
    _write(tmp_path, [_step(0, grad_norm=3.0), _step(1, grad_norm=2.0), _step(2, grad_norm=1.0)],
           names=["_metrics/a.gz", "_metrics/b.gz"])
    assert loader.load().get(SUB).metrics(STEP)[0].xs == (0, 1, 2)


def test_loader_skips_an_uncommitted_file(tmp_path):
    path = tmp_path / SUB / training.FILE_NAME
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_step(1, grad_norm=1.0)) + "\n")  # no manifest: still downloading
    assert training.load_training_data(tmp_path).sub_jobs == {}


def test_the_two_streams_keep_their_own_files(tmp_path):
    """The GPU file and the training file live side by side; neither tab reads the other's."""
    _write(tmp_path, [_gpu()], name="gpu.jsonl")
    _write(tmp_path, RUN)
    data = load_resource_data(tmp_path)
    assert SUB in data.kinds
    assert not data.ignored_rows_by_sub_job.get(SUB)
    parsed = training.load_training_data(tmp_path).get(SUB)
    assert parsed.calls == 6 and parsed.skipped == 0


def test_a_sub_job_with_only_gpu_rows_has_no_training_metrics(tmp_path):
    _write(tmp_path, [_gpu()], sub="job:sampling:0", name="gpu.jsonl")
    assert training.load_training_data(tmp_path).sub_jobs == {}


def test_window_and_stats_per_step():
    parsed = _parse([_step(step, grad_norm=float(10 - step)) for step in range(1, 11)])
    assert parsed.first_step(None) is None
    assert parsed.first_step(3) == 8
    xs, values = parsed.metrics(STEP)[0].window(parsed.first_x(STEP, 3))
    assert xs == (8, 9, 10)
    stats = training.window_stats(xs, values)
    assert (stats.latest, stats.latest_x, stats.low, stats.low_x, stats.high, stats.high_x) == \
        (0.0, 10, 0.0, 10, 2.0, 8)
    assert training.window_stats((), ()) is None
    assert [training.next_window(w) for w in training.WINDOWS] == [100, 1000, None]


def test_a_steps_window_in_the_per_call_view_starts_at_that_steps_first_call():
    parsed = _parse(RUN)
    # The last 2 steps are steps 2 and 3, whose first call is call 3.
    assert parsed.first_x(CALL, 2) == 3
    assert _by_name(parsed.metrics(CALL))["avg_loss"][1] == (9.8489, 9.8489, 5.0613, 5.0613, 2.0419, 2.0419)
    xs, _values = parsed.metrics(CALL)[0].window(parsed.first_x(CALL, 2))
    assert xs == (3, 4, 5, 6)
    assert parsed.first_x(CALL, None) is None
    # Calls past the newest step are the step still running, so every window shows them.
    running = _parse([*RUN, _call(None, 1, avg_loss=1.5)])
    assert running.metrics(CALL)[0].window(running.first_x(CALL, 1))[0] == (5, 6, 7)


def test_both_views_are_drawn_per_step_first():
    assert training.VIEWS == (STEP, CALL)


def test_bookkeeping_numbers_are_hidden_until_every_metric_is_shown():
    parsed = _parse([
        _call(1, 1, avg_loss=2.0, rank=0, sft_model_calls=1, sft_packed_rows=1, rl_model_calls=1, approx_kl=0.01),
        _step(1, grad_norm=1.5, last_lr=1e-5, global_steps=1, update_successful=1.0,
              weight_delta_norm_rank0_sample=0.1, weight_delta_norm_over_lr_rank0_sample=1e4,
              weight_delta_rank0_sample_numel=8388608, weight_delta_rank0_sample_bytes=16777216,
              weight_delta_rank0_sample_tensors=1),
    ])
    assert [s.name for s in parsed.metrics(STEP)] == [
        "grad_norm", "last_lr", "weight_delta_norm_rank0_sample", "weight_delta_norm_over_lr_rank0_sample"]
    # A loss function's own metrics (approx_kl) stay: only names known to be bookkeeping are left out.
    # The packing counts stay too: they show rows sharing a call, or a rank running fewer calls.
    assert [s.name for s in parsed.metrics(CALL)] == [
        "avg_loss", "sft_model_calls", "sft_packed_rows", "rl_model_calls", "approx_kl"]
    assert (parsed.hidden(STEP), parsed.hidden(CALL)) == (5, 1)
    assert len(parsed.metrics(STEP, everything=True)) == 9
    assert "rank" in [s.name for s in parsed.metrics(CALL, everything=True)]
    assert parsed.failed_updates == 0


def test_rank_is_hidden_only_while_every_call_came_from_worker_0():
    from_zero = _parse([_call(1, 1, avg_loss=2.0, rank=0), _call(1, 2, avg_loss=1.9, rank=0)])
    assert [s.name for s in from_zero.metrics(CALL)] == ["avg_loss"]
    assert from_zero.hidden(CALL) == 1
    # A call from another worker is worth seeing, so rank shows by default once one has it.
    mixed = _parse([_call(1, 1, avg_loss=2.0, rank=0), _call(1, 2, avg_loss=1.9, rank=1)])
    assert [s.name for s in mixed.metrics(CALL)] == ["avg_loss", "rank"]
    assert mixed.hidden(CALL) == 0


def test_steps_that_reported_update_successful_not_1_are_counted():
    parsed = _parse([_step(step, grad_norm=1.0, update_successful=ok) for step, ok in [(1, 1.0), (2, 0.0), (3, 1.0)]])
    assert parsed.failed_updates == 1
    # A step run again after a resume replaces its row, so only the latest run of a step counts.
    assert _parse([_step(2, update_successful=0.0), _step(2, update_successful=1.0)]).failed_updates == 0
    assert _parse([_step(1, grad_norm=1.0)]).failed_updates == 0


def test_format_metric():
    assert training.format_metric(None) == "—"
    assert training.format_metric(0.0) == "0"
    assert training.format_metric(-0.0) == "0"
    assert training.format_metric(1e-05) == "1.00e-05"
    assert training.format_metric(0.0137) == "0.0137"
    assert training.format_metric(2048.0) == "2,048"
    assert training.format_metric(-0.5) == "-0.5"


def _text(lines):
    return ["".join(text for text, _ in line) for line in lines]


def test_chart_runs_from_the_lowest_value_and_labels_steps():
    series = _parse([_step(step, x=value) for step, value in [(1, -0.5), (2, 0.0), (3, 0.5)]]).metrics(STEP)[0]
    lines = _text(training.chart_spans(series, None, 40, height=4))
    assert lines[0].startswith(" 0.5 ┤") and lines[-2].startswith("-0.5 └")
    assert all(len(line) <= 40 for line in lines)
    # Few steps: each is as many columns wide (give or take one) across the whole chart, and the lowest
    # still draws.
    bars = lines[-3][len("-0.5 │"):]
    assert len(bars) == 40 - len("-0.5 │")
    assert bars.count(bars[0]) in (len(bars) // 3, len(bars) // 3 + 1) and bars[0] != " "
    assert lines[-1].split() == ["1", "2", "3"]


def test_a_per_call_chart_is_labelled_by_call():
    series = _parse(RUN).metrics(CALL)[0]
    assert _text(training.chart_spans(series, None, 40, height=4))[-1].split() == ["1", "3", "6"]


def test_a_window_longer_than_the_history_starts_at_the_first_step():
    # Three steps under "last 100 steps": the axis is steps 1 to 3, not 0 to 3 with a
    # blank column for a step that never ran.
    parsed = _parse([_step(step, x=value) for step, value in [(1, 9.8), (2, 5.1), (3, 2.0)]])
    series = parsed.metrics(STEP)[0]
    lines = _text(training.chart_spans(series, parsed.first_x(STEP, 100), 40, height=4))
    assert lines[-1].split() == ["1", "2", "3"]
    bars = lines[-3].split("│", 1)[1]
    assert bars[0] != " "  # step 1 draws from the first column
    # A window inside the history still starts where the window does, gaps and all.
    gappy = _parse([_step(step, x=1.0) for step in (1, 2, 9, 10)])
    lines = _text(training.chart_spans(gappy.metrics(STEP)[0], 5, 40, height=4))
    assert lines[-1].split()[0] == "5"


def test_a_window_of_fewer_steps_than_columns_spans_the_chart():
    # 20 steps across 34 columns: a whole number of columns each would leave 14 blank after the newest.
    series = _parse([_step(step, x=float(step)) for step in range(1, 21)]).metrics(STEP)[0]
    *rows, baseline, steps = _text(training.chart_spans(series, None, 40, height=4))
    assert all(len(row) == 40 for row in rows) and rows[0][-1] == "█"  # the newest step is the highest
    assert baseline == "   1 └" + "─" * 34
    assert steps.endswith("20") and len(steps) == 40


def test_chart_of_a_flat_series_and_of_an_empty_window():
    flat = _parse([_step(step, last_lr=1e-5) for step in range(1, 5)]).metrics(STEP)[0]
    lines = _text(training.chart_spans(flat, None, 30, height=4))
    assert "1.00e-05" in lines[2]  # drawn at half height, with its value on the middle label
    assert _text(training.chart_spans(flat, 100, 30)) == ["no samples in this window"]


def test_chart_averages_steps_into_columns_and_styles_with_a_hue():
    series = _parse([_step(step, x=float(step)) for step in range(1, 1001)]).metrics(STEP)[0]
    spans = training.chart_spans(series, None, 50, height=6, hue="#4ade80")
    assert all(sum(len(text) for text, _ in line) <= 50 for line in spans)
    assert any(style.startswith("#") and style != "#4ade80" for line in spans for _, style in line)
    assert _text(spans)[-1].split() == ["1", "500", "1,000"]


def test_no_metrics_text():
    assert "not a training sub-job" in training.no_metrics_text("sampling #0", "running", training=False,
                                                                 downloaded=True)
    assert "PLACING" in training.no_metrics_text("training #0", "placing", training=True, downloaded=False)
    assert training.no_metrics_text("training #0", "running", training=True, downloaded=False) == \
        "loading training metrics…"
    assert "RUNNING" in training.no_metrics_text("training #0", "running", training=True, downloaded=True)
    assert training.no_metrics_text("training #0", "succeeded", training=True, downloaded=True) == \
        "training #0 saved no steps"
    assert training.no_metrics_text("training #0", "succeeded", training=True, downloaded=True,
                                    view=CALL) == "training #0 saved no calls"


def test_rows_in_the_earlier_format_are_counted_not_read():
    v1 = {"kind": "training", "record_version": 1, "sub_job_id": SUB, "op": "fwd_bwd", "step": 1,
          "metrics": {"sum_loss": 4.0, "num_calls": 2.0}}
    parsed = _parse([v1, {**v1, "op": "step", "metrics": {"grad_norm": 1.0}}])
    assert (parsed.old_format, parsed.skipped, parsed.metrics(STEP), parsed.calls) == (2, 0, (), 0)
    assert "earlier format" in training.no_metrics_text("training #0", "succeeded", training=True,
                                                        downloaded=True, old_format=parsed.old_format)


def test_calls_before_an_unreadable_step_row_are_not_given_to_the_next_step():
    parsed = _parse([
        _call(None, 1, avg_loss=9.8),
        {**_step(1, grad_norm=1.0), "metrics": "broken"},  # their step, but it can't be read
        _call(2, 1, avg_loss=5.0),
        _step(2, grad_norm=0.5),
    ])
    assert parsed.call_steps == (None, 2)


def test_a_row_delivered_twice_counts_once():
    call, step = _call(1, 1, avg_loss=1.0), _step(1, grad_norm=1.0)
    parsed = _parse([call, step, call, step])  # a chunk staged twice
    assert parsed.calls == 1
    # Rows without a seq can't be told apart, so each one counts.
    assert _parse([{**call, "seq": None}, {**call, "seq": None}]).calls == 2


def test_a_step_run_again_replaces_its_earlier_row_whole():
    parsed = _parse([_step(5, grad_norm=2.0, extra=1.0), _step(5, grad_norm=1.5)])
    assert _by_name(parsed.metrics(STEP)) == {"grad_norm": ((5,), (1.5,))}  # no `extra` left from run one


def test_after_a_resume_the_header_step_is_where_the_job_is_now():
    parsed = _parse([_step(step, grad_norm=1.0) for step in (199, 200)] + [_step(151, grad_norm=1.0)])
    assert parsed.last_step == 151


def test_a_per_call_window_after_a_resume_and_with_no_calls_in_it():
    parsed = _parse([_call(10, 1, avg_loss=1.0), _step(10), _call(5, 1, avg_loss=2.0), _step(5)])
    assert parsed.call_steps == (10, 5)  # run order, not step order
    assert parsed.first_x(CALL, 1) == 1  # step 5 is the newest: the first call at or after it
    assert _parse([_call(1, 1, avg_loss=1.0), _step(1), _step(9)]).first_x(CALL, 2) == 2  # none in range


def test_a_sub_job_with_only_empty_step_rows_still_has_its_step():
    parsed = _parse([_step(4)])
    assert parsed.last_step == 4 and parsed.metrics(STEP) == ()


def test_the_fake_writes_rows_the_tab_reads_whole():
    from tests.tui_fake import FakeClient
    from cortex_training.tui import views as tui_views

    client = FakeClient()
    job = next(j for j in client.list_jobs() if j["status"] == "RUNNING" and tui_views.job_types(j) == ["training"])
    sub_job = next(sj for sj in job["sub_jobs"] if tui_views.sub_job_kind(sj) == "training")
    from datetime import datetime, timezone
    rows = client._training_rows(client._find(job["job_id"]), sub_job, datetime.now(timezone.utc))
    parsed = training.parse_lines(sub_job["sub_job_id"], [row.encode() for row in rows]).snapshot()
    assert parsed.skipped == 0 and parsed.hidden_names == 0
    assert parsed.calls == 2 * parsed.last_step  # the fake's SFT job: two calls a step
    assert [s.name for s in parsed.metrics(CALL)][0] == "avg_loss"
