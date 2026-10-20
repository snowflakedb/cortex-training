# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Training metrics for the TUI's Training metrics tab (``tui/training.py``).

The rows come from the zone's own ``training.jsonl``, reconstructed per sub-job
by ``download_training_metrics``, beside the ``gpu.jsonl`` the Resources tab
reads.
"""

from __future__ import annotations

import json
from pathlib import Path

from cortex_training.tui import training
from cortex_training.tui.resource import load_resource_data

SUB = "job:training:0"


def _row(step, op="fwd_bwd", ts="2026-10-16T18:00:00Z", **metrics):
    return {"kind": "training", "record_version": 1, "sub_job_id": SUB, "sample_ts": ts, "seq": step,
            "seq_epoch": "e", "op": op, "step": step, "request_id": None, "metrics": metrics}


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


def test_reads_each_row_and_derives_the_mean_loss():
    parsed = _parse([
        _row(1, sum_loss=4.0, num_calls=2.0, entropy=0.9),
        _row(1, op="step", grad_norm=0.8, last_lr=1e-5),
        _row(2, sum_loss=3.0, num_calls=2.0, entropy=0.8),
        _row(2, op="step", grad_norm=0.7, last_lr=1e-5, ts="2026-10-16T18:00:09Z"),
    ])
    # The leading metrics first, then the rest in the order the file has them.
    assert [m.name for m in parsed.metrics] == ["mean_loss", "sum_loss", "grad_norm", "last_lr", "entropy"]
    mean, total = parsed.metrics[0], parsed.metrics[1]
    # num_calls gets no chart of its own; it is the divisor the mean is derived with.
    assert (mean.op, mean.steps, mean.values) == ("fwd_bwd", (1, 2), (2.0, 1.5))
    assert total.values == (4.0, 3.0)
    assert parsed.last_step == 2
    assert parsed.newest.isoformat() == "2026-10-16T18:00:09+00:00"
    assert parsed.skipped == 0


def test_the_mean_loss_is_left_out_without_a_usable_num_calls():
    # The sum is then the only loss the row supports; a made-up divisor would
    # chart a number the service never reported.
    for metrics in ({"sum_loss": 4.0}, {"sum_loss": 4.0, "num_calls": 0.0},
                    {"sum_loss": 4.0, "num_calls": None}, {"num_calls": 2.0}):
        names = [m.name for m in _parse([_row(1, **metrics)]).metrics]
        assert training.MEAN_LOSS not in names, metrics


def test_unreadable_rows_are_counted_and_non_numbers_ignored():
    parsed = _parse([
        {**_row(1, sum_loss=1.0), "record_version": 2},  # a version this client doesn't read
        {**_row(1, sum_loss=1.0), "sub_job_id": "job:training:1"},  # another sub-job's row
        {**_row(1, sum_loss=1.0), "step": True},
        {**_row(1, sum_loss=1.0), "step": -1},
        {**_row(1, sum_loss=1.0), "op": "\x1b]52;c;x\x07"},
        {**_row(1), "metrics": [1.0]},
        _row(3, sum_loss=1.0, note="text", flag=True, inf=float("inf"), **{"bad name\x1b": 1.0}),
        b'{"kind": "training", not json',
    ])
    assert parsed.skipped == 6
    assert [m.name for m in parsed.metrics] == ["sum_loss"]
    assert parsed.last_step == 3


def test_a_later_row_for_the_same_step_wins_and_names_from_two_call_types_stay_apart():
    parsed = _parse([
        _row(5, sum_loss=1.0, shared=1.0),
        _row(5, op="step", shared=2.0),
        _row(4, sum_loss=3.0),  # a resumed run's earlier step, after the first run's
        _row(5, sum_loss=0.5),
    ])
    by_name = {m.name: m for m in parsed.metrics}
    assert by_name["sum_loss"].steps == (4, 5)
    assert by_name["sum_loss"].values == (3.0, 0.5)
    assert by_name["shared"].op == "fwd_bwd" and by_name["step/shared"].values == (2.0,)


def test_loader_reads_only_what_was_committed_since_its_last_read(tmp_path):
    path = _write(tmp_path, [_row(1, sum_loss=2.0)])
    loader = training.TrainingLoader(tmp_path)
    assert loader.load().get(SUB).metrics[0].values == (2.0,)

    # The download appends a chunk; bytes past the committed length are not read yet.
    committed = path.stat().st_size
    with open(path, "ab") as handle:
        handle.write((json.dumps(_row(2, sum_loss=1.0)) + "\n").encode())
        handle.write(json.dumps(_row(3, sum_loss=0.5)).encode())  # not yet whole
    _commit(path, ["_metrics/a.gz", "_metrics/b.gz"], path.stat().st_size)
    assert loader.load().get(SUB).metrics[0].steps == (1, 2)
    with open(path, "ab") as handle:
        handle.write(b"\n")
    _commit(path, ["_metrics/a.gz", "_metrics/b.gz"], path.stat().st_size)
    assert loader.load().get(SUB).metrics[0].steps == (1, 2, 3)
    assert committed < path.stat().st_size


def test_loader_reads_a_rebuilt_file_again_from_the_start(tmp_path):
    _write(tmp_path, [_row(1, sum_loss=2.0), _row(2, sum_loss=1.0)], names=["_metrics/b.gz"])
    loader = training.TrainingLoader(tmp_path)
    assert loader.load().get(SUB).metrics[0].steps == (1, 2)
    # An older chunk landed: the download rebuilt the file with it first.
    _write(tmp_path, [_row(0, sum_loss=3.0), _row(1, sum_loss=2.0), _row(2, sum_loss=1.0)],
           names=["_metrics/a.gz", "_metrics/b.gz"])
    assert loader.load().get(SUB).metrics[0].steps == (0, 1, 2)


def test_loader_skips_an_uncommitted_file(tmp_path):
    path = tmp_path / SUB / training.FILE_NAME
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_row(1, sum_loss=1.0)) + "\n")  # no manifest: still downloading
    assert training.load_training_data(tmp_path).sub_jobs == {}


def test_the_two_streams_keep_their_own_files(tmp_path):
    """The GPU file and the training file live side by side; neither tab reads the other's."""
    _write(tmp_path, [_gpu()], name="gpu.jsonl")
    _write(tmp_path, [_row(1, sum_loss=1.0)])
    # The Resources tab reads the GPU rows and never sees a training row ...
    data = load_resource_data(tmp_path)
    assert SUB in data.kinds
    assert not data.ignored_rows_by_sub_job.get(SUB)
    # ... and this tab reads only its own file.
    parsed = training.load_training_data(tmp_path).get(SUB)
    assert [m.name for m in parsed.metrics] == ["sum_loss"] and parsed.skipped == 0


def test_a_sub_job_with_only_gpu_rows_has_no_training_metrics(tmp_path):
    _write(tmp_path, [_gpu()], sub="job:sampling:0", name="gpu.jsonl")
    assert training.load_training_data(tmp_path).sub_jobs == {}


def test_window_and_stats():
    parsed = _parse([_row(step, sum_loss=float(10 - step)) for step in range(1, 11)])
    assert parsed.first_step(None) is None
    assert parsed.first_step(3) == 8
    steps, values = parsed.metrics[0].window(parsed.first_step(3))
    assert steps == (8, 9, 10)
    stats = training.window_stats(steps, values)
    assert (stats.latest, stats.latest_step, stats.low, stats.low_step, stats.high, stats.high_step) == \
        (0.0, 10, 0.0, 10, 2.0, 8)
    assert training.window_stats((), ()) is None
    assert [training.next_window(w) for w in training.WINDOWS] == [100, 1000, None]


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
    series = _parse([_row(step, sum_loss=value) for step, value in [(1, -0.5), (2, 0.0), (3, 0.5)]]).metrics[0]
    lines = _text(training.chart_spans(series, None, 40, height=4))
    assert lines[0].startswith(" 0.5 ┤") and lines[-2].startswith("-0.5 └")
    assert all(len(line) <= 40 for line in lines)
    # Few steps: each is as many columns wide, and the lowest still draws.
    bars = lines[-3][len("-0.5 │"):]
    assert bars.count(bars[0]) == len(bars) // 3 and bars[0] != " "
    assert lines[-1].split() == ["1", "2", "3"]


def test_chart_of_a_flat_series_and_of_an_empty_window():
    flat = _parse([_row(step, op="step", last_lr=1e-5) for step in range(1, 5)]).metrics[0]
    lines = _text(training.chart_spans(flat, None, 30, height=4))
    assert "1.00e-05" in lines[2]  # drawn at half height, with its value on the middle label
    assert _text(training.chart_spans(flat, 100, 30)) == ["no samples in this window"]


def test_chart_averages_steps_into_columns_and_styles_with_a_hue():
    series = _parse([_row(step, sum_loss=float(step)) for step in range(1, 1001)]).metrics[0]
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
        "training #0 saved no training metrics"
