# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cortex_training.tui.resource as resource_module
from cortex_training.tui.resource import load_resource_data
from cortex_training.tui.resource import parse_sample_ts
from cortex_training.tui.resource import render_chart


def _row(
    timestamp: datetime,
    gpu: int,
    util: float | None,
    *,
    interval: float = 5.0,
    local_gpu: int = 0,
    kind: str | None = None,
    version: int = 2,
) -> dict:
    row = {
        "record_version": version,
        "sub_job_id": "ignored-wire-id",
        "sample_ts": timestamp.isoformat().replace("+00:00", "Z"),
        "worker_num": gpu,
        "local_gpu_index": local_gpu,
        "global_gpu_index": gpu,
        "gpu_util_pct": util,
        "fb_used_bytes": float((gpu + 1) * 1024**3),
        "fb_free_bytes": float(8 * 1024**3),
        "sample_interval_s": interval,
    }
    if kind is not None:
        row["kind"] = kind
    return row


def _write_metrics(
    root: Path,
    sub_job_id: str,
    rows: list[dict | str],
    *,
    committed: int | None = None,
    manifest: bool = True,
) -> Path:
    path = root / sub_job_id / "gpu.jsonl"
    path.parent.mkdir(parents=True)
    normalized = []
    for item in rows:
        if isinstance(item, dict):
            item = {**item, "sub_job_id": sub_job_id}
        normalized.append(item)
    payload = "".join((item if isinstance(item, str) else json.dumps(item)) + "\n" for item in normalized).encode()
    path.write_bytes(payload)
    if manifest:
        size = len(payload) if committed is None else committed
        path.with_name(".gpu.jsonl.manifest.json").write_text(
            json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": size}) + "\n",
            encoding="utf-8",
        )
    return path


def test_parse_rfc3339_nano_on_python_floor():
    parsed = parse_sample_ts("2026-01-01T00:00:00.123456789Z")
    assert parsed == datetime(2026, 1, 1, 0, 0, 0, 123456, tzinfo=timezone.utc)
    for fraction, microsecond in [
        ("1", 100000),
        ("12", 120000),
        ("1234", 123400),
        ("12345", 123450),
    ]:
        parsed = parse_sample_ts(f"2026-01-01T00:00:00.{fraction}Z")
        assert parsed is not None
        assert parsed.microsecond == microsecond
    assert parse_sample_ts("2026-01-01T00:00:00Z") is not None
    assert parse_sample_ts("not-a-time") is None
    assert parse_sample_ts("2026-01-01 00:00:00Z") is None
    assert parse_sample_ts("0001-01-01T00:00:00+05:00") is None
    assert parse_sample_ts("9999-12-31T23:59:59-05:00") is None


def test_redacted_v2_fixture_preserves_zero(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "gpu_metrics_v2.jsonl"
    root = tmp_path / "metrics"
    path = root / "JOB_ID:training:0" / "gpu.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(fixture.read_bytes())
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size})
    )

    data = load_resource_data(root)
    view = data.sub_jobs["JOB_ID:training:0"].views["all"]
    assert view.rows[0].current_util_pct == 0.0
    assert view.average_util_pct == 0.0
    assert data.ignored_rows == 0


def test_mixed_system_rows_are_silent_but_bad_gpu_rows_warn(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    system = {
        "record_version": 3,
        "kind": "system",
        "sample_ts": now.isoformat(),
        "cpu_util_pct": 25,
    }
    unknown = {"record_version": 3, "kind": "future", "sample_ts": now.isoformat()}
    missing_identity = _row(now, 0, 50)
    missing_identity.pop("global_gpu_index")
    unsupported = _row(now, 0, 50, version=1)
    _write_metrics(
        tmp_path,
        "job:training:0",
        [system, unknown, _row(now, 0, 0), missing_identity, unsupported, "{bad json"],
    )

    data = load_resource_data(tmp_path)
    assert data.ignored_rows == 3
    assert data.sub_jobs["job:training:0"].views["all"].rows[0].current_util_pct == 0


def test_extreme_timestamp_rows_are_ignored_while_valid_rows_render(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    too_early = _row(now, 0, 10)
    too_early["sample_ts"] = "0001-01-01T00:00:00Z"
    overflowing = _row(now, 1, 20)
    overflowing["sample_ts"] = "9999-12-31T23:59:59-05:00"
    _write_metrics(
        tmp_path,
        "job:training:0",
        [too_early, overflowing, _row(now, 2, 30)],
    )

    data = load_resource_data(tmp_path)
    view = data.sub_jobs["job:training:0"].views["all"]
    assert [row.global_gpu_index for row in view.rows] == [2]
    assert data.ignored_rows == 2
    assert render_chart(view, 30)


def test_kind_gpu_and_invalid_values_keep_latest_without_weighting(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    missing_interval = _row(now, 0, float("nan"), kind="gpu")
    missing_interval["sample_interval_s"] = None
    later = _row(now + timedelta(seconds=5), 0, float("inf"), kind="gpu")
    later["fb_used_bytes"] = float("inf")
    later["sample_interval_s"] = -1
    _write_metrics(tmp_path, "job:training:0", [missing_interval, later])

    view = load_resource_data(tmp_path).sub_jobs["job:training:0"].views["all"]
    assert view.rows[0].current_util_pct is None
    assert view.rows[0].fb_used_bytes is None
    assert view.rows[0].window_avg_util_pct is None
    assert view.average_util_pct is None


def test_overflowing_number_and_deep_json_are_ignored_without_crashing(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    overflow = _row(now, 0, 10)
    overflow["gpu_util_pct"] = 10**400
    overflow["fb_used_bytes"] = 1e308
    oversized_interval = _row(now, 1, 50, interval=1e308)
    deeply_nested = "[" * 1100 + "0" + "]" * 1100
    _write_metrics(
        tmp_path,
        "job:training:0",
        [overflow, oversized_interval, deeply_nested],
    )

    data = load_resource_data(tmp_path)
    view = data.sub_jobs["job:training:0"].views["all"]
    assert view.rows[0].current_util_pct is None
    assert view.rows[0].fb_used_bytes is None
    assert view.rows[1].current_util_pct == 50
    assert view.rows[1].window_avg_util_pct is None
    assert data.ignored_rows == 1
    chart = render_chart(view, 30)
    assert all(len(line) <= 30 for line in chart.splitlines())


def test_gpu_kind_still_requires_v2_and_matching_sub_job(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    wrong_version = _row(now, 0, 10, kind="gpu", version=1)
    wrong_sub_job = _row(now, 1, 20, kind="gpu")
    _write_metrics(tmp_path, "job:training:0", [wrong_version, wrong_sub_job])
    path = tmp_path / "job:training:0" / "gpu.jsonl"
    docs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    docs[1]["sub_job_id"] = "different:training:0"
    payload = "".join(json.dumps(doc) + "\n" for doc in docs)
    path.write_text(payload, encoding="utf-8")
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size}),
        encoding="utf-8",
    )

    data = load_resource_data(tmp_path)
    assert data.sub_jobs == {}
    assert data.ignored_rows_by_sub_job == {"job:training:0": 2}


def test_unsupported_rows_are_tracked_per_sub_job_without_valid_samples(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _write_metrics(tmp_path, "job:training:0", [_row(now, 0, 10, version=1)])
    _write_metrics(tmp_path, "job:sampling:0", [_row(now, 0, 20), "{bad json"])

    data = load_resource_data(tmp_path)
    assert "job:training:0" not in data.sub_jobs
    assert data.ignored_rows_by_sub_job == {
        "job:training:0": 1,
        "job:sampling:0": 1,
    }
    assert data.ignored_rows == 2


def test_identity_is_global_gpu_within_sub_job(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _write_metrics(
        tmp_path,
        "job:training:0",
        [_row(now, 0, 10, local_gpu=0), _row(now, 1, 90, local_gpu=0)],
    )
    _write_metrics(tmp_path, "job:sampling:0", [_row(now, 0, 30, local_gpu=0)])

    data = load_resource_data(tmp_path)
    training = data.sub_jobs["job:training:0"].views["all"]
    sampling = data.sub_jobs["job:sampling:0"].views["all"]
    assert [row.global_gpu_index for row in training.rows] == [0, 1]
    assert [row.current_util_pct for row in training.rows] == [10, 90]
    assert [row.current_util_pct for row in sampling.rows] == [30]


def test_regular_five_second_series_is_contiguous_and_includes_latest(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [_row(start + timedelta(seconds=5 * i), 0, 10) for i in range(61)]
    _write_metrics(tmp_path, "job:training:0", rows)

    view = load_resource_data(tmp_path).sub_jobs["job:training:0"].views["5m"]
    assert len(view.chart) == 61
    assert all(value == 10 for value in view.chart)
    assert view.end == start + timedelta(minutes=5)


def test_regular_series_with_scrape_jitter_stays_contiguous(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        _row(
            start + timedelta(seconds=5 * i, milliseconds=40 if i % 2 == 0 else -40),
            0,
            90,
        )
        for i in range(61)
    ]
    _write_metrics(tmp_path, "job:training:0", rows)

    view = load_resource_data(tmp_path).sub_jobs["job:training:0"].views["5m"]
    assert len(view.chart) == 61
    assert all(value == 90 for value in view.chart)


def test_ranges_are_per_sub_job_bounded_and_preserve_gaps(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _write_metrics(
        tmp_path,
        "job:training:0",
        [_row(start + timedelta(seconds=5 * i), 0, 10) for i in range(1000)],
    )
    _write_metrics(
        tmp_path,
        "job:sampling:0",
        [_row(start, 0, 20), _row(start + timedelta(seconds=10), 0, 20)],
    )

    data = load_resource_data(tmp_path)
    training = data.sub_jobs["job:training:0"].views["all"]
    sampling = data.sub_jobs["job:sampling:0"].views["5m"]
    assert len(training.chart) == 240
    assert sampling.start == start
    assert sampling.end == start + timedelta(seconds=10)
    assert sampling.chart == (20, None, 20)


def test_summary_equal_weights_gpus_with_unequal_sample_counts(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [_row(start + timedelta(seconds=5 * i), 0, 0) for i in range(10)]
    rows.append(_row(start + timedelta(seconds=45), 1, 100))
    _write_metrics(tmp_path, "job:training:0", rows)

    view = load_resource_data(tmp_path).sub_jobs["job:training:0"].views["all"]
    assert view.average_util_pct == 50
    assert view.chart[-1] == 50


def test_sample_interval_weights_each_gpu_average(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _write_metrics(
        tmp_path,
        "job:training:0",
        [
            _row(start, 0, 0, interval=5),
            _row(start + timedelta(seconds=5), 0, 100, interval=15),
        ],
    )

    view = load_resource_data(tmp_path).sub_jobs["job:training:0"].views["all"]
    assert view.rows[0].window_avg_util_pct == 75
    assert view.average_util_pct == 75


def test_loading_uses_two_file_passes_for_all_ranges(tmp_path, monkeypatch):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _write_metrics(
        tmp_path,
        "job:training:0",
        [_row(start + timedelta(seconds=5 * i), 0, 10) for i in range(10)],
    )
    original = resource_module._documents
    calls = 0

    def counted_documents(path, committed):
        nonlocal calls
        calls += 1
        yield from original(path, committed)

    monkeypatch.setattr(resource_module, "_documents", counted_documents)
    data = load_resource_data(tmp_path)
    assert set(data.sub_jobs["job:training:0"].views) == {"5m", "15m", "1h", "6h", "all"}
    assert calls == 2


def test_latest_cells_do_not_backfill_and_window_can_be_empty_for_gpu(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    older = _row(start, 0, 30)
    older["fb_used_bytes"] = 123.0
    latest = _row(start + timedelta(minutes=10), 1, 80)
    gpu_zero_latest = _row(start + timedelta(minutes=1), 0, None)
    gpu_zero_latest["fb_used_bytes"] = None
    _write_metrics(tmp_path, "job:training:0", [older, gpu_zero_latest, latest])

    view = load_resource_data(tmp_path).sub_jobs["job:training:0"].views["5m"]
    row0 = view.rows[0]
    assert row0.current_util_pct is None
    assert row0.window_avg_util_pct is None
    assert row0.fb_used_bytes is None
    assert view.fb_gpu_count == 1
    assert view.gpu_count == 2


def test_manifest_bounds_cache_reads(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    path = _write_metrics(tmp_path, "job:training:0", [_row(now, 0, 10)])
    first_size = path.stat().st_size
    with path.open("ab") as handle:
        handle.write(json.dumps(_row(now, 1, 90)).encode() + b"\n")
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": first_size})
    )
    data = load_resource_data(tmp_path)
    assert [row.global_gpu_index for row in data.sub_jobs["job:training:0"].views["all"].rows] == [0]

    path.with_name(".gpu.jsonl.manifest.json").unlink()
    assert load_resource_data(tmp_path).sub_jobs == {}
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size + 1})
    )
    assert load_resource_data(tmp_path).sub_jobs == {}


def test_unterminated_committed_tail_is_not_warned(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    path = _write_metrics(tmp_path, "job:training:0", [_row(now, 0, 10)])
    with path.open("ab") as handle:
        handle.write(b"\n  \n")
        handle.write(b'{"record_version":2')
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size})
    )
    data = load_resource_data(tmp_path)
    assert data.ignored_rows == 0
    assert "job:training:0" in data.sub_jobs


def test_chart_uses_fixed_scale_labels_and_fits_narrow_width(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [_row(start + timedelta(seconds=5 * i), 0, 10) for i in range(61)]
    _write_metrics(tmp_path, "job:training:0", rows)
    view = load_resource_data(tmp_path).sub_jobs["job:training:0"].views["all"]

    chart = render_chart(view, 30)
    first, axis, labels = chart.splitlines()
    assert set(first.removeprefix("100% ┤")) == {"▁"}
    assert len(first) <= 30
    assert len(axis) <= 30
    assert "100%" in first and "0%" in axis
    assert view.start.astimezone().strftime("%H:%M") in labels
    assert view.end.astimezone().strftime("%H:%M") in labels

    narrow = render_chart(view, 20)
    assert all(len(line) <= 20 for line in narrow.splitlines())
    assert view.end.astimezone().strftime("%H:%M") in narrow.splitlines()[-1]
    for width in range(1, 7):
        assert all(len(line) <= width for line in render_chart(view, width).splitlines())

    wide = render_chart(view, 120).splitlines()
    assert all(len(line) == 120 for line in wide)
    assert wide[0].index("▁") == wide[2].index(view.start.astimezone().strftime("%H:%M"))
    assert wide[2].endswith(view.end.astimezone().strftime("%H:%M"))
