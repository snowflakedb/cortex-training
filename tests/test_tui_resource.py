# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cortex_training.tui.resource as resource_module
from cortex_training.tui.resource import load_resource_data
from tests.resource_legacy import gpu_jobs
from tests.resource_legacy import render_chart

parse_sample_epoch = resource_module._sample_epoch


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
    base = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    parsed = parse_sample_epoch("2026-01-01T00:00:00.123456789Z")
    assert parsed == base + 0.123456
    for fraction, microsecond in [
        ("1", 100000),
        ("12", 120000),
        ("1234", 123400),
        ("12345", 123450),
    ]:
        parsed = parse_sample_epoch(f"2026-01-01T00:00:00.{fraction}Z")
        assert parsed is not None
        assert round((parsed - base) * 1_000_000) == microsecond
    assert parse_sample_epoch("2026-01-01T00:00:00Z") == base
    assert parse_sample_epoch("2026-01-01T01:00:00+01:00") == base
    assert parse_sample_epoch("not-a-time") is None
    assert parse_sample_epoch("2026-01-01 00:00:00Z") is None
    assert parse_sample_epoch("2026-01-01T00:00:00") is None  # no zone
    assert parse_sample_epoch("0001-01-01T00:00:00+05:00") is None
    assert parse_sample_epoch("9999-12-31T23:59:59-05:00") is None


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
    view = gpu_jobs(data)["JOB_ID:training:0"].views["all"]
    assert view.rows[0].current_util_pct == 0.0
    assert view.busy.avg == 0.0
    assert data.ignored_rows == 0


def test_mixed_system_rows_are_silent_but_bad_gpu_rows_warn(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    system = {
        "record_version": 3,
        "kind": "system",
        "sample_ts": now.isoformat(),
        "worker_num": 0,
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
    assert gpu_jobs(data)["job:training:0"].views["all"].rows[0].current_util_pct == 0


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
    view = gpu_jobs(data)["job:training:0"].views["all"]
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

    view = gpu_jobs(load_resource_data(tmp_path))["job:training:0"].views["all"]
    assert view.rows[0].current_util_pct is None
    assert view.rows[0].fb_used_bytes is None
    assert view.rows[0].window_avg_util_pct is None
    assert view.busy.avg is None


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
    view = gpu_jobs(data)["job:training:0"].views["all"]
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
    assert gpu_jobs(data) == {}
    assert data.ignored_rows_by_sub_job == {"job:training:0": 2}


def test_unsupported_rows_are_tracked_per_sub_job_without_valid_samples(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _write_metrics(tmp_path, "job:training:0", [_row(now, 0, 10, version=1)])
    _write_metrics(tmp_path, "job:sampling:0", [_row(now, 0, 20), "{bad json"])

    data = load_resource_data(tmp_path)
    assert "job:training:0" not in gpu_jobs(data)
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
    training = gpu_jobs(data)["job:training:0"].views["all"]
    sampling = gpu_jobs(data)["job:sampling:0"].views["all"]
    assert [row.global_gpu_index for row in training.rows] == [0, 1]
    assert [row.current_util_pct for row in training.rows] == [10, 90]
    assert [row.current_util_pct for row in sampling.rows] == [30]


def test_regular_five_second_series_is_contiguous_and_includes_latest(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [_row(start + timedelta(seconds=5 * i), 0, 10) for i in range(61)]
    _write_metrics(tmp_path, "job:training:0", rows)

    view = gpu_jobs(load_resource_data(tmp_path))["job:training:0"].views["10m"]
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

    view = gpu_jobs(load_resource_data(tmp_path))["job:training:0"].views["10m"]
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
    training = gpu_jobs(data)["job:training:0"].views["all"]
    sampling = gpu_jobs(data)["job:sampling:0"].views["10m"]
    assert len(training.chart) == 240
    assert sampling.start == start
    assert sampling.end == start + timedelta(seconds=10)
    assert sampling.chart == (20, None, 20)


def test_summary_weights_gpu_time_for_gpus_with_unequal_sample_counts(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [_row(start + timedelta(seconds=5 * i), 0, 0) for i in range(10)]
    rows.append(_row(start + timedelta(seconds=45), 1, 100))
    _write_metrics(tmp_path, "job:training:0", rows)

    view = gpu_jobs(load_resource_data(tmp_path))["job:training:0"].views["all"]
    # One sample of GPU 1 against ten of GPU 0: it counts for the time it reported.
    assert abs(view.busy.avg - 100 / 11) < 1e-9
    assert view.chart[-1] == 50  # the chart's interval there averages both GPUs


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

    view = gpu_jobs(load_resource_data(tmp_path))["job:training:0"].views["all"]
    assert view.rows[0].window_avg_util_pct == 75
    assert view.busy.avg == 75


def test_loading_reads_each_row_once_for_all_ranges(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    path = _write_metrics(
        tmp_path,
        "job:training:0",
        [_row(start + timedelta(seconds=5 * i), 0, 10) for i in range(10)],
    )
    loader = resource_module.ResourceLoader(tmp_path)
    data = loader.load()
    assert set(gpu_jobs(data)["job:training:0"].views) == {"10m", "30m", "1h", "6h", "all"}
    assert loader.documents_read == 10
    assert loader.bytes_read == path.stat().st_size


def test_latest_cells_do_not_backfill_and_window_can_be_empty_for_gpu(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    older = _row(start, 0, 30)
    older["fb_used_bytes"] = 123.0
    # GPU 1's newest sample is 20 minutes in, so GPU 0 has none in the shortest (10m) window.
    latest = _row(start + timedelta(minutes=20), 1, 80)
    gpu_zero_latest = _row(start + timedelta(minutes=1), 0, None)
    gpu_zero_latest["fb_used_bytes"] = None
    _write_metrics(tmp_path, "job:training:0", [older, gpu_zero_latest, latest])

    view = gpu_jobs(load_resource_data(tmp_path))["job:training:0"].views["10m"]
    row0 = view.rows[0]
    assert row0.current_util_pct is None
    assert row0.window_avg_util_pct is None
    assert row0.fb_used_bytes is None
    assert [row.global_gpu_index for row in view.rows] == [0, 1]
    assert view.rows[1].fb_used_bytes is not None


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
    assert [row.global_gpu_index for row in gpu_jobs(data)["job:training:0"].views["all"].rows] == [0]

    path.with_name(".gpu.jsonl.manifest.json").unlink()
    assert gpu_jobs(load_resource_data(tmp_path)) == {}
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size + 1})
    )
    assert gpu_jobs(load_resource_data(tmp_path)) == {}


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
    assert "job:training:0" in gpu_jobs(data)


def test_chart_uses_fixed_scale_labels_and_fits_narrow_width(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [_row(start + timedelta(seconds=5 * i), 0, 10) for i in range(61)]
    _write_metrics(tmp_path, "job:training:0", rows)
    view = gpu_jobs(load_resource_data(tmp_path))["job:training:0"].views["all"]

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


def test_only_gauges_named_with_a_unit_are_metrics():
    for name in ("gpu_util_pct", "fb_free_bytes", "power_watts", "gpu_temp_c", "sm_active_pct", "mem_used_bytes"):
        assert resource_module.is_metric_field(name), name
    # Ids, counts, durations and counters: averaged over a window as gauges, they would chart nonsense.
    for name in ("pid", "uptime_s", "cpu_count", "seq", "seq_epoch", "worker_num", "global_gpu_index",
                 "record_version", "sample_interval_s", "xid_errors_total", "ecc_dbe_volatile_total",
                 "ecc_sbe_count", "_pct", None, 7):
        assert not resource_module.is_metric_field(name), name
    # A field's name is drawn as the metric's: only plain lowercase words, never an escape sequence.
    for name in ("\x1b]52;c;aGk=\x07_pct", "util\x1b[2J_pct", "GPU_util_pct", "gpu util_pct", "gpu-util_pct",
                 "gpu_util_pct\n", "gpü_util_pct", "\u202egpu_util_pct"):
        assert not resource_module.is_metric_field(name), repr(name)


def test_unitless_fields_get_no_chart_and_the_header_still_names_cpus_and_memory(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(3):
        at = start + timedelta(seconds=5 * i)
        gpu = _row(at, 0, 40)
        gpu.update(power_watts=300.0, pid=99, xid_errors_total=0, ecc_sbe_volatile_total=2 * i)
        gpu["\x1b]0;retitled\x07_pct"] = 50.0
        rows.append(gpu)
        rows.append({
            "kind": "system", "record_version": 3, "sample_ts": at.isoformat().replace("+00:00", "Z"),
            "worker_num": 0, "seq": i, "seq_epoch": "e", "sample_interval_s": 5.0, "cpu_util_pct": 25.0,
            "cpu_count": 88.0, "mem_used_bytes": float(86 * 1024**3), "mem_total_bytes": float(860 * 1024**3),
            "pid": 4242 + i, "uptime_s": 60.0 + 5 * i, "ctx_switches_total": 1000 * i,
        })
    _write_metrics(tmp_path, "job:training:0", rows)

    data = load_resource_data(tmp_path)
    assert data.ignored_rows == 0
    gpu = data.kind("job:training:0", resource_module.GPU)
    system = data.kind("job:training:0", resource_module.SYSTEM)
    assert [metric.key for metric in gpu.metrics] == [
        "gpu_util_pct", "fb_used_pct", "fb_used_bytes", "fb_free_bytes", "power_watts"]
    # The memory used % is derived; the CPUs and the memory limit only go in the header.
    assert [metric.key for metric in system.metrics] == ["cpu_util_pct", "mem_used_pct", "mem_used_bytes"]
    for resource in (gpu, system):
        keys = {metric.key for metric in resource.metrics}
        for unit in (None, 0):
            view = resource.view("all", unit)
            assert set(view.charts) == set(view.stats) == keys
    assert system.view("10m").stats["mem_used_pct"].latest == 10.0
    assert resource_module.node_allocation_text(system) == "nodes: 1 reporting  ·  88 CPUs  ·  860 GiB each"


def test_sub_job_directories_with_unprintable_names_are_passed_over(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _write_metrics(tmp_path, "job:training:0", [_row(now, 0, 10)])
    # A sub-job's id is its directory's name, drawn on the terminal: an escape sequence, a
    # line break, a bidi override or a pile of combining marks is no sub-job's id.
    for name in ("job:training:1\x1b]52;c;aGk=\x07", "job:\x1b[2Jtraining:2", "job:training:3\n",
                 "job\u202e:training:4", "job:training:5" + "\u0301" * 10):
        _write_metrics(tmp_path, name, [_row(now, 0, 20), "{bad json"])

    assert resource_module.cached_metric_sources(tmp_path) == ["job:training:0"]
    data = load_resource_data(tmp_path)
    assert list(data.kinds) == ["job:training:0"]
    assert data.ignored_rows_by_sub_job == {}
