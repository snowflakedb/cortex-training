# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure parsing and rendering helpers for the TUI GPU resource view."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

RANGES: dict[str, timedelta | None] = {
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "6h": timedelta(hours=6),
    "all": None,
}
_MAX_BUCKETS = 240
_MAX_SAMPLE_INTERVAL_S = 24 * 60 * 60
_MAX_FB_BYTES = float(2**63)
_MIN_SAMPLE_TS = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MAX_SAMPLE_TS = datetime(9998, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)
_RFC3339_RE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)$")
_FRACTION_RE = re.compile(r"\.(\d+)(?=Z$|[+-]\d\d:\d\d$)")


@dataclass(frozen=True)
class GPUSample:
    sample_ts: datetime
    sub_job_id: str
    global_gpu_index: int
    gpu_util_pct: float | None
    fb_used_bytes: float | None
    fb_free_bytes: float | None
    sample_interval_s: float | None


@dataclass(frozen=True)
class GPURow:
    global_gpu_index: int
    current_util_pct: float | None
    window_avg_util_pct: float | None
    fb_used_bytes: float | None
    fb_total_bytes: float | None


@dataclass(frozen=True)
class RangeView:
    name: str
    start: datetime
    end: datetime
    chart: tuple[float | None, ...]
    average_util_pct: float | None
    rows: tuple[GPURow, ...]
    fb_used_bytes: float | None
    fb_total_bytes: float | None
    fb_gpu_count: int
    gpu_count: int


@dataclass(frozen=True)
class SubJobResource:
    sub_job_id: str
    latest_sample_ts: datetime
    views: dict[str, RangeView]


@dataclass(frozen=True)
class ResourceData:
    sub_jobs: dict[str, SubJobResource]
    ignored_rows_by_sub_job: dict[str, int] = field(default_factory=dict)

    @property
    def ignored_rows(self) -> int:
        return sum(self.ignored_rows_by_sub_job.values())


@dataclass
class _Scan:
    first: datetime
    latest: datetime
    min_interval: float | None
    latest_by_gpu: dict[int, GPUSample]
    gpu_ids: set[int]


@dataclass(frozen=True)
class _Grid:
    left: datetime
    latest: datetime
    width_s: float
    count: int

    def index(self, timestamp: datetime) -> int:
        if self.count == 1:
            return 0
        # Anchor on the latest scrape so sub-interval collection jitter does
        # not turn a regular series into alternating double/empty buckets.
        value = self.count - 1 - round((self.latest - timestamp).total_seconds() / self.width_s)
        return max(0, min(self.count - 1, value))


def _finite_number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bounded_number(value, lower: float, upper: float) -> float | None:
    result = _finite_number(value)
    return result if result is not None and lower <= result <= upper else None


def parse_sample_ts(value: object) -> datetime | None:
    if not isinstance(value, str) or _RFC3339_RE.fullmatch(value) is None:
        return None
    normalized = _FRACTION_RE.sub(
        lambda match: "." + match.group(1)[:6].ljust(6, "0"),
        value,
    )
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return None
        parsed = parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        return None
    if not _MIN_SAMPLE_TS <= parsed <= _MAX_SAMPLE_TS:
        return None
    return parsed


def _parse_gpu_row(doc: object, sub_job_id: str) -> tuple[str, GPUSample | None]:
    """Return ``ok``, ``skip`` (expected non-GPU), or ``ignored``."""
    if not isinstance(doc, dict):
        return "ignored", None
    kind = doc.get("kind")
    if kind is not None and kind != "gpu":
        return "skip", None
    if doc.get("record_version") != 2 or doc.get("sub_job_id") != sub_job_id:
        return "ignored", None
    index = doc.get("global_gpu_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        return "ignored", None
    timestamp = parse_sample_ts(doc.get("sample_ts"))
    if timestamp is None:
        return "ignored", None
    interval = _bounded_number(doc.get("sample_interval_s"), 0, _MAX_SAMPLE_INTERVAL_S)
    if interval == 0:
        interval = None
    return (
        "ok",
        GPUSample(
            sample_ts=timestamp,
            sub_job_id=sub_job_id,
            global_gpu_index=index,
            gpu_util_pct=_bounded_number(doc.get("gpu_util_pct"), 0, 100),
            fb_used_bytes=_bounded_number(doc.get("fb_used_bytes"), 0, _MAX_FB_BYTES),
            fb_free_bytes=_bounded_number(doc.get("fb_free_bytes"), 0, _MAX_FB_BYTES),
            sample_interval_s=interval,
        ),
    )


def _manifest_bytes(path: Path) -> int | None:
    manifest = path.with_name("." + path.name + ".manifest.json")
    try:
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        size = path.stat().st_size
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    names = doc.get("names")
    committed = doc.get("bytes")
    if (
        doc.get("version") != 1
        or not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or isinstance(committed, bool)
        or not isinstance(committed, int)
        or committed < 0
        or size < committed
    ):
        return None
    return committed


def _documents(path: Path, committed: int) -> Iterator[object]:
    with path.open("rb") as handle:
        remaining = committed
        while remaining > 0:
            raw = handle.readline(remaining)
            if not raw:
                break
            remaining -= len(raw)
            if not raw.endswith(b"\n"):
                break
            if not raw.strip():
                continue
            try:
                yield json.loads(raw)
            except (RecursionError, UnicodeDecodeError, ValueError):
                yield None


def _scan_file(path: Path, sub_job_id: str, committed: int) -> tuple[_Scan | None, int]:
    first: datetime | None = None
    latest: datetime | None = None
    min_interval: float | None = None
    latest_by_gpu: dict[int, GPUSample] = {}
    gpu_ids: set[int] = set()
    ignored = 0
    for doc in _documents(path, committed):
        status, sample = _parse_gpu_row(doc, sub_job_id)
        if status == "ignored":
            ignored += 1
        if sample is None:
            continue
        first = sample.sample_ts if first is None else min(first, sample.sample_ts)
        latest = sample.sample_ts if latest is None else max(latest, sample.sample_ts)
        if sample.sample_interval_s is not None:
            min_interval = (
                sample.sample_interval_s if min_interval is None else min(min_interval, sample.sample_interval_s)
            )
        gpu_ids.add(sample.global_gpu_index)
        prior = latest_by_gpu.get(sample.global_gpu_index)
        if prior is None or sample.sample_ts >= prior.sample_ts:
            latest_by_gpu[sample.global_gpu_index] = sample
    if first is None or latest is None:
        return None, ignored
    return _Scan(first, latest, min_interval, latest_by_gpu, gpu_ids), ignored


def _grid(scan: _Scan, duration: timedelta | None) -> _Grid:
    left = scan.first if duration is None else max(scan.first, scan.latest - duration)
    span = max(0.0, (scan.latest - left).total_seconds())
    if span <= 0:
        return _Grid(left, scan.latest, 1.0, 1)
    interval = scan.min_interval or span
    width = max(span / (_MAX_BUCKETS - 1), interval)
    count = min(_MAX_BUCKETS, math.floor(span / width) + 1)
    return _Grid(left, scan.latest, width, max(1, count))


def _finish_view(
    scan: _Scan,
    name: str,
    grid: _Grid,
    bucket_values: dict[int, list[list[float]]],
    gpu_values: dict[int, list[float]],
) -> RangeView:
    chart: list[float | None] = []
    for bucket_index in range(grid.count):
        gpu_means = [
            values[bucket_index][0] / values[bucket_index][1]
            for values in bucket_values.values()
            if values[bucket_index][1] > 0
        ]
        chart.append(sum(gpu_means) / len(gpu_means) if gpu_means else None)

    per_gpu_means = {gpu_id: (total / weight if weight > 0 else None) for gpu_id, (total, weight) in gpu_values.items()}
    available_means = [value for value in per_gpu_means.values() if value is not None]
    average = sum(available_means) / len(available_means) if available_means else None

    fb_used = 0.0
    fb_total = 0.0
    fb_count = 0
    rows: list[GPURow] = []
    for gpu_id in sorted(scan.gpu_ids):
        latest = scan.latest_by_gpu[gpu_id]
        used = latest.fb_used_bytes
        free = latest.fb_free_bytes
        total = used + free if used is not None and free is not None else None
        if used is not None and total is not None:
            fb_used += used
            fb_total += total
            fb_count += 1
        current = latest.gpu_util_pct
        rows.append(
            GPURow(
                global_gpu_index=gpu_id,
                current_util_pct=current,
                window_avg_util_pct=per_gpu_means[gpu_id],
                fb_used_bytes=used,
                fb_total_bytes=total,
            )
        )
    return RangeView(
        name=name,
        start=grid.left,
        end=grid.latest,
        chart=tuple(chart),
        average_util_pct=average,
        rows=tuple(rows),
        fb_used_bytes=fb_used if fb_count else None,
        fb_total_bytes=fb_total if fb_count else None,
        fb_gpu_count=fb_count,
        gpu_count=len(scan.gpu_ids),
    )


def _build_views(path: Path, committed: int, sub_job_id: str, scan: _Scan) -> dict[str, RangeView]:
    grids = {name: _grid(scan, duration) for name, duration in RANGES.items()}
    buckets = {
        name: {gpu_id: [[0.0, 0.0] for _ in range(grid.count)] for gpu_id in scan.gpu_ids}
        for name, grid in grids.items()
    }
    totals = {name: {gpu_id: [0.0, 0.0] for gpu_id in scan.gpu_ids} for name in RANGES}
    for doc in _documents(path, committed):
        _status, sample = _parse_gpu_row(doc, sub_job_id)
        if sample is None or sample.gpu_util_pct is None or sample.sample_interval_s is None:
            continue
        weight = sample.sample_interval_s
        for name, grid in grids.items():
            if sample.sample_ts < grid.left or sample.sample_ts > grid.latest:
                continue
            bucket = buckets[name][sample.global_gpu_index][grid.index(sample.sample_ts)]
            bucket[0] += sample.gpu_util_pct * weight
            bucket[1] += weight
            total = totals[name][sample.global_gpu_index]
            total[0] += sample.gpu_util_pct * weight
            total[1] += weight
    return {
        name: _finish_view(
            scan,
            name,
            grids[name],
            buckets[name],
            totals[name],
        )
        for name in RANGES
    }


def load_resource_data(metrics_root: Path) -> ResourceData:
    sub_jobs: dict[str, SubJobResource] = {}
    ignored_by_sub_job: dict[str, int] = {}
    try:
        paths = sorted(metrics_root.glob("*/gpu.jsonl"))
    except OSError:
        paths = []
    for path in paths:
        committed = _manifest_bytes(path)
        if committed is None:
            continue
        sub_job_id = path.parent.name
        try:
            scan, file_ignored = _scan_file(path, sub_job_id, committed)
        except OSError:
            continue
        if file_ignored:
            ignored_by_sub_job[sub_job_id] = file_ignored
        if scan is None:
            continue
        try:
            views = _build_views(path, committed, sub_job_id, scan)
        except OSError:
            continue
        sub_jobs[sub_job_id] = SubJobResource(
            sub_job_id=sub_job_id,
            latest_sample_ts=scan.latest,
            views=views,
        )
    return ResourceData(
        sub_jobs=sub_jobs,
        ignored_rows_by_sub_job=ignored_by_sub_job,
    )


def cached_metric_sources(metrics_root: Path) -> list[str]:
    try:
        paths = sorted(metrics_root.glob("*/gpu.jsonl"))
    except OSError:
        return []
    return [path.parent.name for path in paths if _manifest_bytes(path) is not None]


_BLOCKS = "▁▂▃▄▅▆▇█"


def _collapse(values: tuple[float | None, ...], width: int) -> list[float | None]:
    if width <= 0 or not values:
        return []
    if len(values) <= width:
        return list(values)
    collapsed: list[float | None] = []
    for column in range(width):
        start = math.floor(column * len(values) / width)
        end = max(start + 1, math.floor((column + 1) * len(values) / width))
        present = [value for value in values[start:end] if value is not None]
        collapsed.append(sum(present) / len(present) if present else None)
    return collapsed


def render_chart(view: RangeView, width: int) -> str:
    if width <= 0:
        return ""
    if width < 7:
        values = _collapse(view.chart, width)
        return "".join(
            " " if value is None else _BLOCKS[min(len(_BLOCKS) - 1, max(0, math.ceil(value / 100 * len(_BLOCKS)) - 1))]
            for value in values
        )
    inner = max(1, width - 6)
    values = _collapse(view.chart, inner)
    glyphs = "".join(
        (" " if value is None else _BLOCKS[min(len(_BLOCKS) - 1, max(0, math.ceil(value / 100 * len(_BLOCKS)) - 1))])
        for value in values
    )
    start = view.start.astimezone().strftime("%H:%M")
    middle = (view.start + (view.end - view.start) / 2).astimezone().strftime("%H:%M")
    end = view.end.astimezone().strftime("%H:%M")
    chart_width = len(glyphs)
    if chart_width >= 17:
        left_gap = (chart_width - 15) // 2
        right_gap = chart_width - 15 - left_gap
        labels = start + (" " * left_gap) + middle + (" " * right_gap) + end
    elif chart_width >= 11:
        labels = start + (" " * (chart_width - 10)) + end
    else:
        labels = end[-chart_width:].rjust(chart_width)
    padding = " " * (inner - chart_width)
    return f"100% ┤{padding}{glyphs}\n  0% └{padding}{'─' * chart_width}\n      {padding}{labels}"


def format_percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}%"


def format_bytes(value: float | None) -> str:
    if value is None:
        return "—"
    gib = value / (1024**3)
    return f"{gib:.1f} GiB"
