# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""A load's GPU kind read the way the older Resources tests read it: busy and memory, per GPU.

These tests check the shared machinery (time weighting, holds, bins and their
bounds, incremental and newest-first reads, stale GPUs) through GPU busy
(``gpu_util_pct``) and memory used % (``fb_used_pct``). The shapes here are
built from :class:`KindResource`'s public API only: each window's view of the
mean across GPUs and of each GPU, and each GPU's newest values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from cortex_training.tui.resource import GPU
from cortex_training.tui.resource import RANGES
from cortex_training.tui.resource import KindResource
from cortex_training.tui.resource import KindView
from cortex_training.tui.resource import Metric
from cortex_training.tui.resource import ResourceData
from cortex_training.tui.resource import SeriesStats
from cortex_training.tui.resource import render_metric_chart

BUSY, MEMORY = "gpu_util_pct", "fb_used_pct"
_GIB = float(1024**3)


@dataclass(frozen=True)
class GPURow:
    global_gpu_index: int
    current_util_pct: float | None
    window_avg_util_pct: float | None
    fb_used_bytes: float | None
    fb_total_bytes: float | None
    current_mem_pct: float | None = None
    window_avg_mem_pct: float | None = None
    sm_active_pct: float | None = None
    tensor_active_pct: float | None = None
    dram_active_pct: float | None = None
    reporting: bool = True


@dataclass(frozen=True)
class RangeView:
    name: str
    start: datetime
    end: datetime
    chart: tuple
    rows: tuple
    mem_chart: tuple = ()
    busy: SeriesStats = SeriesStats()
    memory: SeriesStats = SeriesStats()
    gpus: int = 0  # GPUs with a sample in the range
    gpu_series: dict = field(default_factory=dict, hash=False)  # gpu id -> (busy, memory) chart values
    gpu_stats: dict = field(default_factory=dict, hash=False)  # gpu id -> (busy, memory) stats
    # The view of the mean it was read from, and its busy and memory metrics (to draw them).
    kind_view: KindView | None = field(default=None, compare=False, repr=False)
    metrics: dict = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True)
class SubJob:
    sub_job_id: str
    latest_sample_ts: datetime
    views: dict
    first_sample_ts: datetime | None = None
    sample_interval_s: float | None = None
    reported_interval_s: float | None = None
    gpu_ids: tuple = ()
    reporting_gpu_ids: tuple = ()
    partial: bool = False
    value_interval_s: float | None = None
    kind: KindResource | None = field(default=None, compare=False, repr=False)


def _bytes(value: float | None) -> float | None:
    return None if value is None else value * _GIB


def _range(resource: KindResource, name: str) -> RangeView:
    mean = resource.view(name)
    count = len(next(iter(mean.charts.values()), ()))

    def chart(view, key):
        return view.chart(key) or (None,) * count

    rows, series, stats = [], {}, {}
    for unit in resource.units:
        own = resource.view(name, unit.id)
        latest = unit.latest
        used, free = _bytes(latest.get("fb_used_bytes")), _bytes(latest.get("fb_free_bytes"))
        rows.append(GPURow(
            global_gpu_index=unit.id,
            current_util_pct=latest.get(BUSY),
            window_avg_util_pct=own.stats[BUSY].avg if BUSY in own.stats else None,
            fb_used_bytes=used,
            fb_total_bytes=used + free if used is not None and free is not None else None,
            current_mem_pct=latest.get(MEMORY),
            window_avg_mem_pct=own.stats[MEMORY].avg if MEMORY in own.stats else None,
            sm_active_pct=latest.get("sm_active_pct"),
            tensor_active_pct=latest.get("tensor_active_pct"),
            dram_active_pct=latest.get("dram_active_pct"),
            reporting=unit.reporting,
        ))
        series[unit.id] = (chart(own, BUSY), chart(own, MEMORY))
        stats[unit.id] = (own.stats.get(BUSY, SeriesStats()), own.stats.get(MEMORY, SeriesStats()))
    metrics = {key: resource.metric(key) or Metric(key=key, name=key, unit="%", scale=100.0) for key in (BUSY, MEMORY)}
    return RangeView(name=name, start=mean.start, end=mean.end, chart=chart(mean, BUSY), rows=tuple(rows),
                     mem_chart=chart(mean, MEMORY), busy=mean.stats.get(BUSY, SeriesStats()),
                     memory=mean.stats.get(MEMORY, SeriesStats()), gpus=mean.units, gpu_series=series,
                     gpu_stats=stats, kind_view=mean, metrics=metrics)


def render_chart(view: RangeView, width: int, *, series: str = "busy", height: int = 1) -> str:
    """The mean's busy (or ``series="memory"``) chart, as the Resources tab draws a percentage."""
    metric = view.metrics[BUSY if series == "busy" else MEMORY]
    return render_metric_chart(view.kind_view, metric, width, height=height)


def legacy(resource: KindResource | None) -> SubJob | None:
    """``resource`` (a GPU kind) as the older tests read it."""
    if resource is None:
        return None
    return SubJob(
        sub_job_id=resource.sub_job_id,
        latest_sample_ts=resource.latest_sample_ts,
        views={name: _range(resource, name) for name in RANGES},
        first_sample_ts=resource.first_sample_ts,
        sample_interval_s=resource.sample_interval_s,
        reported_interval_s=resource.reported_interval_s,
        gpu_ids=resource.unit_ids,
        reporting_gpu_ids=resource.reporting_unit_ids,
        partial=resource.partial,
        value_interval_s=resource.value_interval_s,
        kind=resource,
    )


def gpu_jobs(data: ResourceData) -> dict[str, SubJob]:
    """Each sub-job's GPU kind in a load, as the older tests read it (sub-jobs without GPU rows left out)."""
    return {sub_job_id: legacy(kinds[GPU]) for sub_job_id, kinds in data.kinds.items() if GPU in kinds}


