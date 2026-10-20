# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Training metrics for the TUI's Training metrics tab: parsing, stats and charts.

The zone saves the metrics a training sub-job's ``/fwd-bwd`` and ``/step``
calls return to its own file, staged beside the GPU chunks;
``download_training_metrics`` reconstructs one
``<sub_job_id>/training.jsonl`` per sub-job in the TUI cache, next to the
``gpu.jsonl`` the Resources tab reads (see :mod:`resource`). A row is
``"kind": "training"``, one per step for each call type::

    {"kind": "training", "record_version": 1, "sub_job_id": "...:training:0",
     "sample_ts": "2026-10-16T18:02:14Z", "op": "fwd_bwd", "step": 7,
     "metrics": {"sum_loss": 0.0274, "num_calls": 2.0, "entropy": 0.8636}}
    {"kind": "training", "record_version": 1, "sub_job_id": "...:training:0",
     "sample_ts": "2026-10-16T18:02:14Z", "op": "step", "step": 7,
     "metrics": {"grad_norm": 0.873, "last_lr": 1e-05}}

The ``fwd_bwd`` row combines every ``/fwd-bwd`` call of the step: ``sum_loss``
is the sum of the calls' losses and ``num_calls`` is how many were combined.
Whether a step's loss is that sum or its mean depends on how the loss was
scaled, which the row deliberately leaves to the reader, so the tab charts
both: ``mean_loss`` is derived here as ``sum_loss / num_calls`` (the usual
reading), with ``sum_loss`` beside it. ``num_calls`` itself is a count, not a
measurement, so it gets no chart of its own.

Every other finite number in ``metrics`` is a metric, charted against the step.
Two rows of one call type and step (a step run again after a resume) keep the
later one in the file.

:class:`TrainingLoader` reads only what the download committed since its last
read. Pure stdlib, NO ``textual`` import, like ``resource.py``.
"""

from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Iterable

from cortex_training.tui.format import parse_ts
from cortex_training.tui.resource import CHART_AXIS
from cortex_training.tui.resource import CHART_GRID
from cortex_training.tui.resource import _BLOCKS
from cortex_training.tui.resource import _printable_name
from cortex_training.tui.resource import _read_manifest

KIND = "training"
RECORD_VERSION = 1
# The file ``download_training_metrics`` writes per sub-job.
FILE_NAME = "training.jsonl"
# Derived from a fwd_bwd row's ``sum_loss`` and ``num_calls`` (see the module docstring).
MEAN_LOSS = "mean_loss"
# Shown first, in this order, when a sub-job has them; the rest follow in the order the file first has them.
LEADING_METRICS = (MEAN_LOSS, "sum_loss", "grad_norm", "last_lr")
# Numbers in ``metrics`` that count or label rather than measure, so they get no chart.
_NOT_METRICS = frozenset({"num_calls", "rank"})
# A metric name is drawn as it is, so only plain names are read; anything else counts the row as skipped.
_NAME = re.compile(r"[A-Za-z0-9_.:/-]{1,64}")
_OP = re.compile(r"[a-z0-9_]{1,32}")
# The fast filter: a line without this can't be a training row, and isn't parsed.
_MARK = b'"training"'

# The windows ``w`` cycles through: every step, then the last N steps.
WINDOWS: tuple[int | None, ...] = (None, 100, 1000)
DEFAULT_WINDOW: int | None = None


def window_label(window: int | None) -> str:
    return "all steps" if window is None else f"last {window:,} steps"


def next_window(window: int | None) -> int | None:
    index = WINDOWS.index(window) if window in WINDOWS else 0
    return WINDOWS[(index + 1) % len(WINDOWS)]


# ─── snapshots ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MetricSeries:
    """One metric of one sub-job: its values by step, in step order."""

    name: str
    op: str
    steps: tuple[int, ...]
    values: tuple[float, ...]

    def window(self, first_step: int | None) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """The steps and values from ``first_step`` on (every one when None)."""
        if first_step is None:
            return self.steps, self.values
        start = _bisect_left(self.steps, first_step)
        return self.steps[start:], self.values[start:]


@dataclass(frozen=True)
class WindowStats:
    count: int
    latest: float
    latest_step: int
    low: float
    low_step: int
    high: float
    high_step: int


def window_stats(steps: tuple[int, ...], values: tuple[float, ...]) -> WindowStats | None:
    if not values:
        return None
    low = min(range(len(values)), key=values.__getitem__)
    high = max(range(len(values)), key=values.__getitem__)
    return WindowStats(len(values), values[-1], steps[-1], values[low], steps[low], values[high], steps[high])


@dataclass(frozen=True)
class SubJobTraining:
    """One sub-job's training metrics, as of one read."""

    sub_job_id: str
    metrics: tuple[MetricSeries, ...]
    last_step: int | None
    newest: datetime | None  # the newest row's ``sample_ts``
    skipped: int  # training rows that couldn't be read

    def first_step(self, window: int | None) -> int | None:
        """The first step ``window`` shows: the last ``window`` steps up to the newest one."""
        if window is None or self.last_step is None:
            return None
        return self.last_step - window + 1


@dataclass(frozen=True)
class TrainingData:
    """Every sub-job with training rows (or training rows skipped), in sub-job id order."""

    sub_jobs: dict[str, SubJobTraining] = field(default_factory=dict)

    def get(self, sub_job_id: str | None) -> SubJobTraining | None:
        return self.sub_jobs.get(sub_job_id) if sub_job_id else None


# ─── parsing ─────────────────────────────────────────────────────────────────


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class _SubJobState:
    """What one file's training rows hold so far."""

    def __init__(self, sub_job_id: str) -> None:
        self.sub_job_id = sub_job_id
        # metric key -> {step: value}; dicts keep the order the file first has each metric.
        self.values: dict[str, dict[int, float]] = {}
        self.ops: dict[str, str] = {}  # metric key -> the call type its rows are
        self.newest: datetime | None = None
        self.last_step: int | None = None
        self.skipped = 0

    def ingest(self, doc: dict) -> None:
        if doc.get("record_version") != RECORD_VERSION:
            self.skipped += 1  # a version this client doesn't read
            return
        sub_job = doc.get("sub_job_id")
        op = doc.get("op")
        step = doc.get("step")
        metrics = doc.get("metrics")
        if (
            (sub_job is not None and sub_job != self.sub_job_id)
            or not isinstance(op, str) or not _OP.fullmatch(op)
            or isinstance(step, bool) or not isinstance(step, int) or step < 0
            or not isinstance(metrics, dict)
        ):
            self.skipped += 1
            return
        for name, raw in metrics.items():
            value = _finite(raw)
            if value is None or name in _NOT_METRICS or not isinstance(name, str) or not _NAME.fullmatch(name):
                continue
            # The same name from two call types is two metrics: the second is named with its call type.
            key = name if self.ops.setdefault(name, op) == op else f"{op}/{name}"
            self.ops.setdefault(key, op)
            self.values.setdefault(key, {})[step] = value
        self._derive_mean_loss(op, step, metrics)
        self.last_step = step if self.last_step is None else max(self.last_step, step)
        moment = parse_ts(doc.get("sample_ts"))
        if moment is not None and (self.newest is None or moment > self.newest):
            self.newest = moment

    def _derive_mean_loss(self, op: str, step: int, metrics: dict) -> None:
        """A row's mean loss per call, from ``sum_loss`` and ``num_calls`` (see the module docstring).

        Left out when the row has no ``num_calls``, or none above zero: the sum
        is then the only loss the row supports, and a made-up divisor would
        chart a number the service never reported.
        """
        total = _finite(metrics.get("sum_loss"))
        calls = _finite(metrics.get("num_calls"))
        if total is None or calls is None or calls < 1:
            return
        self.ops.setdefault(MEAN_LOSS, op)
        self.values.setdefault(MEAN_LOSS, {})[step] = total / calls

    def snapshot(self) -> SubJobTraining:
        order = [name for name in LEADING_METRICS if name in self.values]
        order += [name for name in self.values if name not in LEADING_METRICS]
        metrics = []
        for name in order:
            by_step = self.values[name]
            steps = tuple(sorted(by_step))
            metrics.append(MetricSeries(name, self.ops[name], steps, tuple(by_step[s] for s in steps)))
        return SubJobTraining(self.sub_job_id, tuple(metrics), self.last_step, self.newest, self.skipped)


def parse_lines(sub_job_id: str, lines: Iterable[bytes], state: _SubJobState | None = None) -> _SubJobState:
    """Read the training rows among ``lines`` (bytes, one JSON row each) into ``state``."""
    state = state or _SubJobState(sub_job_id)
    for raw in lines:
        if _MARK not in raw:
            continue
        try:
            doc = json.loads(raw)
        except (RecursionError, UnicodeDecodeError, ValueError):
            continue  # not a row (the Resources tab counts those)
        if type(doc) is dict and doc.get("kind") == KIND:
            state.ingest(doc)
    return state


def _training_files(metrics_root: Path) -> list[Path]:
    """Each ``<sub_job_id>/training.jsonl`` under ``metrics_root``, in name order.

    The directory's name is the sub-job's id, which the TUI draws, and it comes
    from the stage's listing, so one that isn't printable as it is is passed
    over rather than shown (as :func:`resource._metric_files` does).
    """
    try:
        paths = sorted(metrics_root.glob(f"*/{FILE_NAME}"))
    except OSError:
        return []
    return [path for path in paths if _printable_name(path.parent.name)]


@dataclass
class _FileState:
    identity: tuple[int, int]  # (st_dev, st_ino)
    names: list[str]
    offset: int
    parsed: _SubJobState


class TrainingLoader:
    """Reads a telemetry cache directory's training rows, each file from where it left off.

    A file the download rebuilt (another inode, a chunk list that no longer
    extends the one read, or fewer committed bytes) is read again from the
    start. Thread-safe: one load at a time.
    """

    def __init__(self, metrics_root: Path) -> None:
        self._root = Path(metrics_root)
        self._files: dict[str, _FileState] = {}
        self._guard = threading.Lock()

    def load(self) -> TrainingData:
        with self._guard:
            found: dict[str, SubJobTraining] = {}
            for path in _training_files(self._root):
                sub_job_id = path.parent.name
                state = self._read(sub_job_id, path)
                if state is not None and (state.values or state.skipped):
                    found[sub_job_id] = state.snapshot()
            return TrainingData(found)

    def _read(self, sub_job_id: str, path: Path) -> _SubJobState | None:
        manifest = _read_manifest(path)
        if manifest is None:
            return None  # not committed yet
        names, committed = manifest
        try:
            stat = path.stat()
        except OSError:
            return None
        identity = (stat.st_dev, stat.st_ino)
        known = self._files.get(sub_job_id)
        if (
            known is None
            or known.identity != identity
            or names[: len(known.names)] != known.names
            or committed < known.offset
        ):
            known = _FileState(identity, list(names), 0, _SubJobState(sub_job_id))
            self._files[sub_job_id] = known
        if committed > known.offset:
            try:
                with open(path, "rb") as handle:
                    handle.seek(known.offset)
                    chunk = handle.read(committed - known.offset)
            except OSError:
                return known.parsed
            end = chunk.rfind(b"\n") + 1  # a row is read once it is whole
            parse_lines(sub_job_id, chunk[:end].splitlines(), known.parsed)
            known.offset += end
        known.names = list(names)
        return known.parsed


def load_training_data(metrics_root: Path) -> TrainingData:
    """One full read of a telemetry cache directory, keeping no state."""
    return TrainingLoader(metrics_root).load()


def _bisect_left(steps: tuple[int, ...], target: int) -> int:
    low, high = 0, len(steps)
    while low < high:
        middle = (low + high) // 2
        if steps[middle] < target:
            low = middle + 1
        else:
            high = middle
    return low


# ─── text ────────────────────────────────────────────────────────────────────


def format_metric(value: float | None) -> str:
    """``0.0137``, ``1.00e-05``, ``2,048``; ``—`` without a value."""
    if value is None or value != value:
        return "—"
    value += 0.0  # no "-0"
    size = abs(value)
    if size == 0:
        return "0"
    if size >= 1e5 or size < 1e-3:
        return f"{value:.2e}"
    if size >= 100:
        return f"{value:,.0f}"
    return f"{value:.4g}"


def format_step(step: int | None) -> str:
    return "—" if step is None else f"{step:,}"


def _columns(steps: tuple[int, ...], values: tuple[float, ...], first: int, last: int,
             width: int) -> list[float | None]:
    """The chart's columns from step ``first`` to ``last``: the mean of the steps each covers,
    None where none has a value. With fewer steps than columns, each step is as many columns wide."""
    span = last - first + 1
    count = min(width, span)
    per_step = max(1, width // span) if span <= width else 1
    sums = [0.0] * count
    hits = [0] * count
    for step, value in zip(steps, values):
        column = (step - first) * count // span
        if 0 <= column < count:
            sums[column] += value
            hits[column] += 1
    columns = [sums[i] / hits[i] if hits[i] else None for i in range(count)]
    return [value for value in columns for _ in range(per_step)]


def _step_labels(first: int, last: int, chart_width: int) -> str:
    left, right = format_step(first), format_step(last)
    if first == last or chart_width < len(left) + len(right) + 1:
        return right.rjust(chart_width) if chart_width >= len(right) else ""
    middle = format_step((first + last) // 2)
    if chart_width >= len(left) + len(middle) + len(right) + 4 and middle not in (left, right):
        gap = chart_width - len(left) - len(middle) - len(right)
        return left + " " * (gap // 2) + middle + " " * (gap - gap // 2) + right
    return left + " " * (chart_width - len(left) - len(right)) + right


def _bar_row(values: list[float | None], height: int, level: int, scale: float = 100.0) -> str:
    """Row ``level`` (0 at the bottom) of a ``height``-row bar chart, drawn in eighths of a row."""
    cells = []
    for value in values:
        if value is None:
            cells.append(" ")
            continue
        eighths = min(height * 8, max(1, math.ceil(value / scale * height * 8)))
        fill = eighths - 8 * level
        cells.append("█" if fill >= 8 else " " if fill <= 0 else _BLOCKS[fill - 1])
    return "".join(cells)


def _shade(hue: str, share: float) -> str:
    """``hue`` darkened toward black: ``share`` 1 is the hue itself, 0 is 45% of it."""
    level = 0.45 + 0.55 * max(0.0, min(1.0, share))
    return "#" + "".join(f"{round(int(hue[i:i + 2], 16) * level):02x}" for i in (1, 3, 5))


def chart_spans(series: MetricSeries, first_step: int | None, width: int, *, height: int = 6,
                hue: str | None = None) -> list[list[tuple[str, str]]]:
    """``series``'s chart from ``first_step`` on, as lines of ``(text, style)`` spans.

    The y axis runs from the window's lowest value to its highest (a loss
    doesn't start at 0, and a policy loss can be negative), labelled at the
    top, middle and bottom; the x axis is the step, labelled under the bars.
    With a ``hue``, the bars are in it, lighter toward the top, and the frame
    is muted; without one, every style is empty.
    """
    steps, values = series.window(first_step)
    if not values:
        return [[("no samples in this window", CHART_AXIS if hue else "")]]
    low, high = min(values), max(values)
    if high == low:  # a flat line, drawn at half height
        pad = abs(low) * 0.5 or 1.0
        low, high = low - pad, high + pad
    labels = (format_metric(high), format_metric((low + high) / 2), format_metric(low))
    axis = max(len(label) for label in labels)
    height = max(2, height)
    inner = max(1, width - axis - 2)
    first = steps[0] if first_step is None else max(first_step, 0)
    columns = _columns(steps, values, first, steps[-1], inner)
    # Shifted so the lowest value is the bottom of the chart; it still draws one eighth of a row.
    shifted = [None if value is None else value - low for value in columns]
    scale = high - low
    bar = (lambda row: _shade(hue, (row + 1) / height)) if hue else (lambda row: "")
    axis_style, grid_style = (CHART_AXIS, CHART_GRID) if hue else ("", "")
    chart_width = len(shifted)
    middle_row = height // 2
    lines = []
    for index in range(height):
        label = labels[0] if index == 0 else labels[1] if index == middle_row else ""
        mark = f"{label:>{axis}} ┤" if label else " " * axis + " │"
        level = height - 1 - index
        cells = _bar_row(shifted, height, level, scale)
        line = [(mark, axis_style)]
        if index == middle_row and height > 2:
            line += [(run, grid_style if run[0] == "┈" else bar(level)) for run in _runs(cells.replace(" ", "┈"))]
        else:
            line.append((cells, bar(level)))
        lines.append(line)
    lines.append([(f"{labels[2]:>{axis}} └{'─' * chart_width}", axis_style)])
    lines.append([(" " * (axis + 2) + _step_labels(first, steps[-1], chart_width), axis_style)])
    return lines


def _runs(cells: str) -> list[str]:
    """``cells`` split into runs of gridline and of bar."""
    runs: list[str] = []
    for cell in cells:
        grid = cell == "┈"
        if runs and (runs[-1][0] == "┈") == grid:
            runs[-1] += cell
        else:
            runs.append(cell)
    return runs


def no_metrics_text(sub_job_label: str, status: str, *, training: bool, downloaded: bool) -> str:
    """Why the tab has no training metrics for a sub-job."""
    if not training:
        return f"{sub_job_label} is not a training sub-job: only training sub-jobs save training metrics"
    if status in ("pending", "placing", "initializing", "queued"):
        return f"no training metrics yet: {sub_job_label} is {status.upper()} (the first rows arrive after its first step)"
    if not downloaded:
        return "loading training metrics…"
    if status == "running":
        return (f"no training metrics yet: {sub_job_label} is RUNNING; a row is saved at each step, "
                "and rows reach the stage in batches")
    return f"{sub_job_label} saved no training metrics"

