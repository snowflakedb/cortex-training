# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Training metrics for the TUI's Training metrics tab: parsing, stats and charts.

The zone saves the result of each of a training sub-job's ``/fwd-bwd`` and
``/step`` calls to its own file, staged beside the GPU chunks;
``download_training_metrics`` reconstructs one
``<sub_job_id>/training.jsonl`` per sub-job in the TUI cache, next to the
``gpu.jsonl`` the Resources tab reads (see :mod:`resource`). A row is
``"kind": "training"``, one per ``/fwd-bwd`` call and one per ``/step``::

    {"kind": "training", "record_version": 2, "sub_job_id": "...:training:0",
     "sample_ts": "2026-10-16T18:02:14Z", "op": "fwd_bwd", "step": 3, "call": 1,
     "metrics": {"avg_loss": 2.042, "rank": 0, "sft_model_calls": 1, "sft_packed_rows": 1}}
    {"kind": "training", "record_version": 2, "sub_job_id": "...:training:0",
     "sample_ts": "2026-10-16T18:02:14Z", "op": "step", "step": 3, "call": null,
     "metrics": {"global_steps": 3, "grad_norm": 84.37, "last_lr": 1e-04,
                 "update_successful": 1.0, "weight_delta_norm_rank0_sample": 0.145}}

Each row carries the numbers its call's result has, as the client got them, so
nothing is computed here either. The tab shows two views of them, one above the other:

- **per step**: each ``step`` row's metrics, against the step;
- **per call**: each ``fwd_bwd`` row's metrics, against the call, in the order
  the calls ran, each call tagged with the step it counted toward.

Each view leaves out the numbers in :data:`HIDDEN_METRICS`, which aren't training
signals, until ``a`` shows every metric.

A call's ``step`` is null when the zone hadn't yet seen a ``/step`` (a job's
first calls, or the first after the zone restarted); such a call is taken with
the ``step`` row that follows it, unless that row can't be read, and then it
stays unattributed. Two ``step`` rows for one step (a step run again after a
resume) keep the later one whole; every call is kept, since each one ran. A row
seen twice (same ``seq_epoch`` and ``seq``) counts once. Rows of an earlier
record version are counted, not read: they combined a step's calls.

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
# 2: a row per call. Version 1 combined a step's calls into one row; its rows are counted, not read.
RECORD_VERSION = 2
# The file ``download_training_metrics`` writes per sub-job.
FILE_NAME = "training.jsonl"
OP_FWD_BWD = "fwd_bwd"
OP_STEP = "step"

# The tab's two views, drawn one after the other: per step, then per call.
VIEW_STEP = "step"
VIEW_CALL = "call"
VIEWS = (VIEW_STEP, VIEW_CALL)
VIEW_NAMES = {VIEW_STEP: "per step", VIEW_CALL: "per call"}
# Shown first in a view, in this order, when its rows have them; the rest follow in the order the
# file first has them.
LEADING_METRICS = {VIEW_STEP: ("grad_norm", "last_lr"), VIEW_CALL: ("avg_loss",)}
# Numbers a result carries that aren't training signals, left out of a view until ``a`` shows every
# metric: the step itself (``global_steps``), ``update_successful`` (the header counts the steps
# that report it isn't 1, see :attr:`SubJobTraining.failed_updates`), and the fixed size of the
# weight and gradient samples. The packing counts (``rl_model_calls``, ``sft_model_calls``,
# ``sft_packed_rows``) show: they are how a job shows that rows shared a call, or a rank ran fewer.
UPDATE_SUCCESSFUL = "update_successful"
HIDDEN_METRICS = {
    VIEW_STEP: frozenset({
        "global_steps", UPDATE_SUCCESSFUL, "weight_delta_rank0_sample_numel", "weight_delta_rank0_sample_bytes",
        "weight_delta_rank0_sample_tensors", "gradient_sample_numel",
    }),
    VIEW_CALL: frozenset(),
}
# Also left out, but only while every value is 0: the worker a call's result came from, which is
# currently always worker 0. A call from another worker is worth seeing, so then it shows.
HIDDEN_WHILE_ZERO = {VIEW_STEP: frozenset(), VIEW_CALL: frozenset({"rank"})}
# A metric name is drawn as it is, so only plain names are read; anything else is passed over.
_NAME = re.compile(r"[A-Za-z0-9_.:/-]{1,64}")
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
    """One metric of one sub-job, in x order: by step in the per-step view, by call in the per-call view."""

    name: str
    xs: tuple[int, ...]
    values: tuple[float, ...]

    def window(self, first_x: int | None) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """The xs and values from ``first_x`` on (every one when None)."""
        if first_x is None:
            return self.xs, self.values
        start = _bisect_left(self.xs, first_x)
        return self.xs[start:], self.values[start:]


@dataclass(frozen=True)
class WindowStats:
    count: int
    latest: float
    latest_x: int
    low: float
    low_x: int
    high: float
    high_x: int


def window_stats(xs: tuple[int, ...], values: tuple[float, ...]) -> WindowStats | None:
    if not values:
        return None
    low = min(range(len(values)), key=values.__getitem__)
    high = max(range(len(values)), key=values.__getitem__)
    return WindowStats(len(values), values[-1], xs[-1], values[low], xs[low], values[high], xs[high])


def _hidden(view: str, series: MetricSeries) -> bool:
    if series.name in HIDDEN_METRICS[view]:
        return True
    return series.name in HIDDEN_WHILE_ZERO[view] and all(value == 0 for value in series.values)


@dataclass(frozen=True)
class SubJobTraining:
    """One sub-job's training metrics, as of one read."""

    sub_job_id: str
    step_metrics: tuple[MetricSeries, ...]
    call_metrics: tuple[MetricSeries, ...]
    # The step each call (numbered from 1, in the order the calls ran) counted toward; None for
    # calls after the newest step row that the zone couldn't yet tag.
    call_steps: tuple[int | None, ...]
    last_step: int | None  # the newest step row's step
    newest: datetime | None  # the newest row's ``sample_ts``
    skipped: int  # training rows that couldn't be read
    old_format: int = 0  # rows of an earlier record version, saved before this format and not read
    hidden_names: int = 0  # metric names left out because they can't be drawn as they are
    failed_updates: int = 0  # steps that reported ``update_successful`` isn't 1

    @property
    def calls(self) -> int:
        return len(self.call_steps)

    def metrics(self, view: str, *, everything: bool = False) -> tuple[MetricSeries, ...]:
        """``view``'s metrics; without ``everything``, those in :data:`HIDDEN_METRICS` are left out, and
        those in :data:`HIDDEN_WHILE_ZERO` while every value is 0."""
        series = self.call_metrics if view == VIEW_CALL else self.step_metrics
        if everything:
            return series
        return tuple(item for item in series if not _hidden(view, item))

    def hidden(self, view: str) -> int:
        """How many of ``view``'s metrics are left out until every metric is shown."""
        return len(self.metrics(view, everything=True)) - len(self.metrics(view))

    def first_step(self, window: int | None) -> int | None:
        """The first step ``window`` shows: the last ``window`` steps up to the newest one."""
        if window is None or self.last_step is None:
            return None
        return self.last_step - window + 1

    def first_x(self, view: str, window: int | None) -> int | None:
        """Where ``view``'s window starts: a step, or the first call of the window's first step."""
        first = self.first_step(window)
        if view != VIEW_CALL or first is None:
            return first
        for call, step in enumerate(self.call_steps, start=1):
            if step is None or step >= first:
                return call
        return self.calls + 1  # no call in the window

    def step_of_call(self, call: int) -> int | None:
        return self.call_steps[call - 1] if 1 <= call <= self.calls else None


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


def _whole(value: Any, low: int) -> int | None:
    """``value`` when it is an int of at least ``low`` (not a bool), else None."""
    return value if not isinstance(value, bool) and isinstance(value, int) and value >= low else None


def _numbers(metrics: dict) -> Iterable[tuple[str, float]]:
    for name, raw in metrics.items():
        value = _finite(raw)
        if value is not None and isinstance(name, str) and _NAME.fullmatch(name):
            yield name, value


class _SubJobState:
    """What one file's training rows hold so far."""

    def __init__(self, sub_job_id: str) -> None:
        self.sub_job_id = sub_job_id
        # metric -> {step: value} and metric -> {call: value}; dicts keep the order the file first
        # has each metric.
        self.step_values: dict[str, dict[int, float]] = {}
        self.call_values: dict[str, dict[int, float]] = {}
        self.call_steps: list[int | None] = []  # index call - 1
        self.untagged: list[int] = []  # calls with no step yet, for the next step row
        self.seen: set[tuple[str, int]] = set()  # (seq_epoch, seq) of rows read, so a re-sent row counts once
        self.keys: set[tuple[str, str]] = set()  # (op, request_id) of every row read, readable or not
        self.newest: datetime | None = None
        self.last_step: int | None = None
        self.skipped = 0
        self.old_format = 0  # rows of an earlier record version, which this client doesn't read
        self.hidden_names = 0  # metric names left out because they can't be drawn as they are

    def copy(self) -> _SubJobState:
        """An independent copy, to read more rows into without changing this one."""
        twin = _SubJobState(self.sub_job_id)
        twin.step_values = {name: dict(by_step) for name, by_step in self.step_values.items()}
        twin.call_values = {name: dict(by_call) for name, by_call in self.call_values.items()}
        twin.call_steps = list(self.call_steps)
        twin.untagged = list(self.untagged)
        twin.seen = set(self.seen)
        twin.keys = set(self.keys)
        twin.newest, twin.last_step = self.newest, self.last_step
        twin.skipped, twin.old_format, twin.hidden_names = self.skipped, self.old_format, self.hidden_names
        return twin

    def has_rows(self) -> bool:
        return bool(self.step_values or self.call_steps or self.last_step is not None or self.skipped
                    or self.old_format)

    def _skip(self, op: Any) -> None:
        self.skipped += 1
        if op == OP_STEP:
            # The step these calls counted toward was this unreadable row, so the next step row is
            # not theirs: they stay unattributed rather than land on the wrong step.
            self.untagged.clear()

    def ingest(self, doc: dict) -> None:
        version = doc.get("record_version")
        op = doc.get("op")
        key = row_key(doc)
        if key is not None:
            self.keys.add(key)
        if version != RECORD_VERSION:
            if _whole(version, 1) is not None and version < RECORD_VERSION:
                self.old_format += 1
            else:
                self._skip(op)  # a version this client doesn't read yet, or not a version at all
            return
        sub_job = doc.get("sub_job_id")
        metrics = doc.get("metrics")
        raw_step, raw_call = doc.get("step"), doc.get("call")
        step, call_tag = _whole(raw_step, 0), _whole(raw_call, 1)
        if (
            (sub_job is not None and sub_job != self.sub_job_id)
            or not isinstance(metrics, dict)
            or (raw_step is not None and step is None)
            or op not in (OP_FWD_BWD, OP_STEP)
            or (op == OP_STEP and (step is None or raw_call is not None))
            or (op == OP_FWD_BWD and raw_call is not None and call_tag is None)
        ):
            self._skip(op)
            return
        key = (doc.get("seq_epoch"), doc.get("seq"))
        if isinstance(key[0], str) and _whole(key[1], 0) is not None:
            if key in self.seen:
                return  # a chunk delivered twice: the row is already counted
            self.seen.add(key)
        values = dict(_numbers(metrics))
        self.hidden_names += sum(1 for raw in metrics.values() if _finite(raw) is not None) - len(values)
        if op == OP_FWD_BWD:
            self.call_steps.append(step)
            call = len(self.call_steps)
            if step is None:
                self.untagged.append(call)
            for name, value in values.items():
                self.call_values.setdefault(name, {})[call] = value
        else:
            # A step run again (after a resume) replaces its earlier row whole, so the two runs'
            # metrics never mix in one step.
            for by_step in self.step_values.values():
                by_step.pop(step, None)
            for name, value in values.items():
                self.step_values.setdefault(name, {})[step] = value
            # The calls the zone couldn't tag ran before this step, so they counted toward it.
            for call in self.untagged:
                self.call_steps[call - 1] = step
            self.untagged.clear()
            # The newest step row's step, not the highest: after a resume, the job is back there.
            self.last_step = step
        moment = parse_ts(doc.get("sample_ts"))
        if moment is not None and (self.newest is None or moment > self.newest):
            self.newest = moment

    def snapshot(self) -> SubJobTraining:
        return SubJobTraining(
            self.sub_job_id,
            _series({name: by_step for name, by_step in self.step_values.items() if by_step},
                    LEADING_METRICS[VIEW_STEP]),
            _series(self.call_values, LEADING_METRICS[VIEW_CALL]),
            tuple(self.call_steps),
            self.last_step,
            self.newest,
            self.skipped,
            self.old_format,
            self.hidden_names,
            sum(1 for value in self.step_values.get(UPDATE_SUCCESSFUL, {}).values() if value != 1),
        )


def _series(values: dict[str, dict[int, float]], leading: tuple[str, ...]) -> tuple[MetricSeries, ...]:
    order = [name for name in leading if name in values] + [name for name in values if name not in leading]
    out = []
    for name in order:
        by_x = values[name]
        xs = tuple(sorted(by_x))
        out.append(MetricSeries(name, xs, tuple(by_x[x] for x in xs)))
    return tuple(out)


def row_key(doc: dict) -> tuple[str, str] | None:
    """A row's ``(op, request_id)``: the call it came from, which the stage and the local log share."""
    op, request_id = doc.get("op"), doc.get("request_id")
    if op in (OP_FWD_BWD, OP_STEP) and isinstance(request_id, str) and request_id:
        return op, request_id
    return None


def _rows(lines: Iterable[bytes]) -> Iterable[dict]:
    """The training rows among ``lines`` (bytes, one JSON row each)."""
    for raw in lines:
        if _MARK not in raw:
            continue
        try:
            doc = json.loads(raw)
        except (RecursionError, UnicodeDecodeError, ValueError):
            continue  # not a row (the Resources tab counts those)
        if type(doc) is dict and doc.get("kind") == KIND:
            yield doc


def parse_lines(sub_job_id: str, lines: Iterable[bytes], state: _SubJobState | None = None) -> _SubJobState:
    """Read the training rows among ``lines`` (bytes, one JSON row each) into ``state``."""
    state = state or _SubJobState(sub_job_id)
    for doc in _rows(lines):
        state.ingest(doc)
    return state


def _training_files(metrics_root: Path | None) -> list[Path]:
    """Each ``<sub_job_id>/training.jsonl`` under ``metrics_root``, in name order.

    The directory's name is the sub-job's id, which the TUI draws, and it comes
    from the stage's listing, so one that isn't printable as it is is passed
    over rather than shown (as :func:`resource._metric_files` does).
    """
    if metrics_root is None:
        return []
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


@dataclass
class _LocalState:
    """How far a sub-job's local log has been read, and its rows the stage doesn't have yet."""

    identity: tuple[int, int]  # (st_dev, st_ino)
    offset: int = 0
    pending: list[dict] = field(default_factory=list)
    keys: set[tuple[str, str]] = field(default_factory=set)  # of every local row read, to count one once
    # The stage copy with ``pending`` read on top, and what it was built from: the stage file's
    # read position (None without one) and how many of ``pending`` it has.
    merged: _SubJobState | None = None
    base: tuple | None = None
    applied: int = 0


class TrainingLoader:
    """Reads a job's training rows, each file from where it left off.

    A sub-job's rows come from two copies: the stage's (``metrics_root``,
    complete but arriving in batches) and the client's local log
    (``local_root``, written as each result arrives on this machine, but only
    for the calls this machine polled, and capped). The stage's always wins;
    the local log only adds the newest rows the stage hasn't received yet.
    Both carry each call's ``request_id``, so a local row the stage has (by
    ``(op, request_id)``) is passed over, and so is every local row before it:
    the stage has caught up to there. A job opened on another machine has no
    local log, and one run from several machines has only some of its calls in
    each local log; either way the stage's rows are shown whole.

    A stage file the download rebuilt (another inode, a chunk list that no
    longer extends the one read, or fewer committed bytes) is read again from
    the start, and so is a local log that was pruned or replaced. Thread-safe:
    one load at a time.
    """

    def __init__(self, metrics_root: Path, local_root: Path | None = None) -> None:
        self._root = Path(metrics_root)
        self._local_root = Path(local_root) if local_root is not None else None
        self._files: dict[str, _FileState] = {}  # the stage's, by sub-job
        self._local: dict[str, _LocalState] = {}
        self._guard = threading.Lock()

    def load(self) -> TrainingData:
        with self._guard:
            local = {path.parent.name: path for path in _training_files(self._local_root)}
            stage = {path.parent.name: path for path in _training_files(self._root)}
            found: dict[str, SubJobTraining] = {}
            for sub_job_id in sorted(set(local) | set(stage)):
                state = self._read(sub_job_id, stage[sub_job_id]) if sub_job_id in stage else None
                if sub_job_id in local:
                    state = self._merge(sub_job_id, state, local[sub_job_id])
                if state is not None and state.has_rows():
                    found[sub_job_id] = state.snapshot()
            return TrainingData(found)

    def local_sub_jobs(self) -> set[str]:
        """The sub-jobs with a local log on this machine."""
        return {path.parent.name for path in _training_files(self._local_root)}

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

    def _merge(self, sub_job_id: str, stage: _SubJobState | None, path: Path) -> _SubJobState | None:
        """``stage`` with the local log's rows it doesn't have yet read on top (``stage`` itself
        when it has them all)."""
        local = self._read_local(sub_job_id, path)
        if local is None:
            return stage
        known = self._files.get(sub_job_id)
        base = (known.identity, known.offset) if stage is not None and known is not None else None
        if base != local.base or local.applied < len(local.pending):
            # The stage or the local log gained rows: drop the local ones the stage has, and every
            # one before them, since the stage has caught up to there. That assumes both copies
            # record calls in the same order. Concurrent /fwd-bwd calls can reach the stage in
            # the other order, so an earlier call's local row can be dropped before the stage
            # has it; it reappears with the stage's next batch, which carries it.
            covered = stage.keys if stage is not None else ()
            last = max((i for i, doc in enumerate(local.pending) if row_key(doc) in covered), default=-1)
            if last >= 0:
                del local.pending[: last + 1]
                local.merged, local.applied = None, 0
            if base != local.base:
                local.merged, local.base, local.applied = None, base, 0
        if not local.pending:
            return stage
        if local.merged is None:
            local.merged = stage.copy() if stage is not None else _SubJobState(sub_job_id)
        for doc in local.pending[local.applied:]:
            local.merged.ingest(doc)
        local.applied = len(local.pending)
        return local.merged

    def _read_local(self, sub_job_id: str, path: Path) -> _LocalState | None:
        try:
            stat = path.stat()
        except OSError:
            return self._local.get(sub_job_id)
        identity = (stat.st_dev, stat.st_ino)
        local = self._local.get(sub_job_id)
        if local is None or local.identity != identity or stat.st_size < local.offset:
            local = _LocalState(identity)  # first read, or the file was pruned and written again
            self._local[sub_job_id] = local
        if stat.st_size > local.offset:
            # Appended a whole row at a time and never rebuilt: everything on disk is committed.
            try:
                with open(path, "rb") as handle:
                    handle.seek(local.offset)
                    chunk = handle.read(stat.st_size - local.offset)
            except OSError:
                return local
            end = chunk.rfind(b"\n") + 1
            for doc in _rows(chunk[:end].splitlines()):
                key = row_key(doc)
                if key is not None:
                    if key in local.keys:
                        continue  # one result written twice: it is one call
                    local.keys.add(key)
                local.pending.append(doc)
            local.offset += end
        return local


def load_training_data(metrics_root: Path) -> TrainingData:
    """One full read of a telemetry cache directory, keeping no state."""
    return TrainingLoader(metrics_root).load()


def _bisect_left(xs: tuple[int, ...], target: int) -> int:
    low, high = 0, len(xs)
    while low < high:
        middle = (low + high) // 2
        if xs[middle] < target:
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


def _columns(xs: tuple[int, ...], values: tuple[float, ...], first: int, last: int,
             width: int) -> list[float | None]:
    """The chart's ``width`` columns from x ``first`` to ``last``: the mean of the xs each covers, None
    where none has a value. With fewer xs than columns, each x is as many columns wide, give or take
    one where they don't divide evenly, so a short window still spans the chart."""
    span = last - first + 1
    count = min(width, span)
    sums = [0.0] * count
    hits = [0] * count
    for x, value in zip(xs, values):
        column = (x - first) * count // span
        if 0 <= column < count:
            sums[column] += value
            hits[column] += 1
    columns = [sums[i] / hits[i] if hits[i] else None for i in range(count)]
    return [columns[column * count // width] for column in range(width)]


def _x_labels(first: int, last: int, chart_width: int) -> str:
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


def chart_spans(series: MetricSeries, first_x: int | None, width: int, *, height: int = 6,
                hue: str | None = None) -> list[list[tuple[str, str]]]:
    """``series``'s chart from ``first_x`` on, as lines of ``(text, style)`` spans.

    The y axis runs from the window's lowest value to its highest (a loss
    doesn't start at 0, and a policy loss can be negative), labelled at the
    top, middle and bottom; the x axis is the step or the call, labelled under
    the bars. With a ``hue``, the bars are in it, lighter toward the top, and
    the frame is muted; without one, every style is empty.
    """
    xs, values = series.window(first_x)
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
    # The window's first x, but never before the series' own first x: a window reaching back
    # past the job's history would otherwise start the axis at a step that never ran (step 0,
    # for a job's first steps) and leave its columns blank.
    first = xs[0] if first_x is None else max(first_x, series.xs[0])
    columns = _columns(xs, values, first, xs[-1], inner)
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
    lines.append([(" " * (axis + 2) + _x_labels(first, xs[-1], chart_width), axis_style)])
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


def no_metrics_text(sub_job_label: str, status: str, *, training: bool, downloaded: bool,
                    view: str = VIEW_STEP, old_format: int = 0) -> str:
    """Why the tab has nothing to show for a sub-job in ``view``."""
    if old_format:
        return (f"{sub_job_label}'s {old_format:,} training rows were saved in an earlier format, which "
                "combined each step's calls; this version of the tab reads one row per call and step")
    if not training:
        return f"{sub_job_label} is not a training sub-job: only training sub-jobs save training metrics"
    if status in ("pending", "placing", "initializing", "queued"):
        return f"no training metrics yet: {sub_job_label} is {status.upper()} (the first rows arrive after its first call)"
    if not downloaded:
        return "loading training metrics…"
    what = "calls" if view == VIEW_CALL else "steps"
    if status == "running":
        return (f"no {what} saved yet: {sub_job_label} is RUNNING; a row is saved for each call and step, "
                "and rows reach the stage in batches")
    return f"{sub_job_label} saved no {what}"
