# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The job screen's Training metrics tab.

The tab is one widget that reads and draws its own data, so the job screen
only places it, tells it when it is shown (:meth:`activate`) and when a live
job's periodic refresh is due (:meth:`refresh_live`). It reads the job as the
screen knows it through the ``job`` callable it is given, and fetches its rows
through the screen's ``download``, which writes into the same cache directory
as the Resources tab, under the same lock.

The pane's keys (``v``, ``w``, ``Space``, ``r``) are its own bindings, which win
over the screen's while the focus is inside the pane.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from datetime import timezone
from pathlib import Path
import threading
from typing import Callable

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.containers import VerticalScroll
from textual.widgets import Static
from textual.widgets import TabbedContent
from textual.worker import get_current_worker

from cortex_training import _training_log
from cortex_training.tui import training
from cortex_training.tui import views
from cortex_training.tui.resource import CHART_AXIS
from cortex_training.tui.resource import CHART_MUTED
from cortex_training.tui.resource import chart_hue

_WARNING = "#fbbf24"


class _Pane(Vertical):
    """What both tabs share: posting a worker's answer to the UI thread while the pane is up."""

    def __init__(self, client, job_id: str, *, job: Callable[[], dict], id: str) -> None:
        super().__init__(id=id)
        self._client = client
        self._job_id = job_id
        self._job = job
        self._shown = False  # activated at least once: the first show loads
        self._loading = False
        self._error: str | None = None

    def _post(self, fn, *args) -> None:
        """Worker thread: run ``fn(*args)`` on the UI thread, dropped once the pane is gone."""

        def run() -> None:
            if self.is_attached:
                fn(*args)

        try:
            if self.app.is_running:
                self.app.call_from_thread(run)
        except RuntimeError:  # the app is shutting down
            pass

    def _live(self) -> bool:
        return views.is_live(self._job())

    def is_active_tab(self) -> bool:
        """Whether this pane's tab is the one on screen (its TabPane's id is this id without ``-tab``)."""
        try:
            active = self.screen.query_one("#job-tabs", TabbedContent).active
        except Exception:  # noqa: BLE001 - not mounted, or no screen yet
            return False
        return bool(self.id) and active == self.id.removesuffix("-tab")

    def holds_focus(self) -> bool:
        """Whether the focus is on this pane or inside it."""
        try:
            node = self.screen.focused
        except Exception:  # noqa: BLE001 - no screen yet
            return False
        while node is not None:
            if node is self:
                return True
            node = node.parent
        return False

    def claim_focus(self) -> None:
        """Put the focus where this tab's keys work, when this is the tab on screen.

        The focus is left alone once it is inside this pane, where the user may have moved it.
        """
        if not self.is_active_tab() or self.holds_focus():
            return
        try:
            self.screen.set_focus(self.focus_target())
        except Exception:  # noqa: BLE001 - not mounted yet
            pass

    def activate(self) -> None:
        """UI thread. The tab was shown: the first time, read its data."""
        if not self._shown:
            self._shown = True
            self.reload()
        else:
            self.redraw()
        self.claim_focus()

    def refresh_live(self) -> None:
        """UI thread. A live job's periodic refresh, while the tab is shown."""
        if self._shown and self._live():
            self.reload()

    def action_refresh(self) -> None:
        self._shown = True
        self.reload()

    def reload(self) -> None:
        raise NotImplementedError

    def redraw(self) -> None:
        raise NotImplementedError


# ─── Training metrics ────────────────────────────────────────────────────────

_LOADERS: OrderedDict[str, training.TrainingLoader] = OrderedDict()
_LOADERS_GUARD = threading.Lock()
_MAX_LOADERS = 4


def training_loader(metrics_root: Path, local_root: Path | None = None) -> training.TrainingLoader:
    """The shared loader for a cache directory, so reopening a job reads only what is new."""
    key = f"{Path(metrics_root).resolve()}\0{local_root}"
    with _LOADERS_GUARD:
        loader = _LOADERS.pop(key, None) or training.TrainingLoader(Path(metrics_root), local_root)
        _LOADERS[key] = loader
        while len(_LOADERS) > _MAX_LOADERS:
            _LOADERS.popitem(last=False)
        return loader


# Two charts to a row in a pane at least this wide, as on the Resources tab.
_TWO_CHARTS_WIDTH = 90
_CHART_HEIGHT = 6


class TrainingPane(_Pane):
    """One sub-job's training metrics, per step or per call: stats over the window, and a chart per metric."""

    BINDINGS = [
        Binding("v", "cycle_view", "Per step/call"),
        Binding("w", "cycle_window", "Window"),
        Binding("space", "next_sub_job", "Sub-job", key_display="space"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self, client, job_id: str, metrics_root: Path, *, job: Callable[[], dict],
                 download: Callable[..., bool], id: str = "training") -> None:
        super().__init__(client, job_id, job=job, id=id)
        self._metrics_root = Path(metrics_root)
        # The client's own copy of this job's rows, written as each result arrives (see _training_log).
        self._local_root = _training_log.local_dir(job_id)
        self._download = download
        self._data = training.TrainingData()
        self._downloaded = False  # a download finished: a sub-job without rows has none
        self._window = training.DEFAULT_WINDOW
        self._view = training.DEFAULT_VIEW  # per step, or per /fwd-bwd call
        self._pick: str | None = None  # the sub-job Space last showed

    # How often the local log is read while the tab is shown: it gains a row the moment a result
    # arrives, so a step appears within this, not after the stage's batched upload.
    _LOCAL_REFRESH_SECONDS = 2.0

    def on_mount(self) -> None:
        self.set_interval(self._LOCAL_REFRESH_SECONDS, self._refresh_local)

    def _refresh_local(self) -> None:
        # One stat when this machine has no local log for the job (another machine's job), so the
        # poll costs nothing there; a log written later, once calls run here, is still picked up.
        if self._shown and self.is_active_tab() and not self._loading and self._local_root.is_dir():
            self._read_local()

    @work(thread=True, exclusive=True, group="training-local")
    def _read_local(self) -> None:
        worker = get_current_worker()
        loader = training_loader(self._metrics_root, self._local_root)
        if not loader.local_sub_jobs():
            return  # no local log (another machine's job): the stage refresh is all there is
        data = loader.load()  # only the rows appended since the last read
        if not worker.is_cancelled:
            self._post(self._accept, data)

    def compose(self) -> ComposeResult:
        yield Static("", id="training-status")
        with VerticalScroll(id="training-pane"):
            yield Static("", id="training-stats")
            yield Static("", id="training-charts")

    def focus_target(self):
        return self.query_one("#training-pane")

    # ── which sub-job ──
    def _sub_job_ids(self) -> list[str]:
        """The job's training sub-jobs, then any other sub-job with training rows."""
        ids = [str(sj.get("sub_job_id")) for sj in views.sub_jobs(self._job())
               if views.sub_job_kind(sj) == "training" and sj.get("sub_job_id")]
        return list(dict.fromkeys(ids + list(self._data.sub_jobs)))

    def _source(self) -> str | None:
        ids = self._sub_job_ids()
        if self._pick in ids:
            return self._pick
        return ids[0] if ids else None

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        if action == "next_sub_job":
            return len(self._sub_job_ids()) > 1
        if action in ("cycle_window", "cycle_view"):
            return self._source() is not None
        return True

    def action_next_sub_job(self) -> None:
        ids = self._sub_job_ids()
        if len(ids) < 2:
            return
        current = self._source()
        self._pick = ids[(ids.index(current) + 1) % len(ids)] if current in ids else ids[0]
        self.redraw()

    def action_cycle_window(self) -> None:
        self._window = training.next_window(self._window)
        self.redraw()

    def action_cycle_view(self) -> None:
        self._view = training.next_view(self._view)
        self.redraw()

    # ── reading ──
    def reload(self) -> None:
        if self._loading:
            return
        if self._job().get("sub_jobs") and not self._sub_job_ids():
            self._downloaded = True  # no training sub-job: nothing to download for this tab
            self.redraw()
            return
        self._loading = True
        self.redraw()
        self._refresh()

    @work(thread=True, exclusive=True, group="training")
    def _refresh(self) -> None:
        worker = get_current_worker()
        loader = training_loader(self._metrics_root, self._local_root)

        def show() -> None:
            data = loader.load()
            if not worker.is_cancelled:
                self._post(self._accept, data)

        try:
            show()  # what is cached, while the download runs
            source = self._source()
            ok = self._download(priority=[source] if source else [], on_commit=lambda _result: show(),
                                cancelled=lambda: worker.is_cancelled)
            if not ok or worker.is_cancelled:
                return
            data = loader.load()
            if not worker.is_cancelled:
                self._post(self._finish, data, None)
        except Exception as exc:  # noqa: BLE001 - said in the tab's header
            if not worker.is_cancelled:
                self._post(self._finish, None, views.access_hint(exc, "training metrics") or views.describe_error(exc))
        finally:
            if worker.is_cancelled:
                self._post(self._cancelled)

    def _accept(self, data: training.TrainingData) -> None:
        self._data = data
        self.redraw()

    def _finish(self, data: training.TrainingData | None, error: str | None) -> None:
        self._loading = False
        self._error = error
        if data is not None:
            self._data = data
        if error is None:
            self._downloaded = True
        self.redraw()

    def _cancelled(self) -> None:
        self._loading = False

    # ── drawing ──
    def _width(self) -> int:
        pane = self.query_one("#training-pane", VerticalScroll)
        return max(20, pane.scrollable_content_region.width - 1)

    def on_resize(self, event) -> None:
        if self._shown:
            self.call_after_refresh(self.redraw)

    def redraw(self) -> None:
        if not self.is_attached:
            return
        status = self.query_one("#training-status", Static)
        stats = self.query_one("#training-stats", Static)
        charts = self.query_one("#training-charts", Static)
        source = self._source()
        sub_job = self._data.get(source)
        status.update(self._status_text(source, sub_job))
        if sub_job is None or not sub_job.metrics(self._view):
            stats.update(self._empty_text(source))
            charts.update("")
            return
        first = sub_job.first_x(self._view, self._window)
        width = self._width()
        stats.update(self._stats_text(sub_job, first, width))
        charts.update(self._charts_text(sub_job, first, width))

    def _status_text(self, source: str | None, sub_job: training.SubJobTraining | None) -> Text:
        text = Text()
        ids = self._sub_job_ids()
        if source is None:
            text.append("training metrics")
        else:
            text.append(views.sub_job_label(source), style="bold")
            if len(ids) > 1:
                text.append(f" ({ids.index(source) + 1} of {len(ids)}, space for next)", style=CHART_MUTED)
        if sub_job is not None and (sub_job.last_step is not None or sub_job.calls):
            if sub_job.last_step is not None:
                text.append(f"  ·  step {training.format_step(sub_job.last_step)}")
            text.append(f"  ·  {sub_job.calls:,} call{'s' if sub_job.calls != 1 else ''}")
            if sub_job.newest is not None:
                text.append(f"  ·  newest row {views.fmt_ago(sub_job.newest, datetime.now(timezone.utc))}")
        if source is not None:
            text.append(f"  ·  {training.VIEW_NAMES[self._view]} (v)")
            text.append(f"  ·  {training.window_label(self._window)} (w)")
        notes = []
        if self._loading:
            notes.append(("cached · refreshing…" if sub_job else "loading training metrics…", CHART_MUTED))
        elif self._error:
            notes.append((f"{'cached · refresh failed' if sub_job else 'training metrics unavailable'}: "
                          f"{self._error}", _WARNING))
        if sub_job is not None and sub_job.skipped:
            notes.append((f"{sub_job.skipped:,} training row{'s' if sub_job.skipped != 1 else ''} skipped "
                          "(malformed or unsupported)", _WARNING))
        if sub_job is not None and sub_job.old_format:
            notes.append((f"{sub_job.old_format:,} row{'s' if sub_job.old_format != 1 else ''} in an earlier "
                          "format not shown", _WARNING))
        if sub_job is not None and sub_job.hidden_names:
            notes.append((f"{sub_job.hidden_names:,} value{'s' if sub_job.hidden_names != 1 else ''} with "
                          "a name that can't be shown as it is", _WARNING))
        text.append("\n")
        for index, (note, style) in enumerate(notes):
            text.append(("  ·  " if index else "") + note, style=style)
        return text

    def _empty_text(self, source: str | None) -> Text:
        job = self._job()
        if source is None:
            if not job.get("sub_jobs"):
                return Text("loading the job…" if not self._downloaded else "this job has no sub-jobs")
            return Text("this job has no training sub-job, so it saves no training metrics")
        entry = next((sj for sj in views.sub_jobs(job) if sj.get("sub_job_id") == source), {})
        status = views.status_word(entry.get("status") or job.get("status"))
        if not views.is_live(job) and status not in ("unknown", "?"):
            status = views.status_word(job.get("status"))
        is_training = not entry or views.sub_job_kind(entry) == "training"
        sub_job = self._data.get(source)
        return Text(training.no_metrics_text(views.sub_job_label(source), status, training=is_training,
                                             downloaded=self._downloaded, view=self._view,
                                             old_format=sub_job.old_format if sub_job else 0))

    def _at(self, sub_job: training.SubJobTraining, x: int) -> str:
        """Where a value is from: its step, or its call and (in parentheses) that call's step."""
        if self._view != training.VIEW_CALL:
            return training.format_step(x)
        step = sub_job.step_of_call(x)
        return f"{training.format_step(x)} ({training.format_step(step) if step is not None else 'next'})"

    def _stats_text(self, sub_job: training.SubJobTraining, first: int | None, width: int) -> Text:
        rows = []
        for series in sub_job.metrics(self._view):
            found = training.window_stats(*series.window(first))
            if found is None:
                rows.append((series.name, "—", "—", "—"))
                continue
            rows.append((series.name,
                         f"{training.format_metric(found.latest)} @ {self._at(sub_job, found.latest_x)}",
                         f"{training.format_metric(found.low)} @ {self._at(sub_job, found.low_x)}",
                         f"{training.format_metric(found.high)} @ {self._at(sub_job, found.high_x)}"))
        where = "call (step)" if self._view == training.VIEW_CALL else "step"
        header = ("metric", f"latest @ {where}", f"min @ {where}", f"max @ {where}")
        widths = [max(len(row[i]) for row in [header, *rows]) for i in range(4)]
        text = Text()
        text.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(header)).rstrip()[:width] + "\n",
                    style=CHART_AXIS)
        for index, row in enumerate(rows):
            text.append(row[0].ljust(widths[0]), style=chart_hue(index))
            text.append(("  " + "  ".join(cell.ljust(widths[i + 1]) for i, cell in enumerate(row[1:]))).rstrip())
            text.append("\n")
        return text

    def _charts_text(self, sub_job: training.SubJobTraining, first: int | None, width: int) -> Text:
        per_row = 2 if width >= _TWO_CHARTS_WIDTH else 1
        gap = 3
        chart_width = (width - gap * (per_row - 1)) // per_row
        blocks = [self._chart_block(series, first, chart_width, chart_hue(index))
                  for index, series in enumerate(sub_job.metrics(self._view))]
        text = Text()
        for start in range(0, len(blocks), per_row):
            row = blocks[start:start + per_row]
            height = max(len(block) for block in row)
            text.append("\n")
            for line_index in range(height):
                for column, block in enumerate(row):
                    spans = block[line_index] if line_index < len(block) else []
                    used = sum(len(chunk) for chunk, _ in spans)
                    for chunk, style in spans:
                        text.append(chunk, style=style)
                    if column < len(row) - 1:
                        text.append(" " * (chart_width - used + gap))
                text.append("\n")
        return text

    def _chart_block(self, series: training.MetricSeries, first: int | None, width: int,
                     hue: str) -> list[list[tuple[str, str]]]:
        _xs, values = series.window(first)
        latest = training.format_metric(values[-1]) if values else "—"
        title = f"{series.name} · {training.VIEW_NAMES[self._view]} · {training.window_label(self._window)}"
        room = width - len(latest) - 1
        title = title if len(title) <= room else title[: max(0, room - 1)] + "…"
        head = [(title, hue), (" " * max(1, width - len(title) - len(latest)), ""), (latest, f"bold {hue}")]
        return [head, *training.chart_spans(series, first, width, height=_CHART_HEIGHT, hue=hue)]
