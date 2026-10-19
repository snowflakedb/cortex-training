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

"""Read-only Textual TUI for Cortex Training job logs and GPU resources.

Flow: open → a list of jobs with their status (via the SDK's list_jobs) →
pick one → its preview, then its Logs and Resources dashboard. Pass a job_id on the command line
to jump straight to that dashboard and skip the picker.

Imports ``textual`` at module load. The interface is strictly read-only and
never mutates job state.
"""

from __future__ import annotations

import asyncio
import codecs
import contextvars
import gc
import hashlib
import inspect
import os
import re
import shlex
import shutil
import tempfile
import threading
import time
import traceback
from collections import deque
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from datetime import datetime
from datetime import timezone
from itertools import zip_longest
from pathlib import Path

from rich.cells import cell_len
from rich.cells import get_character_cell_size
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual import work
from textual.app import App
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.containers import ScrollableContainer
from textual.containers import VerticalScroll
from textual.content import Content
from textual.coordinate import Coordinate
from textual.geometry import Offset
from textual.geometry import Region
from textual.css.query import NoMatches
from textual.dom import NoScreen
from textual.notifications import Notification
from textual.notifications import Notify
from textual.screen import Screen
from textual.scroll_view import ScrollView
from textual.selection import SELECT_ALL
from textual.selection import SelectEnd
from textual.selection import Selection
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import DataTable
from textual.widgets import Footer
from textual.widgets._footer import FooterKey
from textual.widgets import Header
from textual.widgets import HelpPanel
from textual.widgets import Input
from textual.widgets import Label
from textual.widgets import ListItem
from textual.widgets import ListView
from textual.widgets import Log
from textual.widgets import Static
from textual.widgets import TabbedContent
from textual.widgets import TabPane
from textual.widgets.data_table import DuplicateKey
from textual.widgets._toast import Toast
from textual.widgets._toast import ToastRack
from textual.widgets.data_table import RowDoesNotExist
from textual.widgets.data_table import RowKey
from textual.worker import get_current_worker

from cortex_training._download_lock import DownloadLock
from cortex_training.client import DownloadCancelled
from cortex_training.client import NoChunksError
from cortex_training.tui import format as log_format
from cortex_training.tui import views
from cortex_training.tui.account import ACCOUNT
from cortex_training.tui.account import SCHEMA
from cortex_training.tui.account import JobClients
from cortex_training.tui.account import JobLister
from cortex_training.tui.account import JobListError
from cortex_training.tui.account import Listing
from cortex_training.tui.account import ListScope
from cortex_training.tui.account import empty_text
from cortex_training.tui.account import loaded_text
from cortex_training.tui.account import scope_text
from cortex_training.tui.account import unavailable_text
from cortex_training.tui.format import display_line
from cortex_training.tui.format import entry_at_level
from cortex_training.tui.format import entry_matches
from cortex_training.tui.format import format_created_at
from cortex_training.tui.format import format_log_entry
from cortex_training.tui.format import one_line
from cortex_training.tui.clipboard import aligned
from cortex_training.tui.clipboard import copied_note
from cortex_training.tui.clipboard import copy_local
from cortex_training.tui.clipboard import screen_dump
from cortex_training.tui.clipboard import selected_text
from cortex_training.tui.format import safe_text
from cortex_training.tui.format import source_label
from cortex_training.tui.format import wrap_log_line
from cortex_training.tui.log_cache import LogCache
from cortex_training.tui.log_cache import cached_log_pages
from cortex_training.tui.log_cache import catching_up
from cortex_training.tui.log_cache import job_cache_dir
from cortex_training.tui.log_cache import prune_consoles
from cortex_training.tui.log_cache import touch_console
from cortex_training.tui.log_style import LogHighlighter
from cortex_training.tui.log_style import line_kind
from cortex_training.tui.log_style import sub_job_color
from cortex_training.tui.resource import CHART_MUTED
from cortex_training.tui.resource import DEFAULT_RANGE
from cortex_training.tui.resource import GPU
from cortex_training.tui.resource import LOADING_OLDER
from cortex_training.tui.resource import OLDER_NOT_LOADED
from cortex_training.tui.resource import RANGES
from cortex_training.tui.resource import SYSTEM
from cortex_training.tui.resource import KindResource
from cortex_training.tui.resource import KindView
from cortex_training.tui.resource import ResourceData
from cortex_training.tui.resource import age_slack
from cortex_training.tui.resource import allocation_text
from cortex_training.tui.resource import cached_metric_sources
from cortex_training.tui.resource import chart_hue
from cortex_training.tui.resource import format_value
from cortex_training.tui.resource import freshness_text
from cortex_training.tui.resource import gpu_allocation
from cortex_training.tui.resource import has_metric_samples
from cortex_training.tui.resource import headline_metric
from cortex_training.tui.resource import level_glyph
from cortex_training.tui.resource import metric_chart_spans
from cortex_training.tui.resource import metric_stats_spans
from cortex_training.tui.resource import no_metrics_text
from cortex_training.tui.resource import node_allocation_text
from cortex_training.tui.resource import resource_loader
from cortex_training.tui.resource import scrape_text
from cortex_training.tui.resource import skipped_rows_text
from cortex_training.tui.resource import window_complete
from cortex_training.tui.resource import window_label
from cortex_training.tui.saved_log import committed_size
from cortex_training.tui.saved_log import read_saved_tail
from cortex_training.tui.untrusted import printable_client

_ERROR_LOG = "~/.cortex-training-errors.log"
# Rich's grey62, by value: Textual's markup and styles (the preview's Content)
# don't know that name and would draw the default color instead.
ENDED_STYLE = "#9e9e9e"


def _has_telemetry(data: ResourceData | None) -> bool:
    return data is not None and bool(data.kinds or data.ignored_rows_by_sub_job)


def _same_telemetry(a: ResourceData, b: ResourceData) -> bool:
    """Whether two loads hold the same snapshots (a loader reuses a sub-job's kind until new rows of it arrive)."""
    return (
        a.ignored_rows_by_sub_job == b.ignored_rows_by_sub_job
        and a.kinds.keys() == b.kinds.keys()
        and all(a.kinds[key].keys() == b.kinds[key].keys()
                and all(a.kinds[key][kind] is b.kinds[key][kind] for kind in a.kinds[key]) for key in a.kinds)
    )


def _cache_job_id(client, job_id: str) -> str:
    """Namespace a persistent cache by service endpoint and job location."""
    scope = tuple(
        value if isinstance(value, str) else ""
        for value in (
            getattr(client, "base_url", ""),
            getattr(client, "endpoint", ""),
            getattr(client, "database", ""),
            getattr(client, "schema", ""),
        )
    )
    if not any(scope):
        return job_id
    digest = hashlib.sha256("\0".join(scope).encode()).hexdigest()[:16]
    return f"{digest}:{job_id}"


class _DaemonThreadExecutor(ThreadPoolExecutor):
    """A bounded default executor whose running calls do not hold up process exit."""

    def __init__(self, max_workers: int | None = None) -> None:
        super().__init__(max_workers=max_workers)
        self._pending: deque[tuple] = deque()
        self._running = 0

    def submit(self, fn, /, *args, **kwargs) -> Future:
        future: Future = Future()
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            self._pending.append((future, contextvars.copy_context(), fn, args, kwargs))
            self._start_pending()
        return future

    def _start_pending(self) -> None:
        while self._pending and self._running < self._max_workers:
            task = self._pending.popleft()
            self._running += 1
            threading.Thread(target=self._run, args=(task,), name="cortex-tui-worker", daemon=True).start()

    def _run(self, task: tuple) -> None:
        future, context, fn, args, kwargs = task
        try:
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(context.run(fn, *args, **kwargs))
                except BaseException as exc:  # noqa: BLE001 - delivered to the awaiting worker
                    future.set_exception(exc)
        finally:
            with self._shutdown_lock:
                self._running -= 1
                self._start_pending()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        # asyncio asks its default executor to wait during loop teardown. Running
        # network calls are daemon threads, so quitting remains immediate.
        with self._shutdown_lock:
            self._shutdown = True
            while self._pending:
                self._pending.popleft()[0].cancel()


def on_stack(screen: Screen) -> bool:
    """Whether ``screen`` remains on its app's stack, safe from a worker."""
    try:
        return screen in screen.app.screen_stack
    except Exception:  # noqa: BLE001 - no app: it is shutting down
        return False


def is_shown(screen: Screen) -> bool:
    """Whether ``screen`` is mounted at the top of its app's stack."""
    try:
        return screen.app.screen is screen and screen.is_mounted
    except Exception:  # noqa: BLE001 - no app: it is shutting down
        return False


# "?" on every screen: the keys panel lists every key that works where you are,
# the ones the footer has no room for too (it lists bindings the footer hides).
KEYS_BINDING = Binding("question_mark", "app.toggle_keys", "Keys", key_display="?")


def close_keys_panel(screen: Screen) -> bool:
    """UI thread. Close the keys panel (``?``, or ``^p`` → Keys) when it is open; whether it was."""
    panels = screen.query("HelpPanel")
    if not panels:
        return False
    panels.remove()
    return True


def after_refresh(screen: Screen, fn, *args, **kwargs) -> None:
    """Run after refresh only while ``screen`` remains on the stack."""

    def run():
        return fn(*args, **kwargs) if on_stack(screen) else None

    screen.call_after_refresh(run)


class JobTabs(TabbedContent):
    """Job tabs that report width changes caused by the keys panel."""

    def on_resize(self, event: events.Resize) -> None:
        self.screen._tabs_resized()


def _freeze_heap() -> None:
    """Exclude the stable TUI heap from later garbage collections."""
    gc.collect()
    gc.freeze()


def call_on_screen(screen: Screen, fn, *args, **kwargs) -> bool:
    """Run a worker callback on the UI thread; false once the app is gone."""

    # Run even after a pop: some callbacks clean up then (a finished download's temp dir).
    async def guarded():
        try:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                await result
        except NoMatches:
            if on_stack(screen):
                raise  # the screen is still live: a real bug

    try:
        app = screen.app
        if not app.is_running:
            return False
        app.call_from_thread(guarded)
        return True
    except RuntimeError:  # NoActiveAppError / closed loop: the app is shutting down
        return False


class _LogPane(Log):
    """The Logs pane, retaining each wrapped row's highlighter kind."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.row_kinds: list[str | None] = []

    def write_rows(self, rows: list[str], kinds: list[str | None], *, scroll_end: bool | None = None) -> None:
        """``rows`` as :meth:`write_lines` writes them, ``kinds`` one per row it makes."""
        self.row_kinds.extend(kinds)
        self.write_lines(rows, scroll_end=scroll_end)
        del self.row_kinds[:max(0, len(self.row_kinds) - len(self.lines))]  # as the oldest rows went

    def clear(self):
        self.row_kinds = []
        return super().clear()

    def _render_line_strip(self, y: int, rich_style):
        highlighter = self.highlighter
        if hasattr(highlighter, "row_kind"):
            highlighter.row_kind = self.row_kinds[y] if 0 <= y < len(self.row_kinds) else None
        return super()._render_line_strip(y, rich_style)


class _SourceList(ListView):
    """The sub-job list, forwarding page and end keys to the Logs pane."""

    _PANE = "#logview"
    BINDINGS = [
        Binding("pageup", "pane('page_up')", show=False),
        Binding("pagedown", "pane('page_down')", show=False),
        Binding("home", "pane('scroll_home')", show=False),
        Binding("end", "pane('scroll_end')", show=False),
    ]

    def action_pane(self, action: str) -> None:
        try:
            pane = self.screen.query_one(self._PANE)
        except NoMatches:
            return
        getattr(pane, f"action_{action}")()


class _TimedTail:
    """A tail client that records when its current request began."""

    def __init__(self, client, screen, gen: int) -> None:
        self._client, self._screen, self._gen = client, screen, gen

    def tail_logs(self, *args, **kwargs):
        self._screen._tail_request = (self._gen, time.monotonic())
        try:
            return self._client.tail_logs(*args, **kwargs)
        finally:
            self._screen._tail_request = None


def spans_content(spans) -> Content:
    """One line of ``(text, style)`` spans (as the Resources charts and stats are drawn) as Content."""
    return Content.assemble(*((text, style) if style else text for text, style in spans))


def hanging(content: Content, width: int, indent: int) -> Content:
    """Wrap one line with hanging continuation indentation."""
    wrap = getattr(content, "wrap", None)
    if wrap is None or content.cell_length <= width or width - indent < 10:
        return content
    first = wrap(width)[0]
    rest = content[len(first.plain):]
    rest = rest[len(rest.plain) - len(rest.plain.lstrip()):]  # the space the break took
    if not rest.plain:
        return first
    return Content("\n" + " " * indent).join([first, *rest.wrap(width - indent)])


def wrapped_parts(parts: list, width: int, indent: int, separator: Content) -> Content:
    """Pack parts onto hanging-indented lines, breaking between parts first."""
    lead = Content(" " * indent)
    lines: list = []
    line = None
    for part in parts:
        if line is not None and (line + separator + part).cell_length <= width:
            line = line + separator + part
            continue
        if line is not None:
            lines.append(line)
        line = lead + part
    if line is not None:
        lines.append(line)
    return Content("\n").join(hanging(line, width, indent) for line in lines)


class _FixedCell:
    """A compact fixed-width cell whose Rich ``Text`` is made only when drawn."""

    __slots__ = ("plain", "style", "spans")

    def __init__(self, plain: str, style: str = "", spans: tuple = ()) -> None:
        self.plain = plain
        self.style = style
        self.spans = spans  # ((start, end, style), ...) over plain

    @classmethod
    def assemble(cls, *parts: tuple) -> "_FixedCell":
        """A cell of ``(text, style)`` parts, one after another."""
        spans, at = [], 0
        for text, style in parts:
            if style and text:
                spans.append((at, at + len(text), style))
            at += len(text)
        return cls("".join(text for text, _style in parts), "", tuple(spans))

    def text(self) -> Text:
        text = Text(self.plain, style=self.style, overflow="ellipsis", no_wrap=True)
        for start, end, style in self.spans:
            text.stylize(style, start, end)
        return text

    def __rich_console__(self, console, options):
        return self.text().__rich_console__(console, options)

    def __str__(self) -> str:
        return self.plain


def _cell(text: str, style: str = "") -> _FixedCell:
    """A one-line printable cell, preserving spaces and never parsing markup."""
    return _FixedCell(views.printable(text).replace("\n", " "), style)


_BLANK = _FixedCell("")


def _sort_part(key: str, descending: bool) -> Content:
    """The clickable ``sort: created ↓`` control."""
    arrow = "↓" if descending else "↑"
    return Content.from_markup(
        f"[dim]sort:[/dim] [b][@click=screen.cycle_sort]{key}[/] [@click=screen.toggle_order]{arrow}[/][/b]")


_SORT_PART_CELLS = max(_sort_part(key, descending).cell_length
                       for key in views.SORT_KEYS for descending in (True, False))


try:  # what _JobTable.replace_rows sets directly; another Textual fills the table a row at a time
    from textual._two_way_dict import TwoWayDict as _TwoWayDict
    from textual.widgets.data_table import Row as _Row
except ImportError:  # pragma: no cover - depends on the installed Textual
    _TwoWayDict = _Row = None


@dataclass(frozen=True)
class _ListLayout:
    """The job table's columns for one terminal width, sort and set of locations."""

    columns: tuple  # (label, width) in display order
    compact_times: bool  # Created and Updated as MM-DD HH:MM

    @property
    def labels(self) -> tuple:
        return tuple(label for label, _width in self.columns)


class _JobRows:
    """One job's table rows as last built, and what they were built from."""

    # Slots: one of these per listed job, kept between renders.
    __slots__ = ("job", "compact", "cells", "sub_cells", "sub_keys", "layout", "expanded", "row", "subs")

    def __init__(self, job: dict, compact: bool, cells: dict) -> None:
        self.job = job  # the record (compared by identity: a refresh keeps an unchanged record)
        self.compact = compact  # built with MM-DD HH:MM times
        self.cells = cells  # column label -> cell: the job's row, but for its Job ID
        self.sub_cells: tuple | None = None  # ((row key suffix, {label: cell}), ...): its sub-job rows (and note)
        self.sub_keys: tuple = ()  # ((job id, sub-job id) or None for the note, ...) beside sub_cells
        self.layout = None  # the layout row and subs are in
        self.expanded: bool | None = None  # whether row and subs are for the job unfolded
        self.row: tuple = ()
        self.subs: tuple = ()  # ((row key suffix, cells), ...); none while folded


def scroll_area(widget: Widget) -> Widget | None:
    """The nearest scrolling selection area, excluding docked content."""
    table = None
    for node in widget.ancestors_with_self:
        if isinstance(node, Screen):
            return table
        if node.styles.dock in ("top", "bottom") and isinstance(node.parent, ScrollableContainer):
            return node
        if isinstance(node, DataTable):
            table = table or node
        elif isinstance(node, (ScrollView, ScrollableContainer)) and not isinstance(node, (Footer, Input)):
            return node
    return table


def scrolling_region(area: Widget) -> Region:
    """The area's scrolling content, excluding docked lines."""
    if isinstance(area, SelectableTable):
        return area.drag_region()
    region = area.scrollable_content_region
    top, bottom = region.y, region.bottom
    for child in area.children:
        if child.display and child.styles.dock == "top":
            top = max(top, child.region.bottom)
        elif child.display and child.styles.dock == "bottom":
            bottom = min(bottom, child.region.y)
    return Region(region.x, top, region.width, bottom - top) if bottom > top else region


class FollowsPointer:
    """Keep cross-widget selection under its pointer while an area scrolls."""

    def _watch__select_state(self, select_state) -> None:
        if select_state is not None and select_state.end is not None and select_state.is_attached_to_dom():
            held = getattr(self.app, "press_held", None)
            if callable(held) and held():
                self._selecting = True
                self.selections = {}
                return
            select_state = self._end_on_text(select_state)
            if select_state is None:  # no text between the pointer and the drag's start yet
                self._selecting = True
                self.selections = {}
                return
            start, end = select_state.start, select_state.end
            first, last = start.content_widget, end.content_widget
            if (first is not None and last is not None and first is not last
                    and start.content_offset is not None and end.content_offset is not None
                    and scroll_area(first) is not None and scroll_area(first) is scroll_area(last)):
                self._selecting = True
                texts = [widget for widget in scroll_area(first).query("*")
                         if widget.allow_select and not widget.is_container]
                if first in texts and last in texts:
                    head, tail = (start.content_offset, end.content_offset)
                    if texts.index(first) > texts.index(last):
                        first, last, head, tail = last, first, tail, head
                    between = texts[texts.index(first) + 1:texts.index(last)]
                    # Both ends take in the character under them, as Textual's selection in one text does.
                    self.selections = {first: Selection(head, None), **{widget: SELECT_ALL for widget in between},
                                       last: Selection(None, tail + Offset(1, 0))}
                    return
        super()._watch__select_state(select_state)

    def _end_on_text(self, state):
        """Move an endpoint on blank space to the nearest text toward its start."""
        start, end = state.start, state.end
        area = scroll_area(start.content_widget) if start.content_widget is not None else None
        if end.content_widget is not None or area is None or start.content_offset is None:
            return state
        widget = start.content_widget
        start_y = widget.content_region.y + start.content_offset.y - int(widget.scroll_offset.y)
        x, y = state.screen_offset
        step = 1 if start_y > y else -1
        region = scrolling_region(area)
        while region.y <= y < region.bottom and y != start_y + step:
            found = self._text_on_row(area, x, y)
            if found is not None:
                text, offset = found
                new = state.update_end(state.screen_offset, SelectEnd(text.parent, text, offset))
                self.set_reactive(Screen._select_state, new)
                return new
            y += step
        return None

    def _text_on_row(self, area: Widget, x: int, y: int):
        """The selectable widget and offset nearest ``(x, y)`` on that row."""
        found, offset = self.get_widget_and_offset_at(x, y)
        if found is not None and offset is not None and not found.is_container and scroll_area(found) is area:
            return found, offset
        nearest = None  # the text on the row that starts nearest to the left of the pointer
        for node in area.query("*"):
            box = node.region
            if (not node.is_container and node.allow_select and node.display and box.y <= y < box.bottom
                    and box.x <= x and (nearest is None or box.x > nearest.region.x)):
                nearest = node
        if nearest is None:
            return None
        found, offset = self.get_widget_and_offset_at(min(x, nearest.region.right - 1), y)
        return (found, offset) if found is nearest and offset is not None else None

    def _update_select(self) -> None:
        state = getattr(self, "_select_state", None)
        position = getattr(self.app, "mouse_position", None)
        if state is not None and state.end is not None and position is not None:
            widget, offset = self.get_widget_and_offset_at(position.x, position.y)
            if widget is not None:
                end = (SelectEnd(widget if isinstance(widget, Screen) else widget.parent, widget, offset)
                       if offset is not None else SelectEnd(widget, None, None))
                self.set_reactive(Screen._select_state, state.update_end(position, end))
        super()._update_select()


class SelectableTable(DataTable):
    """A DataTable with scroll-stable, character-accurate selection and copying."""

    ALLOW_SELECT = True
    _DRAG_CELLS = 2

    def render_line(self, y: int) -> Strip:
        width = self.size.width
        scroll_x, scroll_y = self.scroll_offset
        line = y if y < self._fixed_lines() else y + scroll_y
        strip = self._render_line(line, scroll_x, scroll_x + width, self.rich_style)
        return self._selectable(strip, line, scroll_x)

    def _fixed_lines(self) -> int:
        """The lines at the top that stay put as the table scrolls: the header's and any fixed rows'."""
        rows = range(min(self.fixed_rows, self.row_count))
        height = sum(self.get_row_height(self._row_locations.get_key(index)) for index in rows)
        return height + (self.header_height if self.show_header else 0)

    def _text_lines(self) -> int:
        """How many lines the table's text has: the header's and the rows' (not the blank under them)."""
        return (self.header_height if self.show_header else 0) + len(self._y_offsets)

    def _selectable(self, strip: Strip, line: int, scroll_x: int) -> Strip:
        """Add text offsets and selection highlighting to a rendered line."""
        lines = self._text_lines()
        if line >= lines:  # the blank under the rows: the empty line after them
            return strip.apply_offsets(0, lines)
        fixed, rest = self._line_parts(line)
        cut = min(cell_len(fixed), strip.cell_length)
        parts = ((strip.crop(0, cut), 0), (strip.crop(cut), len(fixed) + _chars_within(rest, scroll_x)))
        selection = self.text_selection
        span = None if selection is None else _selected_span(selection, line, fixed + rest)
        style = self._selection_style() if span is not None and span[0] < span[1] else None
        return Strip.join(
            (_draw_highlight(part, first, span, style) if style is not None else part).apply_offsets(first, line)
            for part, first in parts
        )

    def _line_parts(self, line: int) -> tuple[str, str]:
        """A rendered line's fixed and scrollable text."""
        try:
            row_key, line_in_row = self._get_offsets(line)
        except LookupError:
            return "", ""
        plain = None if line_in_row else self._plain_parts(row_key)
        return plain if plain is not None else self._drawn_parts(row_key, line_in_row)

    def _drawn_parts(self, row_key, line_in_row: int) -> tuple[str, str]:
        """Render one row line into fixed and scrollable text."""
        fixed, scrollable = self._render_line_in_row(row_key, line_in_row, self.rich_style,
                                                     self.cursor_coordinate, self.hover_coordinate)
        # The scrolling cells start with the fixed columns too, drawn again (and cut off, as here).
        skip = sum(column.get_render_width(self) for column in self.ordered_columns[:self.fixed_columns])
        rest = Strip([segment for cell in scrollable for segment in cell]).crop(skip).text
        return "".join(segment.text for cell in fixed for segment in cell), rest

    def _plain_parts(self, row_key) -> tuple[str, str] | None:
        """Fast-path one-line fitting ``_FixedCell`` rows without Rich rendering."""
        row = self.rows.get(row_key)
        if row is None or row.height != 1 or (self._labelled_row_exists and self.show_row_labels):
            return None
        try:
            cells = self.get_row(row_key)
        except (KeyError, RowDoesNotExist):  # a cell missing: what the table draws there
            return None
        pad = " " * self.cell_padding
        texts = []
        for column, cell in zip(self.ordered_columns, cells):
            if not isinstance(cell, _FixedCell) or not cell.plain.isprintable():
                return None
            room = column.get_render_width(self) - 2 * self.cell_padding
            used = cell_len(cell.plain)
            if used > room:
                return None  # cut short with "…", as Rich cuts it
            texts.append(f"{pad}{cell.plain}{' ' * (room - used)}{pad}")
        # Past the last column, the row's blank to the table's edge (as DataTable extends it).
        width = sum(column.get_render_width(self) for column in self.ordered_columns[self.fixed_columns:])
        extend = " " * max(0, self.size.width - width - self._row_label_column_width)
        return "".join(texts[:self.fixed_columns]), "".join(texts[self.fixed_columns:]) + extend

    def _selection_style(self) -> Style:
        """Selection background plus an explicitly configured foreground."""
        screen = self.screen
        style = screen.get_component_rich_style("screen--selection")
        own = screen.get_component_rich_style("screen--selection", partial=True)
        return Style(color=style.color if own.color is not None else None, bgcolor=style.bgcolor)

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """The text under ``selection``, a line for each of the table's lines it covers."""
        lines = self._text_lines()
        start, end = selection
        first = 0 if start is None else max(0, start.y)
        last = lines - 1 if end is None else min(end.y, lines - 1)
        parts = []
        for line in range(first, last + 1):
            text = "".join(self._line_parts(line))
            span = _selected_span(selection, line, text)
            if span is not None:
                parts.append(text[span[0]:span[1]])
        return ("\n".join(parts), "\n") if parts else None

    def drag_region(self, header: bool = False) -> Region:
        """The clipped header or scrolling-row drag region."""
        region = self.scrollable_content_region
        try:
            geometry = self.screen.find_widget(self)
            region = region.intersection(geometry.region.intersection(geometry.clip))
        except Exception:  # noqa: BLE001 - not laid out
            pass
        lines = self.header_height if self.show_header else 0
        if header and lines:
            return Region(region.x, region.y, region.width, min(lines, region.height))
        if region.height <= lines:
            return region
        return Region(region.x, region.y + lines, region.width, region.height - lines)

    def on_header(self, screen_y: int) -> bool:
        """Whether screen line ``screen_y`` is the table's header."""
        top = self.scrollable_content_region.y
        return self.show_header and top <= screen_y < top + self.header_height

    def clear(self, columns: bool = False):
        self._drop_highlight()
        return super().clear(columns)

    def sort(self, *columns, key=None, reverse: bool = False):
        self._drop_highlight()
        return super().sort(*columns, key=key, reverse=reverse)

    def _drop_highlight(self) -> None:
        """The rows are about to change: a highlight in them would sit over other text."""
        try:
            screen = self.screen
        except NoScreen:
            return
        if self in screen.selections:
            screen.clear_selection()

    def on_click(self, event: events.Click) -> None:
        self.swallowed(event)

    def swallowed(self, event: events.Click) -> bool:
        """Suppress non-left clicks that DataTable would treat as row clicks."""
        if event.button == 1:
            return False
        event.prevent_default()
        event.stop()
        return True

    def rows_text(self, keys=None) -> list:
        """Aligned header and rows, adding whole ids when available."""
        whole_id = getattr(self.screen, "whole_id", None)
        extra = ["Whole ID"] if callable(whole_id) else []
        rows = [[str(column.label) for column in self.ordered_columns] + extra]
        for key in keys if keys is not None else [row.key for row in self.ordered_rows]:
            cells = [str(cell) for cell in self.get_row(key)]
            rows.append(cells + ([whole_id(key.value) or ""] if extra else []))
        return aligned(rows)


def _chars_within(text: str, cells: int) -> int:
    """Characters wholly contained in the first ``cells`` terminal cells."""
    if cells <= 0:
        return 0
    if text.isascii():
        return min(cells, len(text))
    used = 0
    for index, character in enumerate(text):
        used += get_character_cell_size(character)
        if used > cells:
            return index
    return len(text)


def _selected_span(selection: Selection, line: int, text: str) -> tuple[int, int] | None:
    """Selected character span on one line, excluding trailing blanks."""
    span = selection.get_span(line)
    if span is None:
        return None
    start, end = span
    end = len(text) if end == -1 else min(end, len(text))
    start = min(start, end)
    return start, start + len(text[start:end].rstrip())


def _draw_highlight(strip: Strip, first: int, span: tuple[int, int], style: Style) -> Strip:
    """Overlay ``style`` on a line-relative character span."""
    start, end = span
    at = first
    segments = []
    for text, own, control in strip:
        low, high = max(0, start - at), min(len(text), end - at)
        if control or low >= high:
            segments.append(Segment(text, own, control))
        else:
            if low:
                segments.append(Segment(text[:low], own))
            segments.append(Segment(text[low:high], own + style if own else style))
            if high < len(text):
                segments.append(Segment(text[high:], own))
        at += len(text)
    return Strip(segments, strip.cell_length)


class _JobTable(SelectableTable):
    """The selectable job table with fold-aware navigation."""

    BINDINGS = [
        Binding("home", "first_job", "First job", show=False),
        Binding("end", "last_job", "Last job", show=False),
        Binding("right", "cursor_right", "Unfold sub-jobs, else scroll right", show=False),
        Binding("left", "cursor_left", "Fold sub-jobs, else scroll left", show=False),
    ]
    _MARKERS = ("▸ ", "▾ ")

    def action_first_job(self) -> None:
        self.move_cursor(row=min(1, self.row_count - 1), animate=False)  # row 0 is the LIVE/HISTORY header
        self.scroll_home(animate=False)

    def action_last_job(self) -> None:
        self.move_cursor(row=self.row_count - 1, animate=False)

    def action_cursor_right(self) -> None:
        if not self.screen.unfold_highlighted():
            super().action_cursor_right()  # a row cursor: the table scrolls right

    def action_cursor_left(self) -> None:
        if not self.screen.fold_highlighted():
            super().action_cursor_left()

    def on_click(self, event: events.Click) -> None:
        """A click on a job's ▸/▾ marker unfolds or folds it; any other click is the table's own, unless swallowed."""
        if self.swallowed(event):
            return
        key = self._marker_at(event)
        if key is not None and self.screen.toggle_job_at(key):
            event.prevent_default()  # the table's own handler would open the job on a second click
            event.stop()

    def _marker_at(self, event: events.Click) -> str | None:
        """The row key of the job whose ▸/▾ marker ``event`` is on, else None."""
        meta = event.style.meta
        row, column = meta.get("row"), meta.get("column")
        columns = self.ordered_columns
        if not (isinstance(row, int) and isinstance(column, int) and 0 <= row < self.row_count
                and 0 <= column < min(len(columns), self.fixed_columns)):
            return None  # the header, or no fixed column (the Job ID column is one; it never scrolls)
        if columns[column].key.value != "Job ID" or not str(self.get_cell_at(Coordinate(row, column))).startswith(
                self._MARKERS):
            return None
        offset = event.get_content_offset(self)
        if offset is None:
            return None
        left = getattr(self, "_row_label_column_width", 0) + sum(c.get_render_width(self) for c in columns[:column])
        if offset.x - left != self.cell_padding:  # the arrow: ▸ and ▾ are one cell wide, after the padding
            return None
        return self.ordered_rows[row].key.value

    def replace_rows(self, rows: list) -> None:
        """Replace fixed-width rows, using tested Textual internals when safe."""
        if not self._fills_fast():
            self.clear()
            for key, cells in rows:
                self.add_row(*cells, key=key)
            return
        keys = [column.key for column in self.ordered_columns]
        data, meta, locations = {}, {}, {}
        for index, (key, cells) in enumerate(rows):
            if len(cells) > len(keys):
                raise ValueError("More values provided than there are columns.")
            row_key = RowKey(key)
            if row_key in locations:
                raise DuplicateKey(f"The row key {row_key!r} already exists.")
            locations[row_key] = index
            data[row_key] = dict(zip_longest(keys, cells))
            meta[row_key] = _Row(row_key, 1)
        self.clear()
        self._data.update(data)
        self.rows.update(meta)
        self._row_locations = _TwoWayDict(locations)
        self._require_update_dimensions = True  # the idle pass sets the size (with no new rows to measure)
        if rows:
            self.cursor_coordinate = self.cursor_coordinate  # within the rows, as add_row keeps it
            if self.columns and self.show_cursor and self.cursor_type != "none":
                self._highlight_cursor()  # as add_row does when the first row arrives
        self._update_count += 1
        self.check_idle()

    def _fills_fast(self) -> bool:
        """Whether tested internals and fixed-width columns permit fast fill."""
        from cortex_training.tui.__main__ import textual_is_floor

        return (_Row is not None and textual_is_floor() and all(hasattr(self, name) for name in self._FAST_FILL)
                and not any(column.auto_width for column in self.columns.values()))

    _FAST_FILL = ("_data", "_row_locations", "_require_update_dimensions", "_update_count", "_highlight_cursor")


class JobListScreen(FollowsPointer, Screen):
    """Every job in the retention window: live first, then history, with sub-job rows.

    ``a`` switches between the configured schema's jobs and the account-wide list
    (see :class:`cortex_training.tui.account.JobLister`); search takes ids, locations
    and ``field<op>value`` filters (see :func:`cortex_training.tui.views.parse_query`).
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        KEYS_BINDING,
        # Priority keeps burst input on the focus selected by the first key.
        Binding("/", "filter", "Search", priority=True),
        Binding("escape", "clear_filter", "Clear search", show=False, priority=True),
        Binding("enter", "submit_search", "Show matches", show=False, priority=True),
        Binding("a", "show_list('account')", "All schemas"),
        Binding("a", "show_list('schema')", "Current schema"),
        ("s", "cycle_sort", "Sort"),
        ("o", "toggle_order", "Order"),
        ("e", "expand_all", "Expand/collapse all"),
        ("r", "refresh", "Refresh"),
    ]
    _REFRESH_SECONDS = 60
    _COLUMNS = [(" ", 1), ("Job ID", 38), ("Status", 12), ("Type", 13), ("Created", 16),
                ("Owner", 14), ("Model", 18), ("GPUs", 9), ("Updated", 16)]
    _FIXED_COLUMNS = 2  # the marker and the job id stay in view while the table scrolls sideways
    _FIRST_COLUMNS = (" ", "Job ID", "Status", "Type", "Created", "Owner")
    _NAME_MIN = 16
    _TIME_COMPACT = 11
    _SEARCH_DEBOUNCE_SECONDS = 0.12
    _LOCATION_AT = [label for label, _width in _COLUMNS].index("GPUs") + 1
    _LOCATION_WIDTHS = (10, 32)

    def __init__(self, client, *, poll_interval=1.0, now=None, clients=None, lister=None):
        """``clients`` and ``lister`` are the app's (shared with a list opened again
        later); without them the screen makes its own for ``client``."""
        super().__init__()
        self._client = client
        self._clients = clients if clients is not None else JobClients(client)
        self._lister = lister if lister is not None else JobLister(client)
        self._poll_interval = poll_interval
        self._now = now  # fixed clock for tests/screenshots; None = wall clock
        self._job_by_item: dict[str, tuple] = {}
        self._jobs_by_key: dict[tuple, dict] = {}
        self._sub_by_item: dict[str, tuple] = {}
        self._all_jobs: list = []
        self._filter = ""
        self._sort = "created"
        self._descending = True
        self._expanded: set = set()
        self._collapsed: set = set()  # live jobs the user folded
        self._last_refreshed = None
        self._loaded_at: float | None = None  # monotonic time of the last good load
        self._failed_at: str | None = None  # HH:MM:SS of the last failed load, while it stands
        self._refreshing = False
        self._refresh_started: float | None = None  # monotonic time the refresh in flight began
        self._list_error: str | None = None  # last failed load, on one line; cleared by a good one
        self._list_refused = False  # that failure was the account-wide list refusing this caller (403/405)
        self._scope: ListScope | None = None  # what the list on screen covers
        self._asked = self._account_wanted()
        self._search_timer = None
        self._layout: _ListLayout | None = None  # the table's columns, once they exist
        self._notice: Content | None = None  # the stale-list notice, before wrapping
        self._info: Content | list | None = None  # the line(s) under the header, before wrapping
        self._nomatch: str | None = None  # the no-match message (plain text), before wrapping
        self._rows: dict[tuple, _JobRows] = {}
        self._hay: dict[tuple, tuple] = {}  # job key -> (record, search_haystack)
        self._resolved: tuple | None = None  # (search text, the jobs list it was resolved against, query)
        self._headers: dict[str, tuple] = {}  # LIVE/HISTORY -> ((count, label, layout), cells)
        self._keys: dict[tuple, str] = {}  # job key -> its row's table key
        self._shown: tuple | None = None  # what the table holds: (layout, ((row key, cells), ...))

    def _job_key(self, job: dict) -> tuple:
        return views.job_key(job, self._clients.home)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="listinfo")
        yield Static("", id="listnotice")
        yield _JobTable(id="jobs", cursor_type="row", zebra_stripes=False, fixed_columns=self._FIXED_COLUMNS)
        with VerticalScroll(id="nomatch-box"):
            yield Static("", id="nomatch")
        yield Input(
            placeholder="search ids…  or fields: status:RUNNING gpu:H200",
            id="jobfilter",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#nomatch", Static).display = False
        self.query_one("#nomatch-box", VerticalScroll).display = False
        self.query_one("#listnotice", Static).display = False
        self._set_columns(self.query_one("#jobs", DataTable), self._layout_for(0))
        self._refresh_jobs()
        # DOMNode reserves _auto_refresh, so this timer has another name.
        self.set_interval(self._REFRESH_SECONDS, self._refresh_tick)

    def on_screen_resume(self) -> None:
        self._update_subtitle()
        self._render_jobs()  # with a status the preview read; a list as it was isn't drawn again
        if self._loaded_at is not None and time.monotonic() - self._loaded_at >= self._REFRESH_SECONDS:
            self._refresh_jobs()

    def on_resize(self, event) -> None:
        after_refresh(self, self._fit_to_width)

    def _fit_to_width(self) -> None:
        """After a resize: re-wrap the lines around the list, and re-render if the columns change."""
        self._paint_info()
        self._paint_notice()
        self._paint_nomatch()
        if self._layout != self._layout_for(self._location_column_width(self._recent())):
            self._render_jobs()

    def _layout_for(self, location_width: int) -> _ListLayout:
        """Fit columns, compacting dates and the job id while keeping the sorted time visible."""
        table = self.query_one("#jobs", DataTable)
        pad = 2 * table.cell_padding
        room = (self.size.width or self.app.size.width) - table.styles.scrollbar_size_vertical
        columns = list(self._COLUMNS)
        if location_width:
            columns.insert(self._LOCATION_AT, ("Location", location_width))
        compact = sum(width + pad for _label, width in columns) > room
        if compact:
            columns = [(label, self._TIME_COMPACT if label in ("Created", "Updated") else width)
                       for label, width in columns]
        name_base = dict(self._COLUMNS)["Job ID"]
        spare = room - pad - sum(width + pad for label, width in columns if label != "Job ID")
        if spare >= name_base:
            name = name_base
        else:
            first = set(self._FIRST_COLUMNS) - {"Job ID"}
            needed = sum(width + pad for label, width in columns if label in first)
            if room - pad - needed < self._NAME_MIN:  # the last of them can't fit: the job id takes its room
                needed -= dict(columns)[self._FIRST_COLUMNS[-1]] + pad
            name = max(self._NAME_MIN, min(name_base, room - pad - needed))
        columns = [(label, name if label == "Job ID" else width) for label, width in columns]
        labels = [label for label, _width in columns]
        if self._sort == "updated":
            ends = sum(width + pad for _label, width in columns[: labels.index("Updated") + 1])
            if ends > room:  # out of view: put it where Created is (the two have the same width)
                at_created, at_updated = labels.index("Created"), labels.index("Updated")
                columns[at_created], columns[at_updated] = columns[at_updated], columns[at_created]
        return _ListLayout(tuple(columns), compact)

    def _set_columns(self, table: DataTable, layout: _ListLayout) -> None:
        """Set the table's columns up for ``layout`` (clearing it) when they aren't the ones it has."""
        if layout == self._layout:
            return
        self._layout = layout
        table.clear(columns=True)
        for label, width in layout.columns:
            table.add_column(label, width=width, key=label)

    def _recent(self) -> list:
        """The jobs of the retention window (the configured schema's list has older ones too)."""
        now = self._clock()
        return [j for j in self._all_jobs if views.within_retention(j, now)]

    def _location_column_width(self, jobs: list) -> int:
        """The Location column's width for ``jobs``; 0 (hidden) when they all share one location."""
        labels = {views.location_label(job) for job in jobs}
        if len(labels) <= 1:
            return 0
        low, high = self._LOCATION_WIDTHS
        return max(low, min(high, max(views.cell_width(label) for label in labels)))

    def _clock(self) -> datetime:
        return self._now or datetime.now(timezone.utc)

    def _update_subtitle(self) -> None:
        if self.app.screen is not self:
            return  # a late refresh must not retitle the screen on top
        if self._refreshing:
            refresh_status = "refreshing…"
            if self._last_refreshed:
                refresh_status += f" (last refreshed {self._last_refreshed})"
        elif self._list_error and self._failed_at and self._loaded_at is None:
            refresh_status = f"load failed {self._failed_at}"  # no list yet: the error is in its place
        elif self._list_error and self._failed_at:
            refresh_status = f"refresh failed {self._failed_at}"
            if self._last_refreshed:
                refresh_status += f" (list from {self._last_refreshed[:5]})"
        else:
            refresh_status = f"last refreshed {self._last_refreshed or '—'}"
        bits = [refresh_status, self._scope_text()]
        self.app.sub_title = "  ·  ".join(bit for bit in bits if bit)

    def _scope_text(self) -> str:
        """What the list on screen covers ("all schemas", ...); "" before the first list."""
        if self._scope is None:
            return ""
        labels = {views.location_label(job) for job in self._all_jobs}
        only = next(iter(labels)) if len(labels) == 1 else ""
        return scope_text(self._scope, location=self._schema_label(), only=only)

    def _refresh_jobs(self, *, manual: bool = False, quiet: bool = False) -> None:
        """Refresh; ``manual`` permits reprobe and ``quiet`` suppresses an in-flight notice."""
        if self._refreshing:
            if manual and not quiet:
                waited = int(time.monotonic() - (self._refresh_started or time.monotonic()))
                self.notify(f"Still waiting for the last refresh ({waited}s); r asks again once it answers.",
                            markup=False)
            return
        self._refreshing = True
        self._refresh_started = time.monotonic()
        self._update_subtitle()
        account = self._account_wanted()
        self._load_jobs(manual, account, account and self._asked)

    def _account_wanted(self) -> bool:
        """Whether the list to show is the account-wide one (a), not the configured schema's."""
        return self._lister.account_wide

    def action_show_list(self, which: str) -> None:
        """``a``: show the account-wide list (``which`` "account") or the configured schema's ("schema")."""
        account = which == "account"
        if account == self._lister.account_wide:
            return
        self._lister.account_wide = account
        self._asked = account
        self.refresh_bindings()  # the footer's a names the other list now
        self._refresh_jobs(manual=True, quiet=True)

    def _refresh_tick(self) -> None:
        if self.app.screen is self:
            self._refresh_jobs()

    def action_refresh(self) -> None:
        self._refresh_jobs(manual=True)

    def _search_focused(self) -> bool:
        try:
            return self.focused is self.query_one("#jobfilter", Input)
        except NoMatches:  # not composed yet
            return False

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        if action == "filter":
            return not self._search_focused()  # in the box, / is part of the query
        if action == "submit_search":
            return self._search_focused()  # on the list, Enter opens the highlighted row
        if action == "show_list":
            # One of a's two bindings, the one for the other list.
            return (parameters[:1] == ("account",)) != self._lister.account_wide
        return True

    def action_filter(self) -> None:
        self.set_focus(self.query_one("#jobfilter", Input))

    async def _typed_ahead(self, box: Input) -> None:
        """Wait for keys already sent to ``box`` before acting on its text."""
        done = asyncio.get_running_loop().create_future()

        def handled() -> None:
            if not done.done():
                done.set_result(None)

        if not box.call_later(handled):  # after what is already in the box's queue
            return
        try:
            await asyncio.wait_for(done, timeout=1.0)
        except asyncio.TimeoutError:  # never expected: act on what the box holds
            pass

    async def action_submit_search(self) -> None:
        """Enter in the search box: show the matches of what was typed now, and move to the list."""
        box = self.query_one("#jobfilter", Input)
        await self._typed_ahead(box)
        text = box.value.strip()
        if text != self._filter or self._search_timer is not None:
            # Enter right after typing: show the matches before Enter can pick a row.
            self._filter = text
            self._search_now()
        self.set_focus(self.query_one("#jobs", DataTable))

    async def action_clear_filter(self) -> None:
        if close_keys_panel(self):
            return
        box = self.query_one("#jobfilter", Input)
        await self._typed_ahead(box)
        if self.focused is box or self._filter or box.value:
            box.value = ""  # its Changed comes after, and finds the search already cleared
            if self._filter or self._search_timer is not None:
                self._filter = ""
                self._search_now()
            self.set_focus(self.query_one("#jobs", DataTable))

    def action_cycle_sort(self) -> None:
        keys = views.SORT_KEYS
        self._sort = keys[(keys.index(self._sort) + 1) % len(keys)]
        self._render_jobs(top=True)  # a new order: show its start, not the old row wherever it went

    def action_toggle_order(self) -> None:
        self._descending = not self._descending
        self._render_jobs(top=True)

    def whole_id(self, key) -> str | None:
        """The whole id of the job or sub-job on row ``key`` (a sub-job's row shows its label there)."""
        if key in self._job_by_item:
            job = self._jobs_by_key.get(self._job_by_item[key])
            return views.job_id_of(job) if job is not None else None
        sub = self._sub_by_item.get(key)  # (job key, sub-job id)
        if sub:
            job = self._jobs_by_key.get(sub[0])
            return sub[1] if sub[1] else (views.job_id_of(job) if job is not None else None)
        return None

    def action_copy_text(self) -> None:
        """ctrl+c: the text the mouse highlighted (in the list too), else everything
        the screen shows, the whole list with whole ids."""
        if not self.app.copy_highlight(self):
            self.app.copy_everything(self)

    def _highlighted(self) -> tuple:
        table = self.query_one("#jobs", DataTable)
        try:
            key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        except Exception:  # noqa: BLE001 - empty table
            return None, None
        if key in self._job_by_item:
            return key, self._job_by_item[key]
        if key in self._sub_by_item:
            return key, self._sub_by_item[key][0]
        if str(key).startswith("note-"):  # an UNKNOWN job's "status unavailable" line
            return key, self._job_by_item.get(str(key).removeprefix("note-"))
        return key, None

    def _is_expanded(self, job: dict) -> bool:
        job_key = self._job_key(job)
        if job_key in self._collapsed:
            return False
        return job_key in self._expanded or views.is_live(job)

    @staticmethod
    def _expandable(job: dict) -> bool:
        # An UNKNOWN job unfolds into a "status unavailable" line even without sub-jobs.
        return bool(views.sub_jobs(job)) or views.is_unknown(job)

    def _unfold(self, job_key: tuple, unfolded: bool) -> None:
        """Toggle a job's sub-rows while preserving cursor and scroll positions."""
        if unfolded:
            self._collapsed.discard(job_key)
            self._expanded.add(job_key)
        else:
            self._expanded.discard(job_key)
            self._collapsed.add(job_key)
        table = self.query_one("#jobs", DataTable)
        scrolled = table.scroll_x
        key = self._row_key(job_key)
        try:
            on_screen = table.get_row_index(key) - int(table.scroll_y)  # rows below the top of the view
        except Exception:  # noqa: BLE001 - not in the table
            on_screen = None
        self._render_jobs(keep=job_key)
        if scrolled:
            after_refresh(self, table.scroll_to, x=scrolled, animate=False, immediate=True)
        if on_screen is not None:
            after_refresh(self, self._keep_job_row, key, on_screen, unfolded)

    def _keep_job_row(self, key: str, on_screen: int, unfolded: bool, tries: int = 10) -> None:
        """Restore a toggled job row's screen position."""
        table = self.query_one("#jobs", DataTable)
        try:
            row = table.get_row_index(key)
        except Exception:  # noqa: BLE001 - gone with a refresh meanwhile
            return
        view = table.scrollable_content_region.height - (table.header_height if table.show_header else 0)
        top = row - max(0, on_screen)
        if unfolded and view > 0:
            rows = table.ordered_rows
            last = row
            while last + 1 < len(rows) and str(rows[last + 1].key.value).startswith((f"sub-{key}-", f"note-{key}")):
                last += 1
            if last - top >= view:
                top = min(row, last - view + 1)
        top = max(0, top)
        if (view <= 0 or top > table.max_scroll_y) and tries:
            # Not laid out, or not yet as tall as its new rows: once it is.
            after_refresh(self, self._keep_job_row, key, on_screen, unfolded, tries - 1)
            return
        table.scroll_to(y=top, animate=False, immediate=True)

    def unfold_highlighted(self) -> bool:
        """→ on a job's own row: unfold it; whether it did (where it didn't, → scrolls the table)."""
        key, job_key = self._highlighted()
        return key in self._job_by_item and self._toggle(job_key, True)

    def fold_highlighted(self) -> bool:
        """← on a job's row or a sub-job row: fold the job; whether it did (else ← scrolls the table)."""
        return self._toggle(self._highlighted()[1], False)

    def toggle_job_at(self, key) -> bool:
        """A click on the ▸/▾ marker of the job row ``key``: unfold or fold it; whether it did."""
        return self._toggle(self._job_by_item.get(key), None)

    def _toggle(self, job_key, unfold: bool | None) -> bool:
        """Unfold (True) or fold (False) a job, or (None) switch it; whether it did."""
        job = self._jobs_by_key.get(job_key)
        if job is None or not self._expandable(job) or self._is_expanded(job) == unfold:
            return False
        self._unfold(job_key, not self._is_expanded(job))
        return True

    def action_expand_all(self) -> None:
        with_subs = [j for j in self._all_jobs if self._expandable(j)]
        if with_subs and all(self._is_expanded(j) for j in with_subs):
            self._expanded.clear()
            self._collapsed = {self._job_key(j) for j in self._all_jobs if views.is_live(j)}
        else:
            self._collapsed.clear()
            self._expanded = {self._job_key(j) for j in self._all_jobs}
        self._render_jobs()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "jobfilter":
            return
        if event.value != event.input.value:
            return  # the box has changed again since: a Changed with what it holds is on its way
        text = event.value.strip()
        if text == self._filter and self._search_timer is None:
            return  # the list shows this search already (Enter or Esc applied it, or a space was typed)
        self._filter = text
        # Rebuild once typing pauses; a rebuild per keystroke lagged with ~900 jobs.
        if self._search_timer is not None:
            self._search_timer.stop()
        self._search_timer = self.set_timer(self._SEARCH_DEBOUNCE_SECONDS, self._search_now)

    def _search_now(self) -> None:
        """Show the matches of the search (``_filter``), without waiting for typing to pause."""
        if self._search_timer is not None:
            self._search_timer.stop()
            self._search_timer = None
        self._render_jobs()

    @work(thread=True, group="jobs")
    def _load_jobs(self, manual: bool = False, account: bool = False, asked: bool = False) -> None:
        """Read one schema or account-wide listing off the UI thread."""
        worker = get_current_worker()
        try:
            listing = self._lister.load(self._clock(), is_cancelled=lambda: worker.is_cancelled, manual=manual,
                                        account=account, asked=asked)
            error = None
        except JobListError as exc:
            listing, error = None, exc
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            listing, error = None, JobListError(views.collapse(f"the job list failed: {views.describe_error(exc)}"))
        if worker.is_cancelled or (listing is None and error is None):
            return
        call_on_screen(self, self._on_jobs, listing, error, account)

    def _on_jobs(self, listing, error: JobListError | None, account: bool = False) -> None:
        """A load's answer: ``listing``, or ``error``; ``account``: it read the account-wide list."""
        self._refreshing = False
        if account != self._account_wanted():
            # a switched lists while this one was read: read the one asked for now.
            self._refresh_jobs(manual=True, quiet=True)
            return
        if account and self._asked:
            self._asked = False
            if error is not None or listing.scope.kind != ACCOUNT:
                listing = self._stay_on_schema(listing, error)
                if listing is None:
                    return
                error = None
        if error is not None:
            # Keep the last good list on screen, saying so, until a load succeeds.
            self._list_error = str(error) or "the job list failed"
            self._list_refused = error.refused
            self._failed_at = time.strftime("%H:%M:%S")
            self._update_subtitle()
            self._render_jobs()
            return
        self._list_error = None
        self._list_refused = False
        self._failed_at = None
        self._loaded_at = time.monotonic()
        self._scope = listing.scope
        self._all_jobs = self._merged(listing.jobs)
        self._jobs_by_key = {self._job_key(job): job for job in self._all_jobs}
        self._rows = {key: rows for key, rows in self._rows.items() if key in self._jobs_by_key}
        self._hay = {key: hay for key, hay in self._hay.items() if key in self._jobs_by_key}
        self._last_refreshed = time.strftime("%H:%M:%S")
        self._update_subtitle()
        self._render_jobs()
        self.app.job_list_drawn(self)

    def _stay_on_schema(self, listing: Listing | None, error: JobListError | None) -> Listing | None:
        """Handle an unavailable account list by retaining the configured schema."""
        self._lister.account_wide = False
        self.refresh_bindings()  # the footer's a offers all schemas again
        why = (str(error) or "the account-wide job list failed") if error is not None else unavailable_text(
            listing.scope)
        self.notify(f"Can't show all schemas ({why}); the list stays on {self._schema_label()}.",
                    severity="warning", timeout=10, markup=False)
        if error is None:
            return Listing(listing.jobs, ListScope(SCHEMA))
        self._update_subtitle()
        if self._loaded_at is None:
            self._refresh_jobs()  # nothing listed yet (a came first): read the configured schema's list
        return None

    def _merged(self, rows: list) -> list:
        """Reuse unchanged records and retain an UNKNOWN row's known creation time."""
        merged = []
        for job in rows:
            if not (isinstance(job, dict) and views.job_id_of(job)):
                continue
            old = self._jobs_by_key.get(self._job_key(job))
            if old is not None:
                if not job.get("created_at") and old.get("created_at"):
                    job = {**job, "created_at": old["created_at"]}
                if job == old:
                    job = old
            merged.append(job)
        return merged

    def job_fetched(self, job: dict) -> None:
        """Merge a detail screen's fresh job into the list."""
        if not isinstance(job, dict):
            return
        key = self._job_key(job)
        current = self._jobs_by_key.get(key)
        if current is None:
            return
        fresh = views.refresh_job(current, job)
        if fresh == current:
            return
        self._all_jobs = [fresh if j is current else j for j in self._all_jobs]
        self._jobs_by_key[key] = fresh

    # ─── rows ────────────────────────────────────────────────────────────────

    @staticmethod
    def _name_cell(job: dict, unfolded: bool | None = None) -> _FixedCell:
        """A plain job id with its optional fold marker."""
        parts = [(views.job_id_of(job), "grey50")]
        if unfolded is not None:
            parts.insert(0, ("▾ " if unfolded else "▸ ", "grey50"))
        return _FixedCell.assemble(*parts)

    @staticmethod
    def _time_cell(raw, compact: bool) -> _FixedCell:
        shown = format_created_at(raw)
        if shown and compact:
            shown = shown[5:]  # MM-DD HH:MM: the list spans six months, so the year is never in doubt
        return _cell(shown or "—")

    @staticmethod
    def _owner_cell(job: dict) -> _FixedCell:
        """The submitting user; a job that records none shows its owning role, dimmed and named a role."""
        user = views.collapse(job.get("submitted_by") or "")
        if user:
            return _cell(user)
        role = views.collapse(job.get("owner_role") or "")
        return _cell(f"role {role}", "grey62") if role else _cell("—")

    def _job_cells(self, job: dict, compact: bool) -> dict:
        """A job's row but for its Job ID: column label -> cell."""
        live = views.is_live(job)
        word = views.status_word(job.get("status"))
        unknown = views.is_unknown(job)
        status_style = {"running": "bold green", "failed": "bold red", "error": "bold red"}.get(
            word, "yellow" if live or unknown else "grey62")
        models = views.job_models(job)
        model = views.short_model(models[0]) if models else "—"
        if len(models) > 1:
            model += f" +{len(models) - 1}"
        if unknown:
            marker = _cell("?", "bold yellow")  # no live state: neither running nor ended, as far as known
        else:
            marker = _cell("●", "bold green") if live else _cell("○", "grey50")
        # Plain text throughout: DataTable parses a str cell as markup.
        return {
            " ": marker,
            "Status": _cell(word.upper(), status_style),
            "Type": _cell(views.job_type_label(job)),
            "Created": self._time_cell(job.get("created_at"), compact),
            "Owner": self._owner_cell(job),
            "Model": _cell(model),
            # Requested GPUs; "shared" for a job that runs only on another job's GPUs.
            "GPUs": _cell(views.job_gpu_cell(job, str(job.get("hardware") or ""))),
            # Plain text: a quoted name can hold anything, brackets included.
            "Location": _cell(views.location_label(job) or "—", "grey70"),
            "Updated": self._time_cell(job.get("updated_at"), compact),
        }

    @staticmethod
    def _row_or_note(job: dict, build) -> dict:
        """Build a row, isolating one malformed job from the rest of the list."""
        try:
            return build()
        except Exception:  # noqa: BLE001 - the traceback goes to the error log
            jid = views.job_id_of(job) if isinstance(job, dict) else "?"
            LogScreen._log_traceback(f"job list row for {jid}")
            return {"Job ID": _cell(jid, "grey50"), "Status": _cell("? can't show", "yellow")}

    @staticmethod
    def _sub_cells(sj: dict, last: bool) -> dict:
        word = views.status_word(sj.get("status"))
        style = {"running": "green", "failed": "red", "error": "red"}.get(word, "grey62")
        branch = "└─" if last else "├─"
        return {
            "Job ID": _cell(f"   {branch} {views.sub_job_label(sj)}", "grey70"),
            "Status": _cell(word.upper(), style),
            "Type": _cell(views.sub_job_type(sj), "grey70"),
            "Model": _cell(views.short_model(sj.get("model_name")) or "—", "grey70"),
            "GPUs": _cell(views.sub_job_gpu_text(sj), "grey70"),
        }

    def _sub_rows(self, job: dict) -> tuple:
        """An expanded job's status note and sub-job rows."""
        job_key = self._job_key(job)
        entries = views.sub_jobs(job)
        subs, keys = [], []
        if views.is_unknown(job):
            subs.append(("note", {"Job ID": _cell(f"   {'├─' if entries else '└─'} ? status unavailable", "yellow")}))
            keys.append(None)
        for si, sj in enumerate(entries):
            last = si == len(entries) - 1
            subs.append((str(si), self._row_or_note(job, lambda: self._sub_cells(sj, last))))
            keys.append((job_key, sj.get("sub_job_id")))
        return tuple(subs), tuple(keys)

    def _header(self, group: str, count: int, layout: _ListLayout) -> tuple:
        """A LIVE/HISTORY row, kept while it reads the same: its title in the Job ID column."""
        kept = self._headers.get(group)
        if kept is not None and kept[0] == (count, layout):
            return kept[1]
        style = "bold green" if group == "LIVE" else "bold grey62"
        cells = {"Job ID": _cell(f"{group} ({count})", style)}
        cells = tuple(cells.get(column, _BLANK) for column in layout.labels)
        self._headers[group] = ((count, layout), cells)
        return cells

    def _job_rows(self, job: dict, layout: _ListLayout) -> _JobRows:
        """Cached rows for a job, layout, and expansion state."""
        job_key = self._job_key(job)
        compact = layout.compact_times
        rows = self._rows.get(job_key)
        if not (rows is not None and rows.job is job and rows.compact == compact):
            rows = _JobRows(job, compact, self._row_or_note(job, lambda: self._job_cells(job, compact)))
            self._rows[job_key] = rows
        expanded = self._is_expanded(job)
        if rows.layout != layout or rows.expanded != expanded:
            unfolded = expanded if self._expandable(job) else None
            name = rows.cells.get("Job ID") or self._row_or_note(
                job, lambda: {"Job ID": self._name_cell(job, unfolded)})["Job ID"]
            labels = layout.labels
            cells = rows.cells
            rows.row = tuple(name if label == "Job ID" else cells.get(label, _BLANK) for label in labels)
            if expanded:
                if rows.sub_cells is None:
                    rows.sub_cells, rows.sub_keys = self._sub_rows(job)
                rows.subs = tuple((suffix, tuple(sub.get(label, _BLANK) for label in labels))
                                  for suffix, sub in rows.sub_cells)
            else:
                rows.subs = ()
            rows.layout, rows.expanded = layout, expanded
        return rows

    def _query(self) -> views.Query:
        """The resolved search query, cached with its input and job records."""
        kept = self._resolved
        if kept is not None and kept[0] == self._filter and kept[1] is self._all_jobs:
            return kept[2]
        query = views.resolve_query(views.parse_query(self._filter), self._all_jobs)
        self._resolved = (self._filter, self._all_jobs, query)
        return query

    def _haystack(self, job: dict) -> str:
        """``views.search_haystack`` of ``job``, kept while its record stays the same."""
        key = self._job_key(job)
        kept = self._hay.get(key)
        if kept is not None and kept[0] is job:
            return kept[1]
        hay = views.search_haystack(job)
        self._hay[key] = (job, hay)
        return hay

    # ─── the lines around the table ──────────────────────────────────────────

    def _show_notice(self) -> None:
        """Show a failed refresh above the retained list."""
        widget = self.query_one("#listnotice", Static)
        stale = bool(self._list_error and self._all_jobs)
        widget.display = stale
        self._notice = None
        if not stale:
            return
        self._notice = Content.from_markup("   [red]refresh failed $failed: $error; showing the list from $at[/red]",
                                           failed=self._failed_at or "", error=self._list_error or "",
                                           at=(self._last_refreshed or "—")[:5])
        self._paint_notice()

    def _paint_notice(self) -> None:
        if self._notice is None:
            return
        widget = self.query_one("#listnotice", Static)
        widget.update(hanging(self._notice, widget.size.width or self.size.width or self.app.size.width, 3))

    def _paint_info(self) -> None:
        if self._info is None:
            return
        widget = self.query_one("#listinfo", Static)
        width = widget.content_size.width or (self.size.width or self.app.size.width) - 2
        if isinstance(self._info, Content):
            widget.update(hanging(self._info, width, 2))
        else:
            widget.update(wrapped_parts(self._info, width, 2, Content.from_markup("   [dim]·[/dim]  ")))

    def _show_nomatch(self, message: str | None) -> None:
        """The message in place of the table (it takes the table's room, so the search box stays put)."""
        widget = self.query_one("#nomatch", Static)
        box = self.query_one("#nomatch-box", VerticalScroll)
        if message != self._nomatch:
            box.scroll_home(animate=False)  # a new message is read from its start
        self._nomatch = message
        widget.display = box.display = message is not None
        self._paint_nomatch()

    def _paint_nomatch(self) -> None:
        if self._nomatch is None:
            return
        widget = self.query_one("#nomatch", Static)
        width = widget.content_size.width
        if not width:
            width = (self.size.width or self.app.size.width) - 6
            self.call_after_refresh(self._repaint_nomatch, width)
        message = self._nomatch
        lines = [Content(line) for line in message.split("\n")]
        widget.update(Content("\n").join(hanging(line, width, 2) for line in lines))

    def _repaint_nomatch(self, painted_for: int) -> None:
        if self.query_one("#nomatch", Static).content_size.width not in (0, painted_for):
            self._paint_nomatch()

    def _older_matches(self, query) -> list:
        """Ids of the jobs older than six months (hidden) that ``query`` matches, newest first."""
        now = self._clock()
        home = self._clients.home
        found = [job for job in self._all_jobs if not views.within_retention(job, now)
                 and views.job_matches_query(job, query, home=home, hay=self._haystack(job))]
        return [views.job_id_of(j) for j in views.sort_job_list(found, "created", True)]

    # ─── the table ───────────────────────────────────────────────────────────

    def _render_jobs(self, *, keep: tuple | None = None, top: bool = False) -> None:
        """Draw the list, optionally keeping one job or starting at the top."""
        table = self.query_one("#jobs", DataTable)
        info_line = self.query_one("#listinfo", Static)
        keep_sub = None
        if keep is None and not top:
            key, keep = self._highlighted()
            keep_sub = self._sub_by_item.get(key)  # a sub-job row keeps its own cursor
        recent = self._recent()
        layout = self._layout_for(self._location_column_width(recent))
        self._show_notice()
        info_line.set_class(False, "error")
        if self._loaded_at is None and not self._list_error:
            self._info = Content.from_markup("  [dim]loading jobs…[/dim]")
            self._paint_info()
            self._hide_table(table, layout, "Loading jobs…")
            return
        if self._list_error and not self._all_jobs:
            info_line.set_class(True, "error")
            hint = f" · a lists {self._schema_label()} only" if self._list_refused else ""
            self._info = None
            info_line.update(Text(f"[error] {self._list_error}  (r retries{hint})", style="red"))
            self._hide_table(table, layout, "")  # an empty message keeps the search box at the bottom
            return
        hidden = len(self._all_jobs) - len(recent)
        query = self._query()
        home = self._clients.home
        if query.empty and not query.errors:
            visible = recent
        else:
            visible = [job for job in recent if views.job_matches_query(job, query, home=home, hay=self._haystack(job))]
        ordered = views.sort_job_list(visible, self._sort, self._descending)
        live = [j for j in ordered if views.is_live(j)]
        history = [j for j in ordered if not views.is_live(j)]

        parts = [Content.from_markup(f"[bold green]● {len(live):,} live[/]   [grey62]○ {len(history):,} historical[/]")]
        if self._filter:
            parts.append(Content.from_markup(f"[b]{len(visible):,}[/b] [dim]match(es) for[/dim] [b]$query[/b]",
                                             query=self._filter))
        sort = _sort_part(self._sort, self._descending)
        parts.append(sort + Content(" " * (_SORT_PART_CELLS - sort.cell_length)))
        self._info = parts
        self._paint_info()

        if not visible:
            one = self._scope is not None and self._scope.kind != ACCOUNT
            widen = self._scope is not None and self._scope.kind == SCHEMA
            if self._filter:
                found = dict(cut=loaded_text(self._scope), older=self._older_matches(query) if hidden else (),
                             location=self._schema_label() if one else "", widen=widen)
                message = views.no_match_message(self._filter, query, len(recent), **found)
            elif self._scope is not None and self._scope.truncated:
                message = empty_text(self._scope)
            elif self._scope is not None and self._scope.kind == ACCOUNT:
                message = "No jobs created in the last 6 months that this role can see, in any database and schema."
            else:
                message = f"No jobs created in the last 6 months in {self._schema_label()} that this role can see."
                if widen:
                    message += "\n  a lists every database and schema."
            self._hide_table(table, layout, message)
            return
        self._show_nomatch(None)
        was_hidden = not table.display
        table.display = True

        plan: list = []
        job_by_item: dict = {}
        sub_by_item: dict = {}
        target_row = None
        groups = (("LIVE", live), ("HISTORY", history))
        if self._sort == "status" and not self._descending:
            groups = groups[::-1]  # status ↑ ended first: the ended jobs are what comes first
        for group, jobs in groups:
            if not jobs:
                continue
            plan.append((f"hdr-{group.lower()}", self._header(group, len(jobs), layout)))
            for job in jobs:
                job_key = self._job_key(job)
                item_id = self._row_key(job_key)
                job_by_item[item_id] = job_key
                rows = self._job_rows(job, layout)
                plan.append((item_id, rows.row))
                if job_key == keep and target_row is None:
                    target_row = len(plan) - 1
                for (suffix, cells), sub_key in zip(rows.subs, rows.sub_keys):
                    if sub_key is None:
                        plan.append((f"note-{item_id}", cells))
                        continue
                    row_key = f"sub-{item_id}-{suffix}"
                    sub_by_item[row_key] = sub_key
                    plan.append((row_key, cells))
                    if keep_sub is not None and sub_key == keep_sub:
                        target_row = len(plan) - 1
        self._job_by_item = job_by_item
        self._sub_by_item = sub_by_item
        # Reuse identical row objects, or reorder them without rebuilding.
        shown = self._shown
        same = (shown is not None and shown[0] == layout and len(shown[1]) == len(plan)
                and table.row_count == len(plan)
                and all(a[0] == b[0] and a[1] is b[1] for a, b in zip(shown[1], plan)))
        if not same:
            if (shown is not None and shown[0] == layout and table.row_count == len(plan)
                    and {key: id(cells) for key, cells in shown[1]} == {key: id(cells) for key, cells in plan}):
                at = layout.labels.index("Job ID")
                position = {id(cells[at]): index for index, (_key, cells) in enumerate(plan)}
                table.sort("Job ID", key=lambda name: position[id(name)])
            else:
                self._set_columns(table, layout)
                table.replace_rows(plan)
            self._shown = (layout, tuple(plan))
        if target_row is None:
            target_row = 1  # first job row (row 0 is a group header)
        target_row = min(target_row, table.row_count - 1)
        if was_hidden:
            # Place again after layout without initially scrolling off the group header.
            table.move_cursor(row=target_row, animate=False, scroll=False)
            after_refresh(self, self._place_cursor, target_row)
        elif not same or top or table.cursor_row != target_row:
            self._place_cursor(target_row)

    def _row_key(self, job_key: tuple) -> str:
        """A stable table key for this screen."""
        key = self._keys.get(job_key)
        if key is None:
            key = self._keys[job_key] = f"job-{len(self._keys)}"
        return key

    def _hide_table(self, table: DataTable, layout: _ListLayout, message: str) -> None:
        """The table off, with ``message`` in its place."""
        self._set_columns(table, layout)
        table.clear()
        self._shown = None
        self._job_by_item, self._sub_by_item = {}, {}
        table.display = False
        self._show_nomatch(message)

    def _schema_label(self) -> str:
        names = [getattr(self._client, attr, None) for attr in ("database", "schema")]
        if all(isinstance(n, str) and n for n in names):
            return ".".join(names)
        return "the configured schema"

    def _place_cursor(self, row: int, tries: int = 20) -> None:
        table = self.query_one("#jobs", DataTable)
        if table.size.height == 0 and tries:
            after_refresh(self, self._place_cursor, row, tries - 1)  # not laid out yet
            return
        table.move_cursor(row=row, animate=False)
        if row <= 1:
            table.scroll_home(animate=False)  # keep the group header above the first job

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        from cortex_training.tui.screens import JobSummaryScreen

        if not is_shown(self):
            return  # a second quick Enter: the first already opened the job
        key = event.row_key.value
        if str(key).startswith("note-"):  # an UNKNOWN job's note line stands for its job
            key = str(key).removeprefix("note-")
        if key in self._job_by_item:
            job = self._jobs_by_key.get(self._job_by_item[key])
            # Every call about the job goes to its own database and schema.
            client = self._job_client(job)
            if client is not None:
                jid = views.job_id_of(job)
                self.app.push_screen(JobSummaryScreen(client, jid, job=job,
                                                      poll_interval=self._poll_interval, on_job=self.job_fetched))
        elif key in self._sub_by_item:
            job_key, sid = self._sub_by_item[key]
            job = self._jobs_by_key.get(job_key)
            client = self._job_client(job)
            if client is not None:
                jid = views.job_id_of(job)
                self.app.push_screen(LogScreen(client, jid, job=job, sub_job_id=sid, poll_interval=self._poll_interval))

    def _job_client(self, job: dict | None):
        """The job's location-bound client, or None after notifying."""
        try:
            return self._clients.for_job(job)
        except Exception as exc:  # noqa: BLE001 - shown, with the traceback in the error log
            LogScreen._log_traceback(f"no client for {views.location_label(job)}")
            self.notify(f"Can't open the job: no client for {views.location_label(job)} "
                        f"({views.describe_error(exc)}; details in {_ERROR_LOG}).", severity="error", markup=False)
            return None


_SEPARATOR = "  ·  "


def fit_line(segments: list, width: int) -> Content:
    """Fit styled segments by cutting those with minimum widths, then wrapping."""
    texts = [text for text, _style, _least in segments]
    sep = views.cell_width(_SEPARATOR)
    over = sum(views.cell_width(text) for text in texts) + sep * (len(texts) - 1) - width
    spares = [max(0, views.cell_width(text) - least) if least is not None else 0 for text, _style, least in segments]
    if 0 < over <= sum(spares):  # cutting gets it onto one row
        for i, spare in enumerate(spares):
            if over <= 0:
                break
            if spare:
                cut = min(spare, over)
                texts[i] = views.fit_cells(texts[i], views.cell_width(texts[i]) - cut)
                over -= cut
    rows: list = []
    for text, (_text, style, _least) in zip(texts, segments):
        piece = Content.styled(text, style)
        if rows and rows[-1].cell_length + sep + piece.cell_length <= width:
            rows[-1] = rows[-1] + Content(_SEPARATOR) + piece
        else:
            rows.append(piece)
    return Content("\n").join(rows)


def _fit_line(parts: list[list[tuple[str, int]]], order: list[int], width: int, variables: dict,
              *, cut: str | None = None, cut_min: int = 40) -> Content:
    """Choose progressively shorter forms until joined parts fit one line.

    ``cut`` reserves ``cut_min`` cells while forms are chosen, then takes the
    remaining room. The final line is clipped only as a last resort.
    """
    levels = [0] * len(parts)

    def build(values: dict) -> tuple[Content, int]:
        forms = [part[min(level, len(part) - 1)] for part, level in zip(parts, levels)]
        line = Content.from_markup(_SEPARATOR.join(markup for markup, _spare in forms if markup), **values)
        return line, line.cell_length + sum(spare for _markup, spare in forms)

    if width <= 0:
        return build(variables)[0]
    text = str(variables.get(cut) or "") if cut else ""
    probe = {**variables, cut: views.fit_cells(text, cut_min)} if text else variables
    cells = build(probe)[1]
    for index in order:
        if cells <= width:
            break
        levels[index] += 1
        cells = build(probe)[1]
    if text:
        used = build({**variables, cut: ""})[1]
        variables = {**variables, cut: views.fit_cells(text, max(1, width - used))}
    line = build(variables)[0]
    return line.truncate(width, ellipsis=True) if line.cell_length > width else line


def _quiet_download(method) -> dict:
    """Disable client stall checks when supported; the TUI reports staleness."""
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return {}
    return {"stale_after_seconds": None} if "stale_after_seconds" in parameters else {}


class _ColoredLinks(Static):
    """A Static whose links retain their own colours."""

    @property
    def link_style(self) -> Style:
        return self.styles.link_style


class FittedFooter(Footer):
    """A footer that hides rightmost keys instead of clipping one."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Where each key ends, measured once all of them are laid out.
        self._edges: list[int] | None = None

    async def recompose(self) -> None:
        self._edges = None  # other keys: measured again
        await super().recompose()
        self.call_after_refresh(self._fit)

    def on_resize(self, event: events.Resize) -> None:
        self._fit()

    def _fit(self) -> None:
        keys = list(self.children)
        if not keys or self.region.width <= 0:
            return
        if self._edges is None or len(self._edges) != len(keys):
            if any(not key.display for key in keys):
                return  # measured only with every key shown (after a recompose)
            self._edges = [key.region.right - self.region.x for key in keys]
        room = self.region.width
        for key, edge in zip(keys, self._edges):
            if key.display != (edge <= room):
                key.display = edge <= room


class _ResourceList(_SourceList):
    """Resource views and units with pane-scrolling navigation."""

    _PANE = "#resource-pane"

    def action_cursor_down(self) -> None:
        target = self.screen._resource_down_from(self.index)
        if target is not None:
            self.index = target
            return
        super().action_cursor_down()

    def watch_index(self, old_index: int | None, new_index: int | None) -> None:
        super().watch_index(old_index, new_index)
        self.screen._resource_list_moved(new_index)


class LogScreen(FollowsPointer, Screen):
    """Read-only Logs and Resource dashboard for one job."""

    BINDINGS = [
        ("b", "back", "Back"),
        Binding("escape", "dismiss_or_back", "Back/clear", show=False),  # the filter box says Esc clears it
        ("q", "quit", "Quit"),
        KEYS_BINDING,
        Binding("t", "cycle_tab", "Tab", show=False),
        Binding("/", "filter", "Filter", priority=True),
        ("l", "cycle_level", "Level"),
        Binding("l", "show_tab('logs')", "Logs", show=False),
        Binding("L", "cycle_level", "Level", show=False),  # the key it had before
        ("p", "toggle_pause", "Pause"),
        ("s", "save_log", "Save"),
        ("r", "refresh_sources", "Refresh"),
        Binding("[", "shrink_sources", "Narrower", show=False),
        Binding("]", "grow_sources", "Wider", show=False),
        Binding("w", "cycle_window", "Window"),  # Resources: the next window
        # Resources: Space shows the next sub-job (round to the first after the last), no Shift
        # needed; shown there with two or more sub-jobs (see check_action). < and > (and , and .)
        # still go back and forth.
        Binding("space", "resource_sub_job(1)", "Sub-job", key_display="space"),
        Binding("less_than_sign", "resource_sub_job(-1)", "Prev sub-job", key_display="<", show=False),
        Binding("greater_than_sign", "resource_sub_job(1)", "Next sub-job", key_display=">", show=False),
        Binding("comma", "resource_sub_job(-1)", "Prev sub-job", show=False),
        Binding("full_stop", "resource_sub_job(1)", "Next sub-job", show=False),
        Binding("g", "show_tab('resource')", "Resources", show=False),
    ]

    # Auto-fit the source labels until [ or ], while preserving the log's minimum.
    _SOURCES_MIN = 12
    _SOURCES_AUTO_MIN = 16
    _SOURCES_AUTO_MAX = 48
    _SOURCES_STEP = 8
    _LOG_MIN_WIDTH = 40
    _LEVEL_CYCLE = [None, "INFO", "WARNING", "ERROR"]
    _KEEP_CONSOLE_AFTER_SECONDS = 600
    _DATA_REFRESH_SECONDS = 30
    _STATUS_REFRESH_SECONDS = 60
    _LOG_ONLY = {"save_log", "cycle_level", "toggle_pause", "shrink_sources", "grow_sources", "filter"}

    class _Note(str):
        """A TUI-authored pane line distinguished by type, not text."""

        __slots__ = ()

    class _CallbackError(Exception):
        """A worker's UI callback failed, not its request."""

    def __init__(self, client, job_id, *, job: dict | None = None, sub_job_id=None, initial_tab: str = "logs",
                 poll_interval=1.0):
        super().__init__()
        self._client = client
        self._job_id = job_id
        self._sub_job_id = sub_job_id
        self._poll_interval = poll_interval
        self._job = job or {"job_id": job_id}
        self._job_status = self._job.get("status")
        self._job_reason = None  # failure/cancel reason, when terminal
        self._sources_width = self._SOURCES_AUTO_MIN
        self._sources_auto = True  # sized to its labels until [ or ]
        cache_job_id = _cache_job_id(client, job_id)
        self._cache = LogCache(cache_job_id)
        self._source_by_item: dict[str, str] = {}
        self._logview: Log | None = None
        self._filter = ""  # grep substring for the log pane
        self._min_level = None  # None | INFO | WARNING | ERROR
        self._paused = False  # auto-scroll paused?
        self._current_source = None  # source currently being tailed
        self._shown_lines: list[str] = []  # unwrapped lines on screen (for re-wrap); the TUI's are _Notes
        self._wrap_width = -1  # content width the buffer is wrapped to (-1: unknown or mixed)
        self._rewrap_timer = None  # wraps the pane again once a resize settles (see _schedule_rewrap)
        self._filter_timer = None
        self._last_line_at: str | None = None  # "last line HH:MM:SS": the newest live line's own time
        self._unmounted = False
        self._stage: tempfile.TemporaryDirectory[str] | None = None
        self._console_dir = job_cache_dir(cache_job_id) / "console"
        self._state_changed_at = None  # when the job entered its current (terminal) state
        self._job_created_at = None  # ties a kept console to this job, not a reused id
        self._job_reads = 0
        self._job_reads_lock = threading.Lock()
        self._job_seq = 0  # the number of the read whose answer the view shows (0: none yet)
        self._stdout_inflight = False
        self._stdout_ready = False
        self._stdout_have: set[str] = set()  # sub-jobs whose saved console is in while the rest download
        self._stdout_first: str | None = None  # the sub-job a download fetches before the others
        self._stdout_others = False  # ... and whether the job has others
        self._stdout_failed = False
        self._held: tuple[list[str], bool] | None = None
        self._init_log_state()
        self._active_tab = initial_tab if initial_tab in ("logs", "resource") else "logs"
        self._log_focus_id = "sources"
        self._metrics_root = job_cache_dir(cache_job_id) / "metrics"
        self._resource_data = ResourceData({})
        self._resource_loaded = False
        self._resource_loading = False
        self._resource_refresh_succeeded = False
        self._resource_error: str | None = None
        self._resource_range = DEFAULT_RANGE
        self._resource_downloaded = False
        self._resource_ticker = None
        self._resource_pick: str | None = None
        self._resource_kind = GPU
        self._resource_kind_pref = GPU
        self._resource_unit: int | None = None
        self._resource_drawn: tuple | None = None  # (sub-job, view, unit, window) last drawn
        self._resource_on_view = False  # the panel's highlight is on a view's entry, not a unit's
        self._resource_shown: str | None = None  # the sub-job the panel and unit were set for
        self._resource_settled: str | None = None  # ... and whose view was chosen (see _follow_resource_source)
        self._resource_entries: list[tuple] = []
        self._resource_rows: list[tuple[ListItem, Label]] = []
        self._resource_list_width = 0
        self._resource_placing = False  # the redraw itself is moving the panel's highlight
        self._resource_stale_shown = False  # the panel marks every unit stale (see _resource_stale)
        self._sources_ready = False
        self._sub_job_missing = False
        self._job_read_started: float | None = None  # monotonic time the job read in flight began
        self._pending_top: int | None = None  # the paused reader's line a re-wrap is scrolling back to
        self._job_read_ticker = None  # repaints the status line's wait while that read is slow
        self._no_sources: str | None = None  # why the job has no sources to show (it couldn't be read, say)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="summary")
        with JobTabs(initial=self._active_tab, id="job-tabs"):
            with TabPane("Logs", id="logs"):
                with Horizontal(id="logs-body"):
                    yield _SourceList(id="sources")
                    # Log does not soft-wrap, so writes arrive pre-wrapped.
                    yield _LogPane(id="logview", highlight=False, max_lines=self._MAX_ROWS)
                status = Static("", id="log-status")
                status.display = False
                yield status
                yield Input(placeholder="filter logs…  (/ focus · Esc clear)", id="logfilter")
            with TabPane("Resources", id="resource"):
                with Horizontal(id="resource-body"):
                    yield _ResourceList(id="resource-list")
                    with VerticalScroll(id="resource-pane"):
                        yield _ColoredLinks("Select Resource to load GPU metrics.", id="resource-status")
                        yield Static("", id="resource-stats")
                        yield Static("", id="resource-charts")
        yield FittedFooter(id="footer")

    def on_mount(self) -> None:
        self._render_summary_line()
        self.set_interval(self._DATA_REFRESH_SECONDS, self._auto_refresh_data)
        self.set_interval(self._STATUS_REFRESH_SECONDS, self._refresh_status)
        self._logview = self.query_one("#logview", Log)
        self._logview.auto_scroll = False
        self._logview.highlighter = LogHighlighter()
        self._logview.highlight = True
        self._apply_sources_width()
        self._update_subtitle()
        self._load_sources()
        after_refresh(self, self._focus_active_tab)
        self.set_interval(self._IDLE_CHECK_SECONDS, self._check_idle)

    def on_screen_resume(self) -> None:
        self._update_subtitle()  # a screen pushed over this one may have replaced it

    def _focus_active_tab(self) -> None:
        try:
            self.set_focus(self.query_one("#resource-list" if self._resource_active() else "#sources"))
        except Exception:  # noqa: BLE001 - not mounted yet
            pass

    def on_unmount(self) -> None:
        # A running download owns its temp directory until it returns.
        self._unmounted = True
        self.workers.cancel_node(self)
        if self._logview is not None:
            self.workers.cancel_node(self._logview)
        if not self._stdout_inflight:
            self._cleanup_stage()

    @property
    def _screen_open(self) -> bool:
        """Whether this screen remains mounted on the app stack."""
        return not self._unmounted and on_stack(self)

    def _logs_active(self) -> bool:
        return self._active_tab == "logs"

    def _resource_active(self) -> bool:
        return self._active_tab == "resource"

    def action_show_tab(self, tab: str) -> None:
        self.query_one("#job-tabs", TabbedContent).active = tab

    def action_cycle_tab(self) -> None:
        tabs = self.query_one("#job-tabs", TabbedContent)
        tabs.active = "resource" if self._active_tab == "logs" else "logs"

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        # Ignore queued stale activations, which would re-activate each other.
        if event.tabbed_content.active != event.pane.id or not self._screen_open:
            return  # (or a tab key just before a back key: the screen is being removed)
        self._active_tab = event.pane.id or "logs"
        self.clear_selection()  # what was highlighted is on the tab left behind
        self.refresh_bindings()  # check_action scopes the footer's keys to the tab
        self._update_subtitle()  # the Logs tab's state shows only on Logs
        if self._resource_active():
            focused = self.focused
            if focused is not None and focused.id in {"sources", "logview", "logfilter"}:
                self._log_focus_id = focused.id
            self.set_focus(self.query_one("#resource-list", ListView))  # its arrows pick a view or unit
            self.call_after_refresh(self._render_resource)
            self._start_resource_refresh(force=False)
            return
        try:
            target = self.query_one(f"#{self._log_focus_id}")
        except Exception:  # noqa: BLE001 - stale focus id after a remount
            target = self.query_one("#sources", ListView)
        self.set_focus(target)
        # Re-wraps only if rows were written at another width while Logs was hidden.
        self.call_after_refresh(self._rewrap)

    def _sources_cap(self) -> int:
        """The widest the sources panel may be: the log keeps _LOG_MIN_WIDTH cells."""
        width = getattr(getattr(self, "size", None), "width", 0) or 0
        return max(self._SOURCES_MIN, width - self._LOG_MIN_WIDTH) if width else self._SOURCES_AUTO_MAX

    def _fitted_sources_width(self) -> int:
        labels = [views.sub_job_label(sid) for sid in self._source_by_item.values()]
        longest = max((views.cell_width(label) for label in labels), default=0) + 2  # with the "● " before it
        return min(self._SOURCES_AUTO_MAX, max(self._SOURCES_AUTO_MIN, longest + 4))

    def _apply_sources_width(self) -> None:
        width = self._fitted_sources_width() if self._sources_auto else self._sources_width
        width = max(self._SOURCES_MIN, min(width, self._sources_cap()))
        changed = width != self._sources_width
        self._sources_width = width
        try:
            self.query_one("#sources", ListView).styles.width = width
        except Exception:  # noqa: BLE001 - not mounted yet
            return
        if changed:
            after_refresh(self, self._rewrap)  # log pane width changed

    def action_grow_sources(self) -> None:
        if not self._logs_active():
            return
        self._sources_auto = False
        self._sources_width = min(self._sources_cap(), self._sources_width + self._SOURCES_STEP)
        self._apply_sources_width()
        self.call_after_refresh(self._rewrap)  # log pane width changed

    def action_shrink_sources(self) -> None:
        if not self._logs_active():
            return
        self._sources_auto = False
        self._sources_width = max(self._SOURCES_MIN, self._sources_width - self._SOURCES_STEP)
        self._apply_sources_width()
        self.call_after_refresh(self._rewrap)  # log pane width changed

    def on_resize(self, event) -> None:
        self.call_after_refresh(self._render_summary_line)
        # Reflow only the in-memory views; resize never performs I/O.
        self._apply_sources_width()  # fitted again, and never wider than the screen allows
        self._schedule_rewrap()
        if self._resource_active():
            self.call_after_refresh(self._render_resource)

    def _tabs_resized(self) -> None:
        """Reflow the log and charts after any tab-width change."""
        self._schedule_rewrap()
        if self._resource_active():
            after_refresh(self, self._render_resource)

    # Debounce drag-resize rewrapping.
    _REWRAP_SETTLE_SECONDS = 0.15

    def _schedule_rewrap(self) -> None:
        """UI thread. Wrap the log pane again for its new width once a resize settles (see _rewrap)."""
        if self._rewrap_timer is not None:
            self._rewrap_timer.stop()
        self._rewrap_timer = self.set_timer(self._REWRAP_SETTLE_SECONDS, partial(after_refresh, self, self._rewrap))

    def action_refresh_sources(self) -> None:
        if self._resource_active():
            self._start_resource_refresh(force=True)
            return
        if self._job_read_started is not None:
            # The job is still being read (a thread can't be cancelled): a second read
            # would only wait beside it. Say so, rather than nothing.
            waited = int(time.monotonic() - self._job_read_started)
            self.notify(f"Still reading this job ({waited}s); r reads it again once it answers.", markup=False)
            return
        self._stdout_failed = False  # r retries a saved-console download that failed
        if self._stdout_ready and self._stdout_recent:
            # Fetched moments after the job ended: add the chunks uploaded since,
            # and read the execution logs again (shown for sub-jobs without a console).
            self._stdout_ready = False
            self._stdout_have.clear()
            self._stdout_again = True
            self._exec_logs.clear()
            self._exec_failed.clear()
        self._load_sources()

    def _update_subtitle(self) -> None:
        # Dynamic status line: everything that changes over the session lives
        # here, leaving the #summary line for static config only. The status
        # comes first, so a narrow header still shows it.
        if not self._screen_open or self.app.screen is not self:
            return  # a late tail update must not overwrite another screen's subtitle
        bits = []
        st = (self._job_status or "").strip()
        if st:
            bits.append(st.removeprefix("JOB_STATE_"))
        bits.append(str(self._job_id))
        if self._logs_active():  # the Logs tab's own state
            if self._last_line_at:
                bits.append(self._last_line_at)
            if self._filter:
                bits.append(f"filter:{self._filter}")
            if self._min_level:
                bits.append(f"≥{self._min_level}")
            if self._paused:
                bits.append("PAUSED")
        # Plain text (the header doesn't read markup), but service text could
        # still carry an escape sequence or a line break.
        self.app.sub_title = "  ·  ".join(" ".join(safe_text(bit).split()) for bit in bits)

    # The fewest cells the models are cut to so the summary line fits on one row.
    _SUMMARY_LEAST = 12

    def _render_summary_line(self) -> None:
        job = self._job
        job_id = views.job_id_of(job) or self._job_id
        # A sub-job's id starts with its job's (JOB_ID:TYPE:INDEX): name it once.
        if not self._sub_job_id:
            ids = [job_id, "all sub-jobs"]
        elif str(self._sub_job_id).startswith(f"{job_id}:"):
            ids = [f"sub-job {self._sub_job_id}"]
        else:
            ids = [job_id, f"sub-job {self._sub_job_id}"]
        # Why the job is where it is, in the service's words: reason and reason_details,
        # or the sentence it sends when the job's status is unavailable.
        why = views.status_unavailable_text(job) if views.is_unknown(job) else views.reason_text(job)
        word = views.status_word(job.get("status"))
        style = "red" if word == "failed" else "yellow" if views.is_live(job) else ENDED_STYLE
        # Where the job is (its calls go there), when the account-wide list says it is
        # somewhere other than the configured location.
        location = views.location_label(job) if self.app.job_elsewhere(job) else ""
        least = self._SUMMARY_LEAST
        # The models are cut to keep the line on one row; the ids,
        # the location and the reason stay whole.
        segments = [*((text, "", None) for text in ids),
                    (views.job_type_label(job), "", None), (", ".join(views.job_models(job)) or "—", "", least),
                    *([(location, "", None)] if location else []), *([(why, style, None)] if why else [])]
        self._set_summary(fit_line(segments, self._summary_width()))

    def _summary_width(self) -> int:
        """The cells the summary line has: the screen's, less its padding, before the first layout."""
        try:
            width = self.query_one("#summary", Static).size.width
        except Exception:  # noqa: BLE001 - not mounted, or already removed
            width = 0
        return width or max(40, self.app.size.width - 2)

    def _set_summary(self, text: Content) -> None:
        try:
            self.query_one("#summary", Static).update(text)
        except Exception:  # noqa: BLE001 - not mounted yet
            pass

    # ─── resources ───────────────────────────────────────────────────────
    # Two charts to a row from this pane width, one under the other below it.
    _RESOURCE_SIDE_BY_SIDE = 90
    _RESOURCE_CHART_GAP = 3  # cells between the two charts of a row
    # Rows of bars in every chart, by the terminal's height: (fewest terminal rows, bar
    # rows), tallest first; shorter than all of them, the last. Each even, so the row
    # labelled with half the scale tops out at exactly half of it.
    _RESOURCE_BAR_ROWS_BY_HEIGHT = ((48, 12), (40, 10), (34, 8), (28, 6), (0, 4))
    _RESOURCE_TICK_SECONDS = 5  # how often the newest sample's age is redrawn
    # A running sub-job's chunks reach the stage about every minute, and an upload
    # can take a minute more: a newest sample older than this (or six sampling
    # intervals, if longer) means no new samples are arriving.
    _RESOURCE_STALE_SECONDS = 180
    # The panel's two views, and what the header calls each view's telemetry.
    _RESOURCE_VIEWS = {GPU: "GPU Resources", SYSTEM: "System Resources"}
    _RESOURCE_WHAT = {GPU: "GPU metrics", SYSTEM: "system metrics"}
    _RESOURCE_SAMPLES = {GPU: "GPU samples", SYSTEM: "system samples"}

    def action_resource_range(self, name: str) -> None:
        if not self._resource_active() or name not in RANGES:
            return
        self._resource_range = name
        self._render_resource()

    def action_cycle_window(self) -> None:
        names = list(RANGES)
        index = names.index(self._resource_range) if self._resource_range in names else -1
        self.action_resource_range(names[(index + 1) % len(names)])

    def _start_resource_refresh(self, *, force: bool) -> None:
        if (
            not self._screen_open
            or not self._sources_ready
            or not self._resource_source()
            or self._sub_job_missing
            or self._resource_loading
            or (self._resource_refresh_succeeded and not force)
        ):
            return
        self._resource_loading = True
        if not self._resource_data.kinds:
            self.query_one("#resource-status", Static).update(f"loading {self._RESOURCE_WHAT[self._resource_kind]}…")
        elif self._resource_active():
            self._render_resource()  # "cached · refreshing…" while it runs
        self._refresh_resource()

    @work(thread=True, group="resource")
    def _refresh_resource(self) -> None:
        worker = get_current_worker()
        lock = DownloadLock(self._metrics_root)
        acquired = False
        # Shared per cache directory: a refresh, or reopening the job, reads only new bytes.
        loader = resource_loader(self._metrics_root)
        shown_here: set[str] = set()  # sub-jobs this refresh has put on screen

        def accept(data: ResourceData) -> None:
            shown_here.update(data.kinds)
            self._post(self._accept_cached_resource, data)

        def show_partial(data: ResourceData) -> None:
            if not worker.is_cancelled:
                accept(data)

        def load() -> ResourceData:
            # The selected sub-job first. A large file read from the start is read
            # newest-first, and that partial view is shown only while the screen has
            # nothing for the sub-job: download_metrics rebuilds the file when a chunk
            # lands out of order, and the re-read must not replace a complete view
            # with its newest few minutes.
            source = self._resource_source()
            blank = source not in self._resource_data.kinds and source not in shown_here
            return loader.load(priority=source, window=RANGES.get(self._resource_range),
                               on_partial=show_partial if blank else None)

        try:
            last = loader.last
            if not self._resource_loaded and _has_telemetry(last):
                # Reopened while another screen's refresh holds the lock: paint what the
                # shared loader read last instead of waiting for that download.
                accept(last)
            acquired = lock.acquire(lambda: self._screen_open and not worker.is_cancelled)
            if not acquired or not self._screen_open or worker.is_cancelled:
                return
            # Paint what is cached while the download runs. Every load is incremental:
            # it reads what was committed since the shared loader last read (another
            # screen's download, say, that closed before reading it), and nothing when
            # nothing was. The screen already shows an unchanged result.
            cached = load()
            if _has_telemetry(cached) and not worker.is_cancelled and not _same_telemetry(cached, self._resource_data):
                accept(cached)

            def committed(_result: dict) -> None:
                # Each sub-job is painted as it lands, the one on screen first.
                if worker.is_cancelled or not self._screen_open:
                    return
                data = load()
                if _has_telemetry(data) and not _same_telemetry(data, self._resource_data):
                    accept(data)

            source = self._resource_source()
            self._client.download_metrics(
                self._job_id,
                self._metrics_root,
                resume=True,
                priority=[source] if source else [],
                on_commit=committed,
                **_quiet_download(self._client.download_metrics),
            )
            if worker.is_cancelled or not self._screen_open:
                return  # the screen closed during the download; the next one to open reads it
            fresh = load()  # reads only what the download appended
            if not worker.is_cancelled:
                self._post(self._finish_resource_refresh, fresh, None)
        except Exception as exc:  # noqa: BLE001 - surfaced in the Resource tab
            # The tab's header has room for the start of the error; the log keeps it whole.
            self._log_traceback(f"GPU metrics refresh for {self._job_id} failed")
            if not worker.is_cancelled:
                try:
                    # Only the lock holder may load; without it, show what the loader read last.
                    partial = loader.load() if acquired else (loader.last or ResourceData({}))
                except Exception:  # noqa: BLE001 - preserve the original refresh error
                    partial = ResourceData({})
                self._post(
                    self._finish_resource_refresh,
                    partial if _has_telemetry(partial) else None,
                    views.access_hint(exc, "GPU metrics") or views.describe_error(exc),
                )
        finally:
            if acquired:
                lock.release()

    def _accept_cached_resource(self, data: ResourceData) -> None:
        if not self._screen_open:
            return
        self._resource_data = data
        self._resource_loaded = True
        self._resource_error = None
        if self._resource_active():
            self._render_resource()

    def _finish_resource_refresh(
        self,
        data: ResourceData | None,
        error: str | None,
    ) -> None:
        if not self._screen_open:
            return
        self._resource_loading = False
        self._resource_refresh_succeeded = error is None
        if error is None:
            self._resource_downloaded = True  # every sub-job's rows are in: one without a kind has none
        if data is not None:
            self._resource_data = data
            self._resource_loaded = True
            self._resource_error = error
            if self._resource_active():
                self._render_resource()
            return
        self._resource_error = error
        self._resource_loaded = _has_telemetry(self._resource_data)
        if self._resource_active():
            self._render_resource()

    def _resource_sub_job_ids(self) -> list[str]:
        """The sub-jobs < and > switch between, in the order the Logs tab lists them."""
        return list(dict.fromkeys(self._source_by_item.values()))

    def _resource_source(self) -> str | None:
        """The sub-job the Resources tab shows: the one picked with < or >, else the one selected in Logs."""
        pick = self._resource_pick
        if pick and pick in self._source_by_item.values():
            return pick
        return self._current_source

    def _resource_kind_data(self, kind: str | None = None) -> KindResource | None:
        """The shown sub-job's telemetry of ``kind`` (default: the view shown); None without rows of it."""
        return self._resource_data.kind(self._resource_source(), kind or self._resource_kind)

    def action_resource_sub_job(self, step: int) -> None:
        """Show the previous or next sub-job without changing Logs."""
        ids = self._resource_sub_job_ids()
        current = self._resource_source()
        index = ids.index(current) if current in ids else 0
        self._resource_pick = ids[(index + step) % len(ids)]
        self.refresh_bindings()
        self._render_resource()
        if self._resource_pick not in self._resource_data.kinds:
            # Not downloaded yet (a cold cache downloads the shown sub-job first): fetch it now.
            self._start_resource_refresh(force=True)

    def _follow_resource_source(self, source: str | None) -> None:
        """Apply the preferred view and mean unit to a newly shown sub-job."""
        if source != self._resource_shown:
            self._resource_shown = source
            self._resource_kind = self._resource_kind_pref
            self._resource_settled = None
            self._resource_unit = None
        kinds = self._resource_data.kinds.get(source or "")
        if kinds and self._resource_settled != source:
            self._resource_settled = source
            if self._resource_kind == GPU and GPU not in kinds and SYSTEM in kinds:
                self._resource_kind = SYSTEM

    def _resource_list_moved(self, index: int | None) -> None:
        """Show the view or unit highlighted in the resource panel."""
        if self._resource_placing or index is None or not 0 <= index < len(self._resource_entries):
            return  # the redraw put it on the entry shown (or on a view's while it has no units)
        if not self._screen_open:
            return
        role, which = (self._resource_entries[index] + (None,))[:2]
        if role == "view":
            self._resource_on_view = True
            if which == self._resource_kind and self._resource_unit is None:
                return
            self._resource_kind = self._resource_kind_pref = which
            self._resource_settled = self._resource_shown  # picked here: it stays for this sub-job
            self._resource_unit = None
        elif role == "unit":
            self._resource_on_view = False
            if which == self._resource_unit:
                return
            self._resource_unit = which
        else:
            return
        self._render_resource()

    def _resource_down_from(self, index: int | None) -> int | None:
        """Route ↓ from GPU Resources to its ALL row."""
        entries = self._resource_entries
        if index is None or not 0 <= index < len(entries) or entries[index] != ("view", GPU):
            return None
        return entries.index(("unit", None)) if ("unit", None) in entries else None

    def _resource_cursor(self) -> int | None:
        """The panel entry of what is shown: the unit's (ALL for the mean), or the view's."""
        entries = self._resource_entries
        if not self._resource_on_view and ("unit", self._resource_unit) in entries:
            return entries.index(("unit", self._resource_unit))
        view = ("view", self._resource_kind)
        return entries.index(view) if view in entries else None

    def _resource_list_rows(self, source: str | None, now: datetime) -> list[tuple[tuple, Content, bool]]:
        """Resource panel rows as ``(entry, label, disabled)``."""
        kinds = self._resource_data.kinds.get(source or "", {})
        # A kind missing now has no rows: once the sub-job has some, or once a download
        # found none for a sub-job that isn't still to send its first samples.
        known = bool(kinds) or (self._resource_downloaded and self._resource_waiting() is None
                                and not self._resource_first_samples_due(gpu_allocation(self._job, source)))
        heads = []
        for kind, name in self._RESOURCE_VIEWS.items():
            none = known and kind not in kinds
            shown = kind == self._resource_kind
            text = f"{'●' if shown else '○'} {name}{' (none)' if none else ''}"
            heads.append((("view", kind), text, "dim" if none else "bold" if shown else ""))
        resource = kinds.get(self._resource_kind)
        units: list[tuple[tuple, str, str, str]] = []
        hue = ""
        if resource is not None:
            metric = headline_metric(resource)
            stale = self._resource_stale(resource, now)
            metrics = list(resource.metrics)
            hue = chart_hue(metrics.index(metric)) if metric in metrics else ""

            def value(latest) -> str:
                if metric is None:
                    return ""
                found = latest.get(metric.key)
                return f"{level_glyph(found, metric.scale)} {format_value(found, metric.unit)}"

            units.append((("unit", None), "ALL", value(resource.latest), "dim" if stale else ""))
            for unit in resource.units:
                dim = stale or not unit.reporting
                units.append((("unit", unit.id), unit.name + (" (stale)" if dim else ""), value(unit.latest),
                              "dim" if dim else ""))
        value_width = max((views.cell_width(value) for *_head, value, _style in units), default=0)
        inner = max([views.cell_width(text) for _entry, text, _style in heads]
                    + [views.cell_width(name) + 2 + value_width for _entry, name, _value, _style in units])
        rows = [(entry, Content.styled(text, style) if style else Content(text), False)
                for entry, text, style in heads]
        if units:
            rows.append((("rule",), Content("─" * inner), True))
        for entry, name, value, style in units:
            gap = " " * (inner - views.cell_width(name) - views.cell_width(value))
            if style:  # dimmed: a unit that stopped reporting
                rows.append((entry, Content.styled(name + gap + value, style), False))
            else:
                # The bar in the chart's colour; the value plain, readable under the highlight too.
                bar, _space, number = value.partition(" ")
                rows.append((entry, Content.assemble(name, gap, (bar, hue) if hue else bar, _space + number), False))
        return rows

    def _sync_resource_list(self, source: str | None, now: datetime) -> None:
        """Show the panel's entries in its rows, and put its highlight on the entry shown."""
        try:
            panel = self.query_one("#resource-list", ListView)
        except NoMatches:
            return
        rows = self._resource_list_rows(source, now)
        kept = self._resource_rows
        for position, (item, label) in enumerate(kept):
            if position < len(rows):
                _entry, content, disabled = rows[position]
                label.update(content)
                if item.disabled != disabled:
                    item.disabled = disabled
                if not item.display:
                    item.display = True
            else:
                if not item.disabled:
                    item.disabled = True  # passed over by the arrow keys
                if item.display:
                    item.display = False
        for _entry, content, disabled in rows[len(kept):]:
            label = Label(content)
            item = ListItem(label, disabled=disabled)
            kept.append((item, label))
            panel.append(item)
        self._resource_entries = [entry for entry, _content, _disabled in rows]
        target = self._resource_cursor()
        if target is not None and panel.index != target:
            self._resource_placing = True
            try:
                panel.index = target
            finally:
                self._resource_placing = False
        self._apply_resource_list_width(max(content.cell_length for _entry, content, _disabled in rows))

    def _apply_resource_list_width(self, longest: int) -> None:
        """Size the panel as the Logs tab's sub-job list is sized: to its labels, within the same bounds."""
        width = min(self._SOURCES_AUTO_MAX, max(self._SOURCES_AUTO_MIN, longest + 4))
        width = max(self._SOURCES_MIN, min(width, self._sources_cap()))
        if width == self._resource_list_width:
            return
        self._resource_list_width = width
        try:
            self.query_one("#resource-list", ListView).styles.width = width
        except NoMatches:
            return
        after_refresh(self, self._render_resource)  # the pane beside it changed width: draw to it

    _RESOURCE_REFRESHING = ("cached · refreshing…", "refreshing…")

    def _resource_state(self, resource: KindResource | None) -> tuple[str, ...]:
        """Refresh-state forms, fullest first, or empty when idle."""
        source = self._resource_source() or ""
        cached = bool(self._resource_data.kinds.get(source)) or source in self._resource_data.ignored_rows_by_sub_job
        what = self._RESOURCE_WHAT[self._resource_kind]
        if self._resource_missing(resource) == LOADING_OLDER:
            return (LOADING_OLDER,)
        if resource is not None and resource.partial:
            if self._resource_error:
                return ("[$warning]older samples failed to load: $problem[/]",
                        "[$warning]older samples failed: $problem[/]")
            return (f"[$warning]{OLDER_NOT_LOADED}[/]",)
        if self._resource_loading:
            if cached:
                return self._RESOURCE_REFRESHING
            # A sub-job with nothing of its own after a whole download has none to load.
            return ("refreshing…",) if self._resource_downloaded else (f"loading {what}…",)
        if self._resource_error:
            return (("cached · [$warning]refresh failed: $problem[/]", "[$warning]refresh failed: $problem[/]")
                    if cached else (f"[$warning]{what} unavailable: $problem[/]",))
        return ()

    def _resource_missing(self, resource: KindResource | None) -> str | None:
        """What stands in for the stats that need the shown sub-job's older samples; None once they are read."""
        if resource is None or not resource.partial:
            return None
        return LOADING_OLDER if self._resource_loading else OLDER_NOT_LOADED

    def _resource_waiting(self) -> str | None:
        """The status word while the shown sub-job (or its job) has not started running, else None."""
        source = self._resource_source()
        word = self._tail_status(source) if source else ""
        return word if word in self._PRE_PLACEMENT | self._STARTING else None

    def _resource_first_samples_due(self, allocation) -> bool:
        """Whether a running, GPU-owning sub-job still owes its first samples."""
        source = self._resource_source()
        if not source or self._tail_status(source) != "running":
            return False
        if allocation.log_prob or allocation.shared_with or allocation.requested == 0:
            return False
        return not self._resource_data.kinds.get(source)

    def _resource_stale(self, resource: KindResource | None, now: datetime) -> bool:
        """Whether a running sub-job has sent no samples of the view's kind for well past their usual delay."""
        source = self._resource_source()
        if resource is None or not source or self._tail_status(source) != "running":
            return False
        interval = resource.sample_interval_s or resource.reported_interval_s or 0
        age = (now - resource.latest_sample_ts).total_seconds()
        return age > max(self._RESOURCE_STALE_SECONDS, 6 * interval)

    def _resource_empty_text(self, allocation, ignored: int) -> str:
        """Why the view shown has nothing to chart, once the telemetry has loaded."""
        kind = self._resource_kind
        source = self._resource_source()
        if kind == GPU and ignored:
            return "no usable GPU metrics for this sub-job"
        waiting = self._resource_waiting()
        if waiting is None and self._resource_first_samples_due(allocation):
            waiting = "running"  # only said: unlike a sub-job still placing, this one is polled
        job_waiting = views.status_word(self._job_status) in self._PRE_PLACEMENT | self._STARTING
        who = "the job" if job_waiting else views.sub_job_label(source) if source else "it"
        return no_metrics_text(allocation, waiting=waiting, who=who, kind=kind)

    def _resource_status(self, resource: KindResource | None, width: int = 0) -> Content:
        """Two fitted header lines for scope, sampling, freshness, and refresh state."""
        kind = self._resource_kind
        source = self._resource_source() or ""
        ignored = self._resource_data.ignored_rows_by_sub_job.get(source, 0)
        states = self._resource_state(resource)
        allocation = gpu_allocation(self._job, source)
        now = datetime.now(timezone.utc)
        stale = self._resource_stale(resource, now)
        values: dict = {"problem": views.collapse(self._resource_error or "")}
        if ignored:
            values.update(note=skipped_rows_text(ignored), note_short=skipped_rows_text(ignored, short=True))
        note = [("$note", 0), ("$note_short", 0), ("", 0)] if ignored else [("", 0)]
        state = [(form, 0) for form in states]
        # Which sub-job this is, by name (the job's id is on the line above), and where it is among them.
        ids = self._resource_sub_job_ids()
        values["label"] = views.sub_job_label(source) if source else ""
        # In the colour of its dot in the Logs tab's list and its tag on the log lines.
        named = f"[b {sub_job_color(values['label'])}]$label[/]"
        if source in ids and len(ids) > 1:
            values.update(position=f"{ids.index(source) + 1} of {len(ids)}", pos=f"{ids.index(source) + 1}/{len(ids)}")
            # A click on it shows the next sub-job, as space does.
            source_forms = [(f"[@click=screen.resource_sub_job(1)]{form}[/]", 0)
                            for form in (f"{named} ($position)", f"{named} ($pos)", named)]
        else:
            source_forms = [(named, 0)]
        if resource is None:
            if kind == GPU:
                values.update(allocation=allocation_text(allocation),
                              allocation_short=allocation_text(allocation, hardware=False))
                first = _fit_line([source_forms, [("$allocation", 0), ("$allocation_short", 0)]], [0, 0, 1], width,
                                  values)
            else:
                first = _fit_line([source_forms], [0, 0], width, values)
            cached = bool(self._resource_data.kinds.get(source))
            if self._resource_loaded and (self._resource_downloaded or cached or not states):
                # A message, not the charts: it may wrap. A refresh's state follows it.
                values["empty"] = self._resource_empty_text(allocation, ignored)
                second = _fit_line([[("$empty", 0)], state or [("", 0)], note], [], 0, values)
            elif states:
                # Nothing to chart yet: the state (an error, say) is the message, whole; it may wrap.
                second = _fit_line([state, note], [], 0, values)
            else:
                second = Content("")
            return Content("\n").join([first, second])
        # A running sub-job that stopped sending samples has no GPU or node reporting now.
        reporting = 0 if stale else len(resource.reporting_unit_ids)
        if kind == GPU:
            values.update(allocation=allocation_text(allocation, reporting, len(resource.units)),
                          allocation_short=allocation_text(allocation, reporting, len(resource.units), hardware=False))
            allocation_forms = [("$allocation", 0), ("$allocation_short", 0)]
        else:
            values.update(nodes=node_allocation_text(resource, reporting),
                          nodes_short=node_allocation_text(resource, reporting, sizes=False))
            allocation_forms = [("$nodes", 0), ("$nodes_short", 0)]
        order = [0, 1, 1, 0, 2]
        samples = self._RESOURCE_SAMPLES[kind]
        values.update(
            window=window_label(self._resource_range),
            key=self._resource_range,  # "10m", "all": the window keys' own names
            scraped=scrape_text(resource),
            scraped_short=scrape_text(resource, notes=False),
            fresh=freshness_text(resource, now, stale=stale, what=samples),
            fresh_short=freshness_text(resource, now, stale=stale, clock=False, what=samples),
        )
        # The window goes before the place: the stats and the chart titles name it too.
        # A click on it shows the next window, as w does.
        window_forms = [("[@click=screen.cycle_window]$window[/]", 0), ("[@click=screen.cycle_window]$key[/]", 0),
                        ("", 0)]
        first = _fit_line([source_forms, window_forms, allocation_forms], order, width, values)
        grows = age_slack(resource, now)
        warn = "[$warning]{}[/]" if stale else "{}"
        fresh = [(warn.format("$fresh"), grows), (warn.format("$fresh_short"), grows)]
        if not states:
            # Nothing shows, but the room a refresh's state needs stays free.
            state = [("", len(_SEPARATOR) + len(text)) for text in self._RESOURCE_REFRESHING]
        scraped = [("$scraped", 0), ("$scraped_short", 0), ("", 0)]
        # Parts: 0 scraped, 1 freshness, 2 state, 3 skipped rows.
        second = _fit_line([scraped, fresh, state, note], [3, 0, 1, 2, 3, 0], width, values, cut="problem")
        return Content("\n").join([first, second])

    def _tick_resource_status(self) -> None:
        # Keeps "newest N s ago" current between refreshes, without redrawing the rest
        # (unless the newest sample just became stale, which marks the units too).
        if not self._screen_open or not self._resource_active() or self.app.screen is not self:
            return
        resource = self._resource_kind_data()
        if resource is not None:
            if self._resource_stale(resource, datetime.now(timezone.utc)) != self._resource_stale_shown:
                self._render_resource()
                return
            status = self.query_one("#resource-status", Static)
            status.update(self._resource_status(resource, status.content_size.width))

    def _resource_charts(self, resource: KindResource, view: KindView, width: int, complete: bool,
                         missing: str | None) -> Content:
        """Every metric's chart, in the metrics' order: two to a row in a pane at least 90 cells wide, titled as
        docs/reference/cli.md says; where one doesn't fit, every title leaves out what the values are, then the
        window, and a name that still doesn't fit ends in ``…`` (so the charts of a row stay level). A window
        whose older samples aren't read yet says so after each title, or on a line of its own under each."""
        metrics = list(resource.metrics)
        if not metrics:
            return Content("")
        per_row = 2 if width >= self._RESOURCE_SIDE_BY_SIDE else 1
        gap = self._RESOURCE_CHART_GAP
        column = max(1, (width - gap * (per_row - 1)) // per_row)
        bars = next(n for rows, n in self._RESOURCE_BAR_ROWS_BY_HEIGHT if self.app.size.height >= rows)
        hues = [chart_hue(index) for index in range(len(metrics))]
        # Each title ends with the metric's latest value, at the chart's right edge; the
        # rest of the title gets the room before it.
        latest = []
        for metric in metrics:
            stats = view.stats.get(metric.key)
            latest.append(format_value(stats.latest, metric.unit) if stats is not None else "")
        room = max(1, column - max((views.cell_width(value) for value in latest), default=0) - 2)
        # The header, pinned above them, names the sub-job: the titles needn't.
        suffix = f" · {window_label(view.name)}" + (f" · {view.label}" if view.label else "")
        tails = [suffix] * len(metrics)
        if any(views.cell_width(metric.name + suffix) > room for metric in metrics):
            tails = [f" · {window_label(view.name)}"] * len(metrics)
        if any(views.cell_width(metric.name + tail) > room for metric, tail in zip(metrics, tails)):
            tails = [""] * len(metrics)
        notes: list[str | None] = [None] * len(metrics)
        if not complete:
            loading = missing == LOADING_OLDER
            marker = " (loading…)" if loading else f" ({OLDER_NOT_LOADED})"
            if all(views.cell_width(metric.name + tail + marker) <= room for metric, tail in zip(metrics, tails)):
                tails = [tail + marker for tail in tails]
            else:  # on a line of its own under every title, so the charts of a row stay level
                note = views.fit_cells("(older samples loading…)" if loading else f"({OLDER_NOT_LOADED})", column)
                notes = [note] * len(metrics)
        blocks = []
        for metric, hue, tail, value, note in zip(metrics, hues, tails, latest, notes):
            name = views.fit_cells(metric.name, room)
            tail = views.fit_cells(tail, max(0, room - views.cell_width(name)))
            used = views.cell_width(name) + views.cell_width(tail)
            head = [Content.assemble((name, f"bold {hue}"), (tail, CHART_MUTED),
                                     " " * max(1, column - used - views.cell_width(value)), (value, f"bold {hue}"))]
            if note is not None:
                head.append(Content.styled(note, CHART_MUTED))
            if has_metric_samples(view, metric.key):
                body = [spans_content(line) for line in metric_chart_spans(view, metric, column, height=bars,
                                                                              hue=hue)]
            else:
                body = [Content.styled(views.fit_cells(f"no {metric.name} samples in this range", column),
                                       CHART_MUTED)]
            blocks.append(head + body)
        height = max(len(block) for block in blocks)
        rows = []
        for start in range(0, len(blocks), per_row):
            group = blocks[start:start + per_row]
            lines = []
            for index in range(height):
                cells = []
                for block in group:
                    line = block[index] if index < len(block) else Content("")
                    if line.cell_length > column:
                        line = line.truncate(column)
                    cells.append(line.pad_right(column - line.cell_length))
                lines.append(Content(" " * gap).join(cells).rstrip())
            rows.append(Content("\n").join(lines))
        return Content("\n\n").join(rows)

    def _render_resource(self, *, retry_layout: bool = True) -> None:
        if not self._screen_open or not self._resource_active():
            return
        try:
            status = self.query_one("#resource-status", Static)
            stats = self.query_one("#resource-stats", Static)
            charts = self.query_one("#resource-charts", Static)
        except NoMatches:
            return
        source = self._resource_source()
        now = datetime.now(timezone.utc)
        self._follow_resource_source(source)
        resource = self._resource_kind_data() if source else None
        drawn = (source, self._resource_kind, self._resource_unit, self._resource_range)
        if drawn != self._resource_drawn:
            # Another sub-job, view, GPU or node, or window: a highlight would sit over
            # other text, or past the end of a shorter block.
            self._resource_drawn = drawn
            self.clear_selection()
        if self._resource_unit is not None and (resource is None or self._resource_unit not in resource.unit_ids):
            self._resource_unit = None  # not one of this view's units (any more)
        self._sync_resource_list(source, now)
        if not source and self._no_sources:
            # Nothing to load or lay out: say why at once (the pane may not have a width yet).
            status.update(self._resource_no_source_text())
        pane_width = stats.content_size.width
        if pane_width <= 0:
            if retry_layout:
                self.call_after_refresh(self._render_resource, retry_layout=False)
            return
        if resource is None:
            status.update(self._resource_status(None, pane_width) if source else self._resource_no_source_text())
            stats.update("")
            charts.update("")
            self._resource_stale_shown = False
            return
        if self._resource_ticker is None:
            self._resource_ticker = self.set_interval(self._RESOURCE_TICK_SECONDS, self._tick_resource_status)
        unit = self._resource_unit
        view = resource.view(self._resource_range, unit) or resource.view(self._resource_range)
        whole = resource.view("all", unit) or resource.view("all")
        if view is None:
            return
        complete = window_complete(resource, view)
        missing = self._resource_missing(resource)  # the older samples: loading, or not loaded
        self._resource_stale_shown = self._resource_stale(resource, now)
        status.update(self._resource_status(resource, pane_width))
        stats.update(Content("\n").join(spans_content(line) for line in metric_stats_spans(
            view, whole, resource.metrics, width=pane_width, partial=resource.partial, window_partial=not complete,
            missing=missing or LOADING_OLDER)))
        charts.update(self._resource_charts(resource, view, pane_width, complete, missing))

    def _resource_no_source_text(self) -> Content | str:
        """The Resources header when no sub-job is selected: why not."""
        if self._sub_job_missing and self._sub_job_id:
            return Content.from_markup("sub-job not found: $sub_job", sub_job=self._sub_job_id)
        if not self._sources_ready:
            return "loading log sources…"
        if self._source_by_item:
            return "select a log source first"
        job = self._job
        if views.is_unknown(job):
            # The service can't say which sub-jobs the job has while its status is unavailable.
            return Content.from_markup("$why; no GPU metrics are cached on this machine",
                                       why=views.status_unavailable_text(job))
        if self._no_sources and self._no_sources != self._NO_SUB_JOBS:
            # The job couldn't be read, or the service answered nothing: the reason the Logs pane gives.
            return Content(safe_text(f"no sub-jobs to show GPU metrics for: {self._no_sources}"))
        return "this job has no sub-jobs, and no GPU metrics are cached on this machine"

    # ─── filter / level / pause ──────────────────────────────────────────
    def check_action(self, action: str, parameters: tuple) -> bool | None:
        if action == "filter" and isinstance(self.focused, Input):
            return False  # in a text box, / is typed
        if action in self._LOG_ONLY:
            return self._logs_active()
        if action == "resource_sub_job":
            # Only where it does something: on Resources, with another sub-job to switch to.
            return self._resource_active() and len(self._resource_sub_job_ids()) > 1
        if action == "cycle_window":
            return self._resource_active()
        return True

    def action_filter(self) -> None:
        if not self._logs_active():
            return
        self.set_focus(self.query_one("#logfilter", Input))

    def action_back(self) -> None:
        if not is_shown(self):
            return  # still being put up, or no longer on top (see is_shown)
        # Opened straight from the command line, nothing is underneath: show the
        # job list rather than an empty screen.
        if len(self.app.screen_stack) > 2:
            self.app.pop_screen()
        else:
            # The app's list (its own client, not this job's, and its account-list state).
            self.app.switch_screen(self.app.job_list_screen())

    def action_dismiss_or_back(self) -> None:
        if not is_shown(self) or close_keys_panel(self):
            return
        if not self._logs_active():
            self.action_back()
            return
        # Esc clears an active filter (and unfocuses the input); otherwise backs out.
        inp = self.query_one("#logfilter", Input)
        if self.focused is inp or self._filter:
            inp.value = ""  # fires on_input_changed → clears the filter
            self.set_focus(self.query_one("#sources", ListView))
        else:
            self.action_back()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "logfilter" and self._screen_open:
            self._filter = event.value.strip()
            self._update_subtitle()
            # Repaint once typing pauses. The filter applies to the lines already
            # here, so a key press never restarts the tail or sends a request.
            if self._filter_timer is not None:
                self._filter_timer.stop()
            self._filter_timer = self.set_timer(self._FILTER_DEBOUNCE_SECONDS, self._refilter)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "logfilter" and self._screen_open:
            self.set_focus(self.query_one("#sources", ListView))

    def action_cycle_level(self) -> None:
        if not self._logs_active():
            return
        i = self._LEVEL_CYCLE.index(self._min_level) if self._min_level in self._LEVEL_CYCLE else 0
        self._min_level = self._LEVEL_CYCLE[(i + 1) % len(self._LEVEL_CYCLE)]
        self._update_subtitle()
        self._refilter()

    def action_toggle_pause(self) -> None:
        if not self._logs_active():
            return
        if self._idle_paused:
            # p resumes a tail paused for lack of input (it picks up from its
            # cursor); auto-scroll stays as it was.
            self._idle_paused = False
            self._resumed_at = time.monotonic()
            self._render_log_status()
            return
        self._paused = not self._paused
        # Resuming jumps to the end and follows again; pausing just stops
        # following (the user can scroll freely). A saved console that arrived
        # while paused replaces the pane now.
        if not self._paused:
            held = self._held
            self._held = None
            if held is not None:
                lines, empty_note = held
                if lines:
                    # Wrapped off the UI thread (see _repaint): a console can be 20,000 lines.
                    self._repaint([(None, line) for line in lines])
                else:
                    self._clear_pane()
                    if empty_note:
                        self._write_line(self._no_log_output())
                self._render_log_status()
            if self._logview is not None:
                self._logview.scroll_end(animate=False)
        self._update_subtitle()

    def _refilter(self) -> None:
        """Apply the filter and level to what this view already has; never re-tail.

        A saved console is re-read from its local file; a live or cached tail is
        repainted from the lines it has received. Either way the work is done
        off the UI thread, and it lands on the newest lines.
        """
        if not self._current_source:
            return
        highlighter = getattr(self._logview, "highlighter", None)
        if isinstance(highlighter, LogHighlighter):
            highlighter.query = self._filter  # the repaint below marks the new filter's matches
        self._held = None
        self._paused = False
        self._update_subtitle()
        if self._saved_shown(self._current_source):
            self._paint_saved()
        else:
            # Older lines may have left the buffer before this repaint.
            self._repaint(list(self._entries), self._filter, self._min_level, truncated=bool(self._entries_dropped))
        if self._logview is not None:
            self._logview.scroll_end(animate=False)
        self._render_log_status()

    # ─── export / copy ───────────────────────────────────────────────────
    # The most a copy of the whole log (ctrl+c, nothing highlighted) takes. OSC 52 carries the text
    # base64-encoded in one escape sequence; many terminals drop a much bigger
    # one without a word, and writing it holds the screen up meanwhile.
    _COPY_MAX_BYTES = 1024 * 1024

    def _export_target(self) -> tuple[str, str, list[str]] | None:
        """UI thread. ``(source, path, shown)``: ``path`` is the saved console
        (empty: export the tail cache); ``shown`` is the execution log the pane
        shows for a sub-job without a console, the export when the cache is empty."""
        src = self._current_source
        if not src:
            return None
        if self._saved_shown(src):
            path = self._stdout_path(src)
            try:
                if path.is_file() and path.stat().st_size > 0:
                    return src, str(path), []
            except OSError:
                pass
        notes, records = self._exec_logs.get(src, ([], []))
        shown = [*notes, *(line for _, lines in records for line in lines)] if records else []
        return src, "", shown

    def _exported_text(self, src: str, shown: list[str]) -> str:
        entries = self._cache.load_entries(src)
        if not entries and shown:
            return "\n".join(shown)
        return "\n".join(format_log_entry(entry) for entry in entries)

    def _export_name(self, src: str) -> str:
        """``~/cortex-training-<job8>-<type>-<index>.log`` for sub-job ``<job>:<type>:<index>``."""
        job = str(self._job_id)
        rest = src[len(job) + 1:] if src.startswith(f"{job}:") else src
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in rest).strip(".-") or "log"
        return os.path.expanduser(f"~/cortex-training-{job[:8]}-{safe}.log")

    _EXPORT_NOTE_SECONDS = 15  # how long what s did stays on the status line

    def _show_export_note(self, text: str) -> None:
        """UI thread. What s did, on the status line (not in the log), for a while."""
        self._export_note = text
        self._render_log_status()
        if self._export_timer is not None:
            self._export_timer.stop()
        self._export_timer = self.set_timer(self._EXPORT_NOTE_SECONDS, self._clear_export_note)

    def _clear_export_note(self) -> None:
        self._export_timer = None
        if self._export_note is not None:
            self._export_note = None
            self._render_log_status()

    def action_save_log(self) -> None:
        if not self._logs_active():
            return
        target = self._export_target()  # what the pane shows now, for the worker
        if target is not None:
            self._save_log(*target)

    @work(thread=True, exclusive=True, group="export")
    def _save_log(self, src: str, file_path: str, shown: list[str]) -> None:
        """Worker thread. Save what :meth:`_export_target` named: the saved console at
        ``file_path`` copied as it is, never read into memory (it can be gigabytes), else
        the lines the tail cached, else ``shown``."""
        path = self._export_name(src)
        try:
            if file_path:
                if not os.path.getsize(file_path):  # emptied since: as for an empty log, below
                    self._post(self._show_export_note, "nothing to save yet for this sub-job")
                    return
                shutil.copyfile(file_path, path)
                count = self._count_lines(path)
            else:
                text = self._exported_text(src, shown)
                if not text:
                    # Nothing yet: never write (or overwrite an earlier save with) an empty file.
                    self._post(self._show_export_note, "nothing to save yet for this sub-job")
                    return
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(text + "\n")
                count = len(text.splitlines())
            msg = f"saved {self._count(count, 'line')} to {path}"
        except OSError as exc:  # noqa: BLE001
            msg = f"save failed: {exc}"
        self._post(self._show_export_note, msg)

    # What ends a line for str.splitlines(), besides \r\n.
    _LINE_BREAKS = ("\n", "\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\N{LINE SEPARATOR}",
                    "\N{PARAGRAPH SEPARATOR}")

    @classmethod
    def _count_lines(cls, path: str) -> int:
        """Worker thread. ``len(text.splitlines())`` for the file at ``path`` decoded as the
        pane decodes it, read a block at a time rather than whole (a console can be
        gigabytes). ``newline=""`` leaves each line break as it is."""
        count, last = 0, ""
        with open(path, encoding="utf-8", errors="replace", newline="") as handle:
            while block := handle.read(1 << 20):
                count += sum(block.count(mark) for mark in cls._LINE_BREAKS) - block.count("\r\n")
                if last == "\r" and block.startswith("\n"):
                    count -= 1  # one \r\n, split between two blocks
                last = block[-1]
        return count + (last != "" and last not in cls._LINE_BREAKS)  # a last line without a break

    @work(thread=True, exclusive=True, group="copy")
    def _copy_log(self) -> None:
        try:
            target = self.app.call_from_thread(self._export_target)
        except RuntimeError:
            return
        if target is None:
            return
        src, path, shown = target
        if path:
            tail = read_saved_tail(
                Path(path),
                max_lines=10**9,
                max_bytes=self._COPY_MAX_BYTES,
                max_line_bytes=self._COPY_MAX_BYTES,
            )
            text, truncated = ("\n".join(tail.lines), tail.truncated) if tail is not None else ("", False)
        else:
            text, truncated = self._exported_text(src, shown), False
        if not text:
            self._post(self.notify, "nothing to copy yet for this sub-job", timeout=3)
            return
        # Without escape sequences or control characters: pasted into another
        # terminal, those would act there. The newest lines, up to the cap.
        text = safe_text(text).removesuffix("\n")
        raw = text.encode("utf-8")
        if len(raw) > self._COPY_MAX_BYTES:
            truncated = True
            raw = raw[-self._COPY_MAX_BYTES:]
            newline = raw.find(b"\n")
            if newline >= 0:
                raw = raw[newline + 1:]
                prefix = ""
            else:
                raw = raw[3:]  # room for the ellipsis below
                while raw and raw[0] & 0xC0 == 0x80:
                    raw = raw[1:]
                prefix = "…"
            text = prefix + raw.decode("utf-8")
        lines = text.split("\n")
        what = (f"the log's last {self._count(len(lines), 'line')} (s saves them all)"
                if truncated else f"the whole log ({self._count(len(lines), 'line')})")
        self._post(self.app.copy_out, text, what)

    def action_copy_text(self) -> None:
        """ctrl+c, on every tab: what the mouse highlighted, else
        everything the tab holds: the log (its newest lines, up to the cap) on Logs; on
        Resources the panel, the header, every stats row and every chart."""
        if self.app.copy_highlight(self):
            return
        if self.query(HelpPanel):
            self.app.copy_everything(self)  # the key list, with the tab beside it
        elif self._logs_active():
            self._copy_log()
        else:
            self.app.copy_everything(self, "the Resources tab")

    def _load_sources(self) -> None:
        """UI thread. Read the job, then show its sub-jobs (see _read_sources)."""
        # The row this screen was opened with (the list's or the preview's copy), as it is now.
        self._read_sources(dict(self._job))

    @work(thread=True, exclusive=True, group="sources")
    def _read_sources(self, row: dict) -> None:
        worker = get_current_worker()
        # Fetch the job once (GS metadata — survives after the zone is gone) for
        # the summary header, status (drives follow/cache), and the sub-job list.
        failed: Exception | None = None
        self._post(self._job_read_began)
        seq = self._next_job_read()
        try:
            job = self._client.get_job(self._job_id) or {}
        except Exception as exc:  # noqa: BLE001 - best-effort; said in the pane when nothing else can show
            job, failed = {}, exc
            self._log_traceback(f"reading job {self._job_id} failed")
        finally:
            self._post(self._job_read_ended)
        if not isinstance(job, dict):
            job = {}
        # An answer without a status or sub-jobs ({}) says no more than a failed call.
        answered = bool(job.get("status") or job.get("sub_jobs"))
        if answered:
            # Status, reason and times are the UI thread's to take, in the order the reads
            # were sent (see _take_job).
            self._post(self._take_job, job, seq)
        # get_job failed or answered nothing: the row this screen was opened with still
        # names the sub-jobs, or says the status is unavailable.
        listed = job if answered else row

        # Each sub-job is a tailable source (its zone-manager / Ray-head pod,
        # whose stdout already includes worker output via Ray log_to_driver).
        sources = []
        for sj in views.sub_jobs(listed):  # an UNKNOWN row's ids are derived from its configs
            if isinstance(sj, dict):
                sid = sj.get("sub_job_id") or sj.get("id")
                if sid:
                    sources.append({"sub_job_id": sid, "job_type": sj.get("job_type") or sj.get("type"),
                                    "status": sj.get("status")})
        # Training first: the service may list sampling first, and training is where
        # a job's progress (and most of its GPUs) show.
        sources.sort(key=lambda s: str(s.get("job_type") or "").lower() != "training")
        if views.is_unknown(listed) and self._sub_job_id and all(s["sub_job_id"] != self._sub_job_id for s in sources):
            # While its status is unavailable the service can't say which sub-jobs exist: try the one asked for.
            sources.append({"sub_job_id": self._sub_job_id})
        if not sources:
            # get_job failed/empty (e.g. offline): fall back to cached sub-jobs.
            source_ids = self._cache.cached_sources()
            source_ids.extend(cached_metric_sources(self._metrics_root))
            sources = [{"sub_job_id": sid} for sid in dict.fromkeys(source_ids)]
        why = None if sources else self._no_sources_text(listed, failed, answered)
        if worker.is_cancelled:
            return
        self._post(self._populate_sources, sources, why)

    def _no_sources_text(self, listed: dict, error: Exception | None, answered: bool) -> tuple[str, str]:
        """Worker thread. Why there is nothing to show (no sub-jobs, and no cached logs or
        metrics): the pane line's tag, ``[error]`` when the read itself failed, and the reason."""
        if views.is_unknown(listed):
            return "[no logs]", f"{views.status_unavailable_text(listed)}; no logs are cached on this machine"
        if getattr(getattr(error, "response", None), "status_code", None) == 404:
            # The job isn't in this client's database and schema: with the account-wide
            # job list, a job id copied from it may well be in another one.
            where = self._configured_location()
            place = ".".join((where["database"], where["schema"])) if where["database"] and where["schema"] else (
                "the configured database and schema")
            return "[no logs]", (
                f"job {self._job_id} was not found in {place} ({views.describe_error(error)}), and no logs are "
                "cached on this machine. If the job list shows it in another location, start the TUI with that "
                "--database and --schema.")
        if error is not None:
            # The read failed (an outage, a timeout): the job may well have sub-jobs and logs.
            bounds = self._read_bounds()
            tries = f" ({bounds[0]} tries)" if bounds and bounds[0] > 1 else ""
            return "[error]", (f"couldn't read job {self._job_id}: {views.describe_error(error)}{tries}; no logs "
                               f"are cached on this machine  (r tries again · details in {_ERROR_LOG})")
        if not answered:
            return "[no logs]", ("the service sent no details for this job (an empty answer), so its sub-jobs "
                                 "aren't known, and no logs are cached on this machine; r asks again")
        return "[no logs]", self._NO_SUB_JOBS

    _NO_SUB_JOBS = "this job has no sub-jobs and no cached logs are available"

    def _read_bounds(self) -> tuple[int, float] | None:
        """(tries, read timeout in seconds) of each request this screen's client sends,
        when the client says (the TUI's own: 3 tries of up to 45 s each)."""
        retries = getattr(self._client, "max_retries", None)
        timeout = getattr(self._client, "request_timeout", None)
        read = timeout[-1] if isinstance(timeout, tuple) and timeout else timeout
        if type(retries) is int and isinstance(read, (int, float)) and not isinstance(read, bool) and read > 0:
            return retries + 1, float(read)
        return None

    def _job_read_began(self) -> None:
        """UI thread. The job read that names the sources started: show that the pane waits for it."""
        self._job_read_started = time.monotonic()
        if not self._shown_lines:  # nothing on screen yet: say what it is waiting for
            self._write_line(f"— reading job {self._job_id} —")
        if self._job_read_ticker is None:
            self._job_read_ticker = self.set_interval(1.0, self._render_log_status)
        else:
            self._job_read_ticker.resume()

    def _job_read_ended(self) -> None:
        """UI thread. The job read answered, or gave up."""
        self._job_read_started = None
        if self._job_read_ticker is not None:
            self._job_read_ticker.pause()
        self._render_log_status()

    # A read that answers within this long shows nothing on the status line.
    _JOB_READ_QUIET_SECONDS = 2.0

    def _job_read_note(self) -> str | None:
        """The status line's note while the job read is slow: how long it has waited,
        and how long it may still (a stalled service is asked again after each timeout)."""
        if self._job_read_started is None:
            return None
        waited = time.monotonic() - self._job_read_started
        if waited < self._JOB_READ_QUIET_SECONDS:
            return None
        note = f"reading the job: no answer yet after {int(waited)} s"
        bounds = self._read_bounds()
        if bounds:
            tries, read = bounds
            note += f" (up to {tries} tries of {read:g} s each)" if tries > 1 else f" (up to {read:g} s)"
        return note

    async def _populate_sources(self, sources, why: tuple[str, str] | None = None) -> None:
        """UI thread. ``why``: the pane line's tag and the reason there are no sources
        (the job couldn't be read, say); see :meth:`_no_sources_text`."""
        lv = self.query_one("#sources", ListView)
        await lv.clear()  # await removal before re-appending src-* ids (see _populate)
        self._source_by_item = {}
        self._sources_ready = True
        self._sub_job_missing = False
        self._no_sources = why[1] if why is not None else None
        items = [s for s in sources if s.get("sub_job_id")]
        if not self._job_seq:
            # No get_job answer taken yet: the row the view was opened with says how the
            # sub-jobs are. Once one is, _take_job keeps them as the newest answer says.
            self._sub_job_status = {s["sub_job_id"]: s["status"] for s in items if s.get("status")}
        for source in self._exec_failed:  # r retries an execution log that couldn't be read
            self._exec_logs.pop(source, None)
        self._exec_failed.clear()
        if why is not None:
            self._clear_pane()
            self._write_line(" ".join(why))
        if self._sub_job_id:
            known = [s["sub_job_id"] for s in items]
            items = [s for s in items if s["sub_job_id"] == self._sub_job_id]
            if not items:
                self._current_source = None
                if why is None:
                    self._sub_job_missing = True
                    lv.append(ListItem(Label(Content.from_markup("error: sub-job not found: $sub_job",
                                                                 sub_job=self._sub_job_id))))
                    # The sources column is narrow: say it in full in the log pane too, with the ids there are.
                    self._clear_pane()
                    ids = ", ".join(known) if known else "none it can name"
                    self._write_line(views.collapse(f"error: sub-job not found: {self._sub_job_id}; this job's "
                                                    f"sub-jobs: {ids} (--sub-job-id takes one of them)"))
                if self._resource_active():
                    self._render_resource()
                return
        for i, s in enumerate(items):
            sid = s["sub_job_id"]
            item_id = f"src-{i}"
            self._source_by_item[item_id] = sid
            label = views.sub_job_label(s)
            # A dot in the colour of the sub-job's tag on its log lines; the name stays
            # plain, readable under the highlight.
            lv.append(ListItem(Label(Content.assemble(("● ", sub_job_color(label)), label)), id=item_id))
        self._apply_sources_width()  # fitted to these labels
        self.refresh_bindings()  # < and > need two or more sub-jobs
        if not items:
            self._current_source = None
        if items:
            # A refresh (r) stays on the source the user picked.
            keep = next((i for i, s in enumerate(items) if s["sub_job_id"] == self._current_source), 0)
            lv.index = keep
            self._start_tail(items[keep]["sub_job_id"])
        if self._resource_active():
            self._render_resource()
            self._start_resource_refresh(force=False)

    def _display(self, line: str) -> str:
        """How a buffered line is shown: a log line's display text (see
        ``format.display_line``), the TUI's own notes only made safe to draw.
        The buffer (and save and copy) keep the line as it came."""
        if isinstance(line, self._Note):
            return safe_text(line)  # never cached: the cache is by text, and a log line may read the same
        text = self._display_cache.get(line)
        if text is None:
            text = self._display_cache[line] = display_line(line)
        return text

    def _line_prefixes(self) -> tuple[str, str]:
        # Tag every log line with the sub-job being tailed, and line its later
        # rows (wrapped, or a traceback's lines) up under its text; the TUI's
        # own status lines stay as they are (_prefixes).
        if not self._current_source:
            return "", ""
        tag = one_line(safe_text(source_label(self._current_source).replace(" #", ":")))
        return f"{tag} │ ", f"{' ' * views.cell_width(tag)} │ "

    def _prefixes(self, line: str) -> tuple[str, str]:
        """The row prefixes of ``line``: none for the TUI's own notes."""
        return ("", "") if isinstance(line, self._Note) else self._line_prefixes()

    def _wrap(self, line: str, width: int) -> list[str]:
        """The rows ``line`` takes in a pane ``width`` cells wide."""
        first, rest = self._prefixes(line)
        return wrap_log_line(self._display(line), width, first=first, rest=rest, max_rows=self._MAX_ROWS_PER_LINE)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if not self._screen_open:
            return
        sid = self._source_by_item.get(event.item.id or "")
        # Re-clicking the source you're already viewing shouldn't reload/reset
        # the pane (that's what caused the jump-to-bottom on every click).
        if sid and sid != self._current_source:
            self._start_tail(sid)

    def _post(self, fn, *args, **kwargs) -> bool:
        """Run ``fn`` on the UI thread from a worker; return ``False`` instead of
        raising once the app/event loop is gone (teardown), so background workers
        stop cleanly rather than surfacing a NoActiveAppError as a WorkerError.

        Only teardown-class errors (``RuntimeError`` — NoActiveAppError and
        "event loop is closed" both derive from it) are swallowed; a real bug in
        ``fn`` (Attribute/Type/Key/…) comes back as a ``_CallbackError``, so a
        worker never takes it for its own request failing (see _tail)."""
        failed: list[Exception] = []

        async def run():
            try:
                result = fn(*args, **kwargs)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # noqa: BLE001 - handed to the worker below
                if isinstance(exc, RuntimeError) or (isinstance(exc, NoMatches) and not on_stack(self)):
                    raise  # teardown, or a screen popped meanwhile: call_on_screen drops them
                failed.append(exc)

        delivered = call_on_screen(self, run)
        if failed:
            raise self._CallbackError(getattr(fn, "__qualname__", repr(fn))) from failed[0]
        return delivered

    def _ask(self, fn, *args):
        """Worker thread. ``fn(*args)`` run on the UI thread, and what it returns: how a worker
        reads the view as it is now, rather than its attributes from another thread. None once
        the app is gone (see _post)."""
        answer: list = []
        self._post(lambda: answer.append(fn(*args)))
        return answer[0] if answer else None

    def _saved_shown(self, source: str | None) -> bool:
        """Whether ``source`` shows its saved console: the whole job's download is in,
        or this sub-job's was fetched first while the rest download."""
        return self._stdout_ready or (source is not None and source in self._stdout_have)

    def _kept_console(self) -> bool:
        """A complete saved console from an earlier open of this same (finished) job."""
        if self._live() or not self._job_created_at:
            return False
        try:
            return (self._console_dir / ".complete").read_text(encoding="utf-8") == str(self._job_created_at)
        except (OSError, ValueError):  # unreadable, or not UTF-8: not a console this view kept
            return False

    def _stdout_path(self, sub_job_id: str) -> Path:
        if self._kept_console():
            return self._console_dir / sub_job_id / "stdout.log"
        root = "" if self._stage is None else self._stage.name
        return Path(root) / sub_job_id / "stdout.log"

    def _console_may_be_incomplete(self, seen: float | None, ended_at) -> bool:
        """Any thread. The job ended too recently for its console's last chunks to be uploaded for sure.

        True when ``ended_at`` (its ``state_changed_at``: when the job entered its
        terminal state) is under _KEEP_CONSOLE_AFTER_SECONDS ago, or when this view
        had the job running that recently (``seen``), whatever the timestamps say.
        """
        if seen is not None and time.monotonic() - seen < self._KEEP_CONSOLE_AFTER_SECONDS:
            return True
        ended = views.parse_ts(ended_at)
        if ended is None:
            return False
        return (datetime.now(timezone.utc) - ended).total_seconds() < self._KEEP_CONSOLE_AFTER_SECONDS

    def _keep_console(self, plan: LogScreen._Download) -> None:
        """Worker thread. Copy the console ``plan`` downloaded into the job's cache for the next open."""
        if not plan.root or views.parse_ts(plan.ended_at) is None or not plan.created_at:
            return
        if self._console_may_be_incomplete(plan.seen, plan.ended_at):
            return  # it may still be missing its last chunks: download it again next time
        tmp = None
        try:
            self._console_dir.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(prefix=".console-", dir=str(self._console_dir.parent)))
            shutil.copytree(plan.root, tmp, dirs_exist_ok=True)
            (tmp / ".complete").write_text(str(plan.created_at), encoding="utf-8")
            shutil.rmtree(self._console_dir, ignore_errors=True)
            os.replace(tmp, self._console_dir)
            tmp = None
        except OSError:
            pass  # this view keeps reading its temporary copy
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)
        # The kept consoles of every job stay within their bound, least recently used out first.
        prune_consoles(root=self._console_dir.parent.parent.parent, keep=self._console_dir)

    def _cleanup_stage(self) -> None:
        stage = self._stage
        self._stage = None
        if stage is not None:
            stage.cleanup()

    def _claim_stdout(self) -> bool:
        """UI thread. One whole-job download for this view; a failure retries on the next open."""
        if not self._screen_open or self._stdout_ready or self._stdout_inflight or self._stdout_failed:
            return False
        if self._stage is None:
            self._stage = tempfile.TemporaryDirectory(prefix="cortex-training-stage-")
        self._stdout_inflight = True
        return True

    # ─── Logs tab state: tail, waiting, idle pause, restarts, scrollback ──────
    _MAX_LINES = 20_000  # lines the pane (and its re-filter buffer) keeps
    _MAX_ROWS = 200_000  # wrapped rows the Log widget may hold for those lines
    _MAX_ROWS_PER_LINE = 20
    _FILTER_DEBOUNCE_SECONDS = 0.25
    # Log reads count as activity, so pause an unattended live tail.
    _IDLE_PAUSE_SECONDS = 15 * 60
    _IDLE_CHECK_SECONDS = 5.0
    _PRE_PLACEMENT = frozenset({"pending", "queued", "creating"})
    _STARTING = frozenset({"placing", "initializing"})
    _NOT_RUNNING_YET = _PRE_PLACEMENT | _STARTING
    _NOT_ROUTED = "does not belong to session"
    _PENDING_PLACEMENT = "is pending placement"
    _EXECUTION_ERROR_RECORDS = 200  # newest error records shown from an execution log
    _EXECUTION_TAIL_RECORDS = 20  # records shown when an execution log has no errors

    def _init_log_state(self) -> None:
        # Current source lines retained for request-free repainting.
        self._entries: deque[tuple[object, str]] = deque(maxlen=self._MAX_LINES)
        self._entries_dropped = 0
        self._display_cache: dict[str, str] = {}  # raw line -> what the pane shows (see _display)
        self._export_note: str | None = None  # what s last did
        self._download_note: str | None = None  # the saved console's download progress, or its failure
        self._tail_error: str | None = None  # the live tail stopped on this error (the status line says so)
        self._tail_request: tuple[int, float] | None = None  # (tail gen, monotonic start) of the call in flight
        self._behind_note: str | None = None  # full pages: the tail is catching up
        self._export_timer = None
        self._repainting = False  # a saved console's repaint is being prepared off the UI thread
        self._paint_gen = 0  # bumped by each _paint_saved; an older paint's result is dropped
        self._saved_matched_none = False  # the saved console has lines, but none pass the filter and level
        self._shown_rows: list[int] = []  # wrapped rows of each _shown_lines line
        self._rows_total = 0
        self._shown_logs = 0  # the log lines among _shown_lines (the rest are the TUI's notes)
        self._repaint_job: LogScreen._Repaint | None = None
        self._repaint_pending: list[str] = []
        self._repaint_gen = 0
        self._pane_truncated = False  # the pane no longer holds the source's oldest lines
        self._head_missing = False  # the tail (or its cache) never had the log's first lines
        self._saved_dropped = 0  # matching saved-console lines past _MAX_LINES
        self._tail_gen = 0  # bumped by every _start_tail; an older tail's pages are dropped
        self._tail_live = False  # a live tail is polling (or held)
        self._saw_live = False
        self._replayed = False  # this tail replayed cached lines before its live pages
        self._sub_job_status: dict[str, str] = {}
        self._job_reason_details = None
        self._waiting: str | None = None  # "pending" | "starting" | "stopped"
        self._waiting_note: str | None = None
        self._wait_words: tuple[str, str] | None = None
        self._unrouted_logged: set[str] = set()  # sub-jobs whose "not routed" traceback is in the error log
        self._idle_paused = False
        self._opened_at = time.monotonic()
        self._resumed_at = 0.0
        self._idle_floor = 0.0  # the idle clock starts no earlier than this (the job wasn't running before)
        self._restart_note: str | None = None
        self._log_status_text = ""
        self._exec_logs: dict[str, tuple[list[str], list[tuple[object, list[str]]]]] = {}
        self._exec_failed: set[str] = set()
        self._exec_inflight: set[str] = set()
        self._active_seen_at: float | None = None  # monotonic time a get_job last had the job active
        self._stdout_recent = False  # the shown console was fetched moments after the job ended
        self._stdout_again = False  # the next download adds the chunks uploaded since the last one

    # The saved console's last lines the live tail's are lined up against (see _newer_tail_lines).
    _ALIGN_LINES = 3

    def _scan_saved(self, path: Path, query: str, level, worker,
                    probe: str | None = None) -> tuple[list[str], int, int, list[str], bool] | None:
        """Scan a console for matching lines, bounded per logical line."""
        kept: deque[str] = deque(maxlen=self._MAX_LINES)
        last: deque[str] = deque(maxlen=self._ALIGN_LINES)
        matched = total = 0
        found = False
        query = query.lower()
        limit = self._COPY_MAX_BYTES
        half = limit // 2
        head = tail = ""
        chars = 0
        shown = None

        def reset() -> None:
            nonlocal head, tail, chars
            head = tail = ""
            chars = 0

        def append(piece: str) -> None:
            nonlocal head, tail, chars
            chars += len(piece)
            take = min(len(piece), half - len(head))
            head += piece[:take]
            tail = (tail + piece[take:])[-half:]

        def take_segment() -> tuple[str, bool, bool]:
            truncated = chars > limit
            text = (head + tail if not truncated
                    else safe_text(head) + " …[truncated]… " + safe_text(tail))
            state = text, chars > 0, truncated
            reset()
            return state

        def emit(state: tuple[str, bool, bool]) -> bool:
            nonlocal found, matched, total
            raw, _nonempty, sanitized = state
            text = raw if sanitized else safe_text(raw)
            displayed = display_line(text)
            total += 1
            if text.strip():
                last.append(text)
            if probe is not None and not found and log_format.unstamped(text) == probe:
                found = True
            if (not query or query in displayed.lower()) and (
                    not level or entry_at_level({"_raw": text}, level)):
                kept.append(text)
                matched += 1
            return total % 4096 == 0 and worker.is_cancelled

        def consume(chunk: str) -> bool:
            nonlocal shown
            for piece in re.split(r"([\r\n])", chunk):
                if piece == "\r":
                    state = take_segment()
                    if state[1]:
                        shown = state
                elif piece == "\n":
                    state = take_segment()
                    if state[1]:
                        shown = state
                    if emit(shown or ("", False, False)):
                        return True
                    shown = None
                else:
                    append(piece)
            return worker.is_cancelled

        with path.open("rb") as handle:
            remaining = committed_size(path, handle.seek(0, 2))
            handle.seek(0)
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            while remaining:
                raw = handle.read(min(64 * 1024, remaining))
                if not raw:
                    break
                remaining -= len(raw)
                if consume(decoder.decode(raw, final=remaining == 0)):
                    return None
            if remaining and consume(decoder.decode(b"", final=True)):
                return None
        state = take_segment()
        if state[1]:
            shown = state
        if shown is not None and emit(shown):
            return None
        return list(kept), matched, total, list(last), found

    def _newer_tail_lines(self, pairs: list, last: list[str], probe_found: bool, query: str,
                          level) -> tuple[list[str], bool]:
        """Worker thread. The live tail's lines past the end of a console fetched
        moments after the job ended (its last chunks may not be uploaded yet),
        that pass the filter and level: they were on screen before the switch.
        ``pairs``: the tail cache's newest (entry, line)s, in log order.

        Returns ``(lines, gap)``. They come after the console's last lines when
        those are in the cache; when they aren't and the cache's oldest line
        isn't in the console either, the whole cache came after the console's
        end (``gap``: lines between the two may be missing)."""
        lines = [line for _, line in pairs]
        displayed = [display_line(line) for line in lines]
        after = log_format.lines_after([display_line(line) for line in last], displayed)
        gap = False
        if after is None:
            if not lines or probe_found:
                return [], False  # they overlap, but can't be lined up: add nothing
            ends = [moment for moment in map(log_format.line_moment, last) if moment is not None]
            starts = log_format.line_moment(lines[0])
            if ends and starts is not None and starts < max(ends):
                return [], False  # the tail began before the console ends: their lines just don't compare
            after, gap = lines, True
        query = query.lower()
        start = len(pairs) - len(after)
        kept = [line for (entry, line), shown in zip(pairs[start:], displayed[start:])
                if (not query or query in shown.lower()) and entry_at_level(entry, level)]
        return kept, gap

    @work(thread=True, exclusive=True, group="paint")
    def _prepare_saved(self, gen: int, source: str, path: Path, query: str, level, width: int,
                       prefixes: tuple[str, str], recent: bool) -> None:
        """Worker thread. Read, filter and wrap the saved console for the pane;
        _apply_saved shows it. A long console takes a while, and the UI keeps
        answering keys meanwhile."""
        worker = get_current_worker()
        try:
            console = path.is_file() and path.stat().st_size > 0
        except OSError:
            console = False
        newer: list[str] = []
        gap = False
        if console:
            # The tail's own lines, for a console that may lag behind them.
            pairs = [(entry, format_log_entry(entry))
                     for entry in self._cache.load_entries(source, limit=self._MAX_LINES)] if recent else []
            probe = log_format.unstamped(safe_text(pairs[0][1])) if pairs else None
            try:
                scanned = self._scan_saved(path, query, level, worker, probe)
            except OSError:  # removed under us (the view closed): nothing to show
                return
            if scanned is None:
                return
            lines, matched, total, last, probe_found = scanned
            if pairs:
                newer, gap = self._newer_tail_lines(pairs, last, probe_found, query, level)
        else:
            # No console for this sub-job (yet): the lines its tail cached, if any.
            entries = self._cache.load_entries(source, limit=2000)
            lines = [format_log_entry(entry) for entry in entries
                     if entry_matches(entry, query) and entry_at_level(entry, level)]
            matched, total = len(lines), len(entries)
        if worker.is_cancelled:
            return
        first, rest = prefixes
        chunks, displays, cut = log_format.wrap_lines(
            lines, width, first=first, rest=rest, max_lines=self._MAX_LINES, max_rows=self._MAX_ROWS,
            max_rows_per_line=self._MAX_ROWS_PER_LINE)
        if worker.is_cancelled:
            return
        self._post(self._apply_saved, gen, source, console, chunks, displays, cut, width,
                   matched, total, len(lines), newer, gap)

    def _ended_job(self) -> dict:
        """This view's copy of the job with the newest status and reason (the tail
        can see the job end before the next get_job refreshes the copy)."""
        job = {**self._job, "status": self._job_status or self._job.get("status")}
        if self._job_reason is not None:
            job["reason"] = self._job_reason
        if self._job_reason_details is not None:
            job["reason_details"] = self._job_reason_details
        return job

    def _no_log_output(self) -> str:
        # A job that never got GPUs (placement_timeout and the like) wrote no logs: say why.
        if views.never_started(self._ended_job()):
            return f"No log output: {views.never_started_text(self._ended_job())}."
        return "No log output is available."

    @staticmethod
    def _span(seconds: float) -> str:
        minutes, rest = divmod(int(seconds), 60)
        return f"{minutes} min" if minutes and not rest else f"{int(seconds)} s"

    def _recent_console_note(self, *, console: bool = True, newer: int = 0) -> str:
        """The note under logs fetched moments after the job ended. ``console``: this sub-job has a saved
        console; ``newer``: the live tail's lines shown after it, which it doesn't have yet."""
        ago = self._span(self._KEEP_CONSOLE_AFTER_SECONDS)
        if console and newer:
            return (f"— the job ended less than {ago} ago, so the saved console may still be missing its last "
                    f"lines; the live tail's last {self._count(newer, 'line')} are shown after it until r "
                    "downloads them —")
        if console:
            return (f"— the job ended less than {ago} ago, so this console may still be missing its last lines; "
                    "r downloads them —")
        # No console yet: the job may have failed before it wrote one, or its
        # chunks aren't uploaded yet. Don't promise lines that may never come.
        return f"— the job ended less than {ago} ago, so its saved logs may not all be uploaded yet; r checks again —"

    def _paint_saved(self) -> None:
        """UI thread. Show the current sub-job's saved console, filtered; it is
        prepared off the UI thread, and the pane keeps what it shows until then."""
        source = self._current_source
        if not source:
            return
        # The job has ended: what its live tail waited for, or restarted from,
        # is over (the tail can notice the end before the status check does).
        self._waiting = self._waiting_note = None
        self._restart_note = None
        self._cancel_repaint()  # what a repaint on its way would show, this paint replaces
        self._paint_gen += 1
        self._repainting = True
        self._prepare_saved(self._paint_gen, source, self._stdout_path(source), self._filter, self._min_level,
                            self._content_width(), self._line_prefixes(), self._stdout_recent)
        self._render_log_status()

    def _apply_saved(self, gen: int, source: str, console: bool, chunks: list, displays: dict, cut: bool,
                     width: int, matched: int, total: int, kept: int, newer: list[str], gap: bool = False) -> None:
        """UI thread. Show a saved console _prepare_saved read and wrapped, unless a newer paint superseded it."""
        if gen != self._paint_gen or source != self._current_source or not self._saved_shown(source) \
                or not self._screen_open:
            return
        self._repainting = False
        self._display_cache.update(displays)
        # The TUI's own lines (shown untagged), after the console's.
        # A job that never got GPUs has no saved logs still to come.
        notes = ([self._Note(self._recent_console_note(console=console, newer=len(newer)))]
                 if self._stdout_recent and not views.never_started(self._ended_job()) else [])
        empty = False
        if console:
            self._saved_dropped = matched - kept
            extra: list[str] = []
            if newer:
                # Shown before the switch, and the console doesn't have them yet: keep them on screen.
                banner = self._Note(f"— {self._count(len(newer), 'newer line')} from the live tail, not in the "
                                    "saved console yet" + ("; lines between the two may be missing" if gap else "")
                                    + " —")
                extra = [banner, *newer]
            self._saved_matched_none = total > 0 and not matched and not newer
        else:
            self._saved_dropped = 0
            extra = self._execution_log_lines(source)
            empty = not total and not extra
            if empty and notes:
                notes.insert(0, self._Note(self._no_log_output()))
            self._saved_matched_none = total > 0 and not chunks
        if self._paused:
            # Shown once the pause ends (p).
            self._held = ([line for line, _, _ in chunks] + extra + notes, empty)
        else:
            self._held = None
            self._clear_pane()
            self._write_chunks(chunks, width, cut=cut)
            if extra or notes:
                self._write_lines(extra + notes)
            elif not chunks and empty:
                self._write_line(self._no_log_output())
            if width != self._content_width():
                self._wrap_width = -1  # the pane changed width meanwhile
                after_refresh(self, self._rewrap)
            if self._logview is not None:
                self._logview.scroll_end(animate=False)
        self._render_log_status()

    def _finish_download(self, ok: bool, problem: str | None = None, recent: bool = False) -> None:
        if not self._screen_open:
            self._stdout_inflight = False
            self._cleanup_stage()
            return
        self._stdout_inflight = False
        self._stdout_first = None
        self._download_note = None
        if not ok:
            self._stdout_failed = True
            if self._current_source in self._stdout_have:
                # The sub-job on screen has its console; the others' didn't come.
                self._download_note = (f"the other sub-jobs' saved consoles couldn't be downloaded "
                                       f"({problem or f'details in {_ERROR_LOG}'}); r tries again")
                self._render_log_status()
                return
            if problem:
                self._write_line(f"— {problem} —")
            else:
                self._write_line(f"— the saved console couldn't be downloaded; details in {_ERROR_LOG} —")
            self._render_log_status()
            return
        # The sub-job on screen was painted already, as soon as its console was in.
        painted = self._current_source in self._stdout_have and recent == self._stdout_recent
        self._stdout_ready = True
        self._stdout_recent = recent
        self._stdout_again = False
        self._stdout_have.clear()
        if painted:
            self._render_log_status()
        else:
            self._paint_saved()

    def _apply_terminal_status(self, status: str, reason: str | None) -> None:
        self._job_status = status
        if reason is not None:
            self._job_reason = reason
        self._update_subtitle()
        if self._resource_active():
            self._render_resource()

    @staticmethod
    def _sub_statuses(job: dict) -> dict[str, str]:
        statuses = {}
        for sj in job.get("sub_jobs") or []:
            if isinstance(sj, dict):
                sid = sj.get("sub_job_id") or sj.get("id")  # as _read_sources reads it
                if sid and sj.get("status"):
                    statuses[sid] = sj["status"]
        return statuses

    def _live(self) -> bool:
        return views.is_live(self._ended_job())

    def _refresh_status(self) -> None:
        # The tail pauses off the Logs tab, so it can't be what notices the job ending.
        if self.app.screen is self and self._live():
            self._fetch_status()

    @work(thread=True, exclusive=True, group="status")
    def _fetch_status(self) -> None:
        # Numbered as it is sent: a read that answers after a later one is dropped (see _take_job).
        # One the next tick's read replaced (exclusive) still counts: with a slow service every
        # read can take longer than a tick, and dropping those would drop them all.
        seq = self._next_job_read()
        try:
            job = self._client.get_job(self._job_id)
        except Exception:  # noqa: BLE001 - keep the current view; the next tick retries
            return
        if isinstance(job, dict) and job.get("status") and self._screen_open:
            self._post(self._apply_status, job, seq)

    def _apply_status(self, job: dict, seq: int) -> None:
        if not self._screen_open:
            return
        ended = self._live() and not views.is_live(job)
        # Status, reason and sub-job statuses come from this job as they are now:
        # a reason the service stopped sending (a job that left pending) is gone.
        # (_take_job also redraws the summary line from it.) An older answer than
        # the one on screen changes nothing.
        if not self._take_job(job, seq):
            return
        if ended and self._current_source:
            # Show the saved console now, as the tail would after its next error.
            self._start_tail(self._current_source, user=False)
            return
        # A Logs tab waiting for the job (or its sub-job) to start tails it once it can.
        self._resume_waiting_tail()

    def _auto_refresh_data(self) -> None:
        # Only the visible tab polls, and only while the job is live. Resources waits
        # until the status check sees the shown sub-job running: before that it has no samples.
        if self.app.screen is not self or not self._live():
            return
        if self._resource_active() and self._resource_waiting() is None:
            self._start_resource_refresh(force=True)

    def _next_job_read(self) -> int:
        """Any thread. The number of a get_job about to be sent: a read sent later has a larger one."""
        with self._job_reads_lock:
            self._job_reads += 1
            return self._job_reads

    def _take_job(self, job: dict, seq: int | None = None) -> bool:
        """UI thread. A fresh get_job replaces this view's status, reason, times and sub-job statuses.

        ``seq`` numbers the read (see _next_job_read). Reads can answer out of order (one
        that retries a slow service takes minutes), so an answer to a read sent before the
        one the view shows is dropped; so is an answer that has the job live after the view
        saw it end, which a job never comes back from. Returns whether ``job`` was taken.
        """
        if not isinstance(job, dict) or not (job.get("status") or job.get("sub_jobs")):
            return False
        if seq is not None and seq < self._job_seq:
            return False  # a read sent later has answered already
        if self._job_status and job.get("status") and views.is_live(job) and not self._live():
            return False  # an older answer, whatever its number: the job has ended
        if seq is not None:
            self._job_seq = seq
        self._sub_job_status = self._sub_statuses(job)
        self._job_reason_details = job.get("reason_details") or None
        if job.get("status") and views.is_live(job):
            self._active_seen_at = time.monotonic()  # it hadn't ended yet
        # A job that ended while the view was open has a new state_changed_at,
        # which decides whether its console is kept (_keep_console).
        self._state_changed_at = job.get("state_changed_at")
        self._job_created_at = job.get("created_at")
        status = job.get("status") or self._job_status  # (an answer without one leaves it)
        reason = job.get("reason") or ""  # omitted: the job has no reason now
        if status != self._job_status or reason != (self._job_reason or ""):
            self._apply_terminal_status(status, reason)
        # The rest of the view (the summary line's reason and reason_details) as the job is now:
        # the tail can see the job end before the status check does. The service's fields come
        # only from the fresh answer, so a reason it stopped sending disappears from the line.
        self._job = views.refresh_job(self._job, job)
        if job.get("status"):
            self._job_reason = job.get("reason")
        self._render_summary_line()
        return True

    def _tail_status(self, sub_job_id: str | None) -> str:
        """The status word that decides how a sub-job's tail runs.

        A placed job's sub-jobs start on their own schedule, so the sub-job's
        own status counts once the job is past placement; before that, and
        once the job has ended, the job's does.
        """
        job = views.status_word(self._job_status)
        sub_job_status = self._sub_job_status.get(sub_job_id or "")
        if not sub_job_status or job in self._PRE_PLACEMENT or not self._live():
            return job
        return views.status_word(sub_job_status)

    def _pending_note(self, sub_job_id: str) -> str:
        """The wait for placement: the job's (with its reason), or one sub-job's once the job is placed."""
        word = self._tail_status(sub_job_id)
        job_word = views.status_word(self._job_status)
        hint = f"(status checked every {self._STATUS_REFRESH_SECONDS} s · r checks now)"
        if job_word in self._PRE_PLACEMENT:
            # As the summary line puts them: "reason — reason_details".
            why = " — ".join(one_line(x).strip() for x in (self._job_reason, self._job_reason_details)
                             if x and str(x).strip())
            return (f"waiting: the job is {job_word.upper()}" + (f" ({why})" if why else "")
                    + f"; the log tail starts by itself when the job starts {hint}")
        if word in self._PRE_PLACEMENT:
            return (f"waiting: {views.sub_job_label(sub_job_id)} is {word.upper()}; "
                    f"its log tail starts by itself once it is placed {hint}")
        # The service refused the tail as pending placement while the status
        # says otherwise (UNKNOWN when the service couldn't read the job's state).
        return (f"waiting: the service says the job is pending placement (its status reads "
                f"{job_word.upper()}); the log tail starts by itself when the job starts {hint}")

    def _waiting_for(self, sub_job_id: str) -> tuple[str, str] | None:
        """The wait, if any, before ``sub_job_id`` can have logs: (kind, note)."""
        word = self._tail_status(sub_job_id)
        if word in self._PRE_PLACEMENT:
            return "pending", self._pending_note(sub_job_id)
        if word in self._STARTING:
            return "starting", (f"waiting: {views.sub_job_label(sub_job_id)} is {word.upper()}; "
                                "its log appears here once it is up")
        if views.status_word(self._job_status) == "unknown":
            # The service couldn't read the job's state: its tail may not answer either. It is tried.
            return "unknown", ("waiting: the job's status is unavailable right now, so its log may not answer; "
                               "the tail keeps trying (r tries now)")
        return None

    def _set_waiting(self, kind: str, note: str) -> None:
        # Said once, on the status line: it stays in view while lines arrive,
        # and the summary line above already carries the job's reason.
        self._waiting, self._waiting_note = kind, note
        self._render_log_status()

    def _clear_waiting(self) -> None:
        if self._waiting:
            self._waiting = self._waiting_note = None
            self._render_log_status()

    def _resume_waiting_tail(self) -> None:
        """UI thread, after a fresh status: start the tail a wait was for, once it can run,
        or one that stopped on an error."""
        source = self._current_source
        if not source or self._saved_shown(source) or not self._live():
            return
        word = self._tail_status(source)
        if self._tail_error and not self._tail_live:
            # Stopped on an error. No status change may be coming to try again on
            # (a job that runs, or whose status is unavailable), so each check
            # does: while someone is watching, as reads of a running job count
            # as activity on it, and whatever the input while it starts up.
            if views.status_word(self._job_status) in self._STARTING or self._watched():
                self._start_tail(source, user=False)  # the wait, if any, comes back with it
            return
        if not self._waiting:
            return
        if self._waiting == "unknown":
            if views.status_word(self._job_status) != "unknown":
                if self._tail_live:
                    self._clear_waiting()  # its status is back, and its tail is polling
                else:
                    self._start_tail(source, user=False)
            elif not self._tail_live and self._watched():
                self._start_tail(source, user=False)  # still unknown: try again while someone watches
            return
        if self._waiting == "starting":
            if not self._tail_live:
                # Its tail stopped while it started up, when the job couldn't be
                # read to tell why: try again.
                self._start_tail(source, user=False)
            elif word in self._STARTING:
                self._refresh_waiting_note()
            else:
                self._clear_waiting()  # it is up, and its tail is polling
            return
        if word in self._PRE_PLACEMENT:
            self._refresh_waiting_note()
            return
        if self._waiting == "stopped" and word != "running":
            return  # the tail starts again once the sub-job runs
        words = self._status_words(source)
        # While the job and sub-job run, no status change is coming to try again on: each check
        # (once a minute) does, but only while someone is watching (see _watched).
        if self._wait_words == words and not (words == ("running", "running") and self._watched()):
            return  # still the statuses the service answered for: r tries now
        self._start_tail(source, user=False)

    def _status_words(self, sub_job_id: str) -> tuple[str, str]:
        return views.status_word(self._job_status), self._tail_status(sub_job_id)

    def _watched(self) -> bool:
        """Whether someone used the view in the last _IDLE_PAUSE_SECONDS. Only then do the
        status checks start a tail by themselves: an unattended view must not keep reading
        a running job."""
        return time.monotonic() - self._last_input() < self._IDLE_PAUSE_SECONDS

    def _refresh_waiting_note(self) -> None:
        """Still waiting: show the note again when it changed (a new reason, or placing → initializing)."""
        wait = self._waiting_for(self._current_source or "")
        if wait is not None and wait[0] == self._waiting and wait[1] != self._waiting_note:
            self._set_waiting(*wait)

    def _start_tail(self, sub_job_id: str, *, user: bool = True) -> None:
        """Show ``sub_job_id``: its saved console, the wait for it to start, or its tail.

        ``user`` is false when the view starts a tail by itself (a status
        check); only a user's action resumes a tail paused for lack of input.
        """
        self._current_source = sub_job_id
        self._held = None
        self._tail_gen += 1
        self._tail_live = False
        self._entries.clear()
        self._entries_dropped = 0
        self._head_missing = False
        self._export_note = None
        self._tail_error = None
        self._behind_note = None
        self._last_line_at = None  # this sub-job's own, once its lines arrive
        self._saw_live = False
        self._restart_note = None
        self._waiting = self._waiting_note = None
        self._wait_words = None
        if user:
            self._idle_paused = False
            self._resumed_at = time.monotonic()
        # A fresh tail context (new source, r, or a status change) starts
        # following the tail — otherwise a pause left over from a prior source
        # would strand this one at the top.
        self._paused = False
        self._replayed = False
        # A saved console still being prepared for the sub-job shown before is dropped when it comes.
        self._paint_gen += 1
        self._repainting = False
        self._update_subtitle()
        self._clear_pane()
        if not self._stdout_ready and self._kept_console():
            self._stdout_ready = True  # finished long ago and downloaded before: replay it
            touch_console(self._console_dir)  # just used: the last kept console to be removed
        live = self._live()
        if self._stdout_ready or not live:
            self._idle_paused = False  # nothing polls now, so nothing is paused
        if self._saved_shown(sub_job_id):
            self._paint_saved()
            if not self._stdout_ready and not live and self._claim_stdout():
                # The other sub-jobs' consoles, again after they failed (r).
                self._tail(sub_job_id, self._tail_gen, True, False)
            return
        download = not live and self._claim_stdout()
        if not live and not download and self._stdout_inflight:
            # The download is under way: this sub-job's console comes with it.
            self._write_line(self._download_banner(sub_job_id))
        if live:
            wait = self._waiting_for(sub_job_id)
            if wait is not None:
                self._set_waiting(*wait)
                if wait[0] == "pending":
                    # The service refuses tail-logs until the job is placed: replay
                    # the cache only, and let the status check start the tail.
                    self._tail(sub_job_id, self._tail_gen, False, False)
                    return
            else:
                self._write_line(f"— tailing {sub_job_id} —")
            self._tail_live = True
        self._render_log_status()
        self._tail(sub_job_id, self._tail_gen, download, live)

    @work(thread=True, exclusive=True, group="tail")
    def _tail(self, sub_job_id: str, gen: int, download: bool, live: bool) -> None:
        worker = get_current_worker()
        # A terminal job's zone is gone, so the operation API would error: serve
        # cache, then one saved-console download. A running/unknown job follows live.
        # A _CallbackError is a bug in this view showing what came, not a failed read:
        # no get_job is sent to tell what it means (see _view_failed).
        try:
            try:
                self._stream(worker, sub_job_id, gen, live=live)
            except self._CallbackError:
                raise
            except Exception as exc:  # noqa: BLE001 - surfaced in the UI
                if worker.is_cancelled:
                    pass
                elif live:
                    self._post(self._tail_ended, gen)
                    self._after_tail_error(worker, sub_job_id, gen, exc)
                    return
                elif not download:
                    self._report_tail_error(sub_job_id, exc, gen)
                    return
            if live:
                self._post(self._tail_ended, gen)
        except self._CallbackError as exc:
            # (The saved console replaces a terminal job's cache replay in a moment.)
            self._view_failed(sub_job_id, exc, gen, live=live, quiet=download)
        if download:  # claimed for this tail (see _claim_stdout): it runs whatever the replay met
            try:
                self._run_download(sub_job_id)
            except self._CallbackError as exc:
                self._view_failed(sub_job_id, exc, gen, live=False)

    def _tail_ended(self, gen: int) -> None:
        if gen == self._tail_gen:
            self._tail_live = False
            if self._idle_paused:
                self._idle_paused = False  # nothing is polling to resume
                self._render_log_status()

    def _stream(self, worker, sub_job_id: str, gen: int, *, live: bool) -> None:
        # Replay the local cache instantly, then (only for a live job) resume
        # the tail from the saved cursor so only new lines are fetched. Each
        # yield is a whole page (or the cache batch): it is formatted here and
        # rendered in one shot, so high-velocity logs don't saturate the UI thread.
        for kind, entries in cached_log_pages(
            self._cache,
            _TimedTail(self._client, self, gen),
            self._job_id,
            sub_job_id,
            follow=live,
            live=live,
            poll_interval=self._poll_interval,
            replay_limit=2000,
            is_cancelled=lambda: worker.is_cancelled,
            hold=self._tail_held,
            # A running job's tail that stops moving is restarted from the end.
            can_restart=lambda: self._tail_status(sub_job_id) == "running",
        ):
            if worker.is_cancelled:
                return
            lines = [format_log_entry(entry) for entry in entries]
            if not self._post(self._on_page, gen, kind, entries, lines):
                return

    def _on_page(self, gen: int, kind: str, entries: list, lines: list[str]) -> None:
        """UI thread. One page of the tail started as ``gen`` (or its cache replay)."""
        # Checked here, on the UI thread, so a page of an older tail, or one that
        # was in flight when the saved console replaced the pane, is dropped.
        if gen != self._tail_gen or self._saved_shown(self._current_source) or not self._screen_open:
            return
        if kind == "partial":
            # Older lines of this log are neither here nor coming (the tail read only its end).
            self._head_missing = True
            self._render_log_status()
            return
        if kind == "caught-up":
            self._behind_note = None
            self._render_log_status()
            return
        if kind != "cache" and entries and self._waiting in ("starting", "unknown"):
            self._clear_waiting()  # lines are arriving: it is up
        pairs = list(zip(entries, lines))
        visible = [line for entry, line in pairs if self._visible(entry, line)]
        if kind in ("restart", "reset"):
            why = ("it had stopped moving" if kind == "restart" else
                   "the service no longer took its saved position in the log")
            self._restart_note = (f"tail restarted {time.strftime('%H:%M:%S')}: {why}, so it restarted from the "
                                  "end of the log; lines in between may be missing")
            self._write_line("— tail restarted from the end of the log; lines in between may be missing —")
        elif visible and kind == "live" and not self._saw_live:
            self._saw_live = True
            if self._replayed:  # where the replayed cache ends and new lines begin
                self._write_line("— live —")
        self._buffer(pairs)
        if kind == "cache" and entries:
            self._replayed = True
        if visible:
            self._write_lines(visible)
        if kind == "live":
            # A full page: the log runs ahead of this view, which catches up at a few reads a second.
            self._behind_note = self._catch_up_note(entries[-1]) if entries and catching_up(entries) else None
        if entries and kind != "cache":
            # The newest line's own timestamp (local time), not when the page arrived.
            clock = log_format.entry_clock(entries[-1])
            label = f"last line {clock}" if clock else f"last line received {time.strftime('%H:%M:%S')}"
            if label != self._last_line_at:
                self._last_line_at = label
                self._update_subtitle()
        self._render_log_status()

    _BEHIND_SECONDS = 5  # a tail getting full pages says how far behind it is from this lag

    @classmethod
    def _catch_up_note(cls, entry) -> str | None:
        """How far behind the log a tail that gets full pages is, by its newest line's own time;
        nothing while it keeps up (a busy log fills every page)."""
        raw = entry.get("_raw") if isinstance(entry, dict) else None
        when = views.parse_ts(str(raw).split(" ", 1)[0]) if isinstance(raw, str) else None
        if when is None:
            return None
        behind = int((datetime.now(timezone.utc) - when).total_seconds())
        if behind < cls._BEHIND_SECONDS:
            return None
        return f"catching up: the log is ahead of this view (its newest line here is {behind} s old)"

    def _buffer(self, pairs: list) -> None:
        """Keep received lines for a re-filter; the oldest go past _MAX_LINES."""
        overflow = len(self._entries) + len(pairs) - self._MAX_LINES
        if overflow > 0:
            self._entries_dropped += overflow
        self._entries.extend(pairs)

    def _visible(self, entry, line: str) -> bool:
        """Whether a received line passes the filter and level (the TUI's own notes always do)."""
        if entry is None:
            return True
        query = self._filter.lower()
        return (not query or query in line.lower()) and entry_at_level(entry, self._min_level)

    def _tail_hidden(self) -> bool:
        """Worker thread. True while the Logs tab is not on screen.

        Each tail-logs call counts as activity on a running job, so a viewer
        parked on another tab or screen must not keep polling; the saved cursor
        makes the catch-up on return cheap.
        """
        try:
            return not self._logs_active() or self.app.screen is not self
        except Exception:  # noqa: BLE001 - app shutting down; let cancellation win
            return False

    def _tail_held(self) -> bool:
        """Worker thread. True while the live tail must not poll: Logs hidden, or paused for no input."""
        return self._idle_paused or self._tail_hidden()

    def _last_input(self) -> float:
        """When the viewer last pressed a key, clicked or scrolled (monotonic seconds)."""
        try:
            seen = float(getattr(self.app, "last_input_at", 0.0) or 0.0)
        except Exception:  # noqa: BLE001 - no app (yet)
            seen = 0.0
        return max(self._opened_at, self._resumed_at, self._idle_floor, seen)

    def _idle_note(self) -> str:
        return f"paused: no input for {self._span(self._IDLE_PAUSE_SECONDS)} (press p to resume)"

    def _check_idle(self) -> None:
        """UI thread (timer). Pause a live tail that nobody has touched for _IDLE_PAUSE_SECONDS,
        and keep the status line's "no answer" note current."""
        if self._screen_open and (self._tail_request is not None or "no answer" in self._log_status_text):
            self._render_log_status()
        if self._idle_paused or not self._tail_live or not self._screen_open:
            return
        if not self._logs_active() or self.app.screen is not self:
            return  # the tail isn't polling now, and coming back to Logs is input
        if views.status_word(self._job_status) in self._NOT_RUNNING_YET:
            # Reads while the job is placed or starts up don't count as activity
            # on it, so the idle clock starts once it runs.
            self._idle_floor = time.monotonic()
            return
        if time.monotonic() - self._last_input() < self._IDLE_PAUSE_SECONDS:
            return
        self._idle_paused = True
        self._write_line(f"— {self._idle_note()} —")
        self._render_log_status()

    def _scrollback_note(self) -> str | None:
        """Say "last N lines" once the pane no longer holds the source's oldest lines."""
        saved = self._saved_shown(self._current_source)
        # A saved console is the whole log: what the tail missed doesn't count there.
        if not (self._pane_truncated or (self._head_missing and not saved) or (saved and self._saved_dropped > 0)):
            return None
        shown = self._shown_logs  # log lines, not the TUI's notes (kept as lines come and go: see _write_chunks)
        if not shown:
            return None
        full = f"{self._cli(f'download-log {shlex.quote(str(self._job_id))} --log-type stdout')} for the full log"
        filtered = bool(self._filter) or bool(self._min_level)
        # With a filter, the pane holds the newest matching lines (a live tail's
        # from the lines it had when the filter was set, and those since), so
        # the note counts only those; older matches may have left the view.
        noun = "matching line" if filtered else "line"
        return f"last {self._count(shown, noun)} · {full}"

    # A tail-logs call (with the client's own retries) unanswered this long is said on the status line.
    _NO_ANSWER_SECONDS = 20

    def _no_answer_note(self) -> str | None:
        request = self._tail_request
        if request is None or request[0] != self._tail_gen or not self._tail_live:
            return None
        waited = time.monotonic() - request[1]
        if waited < self._NO_ANSWER_SECONDS:
            return None
        return f"waiting for the service: the last log request has had no answer for {int(waited)} s"

    def _no_match_note(self) -> str | None:
        """Say so when the filter or level hides every line the pane has for this sub-job."""
        if not (self._filter or self._min_level) or self._repainting or self._repaint_job is not None:
            return None  # (the pane shows the lines before the repaint on its way)
        if self._shown_logs:
            return None
        if not (self._saved_shown(self._current_source) and self._saved_matched_none) and not any(
                entry is not None for entry, _ in self._entries):
            return None  # no lines yet: nothing is hidden
        what = " and ".join(bit for bit in (f"filter '{self._filter}'" if self._filter else "",
                                            f"level ≥{self._min_level}" if self._min_level else "") if bit)
        return f"no lines match {what} (Esc clears the filter, L changes the level)"

    @staticmethod
    def _count(n: int, noun: str) -> str:
        return f"{n:,} {noun}" + ("" if n == 1 else "s")

    def _render_log_status(self) -> None:
        """UI thread. The Logs tab's status line: shown only when it has something to say."""
        try:
            bar = self.query_one("#log-status", Static)
        except NoMatches:
            return
        idle = self._idle_note() if self._idle_paused and self._tail_live else None
        notes = (self._job_read_note(), self._waiting_note, self._tail_error, self._no_answer_note(),
                 self._behind_note, idle, self._restart_note, self._download_note, self._no_match_note(),
                 self._scrollback_note(), self._export_note)
        bits = [bit for bit in notes if bit]
        # One row: a reason or id from the service can carry a line break or an
        # escape sequence, which must neither reach the terminal nor grow the bar.
        text = one_line("  ·  ".join(bits))
        lv = self._logview
        # Log follows new lines only from the very end. A status line that
        # appears, goes, or wraps to another row resizes the pane and leaves a
        # following view short of its end, so put it back there afterwards.
        following = lv is not None and not self._paused and lv.is_vertical_scroll_end
        changed = False
        if text != self._log_status_text:  # pages arrive every second; repaint only on a change
            self._log_status_text = text
            # Plain Content: job ids and service reasons are shown literally.
            bar.update(Content(text))
            changed = True
        if bar.display != bool(bits):
            bar.display = bool(bits)
            changed = True
        if changed and following:
            after_refresh(self, lv.scroll_end, animate=False)

    @staticmethod
    def _log_traceback(label: str) -> None:
        """Worker thread, inside an ``except``: append the traceback to the error log."""
        try:
            with open(os.path.expanduser(_ERROR_LOG), "a", encoding="utf-8") as handle:
                handle.write(f"{label}:\n{traceback.format_exc()}\n")
        except Exception:  # noqa: BLE001
            pass

    def _report_tail_error(self, sub_job_id: str, exc: BaseException, gen: int, *, label: str | None = None) -> None:
        """Worker thread. ``label``: what the error log calls the traceback."""
        self._log_traceback(label or f"tail {sub_job_id} failed")
        error = views.describe_error(exc)
        self._post(
            self._show_tail_error,
            gen,
            f"[error] tail {sub_job_id} failed: {error}  (r retries · full traceback: {_ERROR_LOG})",
            error,
        )

    def _view_failed(self, sub_job_id: str, exc: Exception, gen: int, *, live: bool, quiet: bool = False) -> None:
        """Worker thread, inside the ``except``: a callback the tail ran on the UI thread raised.

        That is a bug in this view, not a failed request, so no get_job is sent to
        tell what it means (see _after_tail_error). Its traceback goes to the error
        log, and the tail stops as on any other error (``quiet``: the error log only).
        """
        label = f"tail {sub_job_id}: a bug in the TUI while showing its log"
        if quiet:
            self._log_traceback(label)
            return
        try:
            if live:
                self._post(self._tail_ended, gen)
            self._report_tail_error(sub_job_id, exc.__cause__ or exc, gen, label=label)
        except self._CallbackError:
            pass  # the view can't say so either: the traceback is in the error log

    def _show_tail_error(self, gen: int, line: str, error: str = "") -> None:
        """UI thread. Only into the pane of the tail that failed, not one the viewer switched to since."""
        if gen == self._tail_gen:
            # One note: a wait's ("the tail keeps trying") no longer holds once
            # the tail has stopped. The status check that starts it again (see
            # _resume_waiting_tail) brings the wait back with it.
            self._clear_waiting()
            self._behind_note = None
            word = views.status_word(self._job_status)
            while_ = {"running": " while the job runs",
                      "unknown": " while the job's status is unavailable"}.get(word, "")
            self._tail_error = f"tail stopped: {error or 'it failed'} (r retries; so does each status check{while_})"
            self._write_line(line)
            self._render_log_status()

    def _after_tail_error(self, worker, sub_job_id: str, gen: int, exc: BaseException) -> None:
        """Worker thread. A live tail failed: one get_job decides what to show, never a retry loop.

        A job that ended switches to its saved console at once (the service
        answers tail-logs with 409 once a job is no longer running, and 404 once
        its session is gone). A job or sub-job that hasn't started yet shows the
        wait instead of an error (409 while pending placement, 400 "does not
        belong to session" for a sub-job the service isn't routing to yet), and
        the status check starts the tail when it can run. The service's own
        answer counts as much as the status word: a 409 "is pending placement"
        waits whatever the status says (UNKNOWN when the service couldn't read
        the job's state). Anything else stays an error, in place of any wait:
        r tries again, and so does the status check (see _resume_waiting_tail).
        """
        if worker.is_cancelled or not self._screen_open:
            return
        seq = self._next_job_read()
        try:
            job = self._client.get_job(self._job_id) or {}
        except Exception:  # noqa: BLE001 - a running job keeps the tail error
            job = None
        if job is None:
            if not worker.is_cancelled and self._screen_open:
                self._report_tail_error(sub_job_id, exc, gen)  # (the tail's traceback, not the read's)
            return
        if worker.is_cancelled or not self._screen_open:
            return
        # Decided on the UI thread, by the view's newest status: this answer's, unless
        # an answer to a read sent later is on screen already (see _take_job).
        step = self._ask(self._tail_failed, gen, sub_job_id, job, seq, views.describe_error(exc))
        if step is None:
            return
        what, detail = step
        if what == "error":
            self._report_tail_error(sub_job_id, exc, gen)
        elif what == "wait" and detail:
            self._log_traceback(detail)
        elif what == "download":
            self._run_download(detail)

    def _tail_failed(self, gen: int, sub_job_id: str, job, seq: int, text: str) -> tuple[str, str] | None:
        """UI thread. What a live tail's failure (``text``) means, now that get_job answered
        ``job`` (read ``seq``); see _after_tail_error. Returns what the tail's worker does
        next: ``("error", "")`` report it, ``("wait", label)`` log it under ``label`` (none:
        it is logged already) for the wait shown, ``("download", source)`` fetch the saved
        console (claimed here), or None."""
        if not self._screen_open:
            return None
        self._take_job(job, seq)
        if self._live():
            word = self._tail_status(sub_job_id)
            if self._NOT_ROUTED in text:
                why, label = "not-routed", f"tail {sub_job_id} waiting (not routed yet)"
                if sub_job_id in self._unrouted_logged:
                    label = ""  # the next answers would only repeat it
                self._unrouted_logged.add(sub_job_id)
            elif word in self._NOT_RUNNING_YET:
                why, label = word, f"tail {sub_job_id} waiting ({word})"
            elif self._PENDING_PLACEMENT in text:
                why, label = "pending-placement", f"tail {sub_job_id} waiting (pending placement, status {word})"
            else:
                return "error", ""
            self._wait_after_error(gen, sub_job_id, why)
            return "wait", label
        # The job ended: show its saved console now.
        if not self._claim_stdout():
            if self._saved_shown(self._current_source):
                self._paint_saved()
            return None
        return "download", self._current_source or ""

    def _wait_after_error(self, gen: int, sub_job_id: str, why: str) -> None:
        """UI thread. The tail stopped because ``sub_job_id`` hasn't started: show the wait."""
        if gen != self._tail_gen or sub_job_id != self._current_source:
            return
        name = views.sub_job_label(sub_job_id)
        if why in self._PRE_PLACEMENT:
            self._set_waiting("pending", self._pending_note(sub_job_id))
        elif why == "pending-placement":
            # Tried again once the status changes (_resume_waiting_tail).
            self._wait_words = self._status_words(sub_job_id)
            self._set_waiting("pending", self._pending_note(sub_job_id))
        elif why == "not-routed":
            # Tried again once its status or the job's changes; while both run,
            # no change is coming, so each status check tries again.
            self._wait_words = job_word, word = self._status_words(sub_job_id)
            if word == "running" and job_word == "running":
                note = (f"waiting: the service isn't routing {name}'s log yet; the tail tries again each "
                        "minute (r tries now)")
            elif word == "running":
                note = (f"waiting: the service isn't routing {name}'s log yet; the tail tries again when the "
                        "job's status changes (r tries now)")
            else:
                note = (f"waiting: {name} hasn't started yet (the service isn't routing to it); "
                        "the tail starts again once it is running")
            self._set_waiting("stopped", note)
        else:
            self._set_waiting("stopped", f"waiting: {name} is {why.upper()}; the tail starts again once it is running")

    def _run_download(self, sub_job_id: str) -> None:
        """Worker thread. The saved-console download this view claimed (see _claim_stdout):
        _finish_download gives the claim back whatever happens, and the error log tells a bug
        in this view (a _CallbackError) from a failed download."""
        ok = False
        problem = None
        recent = False
        try:
            # The saved console is the whole job's output, minutes for a long job: start it
            # only once Logs is on screen, so a look at another tab doesn't pay for it.
            while self._screen_open and self._tail_hidden():
                time.sleep(0.25)
            # A client that can fetch one sub-job: the one on screen first, painted
            # as soon as it is in, then the rest (resumed: that one is done).
            by_sub_job = self._downloads_by_sub_job()
            plan = self._ask(self._download_started, sub_job_id, by_sub_job) if self._screen_open else None
            if plan is not None and self._fetch_consoles(plan):
                ok = True
                recent = self._console_may_be_incomplete(plan.seen, plan.ended_at)
                self._keep_console(plan)
        except DownloadCancelled:
            pass  # the view closed; the next one resumes from what was saved
        except self._CallbackError:
            self._log_traceback(f"saved log {sub_job_id}: a bug in the TUI while showing it")
        except Exception as exc:  # noqa: BLE001 - keep the cache; record the traceback
            # Saved consoles are SQL on the job's Experiment: say so when the role can't read it.
            problem = views.access_hint(exc, "saved logs")
            self._log_traceback(f"saved log {sub_job_id} failed")
        if not self._post(self._finish_download, ok, problem, recent):
            self._cleanup_stage()

    @dataclass(frozen=True)
    class _Download:
        """What a saved-console download fetches, as the view was when it started (see _download_started)."""

        first: str | None  # the sub-job on screen, fetched before the others (None: the whole job at once)
        again: bool  # only the chunks uploaded since the last download (r, moments after the job ended)
        root: str  # the view's temporary directory, downloaded into
        others: bool  # the job has other sub-jobs, fetched after ``first``
        seen: float | None  # when a get_job last had the job live (monotonic seconds)
        ended_at: object  # the job's state_changed_at
        created_at: object  # ... and its created_at (see _keep_console)

    def _fetch_consoles(self, plan: LogScreen._Download) -> bool:
        """Worker thread. Download the saved consoles ``plan`` names; False once the app went away meanwhile."""
        quiet = _quiet_download(self._client.download_stdout_logs)
        if plan.first is None:
            # r on a console fetched moments after the job ended: only the chunks uploaded since.
            self._client.download_stdout_logs(self._job_id, plan.root, **quiet,
                                              **({"resume": True} if plan.again else {}))
            return True
        # Resumed in this view's own directory: a sub-job fetched already only gets newer chunks.
        try:
            self._client.download_stdout_logs(self._job_id, plan.root, sub_job_ids=[plan.first], resume=True,
                                              on_chunk=self._download_progress(first=True, others=plan.others), **quiet)
        except NoChunksError:
            pass  # the sub-job on screen saved no console (yet): the others still come
        if not self._post(self._first_console_in, plan.first,
                          self._console_may_be_incomplete(plan.seen, plan.ended_at)):
            return False
        self._client.download_stdout_logs(self._job_id, plan.root, resume=True,
                                          on_chunk=self._download_progress(first=False, others=plan.others), **quiet)
        return True

    def _downloads_by_sub_job(self) -> bool:
        """Whether this client's download_stdout_logs takes sub_job_ids and on_chunk
        (an older client, or the demo's, downloads the whole job at once)."""
        try:
            parameters = inspect.signature(self._client.download_stdout_logs).parameters
        except (TypeError, ValueError):
            return False
        return "sub_job_ids" in parameters and "on_chunk" in parameters

    # How often the download's chunk count is redrawn on the status line.
    _PROGRESS_SECONDS = 0.25

    def _download_started(self, sub_job_id: str, by_sub_job: bool) -> LogScreen._Download | None:
        """UI thread. A saved-console download starts: say so in the pane, and return what it
        fetches, as the view is now. ``by_sub_job``: the client fetches the sub-job on screen
        (else ``sub_job_id``) before the others; otherwise the whole job comes at once. None
        once the view has closed."""
        if not self._screen_open:
            return None
        first = (self._current_source or sub_job_id) if by_sub_job else None
        self._stdout_first = first
        self._stdout_others = first is not None and any(
            sid != first for sid in (*self._source_by_item.values(), *self._sub_job_status))
        self._write_line(self._download_banner(self._current_source))
        return self._Download(first, self._stdout_again, "" if self._stage is None else self._stage.name,
                              self._stdout_others, self._active_seen_at, self._state_changed_at,
                              self._job_created_at)

    def _download_banner(self, source: str | None) -> str:
        """UI thread. The pane's line for ``source`` while the saved consoles download."""
        first = self._stdout_first
        if first is None:
            if self._stdout_again:
                return "— downloading the saved console's newest chunks —"
            return "— downloading the saved console (the whole job; minutes for a long job) —"
        one, many = (("newest saved-console chunks",) * 2 if self._stdout_again
                     else ("saved console", "saved consoles"))
        name = views.sub_job_label(first)
        if source == first or source is None:
            if not self._stdout_others:
                return f"— downloading {name}'s {one} —"
            return f"— downloading {name}'s {one} first, then the other sub-jobs' —"
        mine = views.sub_job_label(source)
        if first not in self._stdout_have:
            return f"— downloading {name}'s {one} first, then the other sub-jobs' ({mine} among them) —"
        return f"— downloading the other sub-jobs' {many} ({mine} among them) —"

    def _download_progress(self, *, first: bool, others: bool):
        """Worker thread. The client's per-chunk callback: the chunk count on the status line, a few times a second.
        ``others``: the job has sub-jobs fetched after the first."""
        last = [0.0]

        def progress(sub_job_id: str, done: int, total: int, _bytes: int = 0) -> None:
            if not self._screen_open:
                # The view closed: the client stops at this chunk and keeps what it saved.
                raise DownloadCancelled("the saved-console view closed")
            now = time.monotonic()
            if last[0] < 0 or (done < total and now - last[0] < self._PROGRESS_SECONDS):
                return
            last[0] = now
            name = views.sub_job_label(sub_job_id)
            if first:
                text = (f"downloading {name}'s saved console: chunk {done:,} of {total:,}"
                        + (" (the other sub-jobs' come after it)" if others else ""))
            else:
                text = f"downloading the other sub-jobs' saved consoles: {name}, chunk {done:,} of {total:,}"
            try:
                self._post(self._show_download_note, text)
            except self._CallbackError:
                # A bug in drawing the count: it must not fail the download, which this call is part of.
                last[0] = -1.0  # no more counts
                self._log_traceback("showing a saved console's download progress failed (a bug in the TUI)")

        return progress

    def _show_download_note(self, text: str | None) -> None:
        if self._screen_open and text != self._download_note:
            self._download_note = text
            self._render_log_status()

    def _first_console_in(self, source: str, recent: bool) -> None:
        """UI thread. The sub-job on screen has its saved console while the others' still come: show it."""
        if not self._screen_open:
            return
        self._stdout_have.add(source)
        self._stdout_recent = recent
        # Until the others' first chunk is counted (their listing can take a while).
        self._download_note = "downloading the other sub-jobs' saved consoles" if self._stdout_others else None
        if source == self._current_source:
            self._paint_saved()
        else:
            self._render_log_status()

    def _configured_location(self) -> dict:
        """How this screen's client connects, for the copyable commands: its database
        and schema (the job's own, for a job from the account-wide list, whose
        client is bound there), endpoint and connection profile name."""
        auth = getattr(self._client, "_auth_provider", None)  # a profile client's SnowflakeProfileAuth
        found = {"database": getattr(self._client, "database", None), "schema": getattr(self._client, "schema", None),
                 "connection": getattr(auth, "connection_name", None),
                 "endpoint": getattr(self._client, "endpoint", None)}
        # Only real names: a demo or test client's attributes may be anything.
        where = {key: value if isinstance(value, str) and value else None for key, value in found.items()}
        if where["endpoint"] == "cortex-training":
            where["endpoint"] = None
        return where

    def _cli(self, arguments: str) -> str:
        """``cortex-training [--connection C] [--database D --schema S] ARGUMENTS`` that
        reaches this job: copied from a job in another location, a bare command
        would go to the configured database and schema and not find it. Without a
        profile it connects as the TUI does (``--config FILE``, ``--host HOST`` or
        ``--base-url URL``; see ``views.sign_in_options``)."""
        where = self._configured_location()
        names = views.location_names(self._job)  # the row's own location wins
        if names is not None:
            where["database"], where["schema"] = views.sql_identifier(names[0]), views.sql_identifier(names[1])
        if not where.get("connection"):
            where.update(views.sign_in_options(getattr(self.app, "sign_in", None)))
        return " ".join(["cortex-training", *views.cli_options(**where), arguments])

    def _execution_log_lines(self, source: str) -> list[str]:
        """UI thread. What a failed job's execution log adds for a sub-job without a saved console.

        Jobs that fail while they are created or start up may have no saved
        console; their execution log (``_logs/<sub_job_id>/execution.jsonl``
        on the job's Experiment) still records the error. Empty when this
        doesn't apply or the log has nothing; the first call starts the read
        and returns a progress note. The filter and level apply to whole
        records (a record shows with its exception chain or not at all); the
        header always shows.
        """
        if views.status_word(self._job_status) != "failed":
            return []
        if source in self._exec_logs:
            notes, records = self._exec_logs[source]
            query = self._filter.lower()
            kept = [
                line
                for record, lines in records
                if (not query or any(query in line.lower() for line in lines))
                and entry_at_level(record, self._min_level)
                for line in lines
            ]
            return [*notes, *kept]
        if source not in self._exec_inflight:
            self._exec_inflight.add(source)
            self._load_execution_log(source)
        return [self._Note("— no saved console for this sub-job; reading its execution log for the error —")]

    @work(thread=True, group="execution-log")
    def _load_execution_log(self, source: str) -> None:
        failed = False
        try:
            logs = self._client.fetch_execution_logs(self._job_id, sub_job_id=source, filename="execution.jsonl")
            notes, records = self._execution_log_body(source, logs)
        except Exception as exc:  # noqa: BLE001 - shown in the pane; traceback in the error log
            failed = True
            self._log_traceback(f"execution log {source} failed")
            problem = views.access_hint(exc, "execution log") or (
                f"no saved console for this sub-job, and its execution log couldn't be read: "
                f"{views.describe_error(exc)} (details in {_ERROR_LOG})"
            )
            notes, records = [f"— {problem} —"], []
        self._post(self._apply_execution_log, source, notes, records, failed)

    def _execution_log_body(self, source: str, logs) -> tuple[list[str], list[tuple[object, list[str]]]]:
        """Worker thread. A sub-job's execution log for the pane: its header, then each record with its lines."""
        contents = [
            item.get("content") for item in (logs if isinstance(logs, list) else [])
            if isinstance(item, dict) and item.get("sub_job_id") == source
        ]
        records = [record for content in contents for record in log_format.execution_log_records(content)]
        if not records:
            return [], []
        name = views.sub_job_label(source)
        fetch = f"{self._cli(f'download-log {shlex.quote(str(self._job_id))}')} downloads the whole log"
        errors = [record for record in records if log_format.is_error_record(record)]
        if errors:
            shown = errors[-self._EXECUTION_ERROR_RECORDS:]
            head = (f"— no saved console for {name}: the job failed before it wrote one. The error records "
                    f"from its execution log follow ({fetch}) —")
        else:
            shown = records[-self._EXECUTION_TAIL_RECORDS:]
            head = (f"— no saved console for {name}, and its execution log has no error records; its last "
                    f"{len(shown)} record(s) follow ({fetch}) —")
        return [head], [(record, log_format.format_execution_record(record)) for record in shown]

    def _apply_execution_log(self, source: str, notes: list[str], records: list, failed: bool) -> None:
        self._exec_inflight.discard(source)
        # Its header is the TUI's own line; the records' lines are the log's.
        self._exec_logs[source] = ([self._Note(note) for note in notes], records)
        if failed:
            self._exec_failed.add(source)
        if source == self._current_source and self._saved_shown(source):
            self._paint_saved()

    def _content_width(self) -> int:
        """Usable text columns of the log pane (accounts for padding/scrollbar)."""
        lv = self._logview
        if lv is None:
            return 0
        try:
            w = lv.scrollable_content_region.width
        except Exception:  # noqa: BLE001 - not laid out yet
            w = 0
        if w <= 0:
            w = getattr(lv.size, "width", 0)
        return max(0, w)

    def _write_line(self, line: str) -> None:
        # Single lines are the TUI's own status lines, never log content.
        note = line if isinstance(line, self._Note) else self._Note(line)
        self._buffer([(None, note)])
        self._write_lines([note])
        if self._pane_truncated:
            self._render_log_status()  # it pushed a log line out: "last N lines" changes

    @staticmethod
    def _row_count(pieces: list[str]) -> int:
        # Log.write_lines splits each piece on every line boundary (not just \n).
        return sum(len(piece.splitlines()) for piece in pieces)

    @classmethod
    def _log_lines(cls, lines) -> int:
        """How many of ``lines`` are log lines, not the TUI's notes."""
        return sum(1 for line in lines if not isinstance(line, cls._Note))

    def _clear_pane(self) -> None:
        # The pane is drawn again (another sub-job, filter or level): a highlight on the
        # old lines would sit over other text, at the same line numbers. A repaint on its
        # way (see _repaint) was for what the pane showed before.
        self._cancel_repaint()
        self._reset_pane()

    def _reset_pane(self, *, keep_selection: bool = False) -> None:
        """Empty the pane and its buffers (see _clear_pane; a repaint that lands does it too)."""
        if not keep_selection:
            self.clear_selection()
        self._shown_lines = []
        self._shown_rows = []
        self._rows_total = 0
        self._shown_logs = 0
        self._pane_truncated = False
        if self._logview is not None:
            self._logview.clear()
            self._logview.max_lines = self._MAX_ROWS
            # Log.clear() keeps the scroll offset, and Log.write_lines(scroll_end=True)
            # follows only a view already at its end, so lines written in the same
            # callback would stay at the old offset, above the newest ones. An
            # empty pane starts at its top, which is also its end.
            self._logview.scroll_home(animate=False, immediate=True, force=True)

    def _trim_shown(self) -> None:
        """Drop the oldest shown lines past _MAX_LINES lines or _MAX_ROWS wrapped rows."""
        lines, rows = self._shown_lines, self._shown_rows
        drop = max(0, len(lines) - self._MAX_LINES)
        total = self._rows_total - sum(rows[:drop])
        while total > self._MAX_ROWS and drop < len(lines) - 1:
            total -= rows[drop]
            drop += 1
        self._rows_total = total
        if drop:
            self._shown_logs -= self._log_lines(lines[:drop])
            del lines[:drop]
            del rows[:drop]
            if not self._pane_truncated:
                self._pane_truncated = True
                self._render_log_status()

    def _write_lines(self, lines: list[str]) -> None:
        lv = self._logview
        # A page posted by the tail can land after the screen closed; writing to its
        # detached Log starts a Textual worker that crashes the app.
        if lv is None or not lines or not self._screen_open:
            return
        if self._repaint_job is not None:
            # Shown now, and again after the lines of the repaint on its way, which
            # replaces what the pane shows when it lands (see _repaint).
            self._repaint_pending.extend(lines)
        w = self._content_width()
        # A batch is wrapped from its newest line back, only as far as the
        # pane's line and row budgets reach: a repaint of a long log would
        # otherwise wrap thousands of lines only to drop them again.
        chunks: list[tuple[str, list[str], int]] = []
        added = 0
        for line in reversed(lines):
            if len(chunks) >= self._MAX_LINES:
                break
            pieces = self._wrap(line, w)
            rows = self._row_count(pieces)
            if chunks and added + rows > self._MAX_ROWS:
                break
            chunks.append((line, pieces, rows))
            added += rows
        chunks.reverse()
        self._write_chunks(chunks, w, cut=len(chunks) < len(lines))

    def _write_chunks(self, chunks: list[tuple[str, list[str], int]], width: int, *, cut: bool = False,
                      kinds: list | None = None) -> None:
        """UI thread. Append lines already wrapped to ``width``: (line, its rows, row count) each.
        ``cut``: older lines of the same batch didn't fit; ``kinds``: each line's row kind, when
        it was worked out with the wrapping (see _write_rows)."""
        lv = self._logview
        if lv is None or not chunks or not self._screen_open:
            return
        if cut and self._shown_lines:
            # The lines on screen came before the batch's dropped ones: the pane
            # keeps one unbroken run of the newest lines.
            self._clear_pane()
        fresh = not self._shown_lines
        # Always stick to the tail so every viewer sees the newest lines on
        # open and as they stream — cache replay and live alike. Pause is the
        # one escape hatch: it stops following so the user can scroll back.
        follow = not self._paused
        if fresh:
            self._wrap_width = width
        elif width != self._wrap_width:
            self._wrap_width = -1  # rows now wrapped to mixed widths; the next _rewrap redoes them
        wrapped = [piece for _, pieces, _ in chunks for piece in pieces]
        added = sum(rows for _, _, rows in chunks)
        self._shown_lines.extend(line for line, _, _ in chunks)
        self._shown_rows.extend(rows for _, _, rows in chunks)
        self._shown_logs += self._log_lines(line for line, _, _ in chunks)
        self._rows_total += added
        if cut and not self._pane_truncated:
            self._pane_truncated = True
            self._render_log_status()
        self._trim_shown()
        # The widget keeps exactly the rows of the lines kept above.
        before = len(lv.lines)
        lv.max_lines = max(1, self._rows_total)
        # One write_lines call per page keeps the UI responsive under
        # high-velocity logs (vs. a cross-thread hop + write per line).
        self._write_rows(lv, wrapped, chunks, scroll_end=follow, kinds=kinds)
        if follow and before + added > lv.max_lines:
            # Once full, Log drops its oldest rows and repaints only the rows past
            # its old end, which that has moved out of view, and a view already
            # at the end doesn't scroll: repaint it, or the newest lines don't show.
            lv.refresh()
        if len(self._display_cache) > 4 * self._MAX_LINES:
            self._prune_display_cache()

    def _write_rows(self, lv: Log, rows: list[str], chunks, *, scroll_end: bool, kinds: list | None = None) -> None:
        """Write ``rows`` (the pieces of ``chunks``: (line, its rows, row count) each) to the pane.
        ``kinds``: each chunk's row kind (see line_kind), worked out here when not given."""
        if not isinstance(lv, _LogPane):
            lv.write_lines(rows, scroll_end=scroll_end)
            return
        if kinds is None:
            kinds = [line_kind(self._display(line), note=isinstance(line, self._Note)) for line, _, _ in chunks]
        row_kinds: list[str | None] = []
        for kind, (_line, _pieces, count) in zip(kinds, chunks):
            row_kinds += [kind] * count
        lv.write_rows(rows, row_kinds, scroll_end=scroll_end)

    def _prune_display_cache(self) -> None:
        """Keep the display text of the lines this view still has, and drop the rest."""
        keep = set(self._shown_lines)
        keep.update(line for _, line in self._entries)
        self._display_cache = {line: text for line, text in self._display_cache.items() if line in keep}

    def _rewrap(self) -> None:
        """Re-wrap the on-screen buffer after the pane width changes (off the UI thread: see _repaint)."""
        lv = self._logview
        if lv is None or not self._screen_open:
            return
        w = self._content_width()
        job = self._repaint_job
        if w <= 0 or w == (job.width if job is not None else self._wrap_width):
            return
        if job is not None:
            # A repaint is on its way (a new filter's, say): it is made again for this width.
            self._repaint(job.pairs, job.query, job.level, rewrap=job.rewrap, truncated=job.truncated, top=job.top,
                          keep_pending=True)
        elif not self._shown_lines:
            self._wrap_width = w  # nothing to wrap
        else:
            # A paused reader keeps their place: the line at the top of the view stays there
            # (the one a re-wrap still scrolling to it named, when two come close together).
            top = None
            if self._paused:
                top = self._pending_top if self._pending_top is not None else self._line_at_row(
                    int(lv.scroll_offset.y))
            self._repaint([(None, line) for line in self._shown_lines], rewrap=True, truncated=self._pane_truncated,
                          top=top)

    # ─── repaints: the pane's lines filtered and wrapped off the UI thread ──
    @dataclass(frozen=True)
    class _Repaint:
        """A repaint of the log pane on its way (see _repaint)."""

        pairs: list  # (entry, line) each, oldest first
        query: str  # the filter, and the level, they must pass
        level: object
        width: int  # the pane width they are wrapped to
        rewrap: bool  # the lines on screen, at a new width (a highlight on them stays, as it always has)
        truncated: bool  # the pane doesn't hold the source's oldest lines, whatever this repaint fits
        top: int | None  # a paused reader's line (an index into pairs), back at the top of the view after

    def _repaint(self, pairs: list, query: str = "", level=None, *, rewrap: bool = False, truncated: bool = False,
                 top: int | None = None, keep_pending: bool = False) -> None:
        """UI thread. Show the ``pairs`` ((entry, line) each, oldest first, the TUI's notes among
        them) that pass ``query`` and ``level``, filtered and wrapped off the UI thread
        (_prepare_rows) and shown by _apply_rows: the pane's 20,000 lines take a while, and keys
        keep working meanwhile. Until then the pane keeps what it shows; the lines written
        meanwhile (a page of the tail) show at once, and again after the repaint's when it
        lands (``keep_pending``: those written since the repaint this one replaces, too).
        ``rewrap``: the lines on screen, at a new width; ``truncated``: the pane doesn't hold the
        source's oldest lines, whatever this one fits; ``top``: the line of ``pairs`` a paused
        reader had at the top of the view."""
        if not keep_pending:
            self._repaint_pending = []
        self._repaint_gen += 1
        self._repaint_job = job = self._Repaint(pairs, query, level, self._content_width(), rewrap, truncated, top)
        # The worker reads none of the view's state: what it needs is passed as it is now.
        self._prepare_rows(self._repaint_gen, job, self._line_prefixes(), dict(self._display_cache))

    def _cancel_repaint(self) -> None:
        """UI thread. Drop the repaint on its way, and the lines kept for after it: the pane is drawn anew."""
        if self._repaint_job is not None:
            self._repaint_job = None
            self._repaint_gen += 1
        self._repaint_pending = []

    @work(thread=True, exclusive=True, group="wrap")
    def _prepare_rows(self, gen: int, job: LogScreen._Repaint, prefixes: tuple[str, str], known: dict) -> None:
        """Worker thread. Filter and wrap ``job``'s lines for the pane; _apply_rows shows them.
        ``known``: the display cache's texts (see _display)."""
        worker = get_current_worker()
        wrapped = self._wrap_pairs(job.pairs, job.query, job.level, job.width, prefixes, known,
                                   lambda: worker.is_cancelled)
        if wrapped is not None and not worker.is_cancelled:
            self._post(self._apply_rows, gen, *wrapped)

    @classmethod
    def _wrap_pairs(cls, pairs: list, query: str, level, width: int, prefixes: tuple[str, str], known: dict,
                    cancelled) -> tuple[list, list, dict, bool] | None:
        """Any thread. The newest of ``pairs`` that pass ``query`` and ``level`` (the TUI's notes
        always do), wrapped to ``width`` from the newest back as far as the pane's line and row
        budgets reach. Returns ``(chunks, kinds, displays, cut)`` for _write_chunks: each kept
        line's rows and row kind (see line_kind), the display text of the log lines ``known``
        lacked, and whether an older line that passes didn't fit; None once ``cancelled()``."""
        first, rest = prefixes
        query = query.lower()
        chunks: list[tuple[str, list[str], int]] = []
        kinds: list[str | None] = []
        displays: dict[str, str] = {}
        total = 0
        cut = False
        for n, (entry, line) in enumerate(reversed(pairs)):
            if n % 1024 == 0 and cancelled():
                return None
            note = isinstance(line, cls._Note)
            if not note and not ((not query or query in line.lower()) and entry_at_level(entry, level)):
                continue
            if len(chunks) >= cls._MAX_LINES:
                cut = True
                break
            if note:  # as _display and _prefixes show it
                text = safe_text(line)
                pieces = wrap_log_line(text, width, max_rows=cls._MAX_ROWS_PER_LINE)
            else:
                text = known.get(line)
                if text is None:
                    text = displays.get(line)
                    if text is None:
                        text = displays[line] = display_line(line)
                pieces = wrap_log_line(text, width, first=first, rest=rest, max_rows=cls._MAX_ROWS_PER_LINE)
            rows = cls._row_count(pieces)
            if chunks and total + rows > cls._MAX_ROWS:
                cut = True
                break
            chunks.append((line, pieces, rows))
            kinds.append(line_kind(text, note=note))
            total += rows
        chunks.reverse()
        kinds.reverse()
        return chunks, kinds, displays, cut

    def _apply_rows(self, gen: int, chunks: list, kinds: list, displays: dict, cut: bool) -> None:
        """UI thread. Show what _prepare_rows filtered and wrapped, then the lines written meanwhile,
        unless a newer repaint, or a clear, superseded it."""
        job = self._repaint_job
        lv = self._logview
        if gen != self._repaint_gen or job is None or lv is None or not self._screen_open:
            return
        self._repaint_job = None
        pending, self._repaint_pending = self._repaint_pending, []
        self._display_cache.update(displays)
        self._reset_pane(keep_selection=job.rewrap)
        self._wrap_width = job.width
        # One write for the whole buffer: a call per row took seconds at 20,000 lines.
        self._write_chunks(chunks, job.width, cut=cut, kinds=kinds)
        if job.truncated and not self._pane_truncated:
            self._pane_truncated = True  # older lines had left the view before this repaint
        self._write_lines(pending)
        # The pane was emptied, which leaves it at its top: back to the tail unless the reader has
        # paused to read history; then back to the line they were reading (the oldest kept, if
        # that one was dropped).
        if not self._paused:
            lv.scroll_end(animate=False)
        elif job.top is not None:
            self._pending_top = max(0, job.top - (len(job.pairs) - len(chunks)))
            after_refresh(self, self._scroll_to_pending_top)
        if job.width != self._content_width():
            after_refresh(self, self._rewrap)  # the pane changed width meanwhile
        self._render_log_status()

    def _scroll_to_pending_top(self) -> None:
        """After a re-wrap's layout: put the paused reader's line back at the top of the view."""
        top, self._pending_top = self._pending_top, None
        if top is not None and self._logview is not None:
            self._logview.scroll_to(y=sum(self._shown_rows[:top]), animate=False)

    def _line_at_row(self, row: int) -> int | None:
        """The index in ``_shown_lines`` of the line drawn on pane row ``row``."""
        seen = 0
        for index, rows in enumerate(self._shown_rows):
            seen += rows
            if seen > row:
                return index
        return None


class CortexTrainingLogTUI(App):
    """Entry app: show the job picker, or jump straight to a job dashboard when a
    job_id is supplied."""

    CSS = """
    #jobs { height: 1fr; }
    #listinfo { padding: 0 1; height: auto; }
    #nomatch-box { height: 1fr; scrollbar-gutter: stable; }
    #nomatch { padding: 1 2; color: $warning; height: auto; }
    #jobs > .datatable--fixed { background: $surface; }
    #sources { width: 16; border-right: solid $panel; }
    #job-tabs, #job-tabs ContentSwitcher, #logs, #resource, #logs-body {
        height: 1fr;
    }
    #logview { height: 1fr; padding: 0 1; }
    #log-status { height: auto; padding: 0 1; color: $warning; }
    #summary { color: $text-muted; padding: 0 1; }
    #logfilter { border: tall $panel; }
    #jobfilter { border: tall $panel; }
    #resource-body { height: 1fr; }
    #resource-list { width: 16; border-right: solid $panel; scrollbar-size-vertical: 1; }
    #resource-pane { height: 1fr; padding: 0 1; scrollbar-gutter: stable; scrollbar-size-vertical: 1; }
    #resource-status { color: $text-muted; height: auto; dock: top; }
    #resource-stats { height: auto; margin-top: 1; }
    #resource-charts { height: auto; margin-top: 1; text-wrap: nowrap; text-overflow: clip; }
    #summary-body { padding: 1 1; scrollbar-gutter: stable; }
    #health { height: auto; margin: 1 0; }
    #subjobs-title { height: auto; margin-top: 1; }
    #subjobs { height: auto; max-height: 12; margin: 0 1; }
    LogScreen > .screen--selection {
        background: $block-cursor-blurred-background;
        color: $block-cursor-blurred-foreground;
    }
    ToastRack { dock: top; align: right top; margin-bottom: 0; scrollbar-size-vertical: 0; }
    Toast.-information { padding: 0 1; margin-top: 0; width: auto; max-width: 80%; }
    """
    BINDINGS = [Binding("q", "quit", "Quit", priority=True)]
    ENABLE_COMMAND_PALETTE = False
    # How the client connects, set by `cortex-training tui` (never a secret), so
    # the commands the TUI shows to copy connect the same way; None when unknown
    # (the demo, tests). See views.sign_in_options.
    sign_in: dict | None = None

    def __init__(self, client, job_id=None, *, sub_job_id=None, poll_interval=1.0, now=None):
        """Every job record read through ``client`` and its per-location clients comes back printable
        (:mod:`cortex_training.tui.untrusted`)."""
        super().__init__()
        client = printable_client(client)
        self._client = client
        self._job_id = job_id
        self._sub_job_id = sub_job_id
        self._poll_interval = poll_interval
        self._now = now
        # Shared by every job list this app shows: one client per job location, which
        # list to show (a), and whether the account-wide list was available (and when to ask again).
        self._clients = JobClients(client)
        self._lister = JobLister(client)
        self._heap_frozen = False
        # A left-button press, and whether it has moved since (see on_event).
        self._press_at = None
        self._press_moved = False
        self._copy_note: Notification | None = None  # the last copy's note (see copy_out)
        self._copy_what: str | None = None  # while copy_out copies: what its note calls the text
        # This machine's clipboard takes one copy at a time, in order (see _copy_locally).
        self._copy_lock = threading.Lock()
        self._copies = 0  # copies started so far: the last one is the one the clipboard keeps
        self._press_table = False  # a press on a table, where a hand that moves a cell still clicks
        self._dragged = False  # the last press moved before its release
        self._confine: Widget | None = None  # the scrolling area a drag stays in (see _confine_drag)
        self._confine_header = False  # and the drag started on that table's header, where it stays
        self._footer_down: tuple | None = None  # (footer key, button) pressed (see _footer_click)
        self.title = "cortex training"

    def check_action(self, action: str, parameters) -> bool | None:
        if action == "quit" and isinstance(self.focused, Input):
            return False  # a letter in a search or filter
        return True

    def copy_to_clipboard(self, text: str) -> None:
        """The terminal's clipboard (OSC 52, which some terminals ignore) at once, then this machine's
        (see :mod:`cortex_training.tui.clipboard`) on a worker thread, and a note once that is done: its tool can
        take seconds (xclip waits out its timeout without ``DISPLAY``). Textual's own copies (ctrl+c in a text
        box) come here too."""
        if not text:
            return
        super().copy_to_clipboard(text)
        self._copies += 1
        self._copy_locally(text, self._copy_what or f"{len(text):,} characters", self._copies)

    @work(thread=True, group="clipboard")
    def _copy_locally(self, text: str, what: str, copy: int) -> None:
        """Worker thread: put ``text`` on this machine's clipboard, then show the note
        for it (``what`` names it). Copies take turns, in the order they were made, so
        the clipboard ends with the last; one that a newer copy replaced while it
        waited is skipped, and the newer one's note follows."""
        with self._copy_lock:
            if copy != self._copies:
                return
            local = copy_local(text)
            # Still in turn: the next copy's note can't come before this one's.
            try:
                self.call_from_thread(self._note_copy, copied_note(what, local), local)
            except RuntimeError:  # the app stopped meanwhile
                pass

    _COPY_NOTE_SECONDS = 3
    _SENT_NOTE_SECONDS = 8  # the note when only the terminal's clipboard got it: longer, with a tip

    def copy_out(self, text: str, what: str) -> None:
        """Copy ``text`` and say so once it is copied: ``what`` names it ("3 lines")."""
        text = safe_text(text)
        self._copy_what = what
        try:
            self.copy_to_clipboard(text)
        finally:
            self._copy_what = None
        if not text:
            # Nothing to copy, so no worker runs: the note is the one for a copy only
            # the terminal's clipboard took.
            self._note_copy(copied_note(what, False), False)

    def _note_copy(self, message: str, local: bool = True) -> None:
        note = Notification(message, timeout=self._COPY_NOTE_SECONDS if local else self._SENT_NOTE_SECONDS)
        # One copy's note at a time: the last one's, still showing, would stack beside
        # it. Added now, not as a message, so two copies in one burst of keys replace
        # each other too.
        previous, self._copy_note = self._copy_note, note
        try:
            if previous is not None:
                if previous in self._notifications:
                    self._unnotify(previous, refresh=False)
                self._drop_note(previous)
            self._notifications.add(note)
            self._refresh_notifications()
        except Exception:  # noqa: BLE001 - another Textual keeps its notes otherwise: as a message
            self.post_message(Notify(note))

    def _drop_note(self, note: Notification) -> None:
        """Remove what holds ``note`` on screen. Textual removes a replaced note but not
        the row it sat in (its timer would have), and each such empty row pushed the
        next note one row further down, over the screen."""
        for screen in self.screen_stack:
            for rack in screen.query(ToastRack):
                try:
                    rack.get_child_by_id(ToastRack._toast_id(note)).remove()
                except NoMatches:
                    pass

    def copy_highlight(self, screen: Screen) -> bool:
        """Copy the text the mouse highlighted on ``screen`` (in a table too: see
        :class:`SelectableTable`). False when nothing is."""
        text = selected_text(screen)
        if not text:
            return False
        lines = text.count("\n") + 1
        self.copy_out(text, f"{lines:,} lines" if lines > 1 else f"{len(text):,} characters")
        return True

    def copy_everything(self, screen: Screen, what: str = "everything on the screen") -> None:
        """Copy all ``screen`` holds, as text, with what is scrolled out of view (see
        :func:`cortex_training.tui.clipboard.screen_dump`)."""
        text = screen_dump(screen)
        self.copy_out(text, f"{what} ({text.count(chr(10)) + 1 if text else 0:,} lines)")

    def action_toggle_keys(self) -> None:
        """``?``: show the keys panel, or close it."""
        if not close_keys_panel(self.screen):
            self.action_show_help_panel()

    def validate_sub_title(self, sub_title: str) -> str:
        # The header is one row: the subtitle quotes job ids, reasons and search
        # text, which must neither break the row nor reach the terminal as an escape.
        return one_line(sub_title)

    def job_list_screen(self) -> JobListScreen:
        """A job list screen with this app's client, per-location clients and account-list state."""
        return JobListScreen(self._client, poll_interval=self._poll_interval, now=self._now, clients=self._clients,
                             lister=self._lister)

    # The heap is frozen again this long after the first list is drawn: a pause
    # of its own, once the list is on screen, rather than one with drawing it.
    _FREEZE_HEAP_AFTER_SECONDS = 1.0

    def job_list_drawn(self, screen: Screen) -> None:
        """A job list drew the jobs it loaded. The first time, freeze the heap
        soon after (see :func:`_freeze_heap`): the list's records and rows are
        most of it by then."""
        if self._heap_frozen or self.is_headless:
            return
        self._heap_frozen = True

        def freeze() -> None:
            if self.screen is screen:
                _freeze_heap()
            else:
                # A job's screen opened meanwhile: frozen, it would stay once closed.
                # The list's next load tries again.
                self._heap_frozen = False

        self.set_timer(self._FREEZE_HEAP_AFTER_SECONDS, freeze)

    def job_elsewhere(self, job) -> bool:
        """Whether ``job`` is in a database and schema other than the configured ones."""
        return self._clients.elsewhere(job)

    def on_mount(self) -> None:
        # Before any screen starts a worker: quit must not wait for them.
        loop = asyncio.get_running_loop()
        loop.set_default_executor(_DaemonThreadExecutor())
        # Only in a terminal: a headless run is a test, and a test process runs many apps,
        # whose objects left in reference cycles a freeze would keep.
        if not self.is_headless:
            # What the imports made (most of the heap until a list arrives) is
            # frozen before any screen exists, so the collections while the first
            # list is read and drawn don't visit it.
            _freeze_heap()
        if self._job_id:
            # A job opened by id is in the configured database and schema.
            self.push_screen(
                LogScreen(
                    self._client,
                    self._job_id,
                    sub_job_id=self._sub_job_id,
                    poll_interval=self._poll_interval,
                )
            )
        else:
            self.push_screen(self.job_list_screen())

    async def on_event(self, event: events.Event) -> None:
        # A key, click, scroll or paste means someone is at the screen (a mouse
        # move doesn't): LogScreen pauses its live tail after a long gap.
        if isinstance(event, _INPUT_EVENTS) and not event.is_forwarded:
            self.last_input_at = time.monotonic()
        if not isinstance(event, events.MouseEvent) or event.is_forwarded:
            await super().on_event(event)
            return
        if event.button == 3 and isinstance(event, (events.MouseDown, events.MouseUp)):
            # A right-click does nothing. Textual would take it for a click:
            # one clears the highlight, and moves a table's cursor or picks an item.
            return
        self._footer_wander(event)
        if self._footer_click(event):
            return
        # Whether the pointer moved is where it went, not where a drag confined to an
        # area counts it: held past the edge straight out from the press, it would be
        # back on the press.
        pointer = event
        event = self._confine_drag(event)
        keep = not self._track_press(pointer)
        kept = self.screen.selections
        if isinstance(event, events.MouseUp) and self._dragged:
            # A drag isn't a click: Textual makes one of any press and release on one
            # widget, which toggled the header, picked a list item, switched a tab,
            # or opened or folded the job on a table's row; and it drops the highlight
            # when the button comes up where it went down, as a drag let go straight
            # out past the edge of its area does (see above).
            self._mouse_down_widget = None
            self.screen._mouse_down_offset = None
        screen = self.screen
        blank = isinstance(event, events.MouseDown) and self._on_blank(event)
        if blank:
            # A press on no text (the empty part of a list or a pane): Textual would
            # select every widget in the rows dragged over, the charts beside an
            # empty panel. It selects nothing here, and drops what was highlighted.
            screen.ALLOW_SELECT = False
            screen.clear_selection()
        try:
            await super().on_event(event)
        finally:
            if blank:
                del screen.ALLOW_SELECT
        if keep and kept and not self.screen.selections:
            self.screen.selections = kept  # a click on a note or the footer: the highlight stays

    def _footer_wander(self, event: events.MouseEvent) -> None:
        """A press on a footer key that moves two cells along it, or off its row, is a drag:
        its release runs nothing, wherever it comes back to."""
        if self._footer_down is not None and isinstance(event, events.MouseMove):
            key, button, at = self._footer_down
            if event.screen_y != at.y or abs(event.screen_x - at.x) >= SelectableTable._DRAG_CELLS:
                self._footer_down = (key, 0, at)  # no button runs it now

    def _footer_click(self, event: events.MouseEvent) -> bool:
        """A footer key runs when it is clicked: pressed and released on it, the pointer
        all but still. Textual runs it on the press, so a drag that began on q Quit quit.
        True for the press and the release there, which nothing else sees (so a click on
        the footer keeps a highlight)."""
        if isinstance(event, events.MouseDown):
            key = self._widget_at(event, FooterKey)
            self._footer_down = (key, event.button, event.screen_offset) if key is not None else None
            return key is not None
        if isinstance(event, events.MouseUp) and self._footer_down is not None:
            (key, button, at), self._footer_down = self._footer_down, None
            # As on a table's rows: a hand that moved two cells along the key, or off its row, dragged.
            still = event.screen_y == at.y and abs(event.screen_x - at.x) < SelectableTable._DRAG_CELLS
            if button == 1 and still and self._widget_at(event, FooterKey) is key:
                if getattr(key, "_disabled", False):
                    self.bell()
                else:
                    self.simulate_key(key.key)
            return True
        return False

    def _widget_at(self, event: events.MouseEvent, kinds) -> Widget | None:
        try:
            widget, _region = self.screen.get_widget_at(event.x, event.y)
        except Exception:  # noqa: BLE001 - NoWidget: nothing there
            return None
        return next((node for node in widget.ancestors_with_self if isinstance(node, kinds)), None)

    def _confine_drag(self, event: events.MouseEvent) -> events.MouseEvent:
        """A drag that starts in a scrolling area (the log, a list, the Resources pane, the
        preview) stays in it: past its edges the pointer counts as at the edge, where
        Textual scrolls the area (see :class:`FollowsPointer`). Textual would otherwise
        select every widget between the two ends, a log from its very first line, and a
        press on the log's padding took its offset from the top of the log. A table with
        no scrolling pane around it (the job list's) is an area of its own, its rows one
        and its header another (see :meth:`SelectableTable.drag_region`)."""
        if isinstance(event, events.MouseDown):
            self._confine = None
            self._confine_header = False
            if event.button in (1, 2) and not self.mouse_captured:
                widget, _offset = self.screen.get_widget_and_offset_at(event.x, event.y)
                if widget is not None:
                    self._quiet_keys_panel(widget)
                if widget is not None and widget.allow_select:
                    self._confine = scroll_area(widget)
                    self._confine_header = (isinstance(self._confine, SelectableTable)
                                            and self._confine.on_header(event.y))
        area = self._confine
        header = self._confine_header
        if isinstance(event, events.MouseUp):
            self._confine = None
            self._confine_header = False
        if area is None or not area.is_attached:
            return event
        region = area.drag_region(header=True) if header else scrolling_region(area)
        if not region:
            return event
        x = min(max(event.x, region.x), region.right - 1)
        y = min(max(event.y, region.y), region.bottom - 1)
        if (x, y) == (event.x, event.y):
            return event
        # Held off a table's header, the pointer is on it, at the top of the table, where
        # Textual would scroll the rows under it as the pointer moves up or down.
        delta_y = 0 if header else event.delta_y
        return event.__class__(event.widget, x, y, event.delta_x, delta_y, event.button, event.shift,
                               event.meta, event.ctrl, screen_x=x, screen_y=y, style=event.style)

    @staticmethod
    def _quiet_keys_panel(widget: Widget) -> None:
        """The keys panel's table can't be selected (it is drawn as one Rich table), and a
        drag from the panel selected what was beside it: the panel takes no drags."""
        panel = next((node for node in widget.ancestors_with_self if isinstance(node, HelpPanel)), None)
        if panel is not None and panel.ALLOW_SELECT:
            for node in (panel, *panel.query("*")):
                node.ALLOW_SELECT = False

    def _track_press(self, event: events.MouseEvent) -> bool:
        """Follow a press (left or middle button) to its mouse-up: whether it dragged over
        text, or was a click. False for a click on a note or the footer."""
        if event.button not in (1, 2) and not isinstance(event, events.MouseMove):
            return True
        if isinstance(event, events.MouseDown):
            widget, _offset = self.screen.get_widget_and_offset_at(event.x, event.y)
            self._press_at = event.screen_offset
            self._press_moved = self._dragged = False
            self._press_table = isinstance(widget, SelectableTable)
        elif isinstance(event, events.MouseMove):
            if self._press_at is not None and event.button in (1, 2) and self._moved_off(event.screen_offset):
                self._press_moved = True
        elif isinstance(event, events.MouseUp):
            pressed, self._press_at = self._press_at, None
            self._dragged = pressed is not None and self._press_moved
            if pressed is not None and not self._press_moved:
                if self._widget_at(event, (Footer, Toast)) is not None:  # on the footer (out of its keys) or a note
                    return False
        return True

    def _moved_off(self, at: Offset) -> bool:
        """Whether the pointer, at ``at``, has moved off the press: anywhere at all, but on
        a table only off the press's row or ``_DRAG_CELLS`` cells along it, as on a footer
        key: a hand that moves less on a row meant to click it."""
        pressed = self._press_at
        if not self._press_table:
            return at != pressed
        return at.y != pressed.y or abs(at.x - pressed.x) >= SelectableTable._DRAG_CELLS

    def press_held(self) -> bool:
        """Whether the button pressed on a table is held, and hasn't moved off the press
        (see :meth:`_moved_off`): so far a click, which highlights nothing (see
        :class:`FollowsPointer`)."""
        return self._press_at is not None and self._press_table and not self._press_moved

    def _on_blank(self, event: events.MouseEvent) -> bool:
        """Whether ``event`` (as confined) is on no text: an empty part of a list or a pane
        (not an empty line of the log, which is text with nothing on it)."""
        if self.mouse_captured:
            return False
        widget, offset = self.screen.get_widget_and_offset_at(event.x, event.y)
        return widget is not None and offset is None and widget.is_container and widget.allow_select


_INPUT_EVENTS = (events.Key, events.MouseDown, events.MouseScrollDown, events.MouseScrollUp, events.Paste)
