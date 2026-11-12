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

"""The job list: the screen that opens first, and the table it draws.

It reads each refresh through JobLister (tui/account.py), groups the jobs into LIVE
and HISTORY, folds each job's sub-jobs under it, and opens a job's preview.
"""

from __future__ import annotations
import asyncio
import math
import time
from collections import Counter
from collections import deque
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime
from datetime import timezone
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.content import Content
from textual.coordinate import Coordinate
from textual.css.query import NoMatches
from textual.geometry import Region
from textual.screen import Screen
from textual.strip import Strip
from textual.widgets import DataTable
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Input
from textual.widgets import Static
from textual.worker import get_current_worker
from cortex_training import __version__
from cortex_training.tui import theme
from cortex_training.tui import views
from cortex_training.tui.account import ACCOUNT
from cortex_training.tui.account import ACCOUNT_LIST_CAP
from cortex_training.tui.account import ACTIVE_NEXT
from cortex_training.tui.account import OLDER
from cortex_training.tui.logs import LogScreen
from cortex_training.tui.account import SCHEMA
from cortex_training.tui.account import JobClients
from cortex_training.tui.account import JobLister
from cortex_training.tui.account import JobListError
from cortex_training.tui.account import Listing
from cortex_training.tui.account import ListScope
from cortex_training.tui.account import PageBound
from cortex_training.tui.account import Rest
from cortex_training.tui.account import cut_sorted
from cortex_training.tui.account import empty_text
from cortex_training.tui.account import loaded_text
from cortex_training.tui.account import page_text
from cortex_training.tui.account import scope_name
from cortex_training.tui.account import scope_text
from cortex_training.tui.account import sorted_for_pages
from cortex_training.tui.account import stopped_scope
from cortex_training.tui.account import unavailable_text
from cortex_training.tui.format import cut_cells
from cortex_training.tui.format import format_created_at
from cortex_training.tui.format import one_line
from cortex_training.tui.identity import ASKING
from cortex_training.tui.identity import UNKNOWN
from cortex_training.tui.identity import Identity
from cortex_training.tui.identity import caller_identity
from cortex_training.tui.identity import line_parts
from cortex_training.tui.identity import signs_in
from cortex_training.tui.widgets import FollowsPointer
from cortex_training.tui.widgets import _BLANK
from cortex_training.tui.widgets import _ERROR_LOG
from cortex_training.tui.widgets import log_traceback
from cortex_training.tui.widgets import KEYS_BINDING
from cortex_training.tui.widgets import SelectableTable
from cortex_training.tui.widgets import _FixedCell
from cortex_training.tui.widgets import _cell
from cortex_training.tui.widgets import after_refresh
from cortex_training.tui.widgets import call_on_screen
from cortex_training.tui.widgets import close_keys_panel
from cortex_training.tui.widgets import hanging
from cortex_training.tui.widgets import is_shown
from cortex_training.tui.widgets import wrapped_parts

# The column naming where a job lives, as ``DATABASE.SCHEMA`` (see
# :func:`cortex_training.tui.views.location_label`). One name for both the column's label and the
# key its cell is built under: a row's cells are looked up by label (see :meth:`_job_rows`), so
# the two drifting apart would quietly draw the column empty rather than fail.
_SCHEMA_COLUMN = "Schema"


# The header's chip naming the list on screen: all schemas filled in the brand blue, the
# configured schema's own list in the text color, so which one is shown reads at a glance.
_ACCOUNT_CHIP = f"bold {theme.BACKGROUND} on {theme.ACCENT}"
_SCHEMA_CHIP = f"bold {theme.BACKGROUND} on {theme.TEXT}"


class _ListHeader(Header):
    """The job list's header, its subtitle dim as every screen's but for the words naming the
    list on screen (see :meth:`JobListScreen.scope_chip`), drawn as a chip; put first where the
    line doesn't name the list (all schemas cut short says only how far it got)."""

    def format_title(self) -> Content:
        title, sub_title = self.screen_title, self.screen_sub_title
        chip = self.screen.scope_chip() if sub_title else None
        if chip is None:
            return super().format_title()
        words, style = chip
        at = sub_title.find(words)
        before, after = (sub_title[:at], sub_title[at + len(words):]) if at >= 0 else ("", "  ·  " + sub_title)
        return Content.assemble(title, (" — ", "dim"), (before, "dim"), (f" {words} ", style), (after, "dim"))


class _RuleCell(_FixedCell):
    """A hairline across a column (a LIVE/HISTORY heading's), as wide as the column is
    drawn, so it follows a column dragged wider: drawn, but no text, so a copy of the
    list (``str()`` of each cell) leaves it out."""

    __slots__ = ()

    def __init__(self, style: str) -> None:
        super().__init__("", style)

    def __rich_console__(self, console, options):
        return Text("─" * options.max_width, style=self.style, no_wrap=True).__rich_console__(console, options)


class _LocationCell(_FixedCell):
    """A job's ``DB.SCHEMA`` in the Schema column, copied whole. Where the column is too
    narrow for it, the database name is cut (``TEAM_D….EXPERIMENTS``), so the schema the
    column is named for stays whole while it fits, and two schemas of one database
    still read apart."""

    __slots__ = ("schema",)

    def __init__(self, database: str, schema: str) -> None:
        super().__init__(f"{database}.{schema}", theme.MUTED)
        self.schema = schema

    def __rich_console__(self, console, options):
        room, plain = options.max_width, self.plain
        keep = views.cell_width(self.schema) + 2  # the schema, and "…." before it
        if views.cell_width(plain) > room >= keep:
            database = plain[:len(plain) - len(self.schema) - 1]
            plain = f"{cut_cells(database, room - keep)}….{self.schema}"
        # Narrower still, the end is cut, as any cell's is.
        return Text(plain, style=self.style, overflow="ellipsis", no_wrap=True).__rich_console__(console, options)


def _location_cell(job: dict) -> _FixedCell:
    """The Schema column's cell: the job's ``DB.SCHEMA`` (see :class:`_LocationCell`), else ``—``."""
    parts = views.location_parts(job)
    if parts is None:
        return _cell("—", theme.MUTED)
    return _LocationCell(*(part.replace("\n", " ") for part in parts))  # on one line, as _cell makes a cell


class _CopiedCell(_FixedCell):
    """A cell drawn as one text and copied (``str()``) as another: a relative time is
    drawn "3h ago" and copied as the time itself, which a pasted copy keeps true, and
    a status is drawn with its glyph and copied as its word."""

    __slots__ = ("copied",)

    def __init__(self, cell: _FixedCell, copied: str) -> None:
        super().__init__(cell.plain, cell.style, cell.spans)
        self.copied = copied

    def __str__(self) -> str:
        return self.copied


def _sort_direction(key: str, descending: bool) -> str:
    """Which jobs come first under ``views.sort_job_list(key, descending)``, in words."""
    if key == "status":
        return "running first" if descending else "ended first"
    return "newest first" if descending else "oldest first"


def _sort_part(key: str, descending: bool) -> Content:
    """The sort, at the end of the line above the job list: ``sort: created ↓ newest first``.
    A click on its name or arrow does what s or o does."""
    arrow = "↓" if descending else "↑"
    return Content.from_markup(
        f"[{theme.FAINT}]sort:[/] [b {theme.TEXT}][@click=screen.cycle_sort]{key}[/] "
        f"[@click=screen.toggle_order]{arrow}[/][/] [{theme.MUTED}]{_sort_direction(key, descending)}[/]")


# The widest sort part: each is drawn this wide (padded), so s and o never
# change where the line above the list breaks.
_SORT_PART_CELLS = max(_sort_part(key, descending).cell_length
                       for key in views.SORT_KEYS for descending in (True, False))

# Between the parts of the lines around the job list.
_DOT = Content.assemble("  ", ("·", theme.FAINT), "  ")

# The live statuses the line above the list counts, in this order; a status
# spelled another way counts with the one the theme draws it as.
_CHIP_ORDER = ("running", "initializing", "placing", "pending", "unknown")
_CHIP_OF = {"creating": "placing", "queued": "pending"}


def _chip_rank(key: str) -> tuple:
    return (_CHIP_ORDER.index(key) if key in _CHIP_ORDER else len(_CHIP_ORDER), key)


def _status_chips(live: list, history: int, *, live_so_far: bool = False, unknown_so_far: bool = False,
                  history_so_far: bool = False) -> Content:
    """The counts on the line above the list, each live status as its glyph, count and
    word, then the history: ``● 4 running   ◔ 1 placing   ? 1 unknown  │  21 in history``.
    A count the list is still loading jobs for ends in "so far": every live one
    (``live_so_far``, after the last), the unknown one alone (``unknown_so_far``: the
    active jobs are all listed, but no status filter lists an unknown job), the history
    (``history_so_far``)."""
    so_far = Content.styled(" so far", theme.FAINT)
    # A deployment counts under its status as its row says it (ready, not running); two
    # statuses with one word (placing and initializing) share a chip, drawn as the first.
    counts: Counter = Counter()
    looks: dict = {}
    for job in live:
        key = theme.status_key(job.get("status"))
        key = _CHIP_OF.get(key, key)
        word = views.status_word_for(job, job.get("status")) if views.is_deployment(job) else key
        counts[word] += 1
        looks[word] = min(looks.get(word, key), key, key=_chip_rank)
    chips = []
    for word in sorted(counts, key=lambda w: _chip_rank(looks[w])):
        look = theme.status_look(looks[word])
        tail = so_far if looks[word] == "unknown" and unknown_so_far and not live_so_far else ""
        chips.append(Content.assemble((f"{look.glyph} ", look.color), (f"{counts[word]:,} ", f"bold {theme.TEXT}"),
                                      (word, theme.MUTED), tail))
    chips = chips or [Content.styled("no live jobs", theme.MUTED)]
    ended = Content.assemble("  ", ("│", theme.FAINT), "  ", (f"{history:,} ", f"bold {theme.TEXT}"),
                             ("in history", theme.MUTED))
    return (Content("   ").join(chips) + (so_far if live_so_far else Content("")) + ended
            + (so_far if history_so_far else Content("")))


@dataclass(frozen=True)
class _ListLayout:
    """The job table's columns for one terminal width, sort and set of locations."""

    columns: tuple  # (label, width) in display order

    @property
    def labels(self) -> tuple:
        return tuple(label for label, _width in self.columns)


class _JobRows:
    """One job's table rows as last built, and what they were built from."""

    # Slots: one of these per listed job, kept between renders.
    __slots__ = ("job", "stale_at", "cells", "sub_cells", "sub_keys", "labels", "expanded", "row", "subs")

    def __init__(self, job: dict, stale_at: float, cells: dict) -> None:
        self.job = job  # the record (compared by identity: a refresh keeps an unchanged record)
        # The clock's epoch seconds at which one of its relative times ("12m ago",
        # a live job's duration) would read differently: built again from then.
        self.stale_at = stale_at
        self.cells = cells  # column label -> cell: the job's row, but for its Job ID
        self.sub_cells: tuple | None = None  # ((row key suffix, {label: cell}), ...): its sub-job rows (and note)
        self.sub_keys: tuple = ()  # ((job id, sub-job id) or None for the note, ...) beside sub_cells
        # The columns (labels, in order) row and subs are in; their widths don't change the cells.
        self.labels: tuple | None = None
        self.expanded: bool | None = None  # whether row and subs are for the job unfolded
        self.row: tuple = ()
        self.subs: tuple = ()  # ((row key suffix, cells), ...); none while folded


class _Archive:
    """Every job a search covers on one list (``kind``: the account-wide one, not narrowed to a
    location, or the configured schema's): its first page, then the rest of the list read
    behind it (see :meth:`JobLister.read_rest`), by job key. ``version`` counts its changes."""

    __slots__ = ("kind", "jobs", "done", "capped", "failed", "version", "reached")

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.jobs: dict[tuple, dict] = {}
        self.done = False  # the read behind the list has ended (read to its end, capped, or failed)
        self.capped = False  # it stopped at ARCHIVE_CAP with more left
        self.failed = ""  # why it stopped short
        self.version = 0
        # The creation time the read behind the list has reached, newest first: every job created
        # since is held (the jobs of later pages a refresh read can be held past it, with gaps).
        self.reached: datetime | None = None

    def add(self, jobs, key, *, fresher: bool = False) -> None:
        """Hold ``jobs`` (``key`` gives a job's key); ``fresher``: they replace the records held,
        as a refresh's do. The read behind the list adds only jobs it doesn't hold yet."""
        held = self.jobs
        for job in jobs:
            job_key = key(job)
            if fresher or job_key not in held:
                held[job_key] = job
        self.version += 1

    @property
    def complete(self) -> bool:
        """Every job of the list is held, its first ever made included."""
        return self.done and not self.capped and not self.failed


class _JobTable(SelectableTable):
    """The selectable job table with fold-aware navigation, and a bar after each column's
    name that a drag moves, making the column wider or narrower (see :meth:`_bars`)."""

    BINDINGS = [
        Binding("home", "first_job", "First job", show=False),
        Binding("end", "last_job", "Last job", show=False),
        Binding("right", "cursor_right", "Unfold sub-jobs, else scroll right", show=False),
        Binding("left", "cursor_left", "Fold sub-jobs, else scroll left", show=False),
    ]
    _MARKERS = ("▸ ", "▾ ")
    _MIN_WIDTH = 3  # the narrowest a drag leaves a column: a character or two and "…"
    _BAR = "┃"  # heavy, so it reads as a handle
    # A drag on a bar under way: (its column's label, the column's width, the press's screen x).
    _grab: tuple | None = None
    _hovered: str | None = None  # the label of the column whose bar the pointer is on

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
        """A click on a job's ▸/▾ marker unfolds or folds it, and a double-click on a column's
        bar gives it back the width the list picks; any other click is the table's own, unless swallowed."""
        if self.swallowed(event):
            return
        if event.chain == 2:
            edge = self._edge_at(event)
            if edge is not None:
                self.screen.reset_column(edge[0])
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

    # ─── column widths ───────────────────────────────────────────────────────

    def on_mouse_down(self, event: events.MouseDown) -> None:
        """A press on a column's bar grabs it: the bar, the column's right edge, follows the pointer."""
        edge = self._edge_at(event) if event.button == 1 else None
        if edge is None:
            return
        self._grab = (*edge, event.screen_x)
        self.capture_mouse()
        # Textual started a highlight at the press: a drag that resizes highlights nothing.
        self.screen.clear_selection()
        self._redraw_names()  # the bar held is drawn in the accent

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._grab is not None:
            label, width, x = self._grab
            self.screen.resize_column(label, max(self._MIN_WIDTH, width + event.screen_x - x))
            return
        edge = self._edge_at(event)
        self._hover(edge[0] if edge is not None else None)

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._grab is not None:
            self._let_go()

    def on_mouse_release(self, event: events.MouseRelease) -> None:
        if self._grab is not None:  # the capture ended without a release here (another screen opened)
            self._let_go()

    def on_leave(self, event: events.Leave) -> None:
        if self._grab is None:
            self._hover(None)

    def _let_go(self) -> None:
        self._grab = None
        self.release_mouse()
        self._hover(None)
        self._redraw_names()
        self.screen.column_resized()

    def _hover(self, label: str | None) -> None:
        """The pointer on ``label``'s bar (None: on none): the bar brighter, and the pointer a
        sideways arrow where the terminal draws pointer shapes."""
        if label != self._hovered:
            self._hovered = label
            self._redraw_names()
        shape = "ew-resize" if label is not None else "default"
        if self.styles.pointer != shape:
            self.styles.pointer = shape

    def _redraw_names(self) -> None:
        """Draw the column names again (their line stays at the top as the rows scroll)."""
        self.refresh(Region(0, 0, self.size.width, self.header_height))

    def _edge_at(self, event: events.MouseEvent) -> tuple | None:
        """``(label, width)`` of the column whose bar ``event`` is on, or on a blank cell beside
        (so a press a cell off still takes it), else None: a press on a name highlights it, as
        a press on any text does."""
        if not self.on_header(event.screen_y):
            return None
        offset = event.get_content_offset(self)
        if offset is None:
            return None
        x = offset.x
        bars = dict(self._bars(self.scroll_offset.x))
        for at in (x, x - 1, x + 1):  # the bar itself, else the one beside a blank cell
            column = bars.get(at)
            if column is not None and (at == x or self._blank_name_cell(x)):
                return column.key.value, column.width
        return None

    def _blank_name_cell(self, x: int) -> bool:
        """Whether the column names draw no text at content offset ``x`` (padding, or past a short name)."""
        scroll_x = self.scroll_offset.x
        line = self._render_line(0, scroll_x, scroll_x + self.size.width, self.rich_style)
        return not line.crop(x, x + 1).text.strip()

    def _bars(self, scroll_x: int) -> list:
        """``(x, column)`` for each column's bar drawn on the column names, scrolled ``scroll_x``
        cells sideways: in the last cell of the column (its padding after the name), at the
        column's right edge, which a drag on the bar moves. A fixed column's is always in
        view; another's only where it is scrolled clear of them."""
        columns = self.ordered_columns
        left = self._row_label_column_width
        fixed = left + sum(column.get_render_width(self) for column in columns[:self.fixed_columns])
        bars = []
        for index, column in enumerate(columns):
            left += column.get_render_width(self)
            x = left - 1 if index < self.fixed_columns else left - 1 - scroll_x
            if index < self.fixed_columns or fixed <= x < self.size.width:
                bars.append((x, column))
        return bars

    def _render_line(self, y: int, x1: int, x2: int, base_style: Style) -> Strip:
        """A line of the table; on the column names, a bar at each column's right edge. Drawn
        over the padding there, so the text (what a highlight copies) is as it was."""
        strip = super()._render_line(y, x1, x2, base_style)
        if y or not self.show_header:
            return strip
        held = self._grab[0] if self._grab is not None else None
        for x, column in self._bars(x1):
            if x >= strip.cell_length:
                continue
            under = next((segment.style for segment in strip.crop(x, x + 1) if segment.style), None)
            label = column.key.value
            color = theme.ACCENT if label == held else theme.MUTED if label == self._hovered else theme.FAINT
            bar = Segment(self._BAR, (under or Style()) + Style(color=color, bold=False))
            strip = Strip.join([strip.crop(0, x), Strip([bar], 1), strip.crop(x + 1)])
        return strip

    def set_widths(self, widths: dict) -> None:
        """Columns ``widths`` (label -> cells) wide, in place: the rows, the cursor and the
        scroll stay (``clear(columns=True)`` would build the table again from the top)."""
        changed = False
        for column in self.ordered_columns:
            width = widths.get(column.key.value, column.width)
            if column.width != width:
                column.width, changed = width, True
        if changed:
            # As DataTable does when a cell changes: its drawn lines are drawn again, at the new widths.
            self._clear_caches()
            self._update_count += 1
            self._require_update_dimensions = True
            self.refresh()

    def replace_rows(self, rows: list) -> None:
        """Show ``rows`` (``(key, cells)`` pairs, in order) in place of the table's rows."""
        self.clear()
        for key, cells in rows:
            self.add_row(*cells, key=key)

    def patch_rows(self, rows: list, shown: tuple) -> None:
        """Change the table from ``shown`` to ``rows`` (``(key, cells)`` pairs, in order) in place,
        without clearing it (which scrolls it home): the rows gone are removed, the cells that
        changed updated, the new rows added, then all put in order. Each removal is linear in the
        rows, so this suits a list that grows."""
        old = dict(shown)
        for key in old.keys() - {key for key, _cells in rows}:
            self.remove_row(key)
        columns = [column.key for column in self.ordered_columns]
        for key, cells in rows:
            was = old.get(key)
            if was is None:
                self.add_row(*cells, key=key)
            elif was is not cells:
                for column, before, after in zip(columns, was, cells):
                    if before is not after:
                        self.update_cell(key, column, after)
        if [row.key.value for row in self.ordered_rows] != [key for key, _cells in rows]:
            # Each row's Job ID cell is its own object: its identity gives the row's place.
            at = [column.value for column in columns].index("Job ID")
            position = {id(cells[at]): index for index, (_key, cells) in enumerate(rows)}
            self.sort("Job ID", key=lambda name: position[id(name)])


class JobListScreen(FollowsPointer, Screen):
    """Every job in the retention window: live first, then history, with sub-job rows.

    ``a`` switches between the configured schema's jobs and the account-wide list
    (see :class:`cortex_training.tui.account.JobLister`), which shows its pages as they
    come where no list of its own shows yet (:meth:`_on_partial`); search takes ids, locations
    and ``field<op>value`` filters (see :func:`cortex_training.tui.views.parse_query`).

    A job with sub-jobs shows ▸ (folded) or ▾ before its id. Every live job
    starts unfolded; history starts folded.
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
        Binding("f", "full_list('all')", "All jobs"),
        Binding("f", "full_list('newest')", "Page 1"),
        Binding("[", "page('newer')", "Newer jobs"),
        Binding("]", "page('older')", "Older jobs"),
        Binding("d", "show_view('deployments')", "Deployments"),
        Binding("d", "show_view('jobs')", "Jobs"),
        ("s", "cycle_sort", "Sort"),
        ("o", "toggle_order", "Order"),
        ("e", "expand_all", "Expand/collapse all"),
        ("r", "refresh", "Refresh"),
    ]
    _REFRESH_SECONDS = 60
    # Job ID is the ▸/▾ marker and the whole id. Status is its glyph and word
    # (◐ INITIALIZING is the widest); the times are relative ("3h ago"), as the
    # preview has them in full.
    _COLUMNS = [("Job ID", 38), ("Status", 14), ("Type", 13), ("Created", 8), ("Duration", 8),
                ("Owner", 12), ("Model", 18), ("GPUs", 9), ("Updated", 8)]
    _FIXED_COLUMNS = 1  # the job id (with its ▸/▾ marker) stays in view while the table scrolls sideways
    _FIRST_COLUMNS = ("Job ID", "Status", "Type", "Created", "Duration", "Owner")
    _NAME_MIN = 16
    _SEARCH_DEBOUNCE_SECONDS = 0.12
    # A page more of the list changes its rows in place up to this many rows gone (each
    # removal is linear in the rows); beyond, the table is filled again.
    _PATCH_REMOVALS = 64
    _DECLINED_CELLS = 60  # the header's reason a could not show all schemas; the notice has it in full
    _LOCATION_AT = [label for label, _width in _COLUMNS].index("GPUs") + 1
    # The Schema column is as wide as its widest DB.SCHEMA, at least the first and, where
    # the screen has no room spare, at most the second (see _layout_for).
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
        # Job key -> the record of each job the table draws: the list's own, or a search's match
        # from the jobs read behind it (see _job_for).
        self._drawn: dict[tuple, dict] = {}
        self._jobs_by_key: dict[tuple, dict] = {}
        self._sub_by_item: dict[str, tuple] = {}
        self._all_jobs: list = []
        # A schema's whole list cut to its newest jobs (see cut_whole_listing), and whether f
        # asked for every job of it: kept across refreshes, until another list is shown.
        self._cut_listing: Listing | None = None
        self._full_list = False
        self._deployments = False  # d: the Deployments view, else training and sampling jobs
        self._filter = ""
        self._sort = "created"
        self._descending = True
        self._expanded: set = set()
        self._collapsed: set = set()  # jobs folded by the reader (or by an e that holds no more)
        # e, per view (deployments or not): every job folded (True) or unfolded (False); None before
        self._all_folded: dict[bool, bool | None] = {False: None, True: None}
        self._last_refreshed = None
        self._loaded_at: float | None = None  # monotonic time of the last good load
        self._failed_at: str | None = None  # HH:MM:SS of the last failed load, while it stands
        self._refreshing = False
        self._refresh_started: float | None = None  # monotonic time the refresh in flight began
        # Each load's number: the answers (and pages) of a load superseded since are dropped.
        self._load_gen = 0
        self._load_worker = None  # the load in flight
        self._progressive = False  # the load in flight shows its pages as they come
        self._partial_gen: int | None = None  # the load whose first pages the list shows
        # The failure that stands is the account-wide list's, after its first pages showed
        # (and the list on screen is those pages), not a later refresh's.
        self._pages_failed = False
        # a stopped reading the pages on screen, and the list it asked for hasn't answered yet.
        self._partial_stays = False
        self._list_error: str | None = None  # last failed load, on one line; cleared by a good one
        self._list_refused = False  # that failure was the account-wide list refusing this caller (403/405)
        self._scope: ListScope | None = None  # what the list on screen covers
        self._asked = self._account_wanted()
        # Why the last a couldn't show all schemas, for the header until the next a.
        self._declined: str | None = None
        self._switching = False  # a was pressed and the list it asked for hasn't answered yet
        # The (database, schema) the account-wide list on screen was read under, the one the load in
        # flight asked for, and whether that load is a narrowing (so its pages show as they come).
        self._pushdown: tuple | None = None
        self._load_location: tuple | None = None
        self._narrowing = False
        # ] and [: the pages past the first that ] went to, the last one shown (empty on the
        # first page); where the page after the one on screen starts (None on the last); the page
        # the load in flight reads; and whether that load is a page change (its jobs show as they
        # come).
        self._pages: list[PageBound] = []
        self._next_page: PageBound | None = None
        self._load_page: PageBound | None = None
        self._paging = False
        # Each page of this list read whole so far, by its number: (the bound it was read under, its
        # listing, the monotonic time it was read, and that time as the header says it), so ] and [
        # show a page read before at once.
        self._page_cache: dict[int, tuple[PageBound | None, Listing, float, str]] = {}
        # What ] put back if its page fails before any of its jobs show: (the pages, the next page).
        self._page_undo: tuple | None = None
        # The jobs a search covers (see _Archive): the list's first page and the rest of its 6
        # months, read behind it a request at a time once that page is whole; the read's number,
        # so the read of another list is dropped; when a search last drew what it had read so
        # far; and the search's corpus as last built, (archive version, list on screen, jobs).
        self._archive: _Archive | None = None
        self._archive_gen = 0
        self._archive_reading = False
        self._archive_drawn_at = 0.0
        self._corpus_cache: tuple | None = None
        self._sorted_cache: tuple | None = None  # (archive, its version, sorted_for_pages of its jobs)
        # The rows of the page ] shows next, built ahead while nothing else happens: the jobs left to
        # build, the ones built (kept with the pages' own), and the page they are for.
        self._prebuild: deque = deque()
        self._prebuilt: set = set()
        self._prebuild_for: PageBound | None = None
        self._prebuilding = False
        # The page of a search's matches shown (1,000 in history each), of how many; how many
        # jobs matched, and how many the search covered.
        self._match_page = 1
        self._match_pages = 1
        self._match_count = 0
        self._match_searched = 0
        self._search_timer = None
        self._layout: _ListLayout | None = None  # the table's columns, once they exist
        # The widths the reader dragged columns to (label -> cells): kept over every list,
        # refresh and terminal size, until a double-click on the column's bar.
        self._widths: dict[str, int] = {}
        self._notice: Content | None = None  # the stale-list notice, before wrapping
        # The line(s) under the header, before wrapping: a message, or (start parts, the sort).
        self._info: Content | tuple | None = None
        self._nomatch: str | None = None  # the no-match message (plain text), before wrapping
        self._rows: dict[tuple, _JobRows] = {}
        self._hay: dict[tuple, tuple] = {}  # job key -> (record, search_haystack)
        self._resolved: tuple | None = None  # (search text, the jobs list it was resolved against, query)
        self._headers: dict[str, tuple] = {}  # LIVE/HISTORY -> ((count, so_far, column labels), cells)
        self._keys: dict[tuple, str] = {}  # job key -> its row's table key
        self._shown: tuple | None = None  # what the table holds: (column labels, ((row key, cells), ...))
        self._placed_key: str | None = None  # the row the last render put the cursor on
        self._reader_moved = False  # the reader moved the cursor since the list (or a sort) placed it

    def _job_key(self, job: dict) -> tuple:
        return views.job_key(job, self._clients.home)

    def compose(self) -> ComposeResult:
        yield _ListHeader(icon=theme.BRAND_ICON)
        yield Static("", id="pageline")
        yield Static("", id="listwho")
        yield Static("", id="listinfo")
        yield Static("", id="listnotice")
        yield _JobTable(id="jobs", cursor_type="row", zebra_stripes=False, fixed_columns=self._FIXED_COLUMNS)
        with VerticalScroll(id="nomatch-box"):
            yield Static("", id="nomatch")
        # One row; its help shows under it while it has focus, so the rows are the list's otherwise.
        yield Input(
            placeholder="/ search  ids, owners, models, locations · status:running gpus>=8 gpu:H200 · Esc clears",
            id="jobfilter",
        )
        yield Static("", id="searchhelp")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#nomatch", Static).display = False
        self.query_one("#nomatch-box", VerticalScroll).display = False
        self.query_one("#listnotice", Static).display = False
        self.query_one("#pageline", Static).display = False
        self._set_columns(self.query_one("#jobs", DataTable), self._layout_for(0))
        self._paint_help()
        self._paint_who()
        self._refresh_jobs()
        # DOMNode reserves _auto_refresh, so this timer has another name.
        self.set_interval(self._REFRESH_SECONDS, self._refresh_tick)

    def on_screen_resume(self) -> None:
        self._update_subtitle()
        self._paint_who()  # the user and role may have been read while a job's screen was on top
        self._render_jobs()  # with a status the preview read; a list as it was isn't drawn again
        if self._loaded_at is not None and time.monotonic() - self._loaded_at >= self._REFRESH_SECONDS:
            self._refresh_jobs()

    def on_resize(self, event) -> None:
        after_refresh(self, self._fit_to_width)

    def _fit_to_width(self) -> None:
        """After a resize: re-wrap the lines around the list, and re-render if the columns change."""
        self._paint_who()
        self._paint_info()
        self._paint_notice()
        self._paint_nomatch()
        self._paint_help()
        self._fit_columns()

    def _fit_columns(self) -> None:
        """Draw the list again if its columns would change: the terminal's width, or a column's, did."""
        if self._layout != self._layout_for(self._location_column_width(self._in_view(self._all_jobs))):
            self._render_jobs()

    def resize_column(self, label: str, width: int) -> None:
        """A drag on a column's bar: ``label``'s column ``width`` cells wide from now on. Only it
        changes while the drag lasts, so its edge stays under the pointer; the other columns fit
        around it once the drag ends (see :meth:`column_resized`)."""
        self._widths[label] = width
        if self._layout is not None:
            columns = tuple((name, width if name == label else was) for name, was in self._layout.columns)
            self._set_columns(self.query_one("#jobs", DataTable), _ListLayout(columns))

    def column_resized(self) -> None:
        """A drag on a column's bar ended: the columns fit the screen around the widths set."""
        self._fit_columns()

    def reset_column(self, label: str) -> None:
        """``label``'s column back to the width the list picks for it (a double-click on its bar)."""
        if self._widths.pop(label, None) is not None:
            self._fit_columns()

    def on_descendant_focus(self, event) -> None:
        self._paint_help()

    def on_descendant_blur(self, event) -> None:
        self._paint_help()

    def _paint_help(self) -> None:
        """The search help under the box, only while the box has focus (its rows are
        the list's otherwise), broken between its phrases and wrapped under its start."""
        try:
            widget = self.query_one("#searchhelp", Static)
        except NoMatches:  # not composed yet
            return
        widget.display = self._search_focused()
        if not widget.display:
            return

        def phrase(label: str, text: str) -> Content:
            return Content.assemble((f"{label} ", theme.FAINT), (text, theme.MUTED))

        # Broken only between its phrases, so an example stays beside what it shows.
        parts = [phrase("fields", " ".join(views.SEARCH_FIELDS)), phrase("ops", ":  =  !=  >  >=  <  <="),
                 phrase("any other key is a job or config member", "seed=42"),
                 phrase("nested", "training_config.optimizer.lr<1e-4")]
        width = widget.content_size.width or (self.size.width or self.app.size.width) - 3
        lines = wrapped_parts(parts, width, 2, _DOT)
        widget.update(lines[2:])  # the first line starts at the edge, the others under it

    def _layout_for(self, location_width: int) -> _ListLayout:
        """Fit columns, narrowing the job id while keeping the sorted time visible. A column the
        reader dragged keeps the width they gave it. The Schema column (``location_width``: its
        widest name) is at most ``_LOCATION_WIDTHS[1]`` wide, and wider only into room the other
        columns leave, so a long name is cut only where the screen has no room for it."""
        table = self.query_one("#jobs", DataTable)
        pad = 2 * table.cell_padding
        room = (self.size.width or self.app.size.width) - table.styles.scrollbar_size_vertical
        columns = list(self._COLUMNS)
        if location_width:
            columns.insert(self._LOCATION_AT, (_SCHEMA_COLUMN, min(location_width, self._LOCATION_WIDTHS[1])))
        sized = self._widths
        columns = [(label, sized.get(label, width)) for label, width in columns]
        name_base = dict(self._COLUMNS)["Job ID"]
        spare = room - pad - sum(width + pad for label, width in columns if label != "Job ID")
        if "Job ID" in sized:
            name = sized["Job ID"]
        elif spare >= name_base:
            name = name_base
        else:
            first = set(self._FIRST_COLUMNS) - {"Job ID"}
            needed = sum(width + pad for label, width in columns if label in first)
            if room - pad - needed < self._NAME_MIN:  # the last of them can't fit: the job id takes its room
                needed -= dict(columns)[self._FIRST_COLUMNS[-1]] + pad
            name = max(self._NAME_MIN, min(name_base, room - pad - needed))
        columns = [(label, name if label == "Job ID" else width) for label, width in columns]
        labels = [label for label, _width in columns]
        if location_width and _SCHEMA_COLUMN not in sized:
            at = labels.index(_SCHEMA_COLUMN)
            spare = room - sum(width + pad for _label, width in columns)
            width = columns[at][1]
            columns[at] = (_SCHEMA_COLUMN, width + max(0, min(spare, location_width - width)))
        if self._sort == "updated":
            ends = sum(width + pad for _label, width in columns[: labels.index("Updated") + 1])
            if ends > room:  # out of view: put it where Created is (each keeps its own width)
                at_created, at_updated = labels.index("Created"), labels.index("Updated")
                columns[at_created], columns[at_updated] = columns[at_updated], columns[at_created]
        return _ListLayout(tuple(columns))

    def _set_columns(self, table: DataTable, layout: _ListLayout) -> None:
        """Set the table's columns up for ``layout`` when they aren't the ones it has: in place
        where only their widths changed, else from scratch (clearing it)."""
        if layout == self._layout:
            return
        resized = self._layout is not None and layout.labels == self._layout.labels
        self._layout = layout
        if resized:
            table.set_widths(dict(layout.columns))
            return
        table.clear(columns=True)
        for label, width in layout.columns:
            table.add_column(label, width=width, key=label)

    def _in_view(self, jobs: list) -> list:
        """The jobs of the view on screen: deployments (d) or every other job."""
        return [job for job in jobs if views.is_deployment(job) == self._deployments]

    def _other_view_text(self, count: int) -> str:
        noun = "job" if self._deployments else "deployment"
        return f"{count:,} {noun}{'' if count == 1 else 's'} · d shows {'it' if count == 1 else 'them'}"

    def _location_column_width(self, jobs: list) -> int:
        """The width ``jobs``' widest DB.SCHEMA takes, at least ``_LOCATION_WIDTHS[0]``: the
        Schema column's, where the screen has room (see :meth:`_layout_for`); 0 (hidden) when
        they all share one location.

        A narrowed list (see :func:`cortex_training.tui.views.pushdown_location`) therefore hides
        it, and the header names the location instead (see :func:`_scope_text`)."""
        labels = {views.location_label(job) for job in jobs}
        if len(labels) <= 1:
            return 0
        return max(self._LOCATION_WIDTHS[0], max(views.cell_width(label) for label in labels))

    def _clock(self) -> datetime:
        return self._now or datetime.now(timezone.utc)

    def _update_subtitle(self) -> None:
        self._paint_page_line()  # this screen's own, so drawn while another is on top too
        if self.app.screen is not self:
            return  # a late refresh must not retitle the screen on top
        switching = self._switching_text()
        if switching:
            # a was pressed and the other list is still being read: say so first,
            # rather than the scope of the list it replaces.
            if self._partial_stays:
                # The pages a stopped reading, of whichever list they were.
                which = "all-schemas" if self._scope.kind == ACCOUNT else "current-schema"
                shown = f"showing the partial {which} list"
            else:
                shown = f"list from {self._last_refreshed}" if self._last_refreshed else ""
            bits = ["deployments" if self._deployments else "", switching, shown]
            self.app.sub_title = "  ·  ".join(bit for bit in bits if bit)
            return
        if self._scope is not None and self._scope.loading:
            # The list's first pages: the line under the header says how far it is, in place of
            # when it was refreshed.
            bits = ("deployments" if self._deployments else "", self._scope_text())
            self.app.sub_title = "  ·  ".join(bit for bit in bits if bit)
            return
        if self._refreshing:
            refresh_status = "refreshing…"
            if self._last_refreshed:
                refresh_status += f" (last refreshed {self._last_refreshed})"
        elif self._pages_failed and self._list_error and self._failed_at:
            refresh_status = f"load failed {self._failed_at}"  # its first pages show; the notice says what failed
        elif self._list_error and self._failed_at and self._loaded_at is None:
            refresh_status = f"load failed {self._failed_at}"  # no list yet: the error is in its place
        elif self._list_error and self._failed_at:
            refresh_status = f"refresh failed {self._failed_at}"
            if self._last_refreshed:
                refresh_status += f" (list from {self._last_refreshed[:5]})"
        else:
            refresh_status = f"last refreshed {self._last_refreshed or '—'}"
        bits = ["deployments" if self._deployments else "", refresh_status, self._scope_text()]
        self.app.sub_title = "  ·  ".join(bit for bit in bits if bit)

    def _switching_text(self) -> str:
        """While the list ``a``, or the search's own narrowing, asked for is read in place of the
        one on screen: which; else ""."""
        if not self._refreshing:
            return ""
        if self._narrowing and not self._switching:
            where = views.location_filter_text(self._load_location)
            return f"narrowing to {where}…" if where else "reading every schema…"
        if not self._switching:
            return ""
        if self._account_wanted():
            return "loading all schemas…"
        return f"loading the current schema, {self._schema_label()}…"

    def scope_chip(self) -> tuple | None:
        """``(words, style)``: the words of the header naming the list on screen (``all schemas``,
        ``Current schema: DB.SCHEMA``) and the chip the header draws them as; None before the
        first list, and while ``a`` reads the other one (the header says so instead)."""
        if self._scope is None or (self._refreshing and self._switching):
            return None
        declined = bool(self._declined) and self._scope.kind == SCHEMA
        words = one_line(scope_name(self._scope, location=self._schema_label(), declined=declined))
        return words, _ACCOUNT_CHIP if self._scope.kind == ACCOUNT else _SCHEMA_CHIP

    def _page_line_text(self) -> str:
        """The line under the header: the page on screen (see :func:`page_text`), or the one ] or [
        asked for while nothing of it shows yet; "" for a list read whole on one page."""
        if self._paging and self._refreshing and not (self._switching or self._narrowing):
            return f"page {len(self._pages) + 1} · loading…"
        line = page_text(self._scope) if self._scope is not None else ""
        if (line or self._match_pages > 1) and self._searching() and self._search_archive() is not None:
            return self._match_line()  # what the search covered, rather than the page it searched from
        return line

    _PAGE_ARROWS = {"newer": "◂ newer", "older": "older ▸"}

    def _page_arrow(self, which: str) -> tuple:
        """``◂ newer`` or ``older ▸`` (``which``) beside the page line: a link that pages as ``[`` or
        ``]`` does, drawn as every clickable word of the TUI is (underlined, lit under the pointer);
        dim, and no link, where there is no page that way."""
        label = self._PAGE_ARROWS[which]
        if self.check_action("page", (which,)):
            return label, f"{theme.TEXT} @click=screen.page('{which}')"
        return label, theme.FAINT

    def _paint_page_line(self) -> None:
        try:
            widget = self.query_one("#pageline", Static)
        except NoMatches:  # not composed yet
            return
        text = self._page_line_text()
        widget.display = bool(text)
        if text:
            # "page N" stands out; what the page holds stays as quiet as the header.
            head = " ".join(text.split(" ", 2)[:2]).rstrip(":")
            widget.update(Content.assemble(self._page_arrow("newer"), "   ", (head, f"bold {theme.TEXT}"),
                                           text[len(head):], "   ", self._page_arrow("older")))

    def _scope_text(self) -> str:
        """What the list on screen covers ("all schemas", ...); "" before the first list."""
        if self._scope is None:
            return ""
        if self._declined and self._scope.kind == SCHEMA:
            # Why a couldn't show all schemas, and still what the list holds when it is cut short.
            return scope_text(self._scope, location=self._schema_label(), declined=self._declined)
        if self._scope.narrowed:
            # The rows share one location because the walk asked for it, not because the account
            # holds no other, so the header must not claim it covered every schema.
            only = ""
        else:
            labels = {views.location_label(job) for job in self._in_view(self._all_jobs)}
            only = next(iter(labels)) if len(labels) == 1 else ""
        return scope_text(self._scope, location=self._schema_label(), only=only)

    def _refresh_jobs(self, *, manual: bool = False, quiet: bool = False) -> None:
        """Refresh; ``manual`` permits reprobe and ``quiet`` suppresses an in-flight notice."""
        if self._refreshing:
            if manual and not quiet:
                waited = int(time.monotonic() - (self._refresh_started or time.monotonic()))
                if self._partial_gen == self._load_gen:  # nothing is read again after: r is for then
                    # The header's words: the older jobs, or the older active jobs once the newest are in.
                    which = "older active jobs" if self._scope.loading == ACTIVE_NEXT else "older jobs"
                    self.notify(f"Still loading {which} ({waited}s); r works once it's done.", markup=False)
                else:
                    self.notify(f"Still waiting for the last refresh ({waited}s); r asks again once it answers.",
                                markup=False)
            return
        self._refreshing = True
        self._refresh_started = time.monotonic()
        account = self._account_wanted()
        # Set before the header is drawn: it names the location this load asks for.
        self._load_location = self._pushdown_for() if account else None
        self._load_page = self._page_bound()
        self._update_subtitle()
        # Pages show as they come only where no list of this kind shows yet (the first
        # load, or after a), or where this load narrows to a location the list on screen
        # wasn't read under: a refresh keeps the whole list it has until the next is whole.
        # Both lists are read a page at a time; only the account-wide one narrows.
        # So does a page that ] or [ asked for.
        self._progressive = (self._loaded_at is None or self._switching or self._paging
                             or (account and self._narrowing))
        self._load_gen += 1
        self._load_worker = self._load_jobs(manual, account, account and self._asked, self._load_gen,
                                            self._progressive, self._load_location, self._load_page)

    def _supersede(self) -> None:
        """Drop the load in flight (its pages come no more; what it answers is dropped), so another can start."""
        self._load_gen += 1
        if self._load_worker is not None:
            self._load_worker.cancel()  # it stops before its next page
        self._load_worker = None
        self._refreshing = False
        self._paging = False  # a page change it was is over: the next load is a plain refresh
        if self._scope is not None and self._scope.loading:
            # Its first pages stay until the next list answers, no longer marked as still loading.
            self._scope = stopped_scope(self._scope)
            self._partial_stays = True
            self._render_jobs(steady=True)

    def _account_wanted(self) -> bool:
        """Whether the list to show is the account-wide one (a), not the configured schema's."""
        return self._lister.account_wide

    def action_full_list(self, which: str) -> None:
        """``f``: every job of a schema list that came whole (``which`` "all"), or again its
        newest :data:`ACCOUNT_LIST_CAP` and every active job ("newest"). Drawing tens of thousands of rows takes
        seconds, so the header says so first."""
        listing = self._cut_listing
        if listing is None or (which == "all") == self._full_list:
            return
        self._full_list = which == "all"
        self.refresh_bindings()  # the footer's f names the other way now
        if self._full_list:
            self._info = Content.from_markup(f"  [dim]drawing all {len(listing.everything):,} jobs…[/dim]")
            self._paint_info()
            self.call_after_refresh(self._draw_full_list, listing)
        else:
            self._show_listing(listing)

    def _draw_full_list(self, listing: Listing) -> None:
        """``f``'s draw, once its note is on screen: only while ``listing`` is still the list and every
        job is still asked for. A refresh that landed meanwhile drew its own list whole, as f asks, and
        this one's older jobs must not go back over it; a second f meanwhile drew the newest again."""
        if self._cut_listing is listing and self._full_list:
            self._show_listing(listing)

    def _page_bound(self) -> PageBound | None:
        """The page asked for: the last one ] went to, or None for the first."""
        return self._pages[-1] if self._pages else None

    def _first_page(self) -> None:
        """Back to the first page, for another list (a) or another location (a narrowing): the pages
        read of this one are not that one's."""
        self._pages.clear()
        self._next_page = None
        self._page_cache.clear()

    def _cache_page(self, listing: Listing) -> None:
        """Keep ``listing``, a page read whole, for ] and [ to show again at once. A list of another
        kind (the configured schema's in place of the account-wide one) drops the pages of the other."""
        if any(kept.scope.kind != listing.scope.kind for _bound, kept, _at, _clock in self._page_cache.values()):
            self._page_cache.clear()
        self._page_cache[listing.scope.page] = (self._load_page, listing, time.monotonic(),
                                                self._last_refreshed or "")

    def _cached_keys(self) -> set:
        """The jobs of the pages kept, whose rows stay drawn for ] and [."""
        return {self._job_key(job) for _bound, kept, _at, _clock in self._page_cache.values() for job in kept.jobs}

    # ─── every job, for search ───────────────────────────────────────────────

    _ARCHIVE_DRAW_SECONDS = 2.0  # how often a search shows the jobs read behind the list as they come

    def _list_kind(self) -> str | None:
        """Which list is on screen, for its archive: ACCOUNT, or SCHEMA (the configured schema's,
        however it was read); None before the first list."""
        if self._scope is None:
            return None
        return ACCOUNT if self._scope.kind == ACCOUNT else SCHEMA

    def _search_archive(self) -> _Archive | None:
        """The archive of the list on screen; None before its first page is whole, and while the
        list is narrowed to a location (the service's answer for it, which a search covers alone)."""
        archive = self._archive
        if archive is None or archive.kind != self._list_kind() or self._pushdown is not None:
            return None
        return archive

    def _searching(self) -> bool:
        """Whether a search is on: its matches come from every job read, a page of them at a time."""
        if not self._filter:
            return False
        query = self._query()
        return not (query.empty and not query.errors)

    def _corpus(self) -> list:
        """The jobs a search covers: the list on screen's (their records the freshest) and every
        one read behind it (see _Archive)."""
        archive = self._search_archive()
        if archive is None:
            return self._all_jobs
        kept = self._corpus_cache
        if kept is not None and kept[0] == archive.version and kept[1] is self._all_jobs:
            return kept[2]
        jobs = dict(archive.jobs)
        jobs.update((self._job_key(job), job) for job in self._all_jobs)
        corpus = list(jobs.values())
        self._corpus_cache = (archive.version, self._all_jobs, corpus)
        return corpus

    def _keep_archive(self, listing: Listing) -> None:
        """After ``listing`` shows, read whole: hold its jobs, the freshest records, in its list's
        archive. A first page of the list not narrowed to a location starts the read behind it,
        a request at a time (see JobLister.read_rest); so it does again once that read failed, or
        stopped for the other list (a)."""
        kind = self._list_kind()
        archive = self._archive
        same = archive is not None and archive.kind == kind
        if same:
            archive.add(self._all_jobs, self._job_key, fresher=True)
        if listing.scope.page != 1 or self._pushdown is not None:
            return
        if same and not archive.failed and (archive.done or self._archive_reading):
            return
        if not same:
            archive = _Archive(kind)
            archive.add(self._all_jobs, self._job_key, fresher=True)
            self._archive = archive
        archive.done, archive.capped, archive.failed = False, False, ""
        self._archive_gen += 1
        self._archive_reading = True
        self._read_rest(self._archive_gen, listing)

    def _stop_archive(self) -> None:
        """Stop the read behind the list (a asked for the other list): what it read stays, and it
        reads on when that list shows again."""
        if self._archive_reading:
            self.workers.cancel_group(self, "archive")
            self._archive_reading = False
            self._archive_gen += 1

    @work(thread=True, group="archive", exclusive=True)
    def _read_rest(self, gen: int, listing: Listing) -> None:
        """Read the rest of ``listing``'s list behind it, for search (see JobLister.read_rest)."""
        worker = get_current_worker()

        def on_rest(rest: Rest) -> None:
            if not worker.is_cancelled:
                call_on_screen(self, self._on_rest, gen, rest)

        self._lister.read_rest(listing, is_cancelled=lambda: worker.is_cancelled, on_rest=on_rest)

    def _on_rest(self, gen: int, rest: Rest) -> None:
        """One request's jobs of the read behind the list. A search on screen shows them every
        few seconds as they come, and at once when the read ends."""
        archive = self._archive
        if gen != self._archive_gen or archive is None:
            return  # the read of another list, or one stopped since
        archive.add(rest.jobs, self._job_key)
        stamps = [stamp for stamp in (views.parse_ts(job.get("created_at")) for job in rest.jobs) if stamp]
        if stamps:
            oldest = min(stamps)
            archive.reached = oldest if archive.reached is None else min(archive.reached, oldest)
        if rest.done:
            archive.done, archive.capped, archive.failed = True, rest.capped, rest.failed
            self._archive_reading = False
        if not self._prebuilding and not self._prebuilt:
            self._prebuild_next()  # the page ] shows next may be held now
        if self._searching() and archive.kind == self._list_kind():
            now = time.monotonic()
            if rest.done or now - self._archive_drawn_at >= self._ARCHIVE_DRAW_SECONDS:
                self._archive_drawn_at = now
                self._render_jobs(steady=True)

    def _archive_page(self, bound: PageBound | None) -> Listing | None:
        """The page past the first that ``bound`` starts, cut from the jobs read behind the list as
        a whole list is cut (see cut_whole_listing): its 1,000 in history and every active job. None
        for page 1 (a refresh reads what is new), on a narrowed list, and where the read hasn't
        passed that page's end yet, so the service reads it."""
        archive = self._search_archive()
        if bound is None or archive is None:
            return None
        kept = self._sorted_cache
        if kept is None or kept[0] is not archive or kept[1] != archive.version:
            kept = self._sorted_cache = (archive, archive.version, sorted_for_pages(archive.jobs.values()))
        ordered, keys = kept[2]
        listing = cut_sorted(ordered, keys, bound, ListScope(self._scope.kind, reason=self._scope.reason))
        if not archive.complete:
            ends = listing.next_page.before if listing.next_page is not None else None
            if listing.scope.stopped is None or ends is None or archive.reached is None or ends <= archive.reached:
                return None  # the page may run past the jobs read so far
        return listing

    _PREBUILD_SLICE_SECONDS = 0.025  # how long one slice of building the next page's rows may take

    def _prebuild_next(self) -> None:
        """Build the rows of the page ] shows next, cut from the jobs read behind the list, while
        nothing else happens, so that page shows at once too: a slice at a time, the list staying
        responsive between them."""
        if self._next_page is None or self._searching():
            return
        if self._prebuild_for == self._next_page and (self._prebuilding or self._prebuilt):
            return  # building, or built, for that page already
        listing = self._archive_page(self._next_page)
        if listing is None:
            return  # not read yet: tried again as the read behind the list comes in
        self._prebuild_for = self._next_page
        self._prebuilt = set()
        self._prebuild = deque(self._in_view(listing.jobs))
        if not self._prebuilding:
            self._prebuilding = True
            self.set_timer(0.05, self._prebuild_slice)

    def _prebuild_slice(self) -> None:
        layout = self._layout
        if not self._prebuild or layout is None:
            self._prebuilding = False
            return
        ends = time.monotonic() + self._PREBUILD_SLICE_SECONDS
        while self._prebuild and time.monotonic() < ends:
            job = self._prebuild.popleft()
            self._job_rows(job, layout)
            self._prebuilt.add(self._job_key(job))
        if self._prebuild:
            self.set_timer(0.01, self._prebuild_slice)  # the next slice, after whatever waits
        else:
            self._prebuilding = False

    def _match_line(self) -> str:
        """The page line during a search: which page of its matches shows, and what it covered."""
        count, searched = self._match_count, self._match_searched
        head = f"page {self._match_page} of {self._match_pages} · " if self._match_pages > 1 else ""
        matches = f"{head}{count:,} match{'' if count == 1 else 'es'}"
        noun = "deployments" if self._deployments else "jobs"
        archive = self._search_archive()
        if archive is not None and archive.failed:
            return f"{matches} in the {searched:,} {noun} read · the rest didn't load"
        if archive is not None and archive.capped:
            return f"{matches} in the newest {searched:,} {noun}, the most a search reads"
        if archive is None or not archive.done:
            return f"{matches} so far · {searched:,} {noun} read, reading the rest"
        return f"{matches} in all {searched:,} {noun}"

    def action_page(self, which: str) -> None:
        """``]``: the page of older jobs after the one shown (``which`` "older"); ``[``: the page
        before it ("newer"). Each lists every active job; refreshes keep the page.

        A page read before shows at once, as it was read, and is read again behind it (the rows
        stay until the new read is whole) only when it is as old as a refresh, or the page before
        it moved since, so that it starts elsewhere now.

        During a search, ] and [ show the next and the previous page of its matches instead."""
        if self._searching():
            page = self._match_page + (1 if which == "older" else -1)
            if 1 <= page <= self._match_pages:
                self._match_page = page
                self._render_jobs(top=True)
            return
        undo = (list(self._pages), self._next_page)
        if which == "older":
            if self._next_page is None:
                return
            self._pages.append(self._next_page)
        elif self._pages:
            self._pages.pop()
        else:
            return
        bound = self._page_bound()
        self._next_page = None  # where the page after starts is the new page's to say
        self._settle_folds()  # another page: e's choice was for the one before
        if self._refreshing:
            self._supersede()  # the load in flight read another page
        page = len(self._pages) + 1
        local = self._archive_page(bound)
        if local is not None:
            # Read already, behind the list: shown at once, with no request, as fresh as what was read.
            self._page_undo = None
            self._load_page = bound
            self._list_error, self._list_refused, self._failed_at = None, False, None
            self._pages_failed = False
            self._show_listing(local, top=True)
            self._cache_page(local)
            return
        kept = self._page_cache.get(page)
        if kept is not None:
            self._page_undo = None
            read_under, listing, read_at, self._last_refreshed = kept
            self._load_page = bound  # the rows shown are this page's
            # Read whole when kept: a failure since was another load's, not this page's.
            self._list_error, self._list_refused, self._failed_at = None, False, None
            self._pages_failed = False
            self._show_listing(listing, top=True)
            self._loaded_at = read_at  # as old as its read, for the refresh on coming back to the list
            if read_under != bound:
                # The page before moved since this one was read, so it starts elsewhere now: ] waits
                # for the read again, and the pages kept after it are no longer the ones after it.
                self._next_page = None
                for later in [number for number in self._page_cache if number > page]:
                    del self._page_cache[later]
            self.refresh_bindings()
            if read_under != bound or time.monotonic() - read_at >= self._REFRESH_SECONDS:
                self._refresh_jobs(manual=True, quiet=True)
            return
        self._page_undo = undo  # put back should this page fail before any of its jobs show
        self._paging = True
        self.refresh_bindings()
        self._refresh_jobs(manual=True, quiet=True)
        self._update_subtitle()

    def action_show_list(self, which: str) -> None:
        """``a``: show the account-wide list (``which`` "account") or the configured schema's ("schema")."""
        account = which == "account"
        if account == self._lister.account_wide:
            return
        self._first_page()  # another list: its pages are its own
        self._stop_archive()  # and so is what a search reads behind it
        self._lister.account_wide = account
        self._asked = account
        self._declined = None
        self._settle_folds()  # another list: e's choice was for this one
        self._switching = True
        self.refresh_bindings()  # the footer's a names the other list now
        if self._refreshing and self._progressive:
            # A list still reading its pages: no need to wait for the rest of them.
            self._supersede()
        self._refresh_jobs(manual=True, quiet=True)
        # A refresh already in flight reads the list a asked for when it answers;
        # the header says so now.
        self._update_subtitle()

    def action_show_view(self, which: str) -> None:
        """``d``: show the deployments (``which`` "deployments") or every other job ("jobs")."""
        deployments = which == "deployments"
        if deployments == self._deployments:
            return
        self._deployments = deployments
        self._match_page = 1  # a search's matches in the other view start from their first page
        self.refresh_bindings()  # the footer's d names the other view now
        self._render_jobs(top=True)
        self._update_subtitle()

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
        if action == "full_list":
            # Only a list cut short with every job at hand has them to show; of f's two
            # bindings, the one for the other way. On the first page only: f's every job is
            # every page's.
            return (self._cut_listing is not None and not self._pages
                    and (parameters[:1] == ("all",)) != self._full_list)
        if action == "page":
            if self._searching():  # the pages of the search's matches
                return self._match_page > 1 if parameters[:1] == ("newer",) else self._match_page < self._match_pages
            if parameters[:1] == ("newer",):
                return bool(self._pages)
            # Older jobs exist past the page on screen, which is whole (f's every job has them all).
            return (self._next_page is not None and not self._full_list
                    and not (self._scope is not None and self._scope.loading))
        if action == "show_view":
            # One of d's two bindings, the one for the other view.
            return (parameters[:1] == ("deployments",)) != self._deployments
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
        self._match_page = 1  # a search's matches in the new order start from their first page
        self._render_jobs(top=True)  # a new order: show its start, not the old row wherever it went

    def action_toggle_order(self) -> None:
        self._descending = not self._descending
        self._match_page = 1
        self._render_jobs(top=True)

    def _job_for(self, job_key) -> dict | None:
        """The record of a job the table draws: from the page on screen, or (a search's match) from
        the jobs read behind it, which no page on screen holds."""
        job = self._jobs_by_key.get(job_key)
        return job if job is not None else self._drawn.get(job_key)

    def whole_id(self, key) -> str | None:
        """The whole id of the job or sub-job on row ``key`` (a sub-job's row shows its label there)."""
        if key in self._job_by_item:
            job = self._job_for(self._job_by_item[key])
            return views.job_id_of(job) if job is not None else None
        sub = self._sub_by_item.get(key)  # (job key, sub-job id)
        if sub:
            job = self._job_for(sub[0])
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
        if job_key in self._expanded:
            return True
        # e's choice holds for the jobs listed after it, too (a later page's).
        folded = self._all_folded[views.is_deployment(job)]
        return self._starts_unfolded(job) if folded is None else not folded

    @staticmethod
    def _starts_unfolded(job: dict) -> bool:
        """Every live job starts unfolded, its sub-job rows in view; history starts folded."""
        return views.is_live(job)

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
        job = self._job_for(job_key)
        if job is None or not self._expandable(job) or self._is_expanded(job) == unfold:
            return False
        self._unfold(job_key, not self._is_expanded(job))
        return True

    def action_expand_all(self) -> None:
        # The jobs listed: during a search, every job it covers, the ones read behind the page too.
        listed = self._in_view(self._corpus() if self._searching() else self._all_jobs)
        with_subs = [j for j in listed if self._expandable(j)]
        # Every job folded, or every one unfolded: those listed since too; a fold or unfold
        # after it is the job's own.
        self._all_folded[self._deployments] = bool(with_subs) and all(self._is_expanded(j) for j in with_subs)
        for job in listed:
            self._expanded.discard(self._job_key(job))
            self._collapsed.discard(self._job_key(job))
        self._render_jobs()

    def _settle_folds(self) -> None:
        """e's choice holds for the list in front of the reader (and its later pages): once
        another list comes (a, or a refresh), the jobs it covered keep it, each as its own fold,
        and jobs new since start as every job does (live unfolded, history folded)."""
        for job in self._all_jobs:
            folded = self._all_folded[views.is_deployment(job)]
            key = self._job_key(job)
            if (folded is not None and key not in self._expanded and key not in self._collapsed
                    and self._expandable(job)):
                (self._collapsed if folded else self._expanded).add(key)
        self._all_folded = {False: None, True: None}

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "jobfilter":
            return
        if event.value != event.input.value:
            return  # the box has changed again since: a Changed with what it holds is on its way
        text = event.value.strip()
        if text == self._filter and self._search_timer is None:
            return  # the list shows this search already (Enter or Esc applied it, or a space was typed)
        self._filter = text
        self._match_page = 1  # another search: its matches from their first page
        # Rebuild once typing pauses; a rebuild per keystroke lagged with ~900 jobs.
        if self._search_timer is not None:
            self._search_timer.stop()
        self._search_timer = self.set_timer(self._SEARCH_DEBOUNCE_SECONDS, self._search_now)

    def _search_now(self) -> None:
        """Show the matches of the search (``_filter``), without waiting for typing to pause."""
        if self._search_timer is not None:
            self._search_timer.stop()
            self._search_timer = None
        self._renarrow()  # the matches of what is loaded show at once; a narrowed list replaces them
        self._render_jobs()

    def _pushdown_for(self) -> tuple | None:
        """The ``(database, schema)`` the account-wide walk should ask the service for; None for every schema.

        Only the account-wide list narrows: the configured schema's list is already scoped to its
        one schema.
        """
        if not self._account_wanted():
            return None
        archive = self._archive
        if archive is not None and archive.kind == ACCOUNT and archive.complete:
            return None  # every job of every schema is at hand: the search filters them here
        return views.pushdown_location(self._query())

    def _renarrow(self) -> None:
        """Read the account-wide list again when the search pins a location other than the one the
        list on screen was read under.

        Without this a ``db=`` / ``schema=`` search only hides rows the walk already read, so a
        schema whose jobs fall outside the newest :data:`ACCOUNT_LIST_CAP` of the account could not
        be reached at all. The search's own debounce keeps this off the keystroke path.
        """
        if not self._account_wanted() or self._pushdown_for() == self._pushdown:
            return
        if self._scope is None or self._scope.kind != ACCOUNT:
            # The configured schema's list is showing (its own, or in place of an unavailable
            # account-wide one): it is already scoped to its one schema, so a search must not
            # have it read again on every edit.
            return
        self._narrowing = True
        self._first_page()  # another location: its pages are its own
        self._supersede()  # the load in flight read another scope; its pages are not these
        self._refresh_jobs(manual=True, quiet=True)

    @work(thread=True, group="jobs")
    def _load_jobs(self, manual: bool = False, account: bool = False, asked: bool = False, gen: int = 0,
                   progressive: bool = False, location: tuple | None = None,
                   before: PageBound | None = None) -> None:
        """Read one schema or account-wide listing off the UI thread (``progressive``: showing its pages).

        ``location`` narrows the account-wide walk at the service (see
        :func:`cortex_training.tui.views.pushdown_location`); ``before``: the page past the first
        it reads (see :class:`PageBound`)."""
        worker = get_current_worker()

        def on_partial(listing: Listing) -> None:
            if not worker.is_cancelled:
                call_on_screen(self, self._on_partial, gen, listing)

        try:
            listing = self._lister.load(self._clock(), is_cancelled=lambda: worker.is_cancelled, manual=manual,
                                        account=account, asked=asked, on_partial=on_partial if progressive else None,
                                        location=location, before=before)
            error = None
        except JobListError as exc:
            listing, error = None, exc
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            listing, error = None, JobListError(views.collapse(f"the job list failed: {views.describe_error(exc)}"))
        if worker.is_cancelled or (listing is None and error is None):
            return
        call_on_screen(self, self._on_jobs, listing, error, account, gen)

    def _on_partial(self, gen: int, listing: Listing) -> None:
        """The first pages of the list the load ``gen`` reads, shown while it reads the rest."""
        wanted = ACCOUNT if self._account_wanted() else SCHEMA
        if gen != self._load_gen or not self._refreshing or listing.scope.kind != wanted:
            return  # superseded (by a, or a load since): its pages are not the list asked for
        steady = self._partial_gen == gen  # its earlier pages show: the user may be reading them
        self._partial_gen = gen
        top = self._paging and not steady  # the page ] or [ asked for, drawn first from its first job
        self._switching = False  # the list a asked for shows now
        self._narrowing = False  # so does the narrowing, if this load was one
        self._paging = False  # and the page, if this load was a page change
        self._asked = False  # its last answer is handled as a's, though (see _on_jobs)
        # A fresh list shows: an earlier failure no longer stands over it.
        self._list_error, self._list_refused, self._failed_at = None, False, None
        self._pages_failed = False
        self._last_refreshed = time.strftime("%H:%M:%S")
        self._show_listing(listing, steady=steady, top=top)

    def _show_listing(self, listing: Listing, *, steady: bool = False, top: bool = False) -> None:
        """Show ``listing`` as the list; ``steady``: it grows the list on screen, which stays where it is;
        ``top``: a page that ] or [ asked for, drawn from its first job."""
        self._partial_stays = False
        self._next_page = listing.next_page
        if listing.everything is None:
            self._cut_listing, self._full_list = None, False
        else:
            self._cut_listing = listing
            if self._full_list:  # f asked for every job: so does each refresh, until another list
                listing = Listing(listing.everything, replace(listing.scope, stopped=None, walked=0))
        self.refresh_bindings()
        self._scope = listing.scope
        # What these rows were read under, so the header and the next search compare against the
        # listing in hand rather than the live search text (the two differ while a load is in flight).
        self._pushdown = self._load_location if listing.scope.kind == ACCOUNT else None
        self._loaded_at = time.monotonic()
        self._all_jobs = self._merged(listing.jobs)
        self._jobs_by_key = {self._job_key(job): job for job in self._all_jobs}
        # The rows of the pages kept stay too, so [ and ] draw them again without building them, and
        # so do the rows on screen (a search's matches); and every job's search text a search covers.
        keep = self._jobs_by_key.keys() | self._cached_keys()
        on_screen = set(self._job_by_item.values())
        held = self._archive.jobs if self._archive is not None else {}
        self._rows = {key: rows for key, rows in self._rows.items()
                      if key in keep or key in on_screen or key in self._prebuilt}
        self._hay = {key: hay for key, hay in self._hay.items() if key in keep or key in held}
        self._update_subtitle()
        if top:
            self._render_jobs(top=True)
        else:
            self._render_jobs(steady=steady, fresh=not steady)
        self._prebuild_next()  # the page ] shows next, if the read behind the list holds it
        self._ask_identity()  # once the first jobs show, which can be the first of many pages

    def _on_jobs(self, listing, error: JobListError | None, account: bool = False, gen: int | None = None) -> None:
        """A load's answer: ``listing``, or ``error``; ``account``: it read the account-wide list."""
        if gen is not None and gen != self._load_gen:
            return  # superseded: the load that replaced it answers in its place
        self._refreshing = False
        self._load_worker = None
        self._ask_identity()  # a first load that failed, too
        shown = self._partial_gen is not None and self._partial_gen == gen  # its first pages show
        self._partial_gen = None
        if shown and error is not None:
            # A later page failed: the pages read stay, saying so, and r reads them all again.
            self._page_undo = None  # a page change's first jobs show: that page is the one on screen
            self._scope = stopped_scope(self._scope)
            self._list_error = str(error) or "the job list failed"
            self._list_refused = error.refused
            self._failed_at = time.strftime("%H:%M:%S")
            self._pages_failed = True
            self.refresh_bindings()  # read no more: ] goes on from the jobs read
            self._update_subtitle()
            self._render_jobs(steady=True)
            return
        if account != self._account_wanted():
            # a switched lists while this one was read: read the one asked for now.
            self._refresh_jobs(manual=True, quiet=True)
            return
        if account and self._load_location != self._pushdown_for():
            # The search pinned another location while this one was read: read that one now.
            self._refresh_jobs(manual=True, quiet=True)
            return
        if self._load_page != self._page_bound():
            # ] or [ asked for another page while this one was read: read that one now.
            self._refresh_jobs(manual=True, quiet=True)
            return
        top = self._paging and not shown  # a page change whose jobs show now, from its first
        paging = self._paging and not shown
        self._switching = False
        self._narrowing = False
        self._paging = False
        if account and (self._asked or shown):
            # a's answer, also after its first pages showed: an account-wide list
            # unavailable now keeps the configured schema, saying why, as at once.
            self._asked = False
            if error is not None or listing.scope.kind != ACCOUNT:
                listing = self._stay_on_schema(listing, error)
                if listing is None:
                    return
                error = None
        if error is not None:
            if paging and self._page_undo is not None:
                # The page ] or [ asked for failed before any of its jobs showed: the list stays on
                # the page on screen, and so do ], [, r and the refreshes.
                self._pages, self._next_page = self._page_undo
                self._load_page = self._page_bound()
                self.refresh_bindings()
            self._page_undo = None
            # Keep the last good list on screen, saying so, until a load succeeds.
            self._list_error = str(error) or "the job list failed"
            self._list_refused = error.refused
            self._failed_at = time.strftime("%H:%M:%S")
            # Still those pages' failure only if this load read them again (r on them).
            self._pages_failed = self._pages_failed and account
            self._update_subtitle()
            self._render_jobs()
            return
        self._list_error = None
        self._list_refused = False
        self._failed_at = None
        self._pages_failed = False
        self._last_refreshed = time.strftime("%H:%M:%S")
        if not shown:
            self._settle_folds()  # a refresh: the jobs new in it start as every job does
        self._page_undo = None
        self._cache_page(listing)
        self._show_listing(listing, steady=shown, top=top)
        self._keep_archive(listing)
        self.app.job_list_drawn(self)

    def _stay_on_schema(self, listing: Listing | None, error: JobListError | None) -> Listing | None:
        """Handle an unavailable account list by retaining the configured schema."""
        self._lister.account_wide = False
        self.refresh_bindings()  # the footer's a offers all schemas again
        why = (str(error) or "the account-wide job list failed") if error is not None else unavailable_text(
            listing.scope)
        self.notify(f"Can't show all schemas ({why}); the list stays on {self._schema_label()}.",
                    severity="warning", timeout=10, markup=False)
        # The header keeps saying why, in short, after the notice has gone.
        self._declined = views.fit_cells(why, self._DECLINED_CELLS) or "all schemas unavailable"
        if error is None:
            # The configured schema's list as the fallback read it, as that list: one too long to
            # draw whole stays cut, with its every job kept for f (see cut_whole_listing).
            return replace(listing, scope=replace(listing.scope, kind=SCHEMA, reason=None))
        self._update_subtitle()
        if self._loaded_at is None:
            self._refresh_jobs()  # nothing listed yet (a came first): read the configured schema's list
        return None

    def _merged(self, rows: list) -> list:
        """Reuse unchanged records and retain an UNKNOWN row's known creation time (and a deployment's sub-jobs)."""
        merged = []
        for job in rows:
            if not (isinstance(job, dict) and views.job_id_of(job)):
                continue
            old = self._jobs_by_key.get(self._job_key(job))
            if old is not None:
                if not job.get("created_at") and old.get("created_at"):
                    job = {**job, "created_at": old["created_at"]}
                job = views.keep_deployment(old, job)
                if job == old:
                    job = old
            merged.append(job)
        return merged

    def job_fetched(self, job: dict) -> None:
        """Merge a detail screen's fresh job into the list."""
        if not isinstance(job, dict):
            return
        key = self._job_key(job)
        archive = self._archive
        current = self._job_for(key)
        if current is None and archive is not None:
            current = archive.jobs.get(key)
        if current is None:
            return
        fresh = views.refresh_job(current, job)
        if fresh == current:
            return
        if key in self._jobs_by_key:  # on the page on screen
            self._all_jobs = [fresh if j is current else j for j in self._all_jobs]
            self._jobs_by_key[key] = fresh
        if key in self._drawn:
            self._drawn[key] = fresh
        if archive is not None and key in archive.jobs:  # read behind the list: a search finds it fresh
            archive.add([fresh], self._job_key, fresher=True)

    # ─── rows ────────────────────────────────────────────────────────────────

    @staticmethod
    def _name_cell(job: dict, unfolded: bool | None = None) -> _FixedCell:
        """The Job ID cell: the whole id, its first eight characters (what the preview and
        a scan go by) brighter, after a ▾ or ▸ marker for a job that unfolds (``unfolded``
        True or False; None: it has nothing to unfold). Plain text, never markup."""
        jid = views.job_id_of(job)
        parts = [(jid[:8], theme.TEXT), (jid[8:], theme.MUTED)]
        if unfolded is not None:
            parts.insert(0, ("▾ " if unfolded else "▸ ", theme.FAINT))
        # The column cuts a long id: it ends in "…" there; a copy of the list has it whole.
        return _FixedCell.assemble(*parts)

    @staticmethod
    def _age_cell(raw, now: datetime) -> tuple[_FixedCell, float]:
        """A time as how long before ``now`` it was (``3h ago``; copied as the time itself,
        ``YYYY-MM-DD HH:MM``), and the seconds it reads so."""
        dt = views.parse_ts(raw)
        text, lasts = views.list_age((now - dt).total_seconds() if dt else None)
        return _CopiedCell(_cell(text, theme.MUTED if dt else theme.FAINT), format_created_at(raw) or "—"), lasts

    @staticmethod
    def _span_cell(job: dict, now: datetime) -> tuple[_FixedCell, float]:
        """How long the job has run (live: so far, in its status's color) or ran, and the
        seconds it reads so: a run that is over (it has an end time) never changes."""
        seconds = views.job_elapsed(job, now)
        text, lasts = views.list_span(seconds)
        live = views.is_live(job)
        ended = not live and views.parse_ts(job.get("state_changed_at") or job.get("updated_at"))
        style = theme.status_look(job.get("status")).color if live else theme.MUTED
        return _cell(text, style if seconds is not None else theme.FAINT), math.inf if ended else lasts

    @staticmethod
    def _owner_cell(job: dict) -> _FixedCell:
        """The submitting user; a job that records none shows its owning role, dimmed and named a role."""
        user = views.collapse(job.get("submitted_by") or "")
        if user:
            return _cell(user)
        role = views.collapse(job.get("owner_role") or "")
        return _cell(f"role {role}", theme.MUTED) if role else _cell("—", theme.FAINT)

    @staticmethod
    def _status_cell(status, *, glyph_only: bool = False, word: str | None = None) -> _FixedCell:
        """``● RUNNING``: the theme's glyph and word, in the status's color, as every screen
        draws a status (``glyph_only``: the glyph alone, for a sub-job in its job's status).
        Copied as the word alone, also where only the glyph is drawn. ``word`` overrides the
        status's own word (a deployment's, e.g. ``READY``); the color still comes from ``status``."""
        look = theme.status_look(status)
        word = (word if word is not None else views.status_word(status)).upper()
        if glyph_only:
            return _CopiedCell(_cell(look.glyph, look.color), word)
        return _CopiedCell(_FixedCell.assemble((f"{look.glyph} ", look.color), (word, look.style)), word)

    @staticmethod
    def _shown_model(models: list) -> str:
        """The Model column of a job of ``models``: the first, short, and how many more."""
        model = views.short_model(models[0]) if models else "—"
        return model + (f" +{len(models) - 1}" if len(models) > 1 else "")

    def _job_cells(self, job: dict, now: datetime) -> tuple[dict, float]:
        """A job's row but for its Job ID (column label -> cell), and the clock's epoch
        seconds at which one of its relative times would read differently."""
        created, created_lasts = self._age_cell(job.get("created_at"), now)
        updated, updated_lasts = self._age_cell(job.get("updated_at"), now)
        span, span_lasts = self._span_cell(job, now)
        # Plain text throughout: DataTable parses a str cell as markup.
        cells = {
            "Status": self._status_cell(job.get("status"), word=views.status_word_for(job, job.get("status"))),
            "Type": _cell(views.job_type_label(job)),
            "Created": created,
            "Duration": span,
            "Owner": self._owner_cell(job),
            "Model": _cell(self._shown_model(views.job_models(job))),
            # Requested GPUs; "shared" for a job that runs only on another job's GPUs.
            "GPUs": _cell(views.job_gpu_cell(job, str(job.get("hardware") or ""))),
            # Plain text: a quoted name can hold anything, brackets included.
            _SCHEMA_COLUMN: _location_cell(job),
            "Updated": updated,
        }
        return cells, now.timestamp() + min(created_lasts, updated_lasts, span_lasts)

    @staticmethod
    def _row_or_note(job: dict, build):
        """Build a row, isolating one malformed job from the rest of the list."""
        try:
            return build()
        except Exception:  # noqa: BLE001 - the traceback goes to the error log
            jid = views.job_id_of(job) if isinstance(job, dict) else "?"
            log_traceback(f"job list row for {jid}")
            return {"Job ID": _cell(jid, theme.MUTED), "Status": _cell("? can't show", theme.WARN)}

    @classmethod
    def _sub_cells(cls, job: dict, sj: dict, last: bool) -> dict:
        """A sub-job's row, under its job's and fainter, with what it doesn't share with
        its job: its status is the glyph alone while it is its job's, and glyph and word
        where it differs (a sampling sub-job still placing beside a running training
        one); its model shows where it isn't its job's, and its GPUs unless it is the
        job's only sub-job and the job's row says the same count. No type (its label
        has it). A sub-job that has no status of its own shows none, unless its job's
        is unknown too."""
        status = sj.get("status")
        same = theme.status_key(status) == theme.status_key(job.get("status"))
        told = same or str(status or "").strip()
        model = views.short_model(sj.get("model_name"))
        shown = model and model != cls._shown_model(views.job_models(job))
        gpus = views.sub_job_gpu_text(sj)
        gpus_told = len(views.sub_jobs(job)) > 1 or gpus != views.job_gpu_cell(job)
        return {
            "Job ID": _FixedCell.assemble(("  ", ""), (f"{'└' if last else '├'} ", theme.FAINT),
                                          (views.sub_job_label(sj), theme.MUTED)),
            "Status": cls._status_cell(status, glyph_only=same, word=views.status_word_for(job, status)) if told
            else _BLANK,
            "Model": _cell(model, theme.FAINT) if shown else _BLANK,
            "GPUs": _cell(gpus, theme.FAINT) if gpus_told else _BLANK,
        }

    def _sub_rows(self, job: dict) -> tuple:
        """An expanded job's status note and sub-job rows."""
        job_key = self._job_key(job)
        entries = views.sub_jobs(job)
        subs, keys = [], []
        if views.is_unknown(job):
            note = _FixedCell.assemble(("  ", ""), (f"{'├' if entries else '└'} ", theme.FAINT),
                                       ("? status unavailable", theme.WARN))
            subs.append(("note", {"Job ID": note}))
            keys.append(None)
        for si, sj in enumerate(entries):
            last = si == len(entries) - 1
            subs.append((str(si), self._row_or_note(job, lambda: self._sub_cells(job, sj, last))))
            keys.append((job_key, sj.get("sub_job_id")))
        return tuple(subs), tuple(keys)

    @staticmethod
    def _header_cells(group: str, count: int, labels: tuple, so_far: bool = False) -> tuple:
        """A LIVE/HISTORY row, drawn as a section heading: its title and count in the
        Job ID column (``so_far``: "…" after it, while more jobs load), and a rule across
        the columns after (drawn, not copied)."""
        style = f"bold {theme.OK}" if group == "LIVE" else f"bold {theme.MUTED}"
        name = _cell(f"{group}  {count:,}{'…' if so_far else ''}", style)
        return tuple(name if column == "Job ID" else _RuleCell(theme.RULE) for column in labels)

    def _header(self, group: str, count: int, labels: tuple, so_far: bool = False) -> tuple:
        """A LIVE/HISTORY row, kept while it reads the same."""
        kept = self._headers.get(group)
        if kept is not None and kept[0] == (count, so_far, labels):
            return kept[1]
        cells = self._header_cells(group, count, labels, so_far)
        self._headers[group] = ((count, so_far, labels), cells)
        return cells

    def _job_rows(self, job: dict, layout: _ListLayout) -> _JobRows:
        """Cached rows for a job, its columns, and expansion state. Its cells are built again
        only when its record changed, or one of its relative times would read
        differently now (see :meth:`_job_cells`)."""
        job_key = self._job_key(job)
        now = self._clock()
        rows = self._rows.get(job_key)
        if not (rows is not None and rows.job is job and now.timestamp() < rows.stale_at):
            built = self._row_or_note(job, lambda: self._job_cells(job, now))
            cells, stale_at = built if isinstance(built, tuple) else (built, math.inf)  # a note row never ages
            rows = _JobRows(job, stale_at, cells)
            self._rows[job_key] = rows
        expanded = self._is_expanded(job)
        if rows.labels != layout.labels or rows.expanded != expanded:
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
            rows.labels, rows.expanded = layout.labels, expanded
        return rows

    def _query(self) -> views.Query:
        """The resolved search query, cached with its input and job records."""
        corpus = self._corpus()
        kept = self._resolved
        if kept is not None and kept[0] == self._filter and kept[1] is corpus:
            return kept[2]
        query = views.resolve_query(views.parse_query(self._filter), corpus)
        self._resolved = (self._filter, corpus, query)
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
        if self._pages_failed:
            # The first pages of a list whose later ones failed (or were never read).
            text = (f"× load failed {self._failed_at or ''}: {self._list_error or ''}; "
                    f"showing the {len(self._all_jobs):,} jobs read before it (r reads them all again)")
        else:
            text = (f"× refresh failed {self._failed_at or ''}: {self._list_error or ''}; "
                    f"showing the list from {(self._last_refreshed or '—')[:5]}")
        self._notice = Content.assemble("   ", (text, theme.ERR))
        self._paint_notice()

    def _paint_notice(self) -> None:
        if self._notice is None:
            return
        widget = self.query_one("#listnotice", Static)
        # Wrapped lines continue under the message ("   × " is five cells), not at the edge.
        widget.update(hanging(self._notice, widget.size.width or self.size.width or self.app.size.width, 5))

    def _paint_who(self) -> None:
        """The line under the header: how the TUI connects, the user and role it signs in as,
        and its version (see :func:`cortex_training.tui.identity.line_parts`)."""
        widget = self.query_one("#listwho", Static)
        width = widget.content_size.width or (self.size.width or self.app.size.width) - 2
        parts = line_parts(getattr(self.app, "sign_in", None), getattr(self.app, "identity", None),
                           __version__, asks=signs_in(self._client))
        drawn = [Content.assemble((f"{label} ", theme.MUTED),
                                  (one_line(value), theme.FAINT if value in (ASKING, UNKNOWN) else theme.TEXT))
                 for label, value in parts]
        widget.update(wrapped_parts(drawn, width, 2, _DOT))

    def _ask_identity(self) -> None:
        """Once the list first shows jobs or fails, so the first jobs don't wait on it: ask
        who the TUI signs in as, once for the app."""
        if getattr(self.app, "identity_asked", True) or not signs_in(self._client):
            return
        self.app.identity_asked = True
        self._read_identity()

    @work(thread=True, group="identity")
    def _read_identity(self) -> None:
        identity = caller_identity(self._client)
        call_on_screen(self, self._on_identity, identity)

    def _on_identity(self, identity: Identity) -> None:
        self.app.identity = identity
        self._paint_who()

    def _paint_info(self) -> None:
        """The line above the list: the counts (and a search's matches) at its start, the
        sort at its end. Where the two don't fit on one line, each part is whole where it
        fits, wrapped lines continuing under the text."""
        if self._info is None:
            return
        widget = self.query_one("#listinfo", Static)
        width = widget.content_size.width or (self.size.width or self.app.size.width) - 2
        if isinstance(self._info, Content):
            widget.update(hanging(self._info, width, 2))
            return
        left, sort = self._info
        pad = _SORT_PART_CELLS - sort.cell_length  # measured at its widest: s and o never re-wrap it
        head = Content("  ") + _DOT.join(left)
        gap = width - head.cell_length - sort.cell_length - pad
        if gap >= 4:
            widget.update(head + Content(" " * (gap + pad)) + sort)
        else:
            widget.update(wrapped_parts([*left, sort + Content(" " * pad)], width, 2, _DOT))

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

    # ─── the table ───────────────────────────────────────────────────────────

    def _render_jobs(self, *, keep: tuple | None = None, top: bool = False, steady: bool = False,
                     fresh: bool = False) -> None:
        """Draw the list, optionally keeping one job or starting at the top; ``steady``: the
        list grew (a page more of it), and the highlighted row stays where it is on screen;
        ``fresh``: a new list, the cursor where the list puts it."""
        table = self.query_one("#jobs", DataTable)
        info_line = self.query_one("#listinfo", Static)
        if top or fresh:
            self._reader_moved = False
        elif table.display and table.row_count:
            # The cursor off the row the last render put it on: the reader moved it, and the
            # list keeps their place from now on (a search, a fold or a sort since too).
            current = self._highlighted()[0]
            if current is not None and current != self._placed_key:
                self._reader_moved = True
        keep_sub = None
        keep_header = None
        anchor = None
        at_top = False
        if keep is None and not top:
            key, keep = self._highlighted()
            keep_sub = self._sub_by_item.get(key)  # a sub-job row keeps its own cursor
            if steady and key is not None and table.display and table.row_count:
                # The list grows. Where the reader has moved (the cursor, or the view
                # scrolled), the highlighted row stays on its line; where not, the list
                # stays at its top.
                at_top = round(table.scroll_y) == 0 and not self._reader_moved
                if not at_top:
                    anchor = (key, table.cursor_row - round(table.scroll_y), table.scroll_x)
                if str(key).startswith("hdr-"):
                    keep_header = key  # so does a LIVE/HISTORY row, while the list grows
        # A search covers every job read, the list on screen's and the ones read behind it.
        searching = self._searching()
        source = self._corpus() if searching else self._all_jobs
        everyone = source
        recent = self._in_view(everyone)
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
            info_line.update(Text(f"[error] {self._list_error}  (r retries{hint})", style=theme.ERR))
            self._hide_table(table, layout, "")  # an empty message keeps the search box at the bottom
            return
        others = len(everyone) - len(recent)  # the other view's, so nothing looks lost
        query = self._query()
        home = self._clients.home
        if not searching:
            visible = recent
        else:
            visible = [job for job in recent if views.job_matches_query(job, query, home=home, hay=self._haystack(job))]
        ordered = views.sort_job_list(visible, self._sort, self._descending)
        live = [j for j in ordered if views.is_live(j)]
        history = [j for j in ordered if not views.is_live(j)]
        # A search's matches show a page at a time, as the list does: 1,000 in history, and every
        # live one; ] and [ show the others.
        paged = (self._match_page, self._match_pages)
        if searching:
            self._match_pages = max(1, -(-len(history) // ACCOUNT_LIST_CAP))
            self._match_page = min(self._match_page, self._match_pages)
            start = (self._match_page - 1) * ACCOUNT_LIST_CAP
            shown_history = history[start:start + ACCOUNT_LIST_CAP]
        else:
            self._match_page = self._match_pages = 1
            shown_history = history
        self._match_count, self._match_searched = len(visible), len(recent)
        if paged != (self._match_page, self._match_pages):
            self.refresh_bindings()  # ] and [ step through other pages of matches now
        self._paint_page_line()
        archive = self._search_archive() if searching else None
        reading = archive is not None and not archive.done  # more jobs may match, once read

        # While the list loads, its counts are of the jobs read so far: the history until the
        # newest jobs are read, the active statuses until the active jobs are (beside the
        # newest), and the unknown one (and so the LIVE row) until the newest jobs are too, as
        # no status filter lists an unknown job.
        loading = self._scope is not None and bool(self._scope.loading)
        older = loading and self._scope.loading == OLDER
        active_so_far = loading and not self._scope.active_whole
        live_so_far = active_so_far or older
        left = [_status_chips(live, len(history), live_so_far=active_so_far, unknown_so_far=older,
                              history_so_far=older or reading)]
        if others:
            left.append(Content.styled(self._other_view_text(others), theme.MUTED))
        if self._filter:
            # The count before the query: a long query wraps, the count stays in view.
            # The search text is shown literally (assembled, never parsed as markup).
            matched = f"match{'' if len(visible) == 1 else 'es'}{' so far' if loading or reading else ''} for "
            left.append(Content.assemble((f"{len(visible):,} ", f"bold {theme.TEXT}"), (matched, theme.MUTED),
                                         (self._filter, f"bold {theme.ACCENT}")))
        self._info = (left, _sort_part(self._sort, self._descending))
        self._paint_info()

        if not visible:
            one = self._scope is not None and self._scope.kind != ACCOUNT
            widen = self._scope is not None and self._scope.kind == SCHEMA
            if self._filter:
                # What the search covered, where that is short of every job of the list.
                if archive is None:
                    cut = loaded_text(self._scope)
                elif archive.failed:
                    cut = "the rest of the list didn't load"
                elif archive.capped:
                    cut = f"the newest {len(recent):,}, the most a search reads"
                elif not archive.done:
                    cut = "the rest of the list is still being read"
                else:
                    cut = ""
                # The list is cut short and nothing narrowed it: a whole name would read that
                # database from the service instead of only the jobs loaded here.
                narrow = (self._scope is not None and self._scope.kind == ACCOUNT and self._pushdown is None
                          and (self._scope.truncated or self._scope.page > 1)
                          and (archive is None or not archive.complete))
                # A schema's whole list held to its newest jobs: f searches every one of them, where
                # f is offered (on page 1) and the search doesn't already cover them all.
                everything = (len(self._cut_listing.everything)
                              if archive is None and self.check_action("full_list", ("all",)) else 0)
                there = sum(1 for job in everyone if views.is_deployment(job) != self._deployments
                            and views.job_matches_query(job, query, home=home, hay=self._haystack(job)))
                noun = "job" if self._deployments else "deployment"
                noun_here = "deployment" if self._deployments else "job"
                elsewhere = f"d shows the {there:,} matching {noun}{'' if there == 1 else 's'}." if there else ""
                found = dict(cut=cut,
                             location=self._schema_label() if one else "", widen=widen, narrow=narrow,
                             everything=everything, elsewhere=elsewhere)
                message = views.no_match_message(self._filter, query, len(recent), noun=noun_here, **found)
            else:
                what = "jobs"
                if others:
                    what = "deployments" if self._deployments else "training or sampling jobs"
                if self._scope is not None and (self._scope.truncated or self._scope.page > 1):
                    message = empty_text(self._scope, what)
                elif self._scope is not None and self._scope.kind == ACCOUNT:
                    where = f"in {self._scope.narrowed}" if self._scope.narrowed else "in any database and schema"
                    message = f"No {what} that this role can see, {where}."
                else:
                    message = f"No {what} in {self._schema_label()} that this role can see."
                    if widen:
                        message += "\n  a lists every database and schema."
                if others:
                    other = "job" if self._deployments else "deployment"
                    message += f"\n  d shows the {others:,} {other}{'' if others == 1 else 's'}."
            self._hide_table(table, layout, message)
            return
        self._show_nomatch(None)
        was_hidden = not table.display
        table.display = True

        plan: list = []
        job_by_item: dict = {}
        sub_by_item: dict = {}
        drawn: dict = {}
        target_row = None
        groups = (("LIVE", live), ("HISTORY", shown_history))
        if self._sort == "status" and not self._descending:
            groups = groups[::-1]  # status ↑ ended first: the ended jobs are what comes first
        for group, jobs in groups:
            if not jobs:
                continue
            so_far = live_so_far if group == "LIVE" else older or reading
            plan.append((f"hdr-{group.lower()}", self._header(group, len(jobs), layout.labels, so_far)))
            if plan[-1][0] == keep_header:
                target_row = len(plan) - 1
            for job in jobs:
                job_key = self._job_key(job)
                item_id = self._row_key(job_key)
                job_by_item[item_id] = job_key
                drawn[job_key] = job
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
        self._drawn = drawn
        # Reuse identical row objects, or reorder them without rebuilding; new widths alone
        # (a column dragged, the screen resized) change the columns in place.
        if self._layout is not None and self._layout.labels == layout.labels:
            self._set_columns(table, layout)
        shown = self._shown
        same = (shown is not None and shown[0] == layout.labels and len(shown[1]) == len(plan)
                and table.row_count == len(plan)
                and all(a[0] == b[0] and a[1] is b[1] for a, b in zip(shown[1], plan)))
        if not same:
            if (shown is not None and shown[0] == layout.labels and table.row_count == len(plan)
                    and {key: id(cells) for key, cells in shown[1]} == {key: id(cells) for key, cells in plan}):
                at = layout.labels.index("Job ID")
                position = {id(cells[at]): index for index, (_key, cells) in enumerate(plan)}
                table.sort("Job ID", key=lambda name: position[id(name)])
            elif (steady and shown is not None and shown[0] == layout.labels and table.row_count == len(shown[1])
                  and len({key for key, _cells in shown[1]} - {key for key, _cells in plan}) <= self._PATCH_REMOVALS):
                table.patch_rows(plan, shown[1])  # a page more: the rows on screen stay
            else:
                self._set_columns(table, layout)
                table.replace_rows(plan)
            self._shown = (layout.labels, tuple(plan))
        if target_row is None:
            target_row = 1  # first job row (row 0 is a group header)
        target_row = min(target_row, table.row_count - 1)
        self._placed_key = plan[target_row][0] if 0 <= target_row < len(plan) else None
        if at_top:
            # The reader hasn't moved: the list stays at its top, where new live jobs show.
            table.move_cursor(row=target_row, animate=False, scroll=False)
            table.scroll_to(x=table.scroll_x, y=0, animate=False, immediate=True)
            after_refresh(self, self._keep_top)
        elif anchor is not None:
            # The highlighted row stays on its line.
            table.move_cursor(row=target_row, animate=False, scroll=False)
            after_refresh(self, self._keep_view, *anchor)
        elif was_hidden:
            # Place again after layout without initially scrolling off the group header.
            table.move_cursor(row=target_row, animate=False, scroll=False)
            after_refresh(self, self._place_cursor, target_row)
        elif not same or top or table.cursor_row != target_row:
            self._place_cursor(target_row)

    def _keep_top(self) -> None:
        """After the list grew, the reader not having moved: its top stays in view (a refill scrolls home)."""
        table = self.query_one("#jobs", DataTable)
        if table.scroll_y:
            table.scroll_to(y=0, animate=False, immediate=True)

    def _keep_view(self, key: str, on_screen: int, scroll_x: float, tries: int = 10) -> None:
        """After the list grew: scroll so the row ``key`` is ``on_screen`` rows below the top
        of the view again (rows added above it don't push it down), and sideways as it was."""
        table = self.query_one("#jobs", DataTable)
        try:
            row = table.get_row_index(key)
        except Exception:  # noqa: BLE001 - gone with the new rows
            return
        top = max(0, row - on_screen)
        if top > table.max_scroll_y and tries:
            # Not yet as tall as its new rows: once it is.
            after_refresh(self, self._keep_view, key, on_screen, scroll_x, tries - 1)
            return
        table.scroll_to(x=scroll_x, y=top, animate=False, immediate=True)

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
        self._job_by_item, self._sub_by_item, self._drawn = {}, {}, {}
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
            job = self._job_for(self._job_by_item[key])
            if job is None:
                return  # drawn no more (a refresh between the press and this)
            # Every call about the job goes to its own database and schema.
            client = self._job_client(job)
            if client is not None:
                jid = views.job_id_of(job)
                self.app.push_screen(JobSummaryScreen(client, jid, job=job,
                                                      poll_interval=self._poll_interval, on_job=self.job_fetched))
        elif key in self._sub_by_item:
            job_key, sid = self._sub_by_item[key]
            job = self._job_for(job_key)
            if job is None:
                return
            client = self._job_client(job)
            if client is not None:
                jid = views.job_id_of(job)
                self.app.push_screen(LogScreen(client, jid, job=job, sub_job_id=sid, poll_interval=self._poll_interval))

    def _job_client(self, job: dict | None):
        """The job's location-bound client, or None after notifying."""
        try:
            return self._clients.for_job(job)
        except Exception as exc:  # noqa: BLE001 - shown, with the traceback in the error log
            log_traceback(f"no client for {views.location_label(job)}")
            self.notify(f"Can't open the job: no client for {views.location_label(job)} "
                        f"({views.describe_error(exc)}; details in {_ERROR_LOG}).", severity="error", markup=False)
            return None
