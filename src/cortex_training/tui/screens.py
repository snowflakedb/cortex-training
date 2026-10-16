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

"""Job preview and tabbed job detail screens for the TUI.

Flow: job list → :class:`JobSummaryScreen` (preview + sub-jobs) →
:class:`JobDetailScreen` (Logs | Resources) for the
whole job or one sub-job.

Everything here is read-only against the service. The only writes are local:
job display names (:class:`~cortex_training.tui.stores.AliasStore`) and the GPU
telemetry download cache. A job's name is its local alias, else its id; the
preview shows its whole ``comment`` too.
"""

from __future__ import annotations

import time
from datetime import datetime
from datetime import timezone
from time import monotonic

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.content import Content
from textual.containers import Vertical
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.screen import Screen
from textual.widgets import DataTable
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Input
from textual.widgets import Label
from textual.widgets import Static
from textual.widgets import TabbedContent
from textual.widgets import TabPane

from cortex_training.tui import views as v
from cortex_training.tui.app import KEYS_BINDING
from cortex_training.tui.app import FittedFooter
from cortex_training.tui.app import FollowsPointer
from cortex_training.tui.app import JobTabs
from cortex_training.tui.app import LogScreen
from cortex_training.tui.app import PickableTable
from cortex_training.tui.app import after_refresh
from cortex_training.tui.app import call_on_screen
from cortex_training.tui.app import close_keys_panel
from cortex_training.tui.app import hanging
from cortex_training.tui.app import is_shown
from cortex_training.tui.format import is_active_status
from cortex_training.tui.format import one_line
from cortex_training.tui.format import safe_text
from cortex_training.tui.format import source_label
from cortex_training.tui.resource import RANGES
from cortex_training.tui.stores import ALIAS_MAX_CHARS
from cortex_training.tui.stores import AliasStore

# Rich's grey62, by value: Textual's markup and styles (the preview's Content)
# don't know that name and would draw the default color instead.
ENDED_STYLE = "#9e9e9e"
STATUS_STYLE = {
    "running": "bold green",
    "initializing": "yellow",
    "placing": "yellow",
    "pending": "yellow",
    "queued": "yellow",
    "creating": "yellow",
    "failed": "bold red",
    "cancelled": ENDED_STYLE,
    "canceled": ENDED_STYLE,
    "terminated": ENDED_STYLE,
    "done": ENDED_STYLE,
    "succeeded": ENDED_STYLE,
    "completed": ENDED_STYLE,
    "unknown": "yellow",
}
# What a waiting job is doing, by state, when the service sends no reason_details.
_WAITING = {
    "pending": "waiting to be placed on GPUs",
    "queued": "waiting to be placed on GPUs",
    "creating": "waiting to be placed on GPUs",
    "placing": "waiting for its GPUs to come up",
    "initializing": "placed on GPUs; loading the model and starting the engines",
}


def status_text(status) -> Text:
    word = v.status_word(status)
    return Text(word.upper(), style=STATUS_STYLE.get(word, ""))


def _now() -> datetime:
    return datetime.now(timezone.utc)


_SUBJOB_COLUMNS = ("Sub-job", "Type", "Status", "Model", "GPUs", "Config")
_TYPE, _MODEL, _CONFIG = 1, 3, 5
# How the sub-job table fits a screen narrower than its cells: each layout leaves
# out some columns, then cuts (column, to this many cells) in order, each only as
# far as needed, until the table fits. A very long model name gives way first,
# then the end of each config; then the Type column, which repeats each sub-job's
# label (and the Type field above), is left out, and the model and the config are
# cut further. A screen too narrow even for that scrolls the table sideways.
_TABLE_FITS = (
    ((), ((_MODEL, 36), (_CONFIG, 24))),
    ((_TYPE,), ((_MODEL, 36), (_CONFIG, 24), (_MODEL, 10), (_CONFIG, 12))),
)


def _rows_of(content, width: int) -> int:
    """The screen rows ``content`` takes in ``width`` cells, the ones it wraps onto included."""
    wrap = getattr(content, "wrap", None)
    return len(wrap(max(1, width))) if wrap is not None else len(str(content).split("\n"))


def _column_widths(labels, rows: list) -> list:
    """Each column's width with nothing cut: its widest cell, or its label."""
    return [max([v.cell_width(label)] + [cells[i].cell_len for _key, cells in rows]) for i, label in enumerate(labels)]


def fit_columns(labels, rows: list, width: int, pad: int = 2) -> tuple:
    """The sub-job table fitted to ``width`` cells, as :data:`_TABLE_FITS` says.

    ``rows`` are ``(key, cells)`` with one-line ``Text`` cells, and ``pad`` is
    the padding each column adds. Returns ``(labels, rows, cut, wider)``: the
    columns kept and the rows' cells for them (cut with ``…`` where needed),
    whether anything was cut or left out, and whether the table is still wider
    than ``width`` (so it scrolls sideways).
    """
    natural = _column_widths(labels, rows)
    for dropped, cuts in _TABLE_FITS:
        keep = [i for i in range(len(labels)) if i not in dropped]
        widths = {i: natural[i] for i in keep}
        over = sum(widths[i] + pad for i in keep) - width
        for column, least in cuts:
            if over <= 0:
                break
            take = min(over, max(0, widths[column] - least))
            widths[column] -= take
            over -= take
        if over <= 0:
            break
    cut = len(keep) < len(labels) or any(widths[i] < natural[i] for i in keep)
    fitted = [(key, [_fit_text(cells[i], widths[i]) for i in keep]) for key, cells in rows]
    return [labels[i] for i in keep], fitted, cut, over > 0


_SEPARATOR = "  ·  "


def fit_line(segments: list, width: int) -> Content:
    """``(text, style, least)`` segments joined by `` · `` to fit ``width`` cells.

    When cutting the segments that have a ``least`` (the first ones first, down
    to that many cells, with ``…``) puts the line on one row, they are cut just
    enough; the others stay whole. Otherwise nothing is cut and the line wraps
    between segments, so no row ends with a separator. The texts are shown
    literally, never as markup.
    """
    texts = [text for text, _style, _least in segments]
    sep = v.cell_width(_SEPARATOR)
    over = sum(v.cell_width(text) for text in texts) + sep * (len(texts) - 1) - width
    spares = [max(0, v.cell_width(text) - least) if least is not None else 0 for text, _style, least in segments]
    if 0 < over <= sum(spares):  # cutting gets it onto one row
        for i, spare in enumerate(spares):
            if over <= 0:
                break
            if spare:
                cut = min(spare, over)
                texts[i] = v.fit_cells(texts[i], v.cell_width(texts[i]) - cut)
                over -= cut
    rows: list = []
    for text, (_text, style, _least) in zip(texts, segments):
        piece = Content.styled(text, style)
        if rows and rows[-1].cell_length + sep + piece.cell_length <= width:
            rows[-1] = rows[-1] + Content(_SEPARATOR) + piece
        else:
            rows.append(piece)
    return Content("\n").join(rows)


def _fit_text(text: Text, width: int) -> Text:
    """A one-line cell cut to ``width`` terminal cells, ending in ``…`` when cut
    (never in ``·…``), its style kept."""
    if text.cell_len <= width:
        return text
    cut = v.fit_cells(text.plain, width)
    return Text(cut[:-1].rstrip(" ·") + "…", style=text.style)


class PromptScreen(FollowsPointer, ModalScreen):
    """One-line text prompt; dismisses with the entered text, or ``None`` on Esc."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        # Ahead of the box's own ctrl+c, which copies the text selected in it, and the
        # box selects its text as the prompt opens: a highlight the mouse made since wins.
        Binding("ctrl+c", "copy_text", "Copy", show=False, priority=True),
    ]

    def __init__(self, title: str, value: str = "", placeholder: str = "", *, max_length: int = 0) -> None:
        """``max_length``: the most characters the text may have (0: no limit); the hint says so."""
        super().__init__()
        self._title = title
        self._value = value
        self._placeholder = placeholder
        self._max_length = max_length
        self._done = False

    def compose(self) -> ComposeResult:
        with Vertical(id="prompt"):
            yield Label(Content(self._title), id="prompt-title")
            yield Input(value=self._value, placeholder=self._placeholder, id="prompt-input",
                        max_length=self._max_length)
            limit = f" · up to {self._max_length} characters" if self._max_length else ""
            yield Label(f"Enter to save · Esc to cancel{limit}", id="prompt-hint")

    def on_mount(self) -> None:
        self.set_focus(self.query_one("#prompt-input", Input))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._finish(event.value)

    def action_copy_text(self) -> None:
        """ctrl+c or a right-click: what the mouse highlighted, else the text selected in
        the box, else what the prompt shows (as a copy does on every other screen)."""
        if self.app.copy_highlight(self):
            return
        box = self.query_one(Input)
        if box.selected_text:
            self.app.copy_out(box.selected_text, f"{len(box.selected_text):,} characters")
        else:
            self.app.copy_everything(self, "the prompt")

    def action_cancel(self) -> None:
        self._finish(None)

    def _finish(self, value) -> None:
        # Once. Two quick Enters submit twice, and dismiss() pops whatever screen is
        # on top: the second would close the screen under this prompt.
        if self._done or not is_shown(self):
            return
        self._done = True
        self.dismiss(value)


# ─── job preview ─────────────────────────────────────────────────────────────


class JobSummaryScreen(FollowsPointer, Screen):
    """Preview of one job: identity, config, status, and a sub-job breakdown."""

    # Back, quit and the keys panel first: a narrow terminal cuts the footer's end.
    BINDINGS = [
        ("b", "back", "Back"),
        Binding("escape", "close_or_back", "Back", show=False),
        ("q", "app.quit", "Quit"),
        KEYS_BINDING,
        ("l", "open('logs')", "Logs"),
        ("g", "open('resource')", "Resources"),
        ("n", "rename", "Name job"),
        ("r", "refresh", "Refresh"),
        # One key, two bindings: check_action enables the one that applies, and the
        # footer shows only that one, and only while a value is too long to show
        # whole. Last, so a narrow footer cuts it rather than r or q (the comment's
        # note names the key too).
        Binding("e", "expand_values", "Full text"),
        Binding("e", "collapse_values", "Less text"),
    ]

    def __init__(self, client, job_id: str, *, job: dict | None = None, aliases: AliasStore | None = None,
                 poll_interval: float = 1.0, on_job=None) -> None:
        """``on_job(record)`` is called with the job each time a fresh read of it
        arrives (the job list then shows that status too)."""
        super().__init__()
        self._client = client
        self._job_id = job_id
        self._job = job or {"job_id": job_id}
        self._on_job = on_job
        self._aliases = aliases or AliasStore()
        self._poll_interval = poll_interval
        # A thread can't be cancelled, so never stack a second fetch.
        self._loading = False
        self._fetch_since: float | None = None  # monotonic start of the get_job in flight
        self._fetched_at: str | None = None  # local HH:MM:SS of the last good get_job answer
        self._fetch_error: str | None = None  # why the last get_job gave no copy, until one does
        self._scope_by_row: dict = {}
        self._reread_on_resume = False  # set when the details open over this preview
        self._expanded = False  # e: every value whole, however long, and the whole sub-job table
        self._long = False  # some value or table cell is (or, expanded, would be) cut
        self._table_shown = None  # what the sub-job table holds, so an unchanged refresh leaves it be

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="summary-body"):
            yield Static("", id="preview")
            yield Static("", id="fetch")  # reading, refreshing, or why the copy shown is older
            yield Static("", id="health")
            yield Static("", id="subjobs-title")  # says what a click on a row opens
            yield PickableTable(id="subjobs", cursor_type="row", zebra_stripes=True)  # a drag picks rows to copy
        yield Footer()

    _REFRESH_SECONDS = 60

    def on_mount(self) -> None:
        table = self.query_one("#subjobs", DataTable)
        self._refresh_view()
        self._load()
        # Focus without scrolling: on a short terminal a long comment would put the
        # preview's top (the name and comment) above the screen, out of reach of the
        # keys, which all go to the table.
        self.set_focus(table, scroll_visible=False)
        self.set_interval(self._REFRESH_SECONDS, self._refresh_tick)
        self.set_interval(1, self._fetch_tick)

    def _refresh_tick(self) -> None:
        # Keep a live job's status and times current while the preview is showing.
        if self.app.screen is self and v.is_live(self._job):
            self._load()

    def on_screen_resume(self) -> None:
        self._refresh_view()
        # Back from the details, which may have seen the status move on: re-read the job
        # now. Not after the rename prompt: a live get_job can take minutes to answer.
        reread, self._reread_on_resume = self._reread_on_resume, False
        if reread and v.is_live(self._job):
            self._start_fetch()

    @property
    def _display_name(self) -> str:
        return v.display_name(self._job, self._aliases.all())

    _LABEL_WIDTH = 15  # "  " + a 12-column label + " "

    def _value_width(self) -> int:
        """Columns left for a preview value after its label (100 before the first layout)."""
        try:
            width = self.query_one("#preview", Static).size.width
        except Exception:  # noqa: BLE001 - not mounted yet
            width = 0
        return max(20, (width or 100) - self._LABEL_WIDTH)

    def _value_lines(self, value: str) -> list:
        """A preview value (the comment, a long GPU line) as the rows it takes,
        wrapped to the value column by terminal cells, so wide (CJK, emoji) text
        stays in it. Printable first: a control character would break the count."""
        width = self._value_width()
        return [row for line in v.printable(value).split("\n") for row in v.wrap_cells(line, width)]

    def _cut_rows(self, rows: list, keep: int) -> list:
        """The first ``keep`` rows, the last ending in ``…``: the value goes on."""
        kept = rows[:keep]
        kept[-1] = v.fit_cells(f"{kept[-1].rstrip()} …", self._value_width())
        return kept

    def _value_block(self, rows: list) -> str:
        """Rows of a value, the continuation rows under the value column."""
        return ("\n" + " " * self._LABEL_WIDTH).join(rows)

    # Without e, a value other than the comment shows at most this many rows. The
    # comment gets the rows the rest of the preview leaves on the screen, and at
    # least the fewest here, so every field shows when the preview opens.
    _VALUE_ROWS = 3
    _COMMENT_ROWS_MIN = 3
    # The line under a value cut short: how many rows it has left, and the key.
    _MORE = "\n" + " " * _LABEL_WIDTH + "[dim](+$more rows · e shows the whole {what})[/dim]"
    # Rows outside the parts measured in _rows_besides: the header and footer, the
    # body's top and bottom padding, and the health line's margins, the bottom one
    # shared with the sub-jobs title's top margin (adjacent margins collapse; the
    # app's CSS for #summary-body, #health, #subjobs-title).
    _CHROME_ROWS = 2 + 2 + 2

    def _rows_besides(self, fields: list, health: Content, title: Content, table_rows: int) -> int:
        """The screen rows everything but the comment takes (``table_rows``: the
        sub-job table's, its header and any sideways scrollbar included)."""
        width = self._value_width() + self._LABEL_WIDTH
        fetch = self.query_one("#fetch", Static)
        lines = [*fields, health, title, *([fetch.render()] if fetch.display else [])]
        return self._CHROME_ROWS + sum(_rows_of(line, width) for line in lines) + table_rows

    def _comment_rows(self, rows: list, besides: int) -> tuple:
        """``(rows to show, rows left out, too long)`` for the comment: all of it
        when it fits in the screen rows the rest leaves (or with e), else as many
        as do (at least a few), with one more for the note that says so."""
        room = (self.size.height or self.app.size.height) - besides
        if len(rows) <= max(room, self._COMMENT_ROWS_MIN):
            return rows, 0, False
        if self._expanded:
            return rows, 0, True
        keep = max(self._COMMENT_ROWS_MIN, room - 1)
        return self._cut_rows(rows, keep), len(rows) - keep, True

    def check_action(self, action: str, parameters) -> bool | None:
        if action == "expand_values":
            return self._long and not self._expanded
        if action == "collapse_values":
            return self._long and self._expanded
        return True

    def action_expand_values(self) -> None:
        self._expanded = True
        self._refresh_view()
        self.refresh_bindings()

    def action_collapse_values(self) -> None:
        self._expanded = False
        self._refresh_view()
        self.refresh_bindings()

    def _refresh_view(self) -> None:
        job = self._job
        if self.app.screen is self:  # a late load must not retitle the details on top
            self.app.sub_title = f"job preview  ·  {v.shown_name(job, self._aliases.all())}"
        now = _now()
        rows = []
        name = self._display_name
        hint = "local name on this machine · n renames" if name else "no name · press n to name this job"
        name_hint = f"  [dim]({hint})[/dim]"
        # Keep the one-line name and its hint on one line of the preview (cells, not characters).
        name = v.one_line(name, max(20, self._value_width() - v.cell_width(hint) - 4)) if name else name
        # Values (names, comments, ids) are user or server text: pass them as
        # $variables so they are shown literally, never parsed as markup.
        comment = None  # (its place in rows, its rows): it gets what the rest leaves
        long = False
        for label, value in v.summary_fields(job, name, now):
            more = 0  # rows of the value left out
            if label == "Status":
                word = v.status_word(job.get("status"))
                style = STATUS_STYLE.get(word, "")
                if v.is_unknown(job):  # no live state: the health line below says why
                    live = "  [yellow]? status unavailable[/yellow]"
                else:
                    live = "  [green]● live[/green]" if v.is_live(job) else "  [dim]historical[/dim]"
                value_markup = (f"[{style}]$value[/]" if style else "$value") + live
            elif label == "Name":
                value_markup = ("[b]$value[/b]" if name else "[dim]$value[/dim]") + name_hint
            else:
                # A long value (the comment, a GPU line naming another job) continues under the column.
                lines = self._value_lines(value)
                if label == "Comment":
                    comment = (len(rows), lines)
                    continue
                if len(lines) > self._VALUE_ROWS:
                    long = True
                    if not self._expanded:
                        more = len(lines) - self._VALUE_ROWS
                        lines = self._cut_rows(lines, self._VALUE_ROWS)
                value = self._value_block(lines)
                # A cut value says so under it, with the key that shows the rest (a
                # narrow footer has no room for that key).
                value_markup = (("[b]$value[/b]" if label == "Job ID" else "$value")
                                + (self._MORE.format(what="value") if more else ""))
            rows.append(Content.from_markup(f"  [dim]{label:<12}[/dim] {value_markup}", value=value, more=str(more)))
        self._render_fetch()
        width = self._value_width() + self._LABEL_WIDTH
        # Wrapped lines of the health line continue under its text, after the "  ✖ " lead-in.
        health = hanging(self._health(), width, 4)
        self.query_one("#health", Static).update(health)

        table = self.query_one("#subjobs", DataTable)
        subjob_rows = self._subjob_rows(job)
        table_width = self._table_width(table, len(subjob_rows))
        labels, fitted, cut, wider = fit_columns(_SUBJOB_COLUMNS, subjob_rows, table_width, 2 * table.cell_padding)
        if self._expanded:  # every column, every cell whole: the table scrolls sideways when it must
            labels, fitted = list(_SUBJOB_COLUMNS), subjob_rows
            wider = sum(w + 2 * table.cell_padding for w in _column_widths(labels, fitted)) > table_width
        long = long or cut
        self._show_table(table, labels, fitted, wider)
        # What a click on a row opens, then the keys for a table that is cut to fit, or wider
        # than the screen (short, to keep to one row on an 80-column screen).
        hints = ["Click a job to open its telemetry dashboard"]
        if cut:
            hints.append("e fit to screen" if self._expanded else "e whole table")
        if wider:
            hints.append("← → scroll")
        title = self._subjobs_title(hints, width)
        self.query_one("#subjobs-title", Static).update(title)
        if comment is not None:
            # The whole comment when the screen has the rows for it, else its start and a note.
            at, lines = comment
            limit = table.styles.max_height
            table_rows = len(fitted) + 1 + (1 if wider else 0)  # its header, and its sideways scrollbar
            if limit is not None and limit.is_cells:
                table_rows = min(table_rows, int(limit.value))
            shown, more, too_long = self._comment_rows(
                lines, self._rows_besides(rows, health, title, table_rows))
            long = long or too_long
            note = self._MORE.format(what="comment") if more else ""
            rows.insert(at, Content.from_markup(f"  [dim]{'Comment':<12}[/dim] $value{note}",
                                                value=self._value_block(shown), more=str(more)))
        self.query_one("#preview", Static).update(Content("\n").join(rows))
        if long != self._long:
            self._long = long
            self.refresh_bindings()  # e appears in the footer, or goes

    _TITLE = "Sub-jobs  ("

    def _subjobs_title(self, hints: list, width: int) -> Content:
        """``Sub-jobs  (hint · hint)`` in ``width`` cells: a row breaks between two
        hints (without the ``·``) and goes on under the first hint; only a hint too
        long for a row of its own breaks inside."""
        indent = v.cell_width(self._TITLE)
        rows = [self._TITLE + hints[0]]
        for hint in hints[1:]:
            if v.cell_width(rows[-1]) + v.cell_width(f" · {hint})") <= width:
                rows[-1] += f" · {hint}"
            else:
                rows.append(" " * indent + hint)
        rows[-1] += ")"
        text = Content("\n").join(hanging(Content(row), width, indent) for row in rows).plain
        bold = len("Sub-jobs")
        return Content.styled(text[:bold], "bold") + Content.styled(text[bold:], "dim")

    def _show_table(self, table: DataTable, labels: list, rows: list, wider: bool) -> None:
        """Put these columns and rows in the sub-job table, unless it already holds them
        (so a refresh that changes nothing keeps its cursor and sideways scroll)."""
        shown = (tuple(labels), wider,
                 tuple((key, tuple((cell.plain, str(cell.style)) for cell in cells)) for key, cells in rows))
        if shown == self._table_shown:
            return
        self._table_shown = shown
        # Sideways scrolling only for a table wider than the screen. Setting overflow
        # also has Textual re-check the scrollbar, which it otherwise does only when a
        # size changes; a table refitted to exactly its width would keep a stale one.
        table.styles.overflow_x = "auto" if wider else "hidden"
        keep = table.cursor_row
        # Columns too: a column only ever widens, so one cut to fit would stay as wide
        # as its widest cell so far (before the first layout, or on a wider screen).
        table.clear(columns=True)
        table.add_columns(*labels)
        for key, cells in rows:
            table.add_row(*cells, key=key)
        table.move_cursor(row=min(keep, table.row_count - 1), animate=False)  # keep the row on refresh

    def _subjob_rows(self, job: dict) -> list:
        """The sub-job table's rows as ``(key, cells)``; also maps each row to its scope."""
        self._scope_by_row = {"all": None}
        # Text cells: a plain str cell is parsed as markup.
        rows = [("all", [Text("▸ entire job", style="bold"), Text(v.job_type_label(job)), status_text(job.get("status")),
                         Text(", ".join(v.job_models(job)) or "—"), Text(v.job_gpu_cell(job, "requested")),
                         Text("all sub-jobs", style="dim")])]
        subs = v.sub_jobs(job)
        if v.is_unknown(job):
            # Live sub-job state is unavailable (the line above has the service's sentence):
            # say so instead of showing no sub-jobs.
            rows.append(("note", [Text("  ? status unavailable", style="yellow"), Text(""), Text("UNKNOWN", style="yellow"),
                                  Text(""), Text(""), Text("configs at creation" if subs else
                                                           "the service sent no sub-job configs", style="dim")]))
        for i, sj in enumerate(subs):
            key = f"sj-{i}"
            self._scope_by_row[key] = sj.get("sub_job_id")
            config = v.subjob_config_summary(sj)
            target = v.multiplex_target(sj)
            if target:
                # After the config, which a narrow table keeps; the GPUs line names the host in full.
                config = " · ".join(p for p in (config, f"runs on {v.short_sub_job_id(target)}") if p)
            # Its type as the list spells it (log_prob, as the sub-job id does), beside its label.
            rows.append((key, [Text(f"  {v.sub_job_label(sj)}"), Text(v.sub_job_type(sj)),
                               status_text(sj.get("status")), Text(str(sj.get("model_name") or "—")),
                               Text(v.sub_job_gpu_text(sj)), Text(config)]))
        return rows

    def _table_width(self, table: DataTable, rows: int) -> int:
        """The cells the sub-job table has for its columns: its width, less a vertical
        scrollbar when its rows (and header) are more than it shows."""
        # Before the first layout: the screen less the body's padding, the table's
        # margins and the body's scrollbar gutter (kept whether or not it scrolls).
        width = table.size.width or max(40, self.app.size.width - 6)
        limit = table.styles.max_height
        if limit is not None and limit.is_cells and rows + 1 > limit.value:
            width -= table.styles.scrollbar_size_vertical
        return width

    def on_resize(self, event) -> None:
        if self.query_one("#subjobs", DataTable).columns:  # mounted: re-wrap the comment, refit the table
            after_refresh(self, self._refresh_view)

    # Keys the sub-job table moves its cursor with.
    _TABLE_MOVES = {"up", "down", "pageup", "pagedown", "ctrl+home", "ctrl+end"}

    def on_key(self, event) -> None:
        # The preview opens at its top (name and comment) and the keys go to the
        # sub-job table, which a long comment can push below the screen: bring the
        # table into view once its cursor moves, and go back to the top on up from
        # its first row, or ctrl+home.
        table = self.query_one("#subjobs", DataTable)
        if self.focused is not table or event.key not in self._TABLE_MOVES:
            return
        body = self.query_one("#summary-body", VerticalScroll)
        if event.key == "ctrl+home" or (event.key == "up" and table.cursor_row <= 0):
            body.scroll_home(animate=False)
        else:
            after_refresh(self, body.scroll_to_widget, table, animate=False)

    def _health(self) -> Content:
        job = self._job
        word = v.status_word(job.get("status"))
        reason, details = v.collapse(job.get("reason") or ""), v.collapse(job.get("reason_details") or "")
        if word == "failed" and not reason:
            reason = "not reported"  # a failure always gets its reason line
        # The service's words, one way for every state: "reason: X — details", the
        # details alone without a reason, or nothing.
        why = ("  reason: [b]$reason[/b]" if reason else "") + ((" — $details" if reason else "  $details")
                                                                 if details else "")

        def line(lead: str, tail: str = "", **values) -> Content:
            return Content.from_markup(f"  {lead}{why}{tail}", reason=reason, details=details, **values)

        if word == "failed":
            # A job that never got GPUs has no logs to check (placement_timeout and the like).
            hint = ("it never got GPUs, so no sub-job started or wrote logs" if v.never_started(job)
                    else "check Logs (l) for the failing sub-job")
            return line("[bold red]✖ Job failed[/]", "  — $hint", hint=hint)
        if word in ("cancelled", "canceled"):
            return line(f"[{ENDED_STYLE}]■ Cancelled[/]")
        if word == "terminated":
            return line(f"[{ENDED_STYLE}]■ Terminated by the platform[/]")
        if word in ("succeeded", "completed", "done"):
            return line(f"[{ENDED_STYLE}]■ Completed[/]")
        if word in _WAITING:
            # The service says why a job waits (e.g. quota_filled) and, often, how far
            # off it is; without that, what the state means.
            return line(f"[yellow]◔ {word.capitalize()}[/]", "" if details else " — $doing", doing=_WAITING[word])
        if word == "running":
            return Content.from_markup(
                "  [green]● Running[/] — logs stream live; resources refresh automatically")
        if word == "unknown":
            # The service could not read the job's live state: it may still be running,
            # unless the service says the control plane has no record of the job.
            text = v.status_unavailable_text(job)
            hint = "" if "not found" in text.lower() else " — the job may still be running"
            return Content.from_markup(f"  [yellow]? $text[/]{hint}", text=text)
        # A status this client doesn't know: the service's reason, if it sent one.
        return Content.from_markup("  " + why.lstrip(), reason=reason, details=details) if why else Content("")

    def _load(self) -> None:
        self._start_fetch()

    def _start_fetch(self) -> None:
        if not self._loading:
            self._loading = True
            self._fetch_since = monotonic()
            self._fetch_job()
        self._render_fetch()  # "refreshing…", or how long the read in flight has taken

    @work(thread=True, group="summary")
    def _fetch_job(self) -> None:
        try:
            job, error = self._client.get_job(self._job_id), None
        except Exception as exc:  # noqa: BLE001 - said under the fields; the copy on screen stays
            job, error = None, v.collapse(v.describe_error(exc))
        if not (isinstance(job, dict) and job):
            job, error = None, error or "the service sent an empty answer"
        self._post(self._apply_job, job, error)

    def _post(self, fn, *args) -> None:
        call_on_screen(self, fn, *args)

    def _apply_job(self, job: dict | None, error: str | None = None) -> None:
        self._loading = False
        self._fetch_since = None
        if job:
            # The service's fields come only from the fresh answer: a reason it stopped sending disappears.
            self._job = v.refresh_job(self._job, job)
            self._fetched_at, self._fetch_error = time.strftime("%H:%M:%S"), None
            if self._on_job is not None:
                self._on_job(self._job)
        else:
            self._fetch_error = error or "the service sent an empty answer"
        self._refresh_view()

    # A read in flight longer than this shows how long it has taken (the client
    # retries a failing answer for about a minute).
    _FETCH_SLOW_SECONDS = 5

    def _fetch_tick(self) -> None:
        if self._loading and self.app.screen is self:
            self._render_fetch()

    def _fetch_line(self) -> Content:
        """The line under the fields: a read in flight, the last one's failure, or when the copy was read."""
        took = ""
        if self._loading and self._fetch_since is not None:
            seconds = int(monotonic() - self._fetch_since)
            took = f" ({seconds}s)" if seconds >= self._FETCH_SLOW_SECONDS else ""
        error, at = self._fetch_error or "", self._fetched_at or ""
        if self._loading:
            if error:
                return Content.from_markup(f"  [dim]refreshing…{took}[/dim]  [yellow](last try: $error)[/yellow]",
                                           error=error)
            if at:
                return Content.from_markup(f"  [dim]refreshing…{took}  (last read at $at)[/dim]", at=at)
            return Content.from_markup(f"  [dim]reading the job from the service…{took}[/dim]")
        if error:
            # The fields above are the last good copy: this preview's, or the job list's.
            if at:
                return Content.from_markup("  [yellow]couldn't refresh: $error; showing the copy read at $at[/yellow]",
                                           error=error, at=at)
            return Content.from_markup("  [yellow]couldn't read the job: $error; showing the job list's copy[/yellow]",
                                       error=error)
        if at:
            return Content.from_markup("  [dim]read from the service at $at[/dim]", at=at)
        return Content("")

    def _render_fetch(self) -> None:
        try:
            line = self.query_one("#fetch", Static)
        except Exception:  # noqa: BLE001 - not mounted, or already removed
            return
        content = self._fetch_line()
        line.display = bool(content.plain)  # no blank row before the first read
        # A long error continues under its text, after the two-cell indent.
        line.update(hanging(content, self._value_width() + self._LABEL_WIDTH, 4))

    def action_refresh(self) -> None:
        if self._loading:
            # The read is in flight (a thread can't be cancelled): say so, not nothing.
            self.notify("Still reading this job; r asks again once it answers.")
            return
        self._load()

    def _highlighted_scope(self):
        table = self.query_one("#subjobs", DataTable)
        try:
            key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        except Exception:  # noqa: BLE001 - empty table
            key = "all"
        return self._scope_by_row.get(key)

    def whole_id(self, key) -> str | None:
        """The whole id of the sub-job on row ``key``, the job's on the "all" row."""
        return self._scope_by_row.get(key) or (self._job_id if key == "all" else None)

    def action_copy_text(self) -> None:
        """ctrl+c or a right-click: what the mouse highlighted, else the whole preview."""
        if not self.app.copy_highlight(self):
            self.app.copy_everything(self, "the preview")

    def action_back(self) -> None:
        if is_shown(self):  # not while still mounting, nor the screen under a pushed one
            self.app.pop_screen()

    def action_close_or_back(self) -> None:
        """Esc: close the keys panel when it is open, else go back."""
        if is_shown(self) and not close_keys_panel(self):
            self.app.pop_screen()

    def action_open(self, tab: str) -> None:
        if is_shown(self):
            self._open_detail(self._highlighted_scope(), tab)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # Not a second quick Enter, after the first opened the details.
        if is_shown(self) and event.row_key.value in self._scope_by_row:  # not the "status unavailable" line
            self._open_detail(self._scope_by_row[event.row_key.value], "logs")

    def _open_detail(self, sub_job_id, tab: str) -> None:
        self._reread_on_resume = True
        self.app.push_screen(
            JobDetailScreen(self._client, self._job_id, job=self._job, sub_job_id=sub_job_id, initial_tab=tab,
                            aliases=self._aliases, poll_interval=self._poll_interval)
        )

    def action_rename(self) -> None:
        if not is_shown(self):
            return

        def done(value):
            if value is not None:
                self._aliases.set(self._job_id, v.alias_to_store(self._job, value))
                self._refresh_view()

        current, placeholder = v.rename_prompt(self._job, self._aliases.all())
        self.app.push_screen(PromptScreen(f"Display name for {self._job_id}", current, placeholder,
                                          max_length=ALIAS_MAX_CHARS), done)


# ─── job detail (tabs) ───────────────────────────────────────────────────────

# Copying (ctrl+c, a right-click) works on every tab, what the mouse highlighted or else everything the tab holds: not here.
_LOG_ONLY = {"save_log", "cycle_level", "toggle_pause", "shrink_sources", "grow_sources", "filter"}
# Pane ids match LogScreen's ("logs", "resource"), so its tab logic applies as-is.
_TAB_IDS = ("logs", "resource")
_TAB_FOCUS = {"logs": "#sources", "resource": "#resource-list"}


class JobDetailScreen(LogScreen):
    """Logs | Resources for a job, optionally scoped to one sub-job.

    Extends :class:`LogScreen`, so the Logs tab keeps its live tail, cache,
    saved-console fallback, filter, level, pause, save, and copy behavior, and
    the Resources tab is LogScreen's view of one sub-job's GPUs or nodes at a
    time: the one selected in Logs until ``<`` or ``>`` picks another.

    Textual runs a handler such as ``on_mount`` for every class in the MRO, so
    the handlers here never call ``super()`` for them.
    """

    # 2 and 3 are one footer entry ("23 Tabs"), in the places LogScreen keeps for
    # them, after Back and Quit. The preview's letters open the same tabs here;
    # on Logs, l cycles the level instead (LogScreen's binding comes first, and
    # check_action turns it off on the other tabs).
    # w takes LogScreen's place for it, ahead of the Resources tab's sub-job keys.
    _TABS = Binding.Group("Tabs", compact=True)
    BINDINGS = [
        Binding("2", "show_tab('logs')", "Logs", group=_TABS),
        Binding("3", "show_tab('resource')", "Resources", group=_TABS),
        # On Logs, l cycles the level (check_action turns it off elsewhere, where l opens Logs).
        Binding("l", "cycle_level", "Level"),
        Binding("l", "show_tab('logs')", "Logs", show=False),
        Binding("g", "show_tab('resource')", "Resources", show=False),
        Binding("w", "cycle_window", "Window"),
        # Keep the footer readable next to the tab keys; these still work (? lists them).
        Binding("[", "shrink_sources", "Narrower", show=False),
        Binding("]", "grow_sources", "Wider", show=False),
    ]
    _DATA_REFRESH_SECONDS = 30
    _STATUS_REFRESH_SECONDS = 60
    _SUBTITLE_REASON = False  # the summary line shows reason and reason_details
    # On a narrow line, w leaves out the windows' names: the header names the window shown.
    _RESOURCE_KEY_HELP = ((f"w window {' · '.join(RANGES)}", "w window"), ("r refresh", "r refresh"))

    def __init__(self, client, job_id, *, job: dict | None = None, sub_job_id=None, initial_tab: str = "logs",
                 aliases: AliasStore | None = None, poll_interval=1.0,
                 job_status=None) -> None:
        super().__init__(client, job_id, sub_job_id=sub_job_id, poll_interval=poll_interval,
                         job_status=job_status or (job or {}).get("status"))
        self._job = job or {"job_id": job_id}
        self._scope = sub_job_id
        self._initial_tab = initial_tab if initial_tab in _TAB_IDS else "logs"
        self._active_tab = self._initial_tab
        self._aliases = aliases or AliasStore()

    # ── layout ──
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="summary")
        with JobTabs(initial=self._initial_tab, id="job-tabs"):
            with TabPane("Logs", id="logs"):
                yield from self._compose_logs()
            with TabPane("Resources", id="resource"):
                yield from self._compose_resource()
        yield FittedFooter(id="footer")

    def on_mount(self) -> None:
        self._render_summary_line()
        self.set_interval(self._DATA_REFRESH_SECONDS, self._auto_refresh_data)
        self.set_interval(self._STATUS_REFRESH_SECONDS, self._refresh_status)

    def on_screen_resume(self) -> None:
        self._update_subtitle()  # a screen pushed over this one may have replaced it

    # The fewest cells the name and the models are cut to so the summary line fits on one row.
    _SUMMARY_LEAST = 12

    def _render_summary_line(self) -> None:
        job = self._job
        # Short here: this line also carries the id, scope, type, models and reason;
        # the preview shows the whole name.
        name = v.one_line(v.display_name(job, self._aliases.all()), 48)
        job_id = v.job_id_of(job) or self._job_id
        # A sub-job's id starts with its job's (JOB_ID:TYPE:INDEX): name it once.
        if not self._scope:
            ids = [job_id, "all sub-jobs"]
        elif str(self._scope).startswith(f"{job_id}:"):
            ids = [f"sub-job {self._scope}"]
        else:
            ids = [job_id, f"sub-job {self._scope}"]
        # Why the job is where it is, in the service's words: reason and reason_details,
        # or the sentence it sends when the job's status is unavailable.
        why = v.status_unavailable_text(job) if v.is_unknown(job) else v.reason_text(job)
        word = v.status_word(job.get("status"))
        style = "red" if word == "failed" else "yellow" if v.is_live(job) else ENDED_STYLE
        # Where the job is (its calls go there), when the account-wide list says it is
        # somewhere other than the configured location.
        elsewhere = getattr(self.app, "job_elsewhere", None)
        location = v.location_label(job) if not callable(elsewhere) or elsewhere(job) else ""
        least = self._SUMMARY_LEAST
        # The name, then the models, are cut to keep the line on one row; the ids,
        # the location and the reason stay whole.
        segments = [*([(name, "bold", least)] if name else []), *((text, "", None) for text in ids),
                    (v.job_type_label(job), "", None), (", ".join(v.job_models(job)) or "—", "", least),
                    *([(location, "", None)] if location else []), *([(why, style, None)] if why else [])]
        self._set_summary(fit_line(segments, self._summary_width()))

    def _summary_width(self) -> int:
        """The cells the summary line has: the screen's, less its padding, before the first layout."""
        try:
            width = self.query_one("#summary", Static).size.width
        except Exception:  # noqa: BLE001 - not mounted, or already removed
            width = 0
        return width or max(40, self.app.size.width - 2)

    def _on_job_loaded(self, job: dict) -> None:
        # The service's fields come only from the fresh answer, so a reason (or
        # reason_details) it stopped sending disappears from this line and the subtitle.
        self._job = v.refresh_job(self._job, job)
        if job.get("status"):
            self._job_reason = job.get("reason")
            self._update_subtitle()
        self._render_summary_line()

    def _live(self) -> bool:
        # LogScreen keeps _job_status current (get_job, and the tail seeing the job end).
        return v.is_live({"status": self._job_status or self._job.get("status")})

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
        ended = is_active_status(self._job_status) and not is_active_status(job.get("status"))
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

    # ── tabs & bindings ──
    def action_show_tab(self, tab: str) -> None:
        if tab in _TAB_IDS:
            self.query_one("#job-tabs", TabbedContent).active = tab

    def action_cycle_tab(self) -> None:
        i = _TAB_IDS.index(self._active_tab) if self._active_tab in _TAB_IDS else 0
        self.action_show_tab(_TAB_IDS[(i + 1) % len(_TAB_IDS)])

    def _sync_footer(self) -> None:
        # The footer stays on every tab; check_action scopes its keys to the tab.
        self.refresh_bindings()

    def _focus_active_tab(self) -> None:
        try:
            self.set_focus(self.query_one(_TAB_FOCUS.get(self._active_tab, "#sources")))
        except Exception:  # noqa: BLE001 - not mounted yet
            pass

    def check_action(self, action: str, parameters):
        tab = self._active_tab
        if action == "filter" and isinstance(self.focused, Input):
            return False  # / is a priority key (see LogScreen): in a text box it is typed
        if action in _LOG_ONLY:
            return tab == "logs"
        if action == "resource_range":
            return False  # digits pick tabs here, not windows; w cycles the Resources window
        if action == "cycle_window":
            return tab == "resource"
        if action == "resource_sub_job":
            # < and >: on Resources, with another sub-job to switch to.
            return tab == "resource" and len(self._resource_sub_job_ids()) > 1
        return True

    def _line_prefixes(self) -> tuple[str, str]:
        # Tag every log line with the sub-job being tailed, and line its later
        # rows (wrapped, or a traceback's lines) up under its text; the TUI's
        # own status lines stay as they are (LogScreen._prefixes).
        if not self._current_source:
            return "", ""
        tag = one_line(safe_text(source_label(self._current_source).replace(" #", ":")))
        return f"{tag} │ ", f"{' ' * v.cell_width(tag)} │ "

    def _ended_job(self) -> dict:
        """This view's copy of the job with the newest status and reason (the tail
        can see the job end before the next get_job refreshes the copy)."""
        job = {**self._job, "status": self._job_status or self._job.get("status")}
        if self._job_reason is not None:
            job["reason"] = self._job_reason
        return job

    def _never_started(self) -> bool:
        return v.never_started(self._ended_job())

    def _no_log_output(self) -> str:
        # A job that never got GPUs (placement_timeout and the like) wrote no logs: say why.
        if self._never_started():
            return f"No log output: {v.never_started_text(self._ended_job())}."
        return super()._no_log_output()

    def _auto_refresh_data(self) -> None:
        # Only the visible tab polls, and only while the job is live. Resources waits
        # until the status check sees the shown sub-job running: before that it has no samples.
        if self.app.screen is not self or not self._live():
            return
        if self._resource_active() and self._resource_polls():
            self._start_resource_refresh(force=True)

    # ── windows ──
    def _resource_job(self) -> dict:
        return self._job  # the Resources header's GPU type and requested GPUs

    def action_cycle_window(self) -> None:
        if self._resource_active():
            names = list(RANGES)
            i = names.index(self._resource_range) if self._resource_range in names else -1
            self.action_resource_range(names[(i + 1) % len(names)])

    def on_resize(self, event) -> None:
        # LogScreen.on_resize also runs (log re-wrap, Resources); refit the summary
        # line here.
        self.call_after_refresh(self._render_summary_line)
