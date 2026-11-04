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

"""The job details screen's Checkpoints tab.

``CheckpointTab`` is mixed into ``LogScreen`` (tui/logs.py), as ``ResourceTab`` is:
the tab lists the job's checkpoints (``list_checkpoints``), newest first, and shows
the highlighted one's details under the list: its fields, the job and training
sub-job that saved it and the job's training configuration. The buttons between
the list and the details download it (``d``), or copy the command that starts a
new training job, sampling job, or both from it. Each checkpoint's export is read
once, in the background, for its files: their sizes, and the step and type they
show when the list doesn't say (see tui/checkpoints.py). The export's links are
never shown or kept; a download asks for fresh ones.
"""

from __future__ import annotations
import json
import shlex
import threading
import time
from pathlib import Path
from datetime import datetime
from datetime import timezone
from rich.text import Text
from textual.content import Content
from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.containers import VerticalScroll
from textual.widgets import DataTable
from textual.widgets import Static
from textual.worker import get_current_worker
from cortex_training.tui import theme
from cortex_training.tui import views
from cortex_training.tui.checkpoints import BOTH
from cortex_training.tui.checkpoints import RESUMABLE
from cortex_training.tui.checkpoints import SAMPLING
from cortex_training.tui.checkpoints import TRAINING
from cortex_training.tui.checkpoints import USES
from cortex_training.tui.checkpoints import DownloadCancelled
from cortex_training.tui.checkpoints import download_checkpoint
from cortex_training.tui.checkpoints import new_job_body
from cortex_training.tui.checkpoints import use_plan
from cortex_training.tui.checkpoints import SORTS
from cortex_training.tui.checkpoints import WEIGHTS_ONLY
from cortex_training.tui.checkpoints import Checkpoint
from cortex_training.tui.checkpoints import compact_numbers
from cortex_training.tui.checkpoints import FilesInfo
from cortex_training.tui.checkpoints import extra_text
from cortex_training.tui.checkpoints import files_info
from cortex_training.tui.checkpoints import fmt_bytes
from cortex_training.tui.checkpoints import fmt_loss
from cortex_training.tui.checkpoints import parse_checkpoints
from cortex_training.tui.checkpoints import saves_no_checkpoints
from cortex_training.tui.checkpoints import sort_checkpoints
from cortex_training.tui.checkpoints import training_sub_jobs
from cortex_training.tui.format import local_time
from cortex_training.tui.format import printable
from cortex_training.tui.widgets import SelectableTable
from cortex_training.tui.widgets import _ColoredLinks
from cortex_training.tui.widgets import log_traceback
from cortex_training.tui.widgets import spans_content


class CheckpointTab:
    """The Checkpoints tab of :class:`~cortex_training.tui.logs.LogScreen`."""

    # The most checkpoints one refresh reads the files of. The rest are read by the
    # next refresh (r, or the 30-second one while the job is live).
    _CHECKPOINT_FILES_PER_REFRESH = 50

    def _init_checkpoint_state(self) -> None:
        self._ckpt_rows: list[Checkpoint] = []
        self._ckpt_loaded = False  # a list has arrived (it may be empty)
        self._ckpt_loading = False
        self._ckpt_error: str | None = None  # why the last read gave no list
        self._ckpt_read_at: str | None = None  # local HH:MM:SS of the last list read
        self._ckpt_sort = "created"
        self._ckpt_descending = True
        # Checkpoint id → its FilesInfo, or why its export failed. Read once per screen.
        self._ckpt_files: dict = {}
        self._ckpt_reading_files: set[str] = set()
        self._ckpt_highlighted: str | None = None  # the id the details are for
        self._ckpt_shown: tuple | None = None  # what the table holds, so an unchanged redraw leaves it be
        self._ckpt_copied: tuple | None = None  # (checkpoint id, use, command) last copied
        # The download running or last run: checkpoint id, its folder, and its progress or outcome.
        self._ckpt_download: dict | None = None
        self._ckpt_download_stop = threading.Event()

    _USE_HELP = {
        TRAINING: "copies the command for a new training job that resumes from it",
        SAMPLING: "copies the command for a new sampling job serving its weights",
        BOTH: "copies the command for a new job with both: training resumes from the resumable "
              "checkpoint, sampling serves the weights-only one of the same step",
    }

    def _compose_checkpoint_tab(self) -> ComposeResult:
        # The list's card (its border says how it is sorted and the keys), then, under
        # it, the highlighted checkpoint: its card with the actions, the box saying
        # what a button copied, and the card of the job that saved it.
        with Vertical(id="checkpoint-list-card"):
            yield Static("", id="checkpoint-status")
            yield SelectableTable(id="checkpoint-table", cursor_type="row")
        with VerticalScroll(id="checkpoint-pane"):
            with Vertical(id="checkpoint-card"):
                yield Static("", id="checkpoint-hero")
                # Segments to click, as the Resources tab's window selector: Download, then
                # what kind of new job to copy the command for.
                yield _ColoredLinks("", id="checkpoint-actions")
                yield Static("", id="checkpoint-action-status")
                yield Static("", id="checkpoint-detail")
            yield Static("", id="checkpoint-copied")
            with Vertical(id="checkpoint-origin-card"):
                yield Static("", id="checkpoint-origin")

    def _checkpoints_active(self) -> bool:
        return self._active_tab == "checkpoints"

    # ─── reading ─────────────────────────────────────────────────────────
    def _start_checkpoint_refresh(self, *, force: bool) -> None:
        if not self._screen_open or self._ckpt_loading or (self._ckpt_loaded and not force):
            return
        if saves_no_checkpoints(self._job):
            self._render_checkpoints()  # says why there is nothing to list
            return
        self._ckpt_loading = True
        self._render_checkpoints()
        self._read_checkpoints()

    @work(thread=True, exclusive=True, group="checkpoints")
    def _read_checkpoints(self) -> None:
        worker = get_current_worker()
        try:
            rows = parse_checkpoints(self._client.list_checkpoints(self._job_id))
        except Exception as exc:  # noqa: BLE001 - surfaced on the tab
            log_traceback(f"listing checkpoints of {self._job_id} failed")
            if not worker.is_cancelled:
                self._post(self._finish_checkpoint_refresh, None, views.describe_error(exc))
            return
        if not worker.is_cancelled:
            self._post(self._finish_checkpoint_refresh, rows, None)

    def _finish_checkpoint_refresh(self, rows: list[Checkpoint] | None, error: str | None) -> None:
        if not self._screen_open:
            return
        self._ckpt_loading = False
        self._ckpt_error = error
        if rows is not None:
            self._ckpt_rows = rows
            self._ckpt_loaded = True
            self._ckpt_read_at = time.strftime("%H:%M:%S")
        self._render_checkpoints()
        self._start_checkpoint_files()

    def _start_checkpoint_files(self) -> None:
        """Read the files of every listed checkpoint not read yet, newest first."""
        wanted = [row.checkpoint_id for row in sort_checkpoints(self._ckpt_rows)
                  if row.checkpoint_id not in self._ckpt_files and row.checkpoint_id not in self._ckpt_reading_files]
        wanted = wanted[:self._CHECKPOINT_FILES_PER_REFRESH]
        if not wanted:
            return
        self._ckpt_reading_files.update(wanted)
        self._read_checkpoint_files(wanted)

    @work(thread=True, group="checkpoint-files")
    def _read_checkpoint_files(self, checkpoint_ids: list[str]) -> None:
        worker = get_current_worker()
        for cid in checkpoint_ids:
            if worker.is_cancelled or not self._screen_open:
                return
            try:
                result: FilesInfo | str = files_info(self._client.export_checkpoint(self._job_id, cid))
            except Exception as exc:  # noqa: BLE001 - said in the checkpoint's details
                log_traceback(f"reading the files of checkpoint {cid} of {self._job_id} failed")
                result = views.describe_error(exc)
            if worker.is_cancelled or not self._post(self._accept_checkpoint_files, cid, result):
                return

    def _accept_checkpoint_files(self, checkpoint_id: str, result) -> None:
        if not self._screen_open:
            return
        self._ckpt_reading_files.discard(checkpoint_id)
        self._ckpt_files[checkpoint_id] = result
        self._render_checkpoints()

    # ─── keys ────────────────────────────────────────────────────────────
    def action_checkpoint_sort(self) -> None:
        if not self._checkpoints_active():
            return
        names = list(SORTS)
        self._ckpt_sort = names[(names.index(self._ckpt_sort) + 1) % len(names)]
        self._ckpt_descending = True
        self._render_checkpoints()

    def action_checkpoint_order(self) -> None:
        if not self._checkpoints_active():
            return
        self._ckpt_descending = not self._ckpt_descending
        self._render_checkpoints()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "checkpoint-table" or event.row_key is None:
            return
        if event.cursor_row != event.data_table.cursor_row:
            return  # a redraw has moved the cursor since: the row it is on now is the one shown
        self._ckpt_highlighted = event.row_key.value
        self._render_checkpoint_detail()

    # ─── actions: download, and a new job from it ────────────────────────
    def _highlighted_checkpoint(self) -> Checkpoint | None:
        return next((r for r in self._ckpt_rows if r.checkpoint_id == self._ckpt_highlighted), None)

    def action_checkpoint_use(self, use: str) -> None:
        """A click on a new-job segment: copy the command for that kind of job."""
        if self._checkpoints_active():
            self._copy_new_job_command(use)

    def _checkpoint_plans(self, row: Checkpoint) -> dict:
        """Use → the CreateJob body for a new job from ``row``, or why there is none (a string)."""
        plans = {}
        for use, plan in use_plan(row, self._ckpt_rows, self._ckpt_files).items():
            if isinstance(plan, str):
                plans[use] = plan
                continue
            body = new_job_body(self._job, str(self._job_id), *plan)
            plans[use] = body if body is not None else "the job's record has no training sub-job configuration to copy"
        return plans

    def _checkpoint_use_reasons(self, row: Checkpoint, kind: str | None = None) -> dict:
        """Use → why the highlighted checkpoint can't start a job for it (None: it can)."""
        return {use: plan if isinstance(plan, str) else None for use, plan in self._checkpoint_plans(row).items()}

    def _copy_new_job_command(self, use: str) -> None:
        row = self._highlighted_checkpoint()
        if row is None or use not in USES:
            return
        body = self._checkpoint_plans(row)[use]
        if isinstance(body, str):
            self.notify(f"Can't start a {USES[use]} job from it: {body}.", severity="warning", markup=False)
            return
        # One line to paste into a shell: the CreateJob body, quoted, piped to submit.
        command = (f"printf '%s' {shlex.quote(json.dumps(body, separators=(',', ':')))} | "
                   + self._cli("submit -"))
        self._ckpt_copied = (row.checkpoint_id, use, command)
        self.app.copy_out(command, f"the command for a new {USES[use]} job from {printable(row.checkpoint_id)}")
        self._render_checkpoint_detail()

    def _ckpt_download_root(self, checkpoint_id: str) -> Path:
        return Path.home() / "cortex-training-checkpoints" / str(self._job_id)[:8] / checkpoint_id

    def action_checkpoint_download(self) -> None:
        if not self._checkpoints_active():
            return
        running = self._ckpt_download
        row = self._highlighted_checkpoint()
        if running is not None and running.get("running"):
            if row is not None and running["id"] != row.checkpoint_id:
                self.notify(f"Already downloading {printable(running['id'])}: one download at a time.",
                            markup=False)
                return
            self._ckpt_download_stop.set()  # the button reads "Stop" while it runs
            return
        if row is None:
            return
        self._ckpt_download_stop = threading.Event()
        self._ckpt_download = {"id": row.checkpoint_id, "root": self._ckpt_download_root(row.checkpoint_id),
                               "running": True, "done": 0, "total": None, "files_done": 0, "files": None}
        self._render_checkpoint_actions()
        self._ckpt_download_checkpoint(row.checkpoint_id, self._ckpt_download["root"], self._ckpt_download_stop)

    @work(thread=True, group="checkpoint-download")
    def _ckpt_download_checkpoint(self, checkpoint_id: str, root: Path, stop: threading.Event) -> None:
        last = [0.0]

        def progress(done, total, files_done, files) -> None:
            now = time.monotonic()
            if now - last[0] >= 0.25 or files_done == files:  # a few repaints a second
                last[0] = now
                self._post(self._ckpt_download_progress, checkpoint_id, done, total, files_done, files)

        try:
            # Fresh links: the ones read for the files' sizes may have expired.
            export = self._client.export_checkpoint(self._job_id, checkpoint_id)
            paths = download_checkpoint(self._client, self._job_id, checkpoint_id, export, root,
                                        on_progress=progress,
                                        cancelled=lambda: stop.is_set() or not self._screen_open)
            outcome, error = f"{len(paths)} files", None
        except DownloadCancelled:
            outcome, error = None, "stopped"
        except Exception as exc:  # noqa: BLE001 - said beside the button
            log_traceback(f"downloading checkpoint {checkpoint_id} of {self._job_id} failed")
            outcome, error = None, views.describe_error(exc)
        self._post(self._ckpt_download_finished, checkpoint_id, outcome, error)

    def _ckpt_download_progress(self, checkpoint_id, done, total, files_done, files) -> None:
        state = self._ckpt_download
        if not self._screen_open or state is None or state["id"] != checkpoint_id:
            return
        state.update(done=done, total=total, files_done=files_done, files=files)
        self._render_checkpoint_actions()

    def _ckpt_download_finished(self, checkpoint_id, outcome, error) -> None:
        state = self._ckpt_download
        if not self._screen_open or state is None or state["id"] != checkpoint_id:
            return
        state.update(running=False, outcome=outcome, error=error)
        self._render_checkpoint_actions()

    def _ckpt_download_text(self, row: Checkpoint | None) -> Text:
        state = self._ckpt_download
        if state is None or row is None or state["id"] != row.checkpoint_id:
            return Text(f"d downloads it to {self._display_path(self._ckpt_download_root(row.checkpoint_id))}"
                        if row is not None else "", style=theme.MUTED)
        where = self._display_path(state["root"])
        if state["running"]:
            total = state.get("total")
            amount = (f"{fmt_bytes(state['done'])} of {fmt_bytes(total)}" if total
                      else fmt_bytes(state["done"]))
            files = f" · file {min(state['files_done'] + 1, state['files'])} of {state['files']}" \
                if state.get("files") else ""
            percent = f" ({state['done'] * 100 // total}%)" if total else ""
            return Text(f"downloading {amount}{percent}{files} to {where} · d or the button stops it",
                        style=theme.WARN_TEXT)
        if state.get("error") == "stopped":
            return Text(f"download stopped · d picks it up again in {where}", style=theme.WARN_TEXT)
        if state.get("error"):
            return Text(f"download failed: {state['error']} · d tries again", style=theme.ERR_TEXT)
        return Text(f"downloaded {state.get('outcome')} ({fmt_bytes(state.get('total'))}) to {where}",
                    style=theme.OK)

    @staticmethod
    def _display_path(path: Path) -> str:
        try:
            return "~/" + str(path.relative_to(Path.home()))
        except ValueError:
            return str(path)

    def _render_checkpoint_actions(self) -> None:
        try:
            actions = self.query_one("#checkpoint-actions", Static)
            status = self.query_one("#checkpoint-action-status", Static)
        except Exception:  # noqa: BLE001 - not mounted yet
            return
        row = self._highlighted_checkpoint()
        if row is None:
            return
        state = self._ckpt_download
        running = state is not None and state.get("running")
        elsewhere = running and state["id"] != row.checkpoint_id  # one download at a time
        if running and not elsewhere:
            download = (" ■ Stop download ",
                        f"bold {theme.BACKGROUND} on {theme.WARN} @click=screen.checkpoint_download")
        elif elsewhere:
            download = (" ↓ Download ", f"{theme.FAINT} on {theme.SURFACE}")
        else:
            download = (" ↓ Download ", f"bold {theme.BACKGROUND} on {theme.ACCENT} @click=screen.checkpoint_download")
        spans = [download, ("   new job from it ", theme.MUTED)]
        reasons = self._checkpoint_use_reasons(row)
        for use, label in (("training", "Training"), ("sampling", "Sampling"), ("both", "Training + sampling")):
            look = (f"{theme.TEXT} on {theme.PANEL} @click=screen.checkpoint_use('{use}')" if reasons[use] is None
                    else f"{theme.FAINT} on {theme.SURFACE}")
            spans += [(f" {label} ", look), (" ", "")]
        actions.update(spans_content(spans))
        said = self._ckpt_download_text(row) if not elsewhere else Text(
            f"downloading {printable(state['id'])}: its own Stop ends it", style=theme.WARN_TEXT)
        why = [reason[0].upper() + reason[1:] for reason in dict.fromkeys(r for r in reasons.values() if r)]
        if why:
            said.append("\n" + " · ".join(why), style=theme.FAINT)
        status.update(said)

    # ─── drawing ─────────────────────────────────────────────────────────
    def _files_of(self, row: Checkpoint) -> FilesInfo | None:
        info = self._ckpt_files.get(row.checkpoint_id)
        return info if isinstance(info, FilesInfo) else None

    def _sorted_checkpoints(self) -> list[Checkpoint]:
        return sort_checkpoints(self._ckpt_rows, self._ckpt_sort, self._ckpt_descending, self._ckpt_files)

    def _checkpoint_status_text(self) -> Text:
        """What the list card says above its table: only a wait, an error, or why it is empty."""
        if saves_no_checkpoints(self._job):
            return Text("This job has no training sub-job, so it saves no checkpoints.", style=theme.MUTED)
        if not self._ckpt_loaded:
            if self._ckpt_error:
                return Text(f"Checkpoints unavailable: {self._ckpt_error}  (r tries again)", style=theme.ERR_TEXT)
            return Text("listing checkpoints…", style=theme.MUTED)
        if self._ckpt_error:  # a refresh failed: the list shown is the last one read
            return Text(f"refresh failed: {self._ckpt_error}", style=theme.ERR_TEXT)
        if not self._ckpt_rows:
            return Text("No checkpoints yet. A save() shows here once it completes (r refreshes)."
                        if self._live() else "This job saved no checkpoints.", style=theme.MUTED)
        return Text("")

    def _title_list_card(self) -> None:
        card = self.query_one("#checkpoint-list-card", Vertical)
        count = f" · {len(self._ckpt_rows)}" if self._ckpt_loaded else ""
        card.border_title = Content.assemble((" Checkpoints", f"bold {theme.TEXT}"), (f"{count} ", theme.MUTED))
        hints = []
        if self._ckpt_loading:
            hints.append("refreshing…")
        elif self._ckpt_read_at:
            hints.append(f"listed {self._ckpt_read_at}" + (" · every 30 s" if self._live() else ""))
        if self._ckpt_reading_files:
            hints.append(f"reading files of {len(self._ckpt_reading_files)}…")
        hints += [f"s sort: {SORTS[self._ckpt_sort]} {'↓' if self._ckpt_descending else '↑'}", "o order", "r refresh"]
        card.border_subtitle = Content.styled(f" {' · '.join(hints)} ", theme.MUTED)

    def _checkpoint_cells(self, row: Checkpoint) -> tuple:
        step, step_inferred, kind, kind_inferred = row.with_files(self._files_of(row))
        info = self._files_of(row)
        created = local_time(row.created, "%Y-%m-%d %H:%M:%S") if row.created else printable(row.created_at or "—")
        return (
            printable(row.checkpoint_id),
            created,
            Text(f"{step}" if step is not None else "—", justify="right", style=theme.MUTED if step_inferred else ""),
            _kind_badge(kind, inferred=kind_inferred),
            Text(fmt_loss(row.loss), justify="right"),
            Text(fmt_bytes(info.total_bytes) if info else "…" if row.checkpoint_id in self._ckpt_reading_files
                 else "—", justify="right"),
        )

    def _render_checkpoints(self) -> None:
        if not self._screen_open:
            return
        try:
            status = self.query_one("#checkpoint-status", Static)
            table = self.query_one("#checkpoint-table", DataTable)
        except Exception:  # noqa: BLE001 - not mounted yet
            return
        said = self._checkpoint_status_text()
        status.update(said)
        status.display = bool(said.plain)
        self._title_list_card()
        rows = self._sorted_checkpoints()
        cells = [(row.checkpoint_id, self._checkpoint_cells(row)) for row in rows]
        shown = tuple((cid, tuple((str(cell), str(getattr(cell, "style", ""))) for cell in row_cells))
                      for cid, row_cells in cells)
        if shown != self._ckpt_shown:
            self._ckpt_shown = shown
            keep = self._ckpt_highlighted
            table.clear(columns=not table.columns)
            if not table.columns:
                for label, key in (("Checkpoint", "id"), ("Created", "created"), ("Step", "step"),
                                   ("Type", "type"), ("Loss", "loss"), ("Size", "size")):
                    table.add_column(label, key=key)
            for cid, row_cells in cells:
                table.add_row(*row_cells, key=cid)
            ids = [cid for cid, _ in cells]
            if ids:
                table.move_cursor(row=ids.index(keep) if keep in ids else 0, animate=False)
                self._ckpt_highlighted = ids[table.cursor_row]
            else:
                self._ckpt_highlighted = None
        table.display = bool(cells)
        self._render_checkpoint_detail()

    def _render_checkpoint_detail(self) -> None:
        try:
            pane = self.query_one("#checkpoint-pane", VerticalScroll)
        except Exception:  # noqa: BLE001 - not mounted yet
            return
        self._render_checkpoint_actions()
        row = self._highlighted_checkpoint()
        pane.display = row is not None
        if row is None:
            return
        info = self._files_of(row)
        step, step_inferred, kind, kind_inferred = row.with_files(info)
        self.query_one("#checkpoint-hero", Static).update(self._checkpoint_hero(row, kind, kind_inferred))
        self.query_one("#checkpoint-detail", Static).update(
            self._checkpoint_facts(row, step, step_inferred, kind, kind_inferred))
        self._render_copied(row)
        origin_card = self.query_one("#checkpoint-origin-card", Vertical)
        origin_card.border_title = Content.styled(" Saved by ", f"bold {theme.TEXT}")
        self.query_one("#checkpoint-origin", Static).update(self._checkpoint_origin())

    # Labels and values as the job preview lays them out: a muted label column, then the value.
    _LABEL = 13

    def _fact(self, text: Text, label: str, value, style: str = theme.TEXT) -> None:
        if text.plain:
            text.append("\n")
        text.append(f"{label:<{self._LABEL - 1}} ", style=theme.MUTED)
        if isinstance(value, Text):
            text.append_text(value)
        else:
            text.append(str(value), style=style)

    def _checkpoint_hero(self, row: Checkpoint, kind: str | None, inferred: bool) -> Text:
        """The card's head: the checkpoint id, its type as a pill, and how long ago it was saved."""
        hero = Text(printable(row.checkpoint_id), style=f"bold {theme.TEXT}")
        look = _KIND_LOOKS.get(kind, _KIND_LOOKS[None])
        hero.append("   ")
        hero.append(f" {look[0]} {(kind or 'type unknown').upper()} ", style=f"bold {theme.BACKGROUND} on {look[1]}")
        if inferred:
            hero.append(" from its files", style=f"italic {theme.FAINT}")
        if row.created:
            hero.append(f"   saved {views.fmt_ago(row.created, datetime.now(timezone.utc))}", style=theme.MUTED)
        return hero

    def _checkpoint_facts(self, row: Checkpoint, step, step_inferred: bool, kind, kind_inferred: bool) -> Text:
        text = Text()
        what = {WEIGHTS_ONLY: "model weights only: serves sampling",
                RESUMABLE: "weights and optimizer state: resumes training"}.get(kind, "not reported")
        self._fact(text, "Holds", what)
        if row.created:
            self._fact(text, "Created", local_time(row.created, "%Y-%m-%d %H:%M:%S %Z"))
        else:
            self._fact(text, "Created", printable(row.created_at or "—"))
        step_value = Text("—" if step is None else str(step), style=theme.TEXT)
        if step_inferred:
            step_value.append("  from its files", style=f"italic {theme.FAINT}")
        self._fact(text, "Global step", step_value)
        self._fact(text, "Loss", fmt_loss(row.loss), theme.TEXT if row.loss is not None else theme.FAINT)
        for key, value in sorted(row.extra.items()):
            self._fact(text, printable(key), extra_text(value))
        files = self._ckpt_files.get(row.checkpoint_id)
        if isinstance(files, FilesInfo):
            self._fact(text, "Files", f"{len(files.files)} · {fmt_bytes(files.total_bytes)}")
            width = max((len(printable(name)) for name, _ in files.files), default=0)
            for name, size in files.files:
                text.append(f"\n{'':<{self._LABEL}}{printable(name):<{width}}  ", style=theme.FAINT)
                text.append(f"{fmt_bytes(size):>8}", style=theme.MUTED)
        elif isinstance(files, str):
            self._fact(text, "Files", f"unavailable: {files}", theme.ERR_TEXT)
        elif row.checkpoint_id in self._ckpt_reading_files:
            self._fact(text, "Files", "reading…", theme.MUTED)
        return text

    def _render_copied(self, row: Checkpoint) -> None:
        """The box under the card with the command a button copied, in the callout's style."""
        box = self.query_one("#checkpoint-copied", Static)
        copied = self._ckpt_copied
        box.display = copied is not None and copied[0] == row.checkpoint_id
        if not box.display:
            return
        _cid, use, command = copied
        box.styles.border = ("round", theme.ACCENT)
        box.border_title = Content.styled(f" ⧉ Copied: a new {USES[use]} job ", f"bold {theme.ACCENT}")
        box.border_subtitle = Content.styled(" paste it into a shell to submit ", theme.MUTED)
        body = Text(self._USE_HELP[use].removeprefix("copies the command for ").capitalize() + "\n",
                    style=theme.MUTED)
        body.append(command, style=theme.TEXT)
        box.update(body)

    def _checkpoint_origin(self) -> Text:
        text = Text()
        job = self._job
        status = Text(f"{theme.status_look(job.get('status')).glyph} {views.status_word(job.get('status')).upper()}",
                      style=theme.status_style(job.get("status")) or theme.MUTED)
        self._fact(text, "Job", Text(printable(self._job_id), style=theme.TEXT).append("   ").append_text(status))
        trainers = training_sub_jobs(job)
        for sj in trainers:
            self._fact(text, "Sub-job" if len(trainers) == 1 else "Training",
                       "  ".join(printable(part) for part in (sj.get("sub_job_id") or "—", sj.get("model_name"))
                                 if part))
        configs = [sj for sj in trainers if isinstance(sj.get("training_config"), dict)]
        for sj in configs:
            shown = {key: sj[key] for key in ("model_name", "dtype", "seed") if key in sj}
            shown.update(sj["training_config"])
            lines = json.dumps(compact_numbers(views.printable_data(shown)), indent=2, sort_keys=True,
                               default=str).splitlines()
            label = "Config" if len(configs) == 1 else f"Config {printable(sj.get('sub_job_id') or '')}"
            self._fact(text, label, "the job's, as submitted", theme.FAINT)
            for line in lines:
                text.append(f"\n{'':<{self._LABEL}}{line}", style=theme.TEXT)
        return text


# A checkpoint type's glyph and color, as a status's (theme.STATUS_LOOKS).
_KIND_LOOKS = {WEIGHTS_ONLY: ("●", theme.OK), RESUMABLE: ("◆", theme.INFO), None: ("?", theme.ENDED)}


def _kind_badge(kind: str | None, *, inferred: bool = False) -> Text:
    """``● weights-only``: a checkpoint type in its color; italic when read from its files."""
    glyph, color = _KIND_LOOKS.get(kind, _KIND_LOOKS[None])
    if kind is None:
        return Text("—", style=theme.FAINT)
    return Text(f"{glyph} {kind}", style=color + (" italic" if inferred else ""))
