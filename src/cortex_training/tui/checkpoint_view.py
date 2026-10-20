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
sub-job that saved it, the job's training configuration, and the commands that
download it and use it. Each checkpoint's export is read once, in the background,
for its files: their sizes, and the step and type they show when the list doesn't
say (see tui/checkpoints.py). The export's links are never shown or kept.
"""

from __future__ import annotations
import json
import shlex
import time
from datetime import datetime
from datetime import timezone
from rich.text import Text
from textual import work
from textual.widgets import DataTable
from textual.widgets import Static
from textual.worker import get_current_worker
from cortex_training.tui import views
from cortex_training.tui.checkpoints import RESUMABLE
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
from cortex_training.tui.widgets import log_traceback


class CheckpointTab:
    """The Checkpoints tab of :class:`~cortex_training.tui.logs.LogScreen`."""

    # The most checkpoints one refresh reads the files of. The rest are read by the
    # next refresh (r, or the 30-second one while the job is live).
    _CHECKPOINT_FILES_PER_REFRESH = 50
    _CHECKPOINT_LABEL = 14  # the details' label column

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

    # ─── drawing ─────────────────────────────────────────────────────────
    def _files_of(self, row: Checkpoint) -> FilesInfo | None:
        info = self._ckpt_files.get(row.checkpoint_id)
        return info if isinstance(info, FilesInfo) else None

    def _sorted_checkpoints(self) -> list[Checkpoint]:
        return sort_checkpoints(self._ckpt_rows, self._ckpt_sort, self._ckpt_descending, self._ckpt_files)

    def _checkpoint_status_text(self) -> Text:
        if saves_no_checkpoints(self._job):
            return Text("This job has no training sub-job, so it saves no checkpoints.", style="dim")
        if not self._ckpt_loaded:
            if self._ckpt_error:
                return Text(f"Checkpoints unavailable: {self._ckpt_error}  (r tries again)", style="red")
            return Text("listing checkpoints…", style="dim")
        count = len(self._ckpt_rows)
        bits = [f"{count} checkpoint{'' if count == 1 else 's'}",
                f"sort: {SORTS[self._ckpt_sort]} {'↓' if self._ckpt_descending else '↑'} (s · o)"]
        if self._ckpt_loading:
            bits.append("refreshing…")
        elif self._ckpt_read_at:
            bits.append(f"listed {self._ckpt_read_at}" + (" · every 30 s while live" if self._live() else "")
                        + " · r refreshes")
        if self._ckpt_reading_files:
            bits.append(f"reading files of {len(self._ckpt_reading_files)}…")
        text = Text("  ·  ".join(bits), style="dim")
        if self._ckpt_error:  # a refresh failed: the list shown is the last one read
            text.append(f"\nrefresh failed: {self._ckpt_error}", style="red")
        return text

    def _checkpoint_cells(self, row: Checkpoint) -> tuple:
        step, step_inferred, kind, kind_inferred = row.with_files(self._files_of(row))
        info = self._files_of(row)
        created = local_time(row.created, "%Y-%m-%d %H:%M:%S") if row.created else printable(row.created_at or "—")
        kind_style = "green" if kind == WEIGHTS_ONLY else "cyan" if kind == RESUMABLE else "dim"
        return (
            printable(row.checkpoint_id),
            created,
            Text(f"{step}" if step is not None else "—", justify="right", style="dim" if step_inferred else ""),
            Text(kind or "—", style=kind_style + (" italic" if kind_inferred else "")),
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
        status.update(self._checkpoint_status_text())
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
            detail = self.query_one("#checkpoint-detail", Static)
        except Exception:  # noqa: BLE001 - not mounted yet
            return
        row = next((r for r in self._ckpt_rows if r.checkpoint_id == self._ckpt_highlighted), None)
        if row is None:
            empty = self._ckpt_loaded and not self._ckpt_rows and not saves_no_checkpoints(self._job)
            if not empty:
                message = ""
            elif self._live():
                message = "No checkpoints yet. A save() shows here once it completes (r refreshes)."
            else:
                message = "This job saved no checkpoints."
            detail.update(Text(message, style="dim"))
            return
        detail.update(self._checkpoint_detail(row))

    def _field(self, text: Text, label: str, value, style: str = "") -> None:
        text.append(f"  {label:<{self._CHECKPOINT_LABEL - 2}}", style="bold")
        if isinstance(value, Text):
            text.append_text(value)
        else:
            text.append(str(value), style=style)
        text.append("\n")

    def _heading(self, text: Text, title: str) -> None:
        if text.plain:
            text.append("\n")
        text.append(title + "\n", style="bold underline")

    def _command(self, text: Text, lines: list[str]) -> None:
        for line in lines:
            text.append(f"    {line}\n", style="cyan")

    def _checkpoint_detail(self, row: Checkpoint) -> Text:
        now = datetime.now(timezone.utc)
        info = self._files_of(row)
        step, step_inferred, kind, kind_inferred = row.with_files(info)
        cid = printable(row.checkpoint_id)
        job_id = printable(self._job_id)
        # The commands quote the ids as the service sent them: a cleaned-up id would name
        # another checkpoint. shlex and JSON quoting keep them inert in a terminal.
        raw_cid, raw_job = row.checkpoint_id, str(self._job_id)
        text = Text()

        self._heading(text, "Checkpoint")
        self._field(text, "ID", cid)
        kind_value = Text(kind or "unknown", style="green" if kind == WEIGHTS_ONLY else "cyan" if kind else "dim")
        if kind == WEIGHTS_ONLY:
            kind_value.append(" · model weights only: loadable for sampling", style="dim")
        elif kind == RESUMABLE:
            kind_value.append(" · weights and optimizer state: resumes training", style="dim")
        if kind_inferred:
            kind_value.append("  (from its files)", style="dim italic")
        self._field(text, "Type", kind_value)
        if row.created:
            created = f"{local_time(row.created, '%Y-%m-%d %H:%M:%S %Z')}  ({views.fmt_ago(row.created, now)})"
        else:
            created = printable(row.created_at or "—")
        self._field(text, "Created", created)
        self._field(text, "Global step", Text(f"{step}" if step is not None else "—").append(
            "  (from its files)" if step_inferred else "", style="dim italic"))
        self._field(text, "Loss", fmt_loss(row.loss))
        for key, value in sorted(row.extra.items()):
            self._field(text, printable(key), extra_text(value))
        files = self._ckpt_files.get(row.checkpoint_id)
        if isinstance(files, FilesInfo):
            self._field(text, "Files", f"{len(files.files)} · {fmt_bytes(files.total_bytes)}")
            for name, size in files.files:
                text.append(f"{'':<{self._CHECKPOINT_LABEL}}{printable(name)}", style="dim")
                text.append(f"  {fmt_bytes(size)}\n", style="dim")
        elif isinstance(files, str):
            self._field(text, "Files", f"unavailable: {files}", style="red")
        elif row.checkpoint_id in self._ckpt_reading_files:
            self._field(text, "Files", "reading…", style="dim")

        self._heading(text, "Saved by")
        self._field(text, "Job", f"{job_id}  ({views.status_word(self._job.get('status'))})")
        trainers = training_sub_jobs(self._job)
        for sj in trainers:
            model = sj.get("model_name") or ""
            self._field(text, "Sub-job" if len(trainers) == 1 else "Training sub-job",
                        "  ".join(printable(part) for part in (sj.get("sub_job_id") or "—", model) if part))

        configs = [sj for sj in trainers if isinstance(sj.get("training_config"), dict)]
        if configs:
            self._heading(text, "Training configuration (the job's, as submitted)")
            for sj in configs:
                if len(configs) > 1:
                    text.append(f"  {printable(sj.get('sub_job_id') or '')}\n", style="bold")
                shown = {key: sj[key] for key in ("model_name", "dtype", "seed") if key in sj}
                shown.update(sj["training_config"])
                for line in json.dumps(compact_numbers(views.printable_data(shown)), indent=2, sort_keys=True,
                                       default=str).splitlines():
                    text.append(f"    {line}\n")

        self._heading(text, "Download")
        text.append("  Export short-lived links to its files, then download them (Python):\n", style="dim")
        self._command(text, [
            "import pathlib, shutil, requests",
            "# client: a CortexTrainingClient for this account, database and schema",
            f"links = client.export_checkpoint({json.dumps(raw_job)}, {json.dumps(raw_cid)})",
            'for f in links["files"]:',
            f'    path = pathlib.Path({json.dumps(raw_cid)}) / f["filename"]',
            "    path.parent.mkdir(parents=True, exist_ok=True)",
            '    with requests.get(f["url"], stream=True) as r, open(path, "wb") as out:',
            "        r.raise_for_status(); shutil.copyfileobj(r.raw, out)",
        ])
        text.append("  List this job's checkpoints from the CLI:\n", style="dim")
        self._command(text, [self._cli(f"checkpoints {shlex.quote(raw_job)}")])

        self._heading(text, "Use it")
        source = json.dumps({"source_checkpoint_info": {"checkpoint_id": raw_cid, "source_job_id": raw_job}})
        if kind != RESUMABLE:
            text.append("  Serve it for sampling (needs a weights-only checkpoint):\n", style="dim")
            self._command(text, ["python -m recipes.inference.serve config=CONFIG.json job_config=JOB_CONFIG "
                                 f"source_job_id={shlex.quote(raw_job)} checkpoint_id={shlex.quote(raw_cid)}"])
            text.append("  or start a sampling sub-job from it, with this in its config:\n", style="dim")
            self._command(text, [source])
        if kind != WEIGHTS_ONLY:
            text.append("  Resume training from it in a running training job (the location options are this\n"
                        "  job's: change them when the target job is elsewhere):\n", style="dim")
            self._command(text, [self._cli(f"--job-id TARGET_JOB_ID load {shlex.quote(raw_cid)} "
                                           f"--source-job-id {shlex.quote(raw_job)}")])
            text.append("  or start a training sub-job from it, with this in its config "
                        "(training_config.load_optimizer_states=false keeps only the weights,\n"
                        "  as a different GPU count needs):\n", style="dim")
            self._command(text, [source])
        if kind == RESUMABLE:
            text.append("  Sampling needs a weights-only checkpoint: save one with "
                        "save(checkpoint_type=\"weights-only\").\n", style="dim")
        return text
