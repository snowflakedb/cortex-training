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

"""The Checkpoints tab's data: a job's checkpoint list, read and sorted, and what
its export says about each checkpoint's files.

``list_checkpoints`` names each checkpoint and its creation time; its global step,
type and loss come when the service sends them. A checkpoint's export lists its
files, and those say the rest: the ``global_step<N>/`` folder they are under, and
whether they hold optimizer state (resumable) or only weights (weights-only).
Nothing here draws or calls the service; tui/checkpoint_view.py does. A download of
a checkpoint's files reports its outcome through the client's metrics.
"""

from __future__ import annotations
import json
import re
import time
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any
from typing import Optional
from cortex_training.tui.format import parse_ts
from cortex_training.tui.format import printable

RESUMABLE = "resumable"
WEIGHTS_ONLY = "weights-only"

# The sorts s cycles through, in order, and what the status line calls each.
SORTS = {"created": "created", "step": "step", "type": "type"}

# The fields a row shows in its own columns; the details list any others the service sent,
# except ones that could carry a link or a credential.
_KNOWN_FIELDS = frozenset({"checkpoint_id", "created_at", "global_steps", "global_step", "checkpoint_type",
                           "avg_loss", "loss"})
_HIDDEN_FIELDS = re.compile(r"url|link|stage_info|credential|token|secret|key", re.IGNORECASE)

_STEP_IN_NAME = re.compile(r"(?:^|/)global_step(\d+)(?:/|$)")
_OPTIMIZER_FILE = re.compile(r"optim_states\.pt$|(?:^|/)optimizer[^/]*$")


@dataclass
class FilesInfo:
    """What a checkpoint's export says about it: its files (name and size), and the
    step and type they show, None when they don't."""

    files: list = field(default_factory=list)  # [(filename, size in bytes or None)]
    step: Optional[int] = None
    kind: Optional[str] = None
    expires_at: Optional[str] = None

    @property
    def total_bytes(self) -> Optional[int]:
        sizes = [size for _name, size in self.files if size is not None]
        return sum(sizes) if sizes else None


@dataclass
class Checkpoint:
    """One entry of ``list_checkpoints``, its fields read once."""

    checkpoint_id: str  # as the service sent it: the export and the commands use it
    created_at: Any = None  # as sent
    created: Optional[datetime] = None
    step: Optional[int] = None
    kind: Optional[str] = None
    loss: Optional[float] = None
    extra: dict = field(default_factory=dict)  # the other fields the service sent

    def with_files(self, files: Optional[FilesInfo]) -> tuple[Optional[int], bool, Optional[str], bool]:
        """(step, step from the files?, type, type from the files?): the service's own
        value first, else what ``files`` shows."""
        step, step_inferred = self.step, False
        if step is None and files is not None and files.step is not None:
            step, step_inferred = files.step, True
        kind, kind_inferred = self.kind, False
        if kind is None and files is not None and files.kind is not None:
            kind, kind_inferred = files.kind, True
        return step, step_inferred, kind, kind_inferred


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isascii() and value.strip().isdigit():
        return int(value.strip())
    return None


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value) if isinstance(value, str) and value.strip() else None
    except ValueError:
        return None


def checkpoint_kind(value: Any) -> Optional[str]:
    """``resumable`` or ``weights-only`` for the service's type, in any case or with an
    underscore; None for anything else."""
    if not isinstance(value, str):
        return None
    word = value.strip().lower().replace("_", "-")
    return word if word in (RESUMABLE, WEIGHTS_ONLY) else None


def step_from_name(name: Any) -> Optional[int]:
    """The N of a ``global_step<N>`` name or path segment, else None."""
    match = _STEP_IN_NAME.search(name) if isinstance(name, str) else None
    return int(match.group(1)) if match else None


def parse_checkpoints(answer: Any) -> list[Checkpoint]:
    """``list_checkpoints``'s answer as checkpoints; entries without an id are left out."""
    rows = []
    for entry in answer if isinstance(answer, list) else []:
        if not isinstance(entry, dict):
            continue
        cid = entry.get("checkpoint_id")
        if not isinstance(cid, str) or not cid:
            continue
        step = _as_int(entry.get("global_steps"))
        if step is None:
            step = _as_int(entry.get("global_step"))
        if step is None:
            step = step_from_name(cid)
        loss = _as_float(entry.get("avg_loss"))
        if loss is None:
            loss = _as_float(entry.get("loss"))
        rows.append(Checkpoint(
            checkpoint_id=cid,
            created_at=entry.get("created_at"),
            created=parse_ts(entry.get("created_at")),
            step=step,
            kind=checkpoint_kind(entry.get("checkpoint_type")),
            loss=loss,
            extra={key: value for key, value in entry.items()
                   if key not in _KNOWN_FIELDS and not _HIDDEN_FIELDS.search(str(key))},
        ))
    return rows


def files_info(export: Any) -> FilesInfo:
    """What ``export_checkpoint``'s answer shows. Its links are never kept: they are
    short-lived credentials, and the tab only needs the names and sizes."""
    files = []
    raw = export.get("files") if isinstance(export, dict) else None
    for entry in raw if isinstance(raw, list) else []:
        if isinstance(entry, dict) and isinstance(entry.get("filename"), str):
            files.append((entry["filename"], _as_int(entry.get("size_bytes"))))
    names = [name for name, _size in files]
    steps = {step for step in map(step_from_name, names) if step is not None}
    kind = None
    if any(_OPTIMIZER_FILE.search(name) for name in names):
        kind = RESUMABLE
    elif any(name.endswith(".safetensors") for name in names):
        kind = WEIGHTS_ONLY
    expires = export.get("expires_at") if isinstance(export, dict) else None
    return FilesInfo(files=files, step=steps.pop() if len(steps) == 1 else None, kind=kind,
                     expires_at=expires if isinstance(expires, str) else None)


_TYPE_RANK = {RESUMABLE: 0, WEIGHTS_ONLY: 1}


def sort_checkpoints(rows: list[Checkpoint], key: str = "created", descending: bool = True,
                     files: Optional[dict] = None) -> list[Checkpoint]:
    """``rows`` by ``key`` ("created", "step" or "type"), the newest (or largest) first.

    Rows without the sorted value come after the others in both directions; ties keep
    the newest created first. ``files`` (checkpoint id → :class:`FilesInfo`) fills in a
    step or type the service didn't send.
    """
    files = files or {}

    def created(row: Checkpoint) -> float:
        return row.created.timestamp() if row.created else float("-inf")

    by_created = sorted(rows, key=created, reverse=True)
    if key == "created":
        known = [row for row in by_created if row.created]
        unknown = [row for row in by_created if not row.created]
        return (known if descending else known[::-1]) + unknown

    def value(row: Checkpoint):
        info = files.get(row.checkpoint_id)
        step, _, kind, _ = row.with_files(info if isinstance(info, FilesInfo) else None)
        return step if key == "step" else _TYPE_RANK.get(kind) if kind else None

    known = [row for row in by_created if value(row) is not None]
    unknown = [row for row in by_created if value(row) is None]
    return sorted(known, key=value, reverse=descending) + unknown


def fmt_bytes(size: Optional[int]) -> str:
    """``size`` bytes for reading: ``1.2 GB``, ``977 KB``, ``12 B``; ``—`` without one."""
    if size is None:
        return "—"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000
    return f"{size} B"


def fmt_loss(loss: Optional[float]) -> str:
    return "—" if loss is None else f"{loss:.4g}"


def extra_text(value: Any) -> str:
    """A field the tab has no column for, printable on one line."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return " ".join(printable(text).split())


def compact_numbers(value: Any) -> Any:
    """``value`` (decoded JSON) with each whole float as an int: the service sends a
    config's counts as floats (``"n_gpus": 1.0``), which read as they were given."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: compact_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [compact_numbers(item) for item in value]
    return value


def training_sub_jobs(job: Any) -> list[dict]:
    """The job's training sub-jobs: the ones that save checkpoints."""
    subs = job.get("sub_jobs") if isinstance(job, dict) else None
    return [sj for sj in subs if isinstance(sj, dict)
            and str(sj.get("job_type") or sj.get("type") or "").lower() == "training"] if isinstance(subs, list) else []


def saves_no_checkpoints(job: Any) -> bool:
    """Whether ``job`` is known to have no training sub-job: its sub-jobs are listed and
    none trains, so it has no checkpoints to list."""
    subs = job.get("sub_jobs") if isinstance(job, dict) else None
    return isinstance(subs, list) and bool(subs) and not training_sub_jobs(job)


# ─── starting a job from a checkpoint ────────────────────────────────────────

TRAINING = "training"
SAMPLING = "sampling"
BOTH = "both"
USES = {TRAINING: "training", SAMPLING: "sampling", BOTH: "training + sampling"}

# The sub-job fields a CreateJob body takes; a job record's others (ids, status,
# times) are the service's own.
_SUB_JOB_WIRE_FIELDS = ("job_type", "model_name", "global_batch_size", "dtype", "seed", "model_post_init",
                        "training_config", "inference_config")


def use_plan(row: Checkpoint, rows: list[Checkpoint], files: Optional[dict] = None) -> dict:
    """Use → (training checkpoint id, sampling checkpoint id) for a new job from ``row``,
    or why it can't be used for it (a string).

    The service loads a training sub-job only from a resumable checkpoint and a
    sampling sub-job only from a weights-only one. So training + sampling pairs
    ``row`` with a checkpoint of the other type saved at the same step, the newest
    when there are several. A checkpoint of unknown type is offered for each use.
    """
    files = files or {}

    def info(r: Checkpoint) -> tuple[Optional[int], Optional[str]]:
        step, _, kind, _ = r.with_files(files.get(r.checkpoint_id) if isinstance(
            files.get(r.checkpoint_id), FilesInfo) else None)
        return step, kind

    step, kind = info(row)
    cid = row.checkpoint_id
    plan: dict = {}
    plan[TRAINING] = "training resumes from a resumable checkpoint" if kind == WEIGHTS_ONLY else (cid, None)
    plan[SAMPLING] = "sampling needs a weights-only checkpoint" if kind == RESUMABLE else (None, cid)
    if kind is None:
        plan[BOTH] = (cid, cid)
    else:
        other = WEIGHTS_ONLY if kind == RESUMABLE else RESUMABLE
        partners = [r for r in sort_checkpoints(rows, files=files)
                    if r.checkpoint_id != cid and info(r) == (step, other)] if step is not None else []
        if not partners:
            plan[BOTH] = f"training + sampling needs a {other} checkpoint saved at the same step"
        else:
            mate = partners[0].checkpoint_id
            plan[BOTH] = (cid, mate) if kind == RESUMABLE else (mate, cid)
    return plan


def _wire_sub_job(sj: dict, kind: str) -> dict:
    out = {key: sj[key] for key in _SUB_JOB_WIRE_FIELDS if key in sj and sj[key] is not None}
    out["job_type"] = kind
    return compact_numbers(out)


def _sampling_like(job: Any, trainer: Optional[dict]) -> Optional[dict]:
    """The job's own sampling sub-job, else one for the trainer's model on one GPU."""
    subs = job.get("sub_jobs") if isinstance(job, dict) else None
    for sj in subs if isinstance(subs, list) else []:
        if isinstance(sj, dict) and str(sj.get("job_type") or "").lower() == SAMPLING \
                and isinstance(sj.get("inference_config"), dict):
            return _wire_sub_job(sj, SAMPLING)
    if trainer is None or not trainer.get("model_name"):
        return None
    inference: dict = {"n_gpus": 1}
    seq = (trainer.get("training_config") or {}).get("max_seq_len")
    if seq is not None:
        inference["max_seq_len"] = seq
    out = {"job_type": SAMPLING, "model_name": trainer["model_name"], "inference_config": inference}
    if trainer.get("dtype"):
        out["dtype"] = trainer["dtype"]
    return compact_numbers(out)


def new_job_body(job: Any, job_id: str, training_checkpoint: Optional[str] = None,
                 sampling_checkpoint: Optional[str] = None) -> Optional[dict]:
    """A CreateJob body for a new job that starts from saved checkpoints: the source
    job's training sub-job from ``training_checkpoint``, its sampling sub-job from
    ``sampling_checkpoint``, or both, as the job ran them, each with
    ``source_checkpoint_info``. The training sub-job keeps the source's GPU count and
    parallelism, which a resume with optimizer state needs. None when the job's record
    has no training sub-job to copy."""
    trainers = training_sub_jobs(job)
    trainer = next((sj for sj in trainers if isinstance(sj.get("training_config"), dict)), None)
    subs = []
    if training_checkpoint is not None:
        if trainer is None:
            return None
        subs.append({**_wire_sub_job(trainer, TRAINING),
                     "source_checkpoint_info": {"checkpoint_id": training_checkpoint, "source_job_id": job_id}})
    if sampling_checkpoint is not None:
        sampler = _sampling_like(job, trainer)
        if sampler is None:
            return None
        subs.append({**sampler,
                     "source_checkpoint_info": {"checkpoint_id": sampling_checkpoint, "source_job_id": job_id}})
    if not subs:
        return None
    body: dict = {"sub_job_configs": subs}
    hardware = job.get("hardware") if isinstance(job, dict) else None
    if isinstance(hardware, str) and hardware:
        body["hardware"] = hardware
    return body


# ─── downloading ─────────────────────────────────────────────────────────────


class DownloadCancelled(Exception):
    """The download was stopped before it finished."""


def local_path(root: Path, filename: str) -> Optional[Path]:
    """Where ``filename`` (an export's relative name) goes under ``root``; None for a
    name that would land outside it."""
    parts = PurePosixPath(filename).parts
    if not parts or PurePosixPath(filename).is_absolute() or any(part in ("..", "") for part in parts) \
            or "\\" in filename:
        return None
    return root.joinpath(*parts)


def download_export(export: Any, root: Path, *, get=None, on_progress=None, cancelled=lambda: False,
                    chunk_bytes: int = 1 << 20, stats: Optional[dict] = None) -> list:
    """Download every file of ``export_checkpoint``'s answer under ``root``; return the paths.

    A file already there at its exported size is kept, so a stopped download picks up
    with the next file. Each file is written beside its final name and renamed into
    place once whole. ``on_progress(done_bytes, total_bytes, files_done, files)`` is
    called as bytes arrive; ``cancelled()`` stops it between chunks, and a file that
    fails once it is true counts as stopped too. ``stats``, when given, says as it goes
    how many files it covers (``file_count``: the export's files that come with a link)
    and their size (``total_bytes``, when the export gives every file's), how many it
    downloaded (``files_written``) or kept (``files_skipped``), and how many bytes it
    wrote (``bytes_written``), however it ends.
    """
    import requests

    get = get or requests.get
    stats = {} if stats is None else stats
    raw = export.get("files") if isinstance(export, dict) else None
    entries = [e for e in raw if isinstance(e, dict) and isinstance(e.get("filename"), str)
               and isinstance(e.get("url"), str)] if isinstance(raw, list) else []
    stats.update(file_count=len(entries), files_written=0, files_skipped=0, bytes_written=0)
    plan = []
    for entry in entries:
        path = local_path(root, entry["filename"])
        if path is None:
            raise ValueError(f"refusing to write {printable(entry['filename'])!r} outside {root}")
        plan.append((path, entry["url"], _as_int(entry.get("size_bytes"))))
    total = sum(size or 0 for _path, _url, size in plan)
    if all(size is not None for _path, _url, size in plan):
        stats["total_bytes"] = total
    done = 0
    written = []
    for index, (path, url, size) in enumerate(plan):
        if cancelled():
            raise DownloadCancelled()
        if size is not None and path.is_file() and path.stat().st_size == size:
            done += size
            written.append(path)
            stats["files_skipped"] += 1
            if on_progress:
                on_progress(done, total, index + 1, len(plan))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        try:
            with get(url, stream=True, timeout=(10, 60)) as response:
                response.raise_for_status()
                with open(partial, "wb") as out:
                    for chunk in response.iter_content(chunk_bytes):
                        if cancelled():
                            raise DownloadCancelled()
                        out.write(chunk)
                        done += len(chunk)
                        stats["bytes_written"] += len(chunk)
                        if on_progress:
                            on_progress(done, total, index, len(plan))
        except DownloadCancelled:
            raise
        except Exception as exc:
            if cancelled():  # it failed after the stop (a read still waiting, say): a stop, not a failure
                raise DownloadCancelled() from exc
            raise
        partial.replace(path)
        written.append(path)
        stats["files_written"] += 1
        if on_progress:
            on_progress(done, total, index + 1, len(plan))
    return written


def download_checkpoint(client: Any, job_id: str, checkpoint_id: str, export: Any, root: Path,
                        **options: Any) -> list:
    """:func:`download_export` with ``options``, reported as one ``download_checkpoint``
    metric through ``client``'s own telemetry: how long it took, its files and bytes,
    and whether it finished, failed or was stopped (a stop is its own outcome, not a
    failure). The report never changes what the download returns or raises."""
    values: dict = {"artifact": "checkpoint", "cancelled": False}
    started = time.monotonic()
    try:
        paths = download_export(export, root, stats=values, **options)
    except (DownloadCancelled, KeyboardInterrupt):
        _report_download(client, job_id, checkpoint_id, started, values, cancelled=True)
        raise
    except Exception as exc:
        _report_download(client, job_id, checkpoint_id, started, values, error=exc)
        raise
    _report_download(client, job_id, checkpoint_id, started, values)
    return paths


def _report_download(client: Any, job_id: str, checkpoint_id: str, started: float, values: dict, *,
                     error: Optional[BaseException] = None, cancelled: bool = False) -> None:
    try:
        report = getattr(client, "_emit_operation_outcome", None)  # the --debug client has no telemetry
        if report is not None:
            report("download_checkpoint", started, {"job_id": job_id, "checkpoint_id": checkpoint_id},
                   values=values, error=error, cancelled=cancelled,
                   error_message=None if error is None else _metric_error_text(error))
    except Exception:  # noqa: BLE001 - telemetry is best effort
        pass


def _metric_error_text(exc: BaseException) -> str:
    """What a failed download's metric says about ``exc``: its type, and its HTTP status or
    OS error. Never its own text, which can quote a file's presigned link (a credential) or
    a local path."""
    import requests

    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return f"{type(exc).__name__}: HTTP {status}"
    strerror = getattr(exc, "strerror", None)
    # A requests error is an OSError too, but its text is about the link, not the disk.
    if isinstance(exc, OSError) and not isinstance(exc, requests.RequestException) \
            and isinstance(strerror, str) and strerror:
        return f"{type(exc).__name__}: {strerror}"
    return type(exc).__name__
