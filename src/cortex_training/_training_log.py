# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""A local copy of a job's training metrics, written as the client receives each result.

The service saves every ``/fwd-bwd`` call's and ``/step``'s numbers to the job's
stage, where they arrive in batches, tens of seconds after the call. The client
that made the call already has the result, so it also appends the same row
here the moment the result arrives. The TUI's Training metrics tab shows the
stage's rows and adds this file's rows the stage hasn't received yet, matched by
``request_id``: the person running a job sees each step at once, and the stage
copy, which has every call from every machine, is what everyone sees in the end.

A row is written only for a result that names one of the job's sub-jobs (its
``job_id``, or the envelope's), and holds the result as the sub-job returned it,
without the envelope's transport metrics, so it matches the stage's row.

One file per sub-job, in the row format the service writes (``record_version``
2, see ``docs/reference/rest-api.md``), under the TUI's cache directory::

    <cache_root>/local-training/<job_id>/<sub_job_id>/training.jsonl

A file stops growing at :data:`MAX_BYTES`, and the first write of each client
prunes the directory: a job not written to for :data:`MAX_AGE_SECONDS` goes, and
then the least recently written jobs until the rest fit in :data:`MAX_TOTAL_BYTES`.
The stage keeps every job whole, so a pruned or capped job only loses the head
start, and the TUI merges the two copies (see ``tui/training.py``).

Best-effort throughout: nothing here raises into, or slows, the call it came
from. Set ``CORTEX_TRAINING_DISABLE_LOCAL_TRAINING_METRICS`` to turn it off.
Pure stdlib, NO ``textual`` import.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Mapping

from cortex_training.tui.log_cache import cache_root

logger = logging.getLogger(__name__)

DISABLE_ENV = "CORTEX_TRAINING_DISABLE_LOCAL_TRAINING_METRICS"
DIR_NAME = "local-training"
FILE_NAME = "training.jsonl"
RECORD_VERSION = 2
OP_FWD_BWD = "fwd_bwd"
OP_STEP = "step"
OP_LOAD = "load"  # not a row: a load-checkpoint, after which the next calls' step is unknown
# A file stops growing past this, rather than filling the disk; the stage keeps the whole job.
MAX_BYTES = 64 * 1024 * 1024
# The whole directory, across jobs: older jobs are pruned past either (see prune).
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_AGE_SECONDS = 30 * 24 * 60 * 60
MAX_METRICS = 256
# Job and sub-job ids name directories, so only plain ones are written.
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}")


def disabled() -> bool:
    return os.environ.get(DISABLE_ENV, "").strip().lower() not in ("", "0", "false", "no")


def local_dir(job_id: str, root: Path | None = None) -> Path:
    """Where a job's local training metrics are written: one ``<sub_job_id>/training.jsonl`` each."""
    return (root if root is not None else cache_root()) / DIR_NAME / job_id


def _tree(path: Path) -> tuple[float, int]:
    """The newest mtime and the total size of the files under ``path``."""
    newest, size = 0.0, 0
    for item in path.rglob("*"):
        try:
            stat = item.stat()
        except OSError:
            continue
        if item.is_file():
            newest, size = max(newest, stat.st_mtime), size + stat.st_size
    return newest, size


def prune(root: Path | None = None, *, keep: str | None = None, max_bytes: int | None = None,
          max_age: float | None = None, now: float | None = None) -> list[Path]:
    """Remove jobs' local training metrics: any not written to for ``max_age`` seconds, then the
    least recently written until the rest fit in ``max_bytes``. Never the job ``keep``. Returns
    the removed directories."""
    max_bytes = MAX_TOTAL_BYTES if max_bytes is None else max_bytes
    max_age = MAX_AGE_SECONDS if max_age is None else max_age
    now = time.time() if now is None else now
    base = (root if root is not None else cache_root()) / DIR_NAME
    jobs = []
    for job in base.iterdir() if base.is_dir() else ():
        if job.is_dir() and not job.is_symlink() and job.name != keep:
            jobs.append((*_tree(job), job))
    total = sum(size for _, size, _ in jobs) + (_tree(base / keep)[1] if keep and (base / keep).is_dir() else 0)
    removed = []
    for newest, size, job in sorted(jobs, key=lambda item: item[0]):
        if now - newest <= max_age and total <= max_bytes:
            continue
        shutil.rmtree(job, ignore_errors=True)
        total -= size
        removed.append(job)
    return removed


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def _metrics(result: Mapping[str, Any], *, nested: bool) -> dict[str, float]:
    """Every finite number in ``result`` (and, for a call, its ``metrics``), as the service saves them."""
    metrics: dict[str, float] = {}
    items = [(k, v) for k, v in result.items() if k != "metrics"]
    if nested and isinstance(result.get("metrics"), Mapping):
        items += list(result["metrics"].items())
    for key, raw in items:
        value = _number(raw)
        if value is not None and isinstance(key, str) and key and key not in metrics and len(metrics) < MAX_METRICS:
            metrics[key] = value
    return metrics


class TrainingLog:
    """Appends a row per ``/fwd-bwd`` and ``/step`` result, tagged with its step and call as the service does."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root
        self._lock = threading.Lock()
        self._epoch = uuid.uuid4().hex  # this client's run of seq numbers
        self._seq = 0
        # (job, sub-job) -> the step the next calls count toward, and the calls counted toward it so far.
        self._next_step: dict[tuple[str, str], int] = {}
        self._calls: dict[tuple[str, str], int] = {}
        self._full: set[Path] = set()
        self._pruned = False

    def record(self, job_id: str, op: str, request_id: str | None, result: Any,
               sub_job_id: str | None = None) -> None:
        """Append ``result``'s row. Never raises.

        The row is filed under the sub-job the result names (its ``job_id``), else
        ``sub_job_id``; a result that names neither, or names something that is not one of
        ``job_id``'s sub-jobs, is skipped: a row under the wrong name would show as another
        sub-job beside the stage's.
        """
        try:
            if op not in (OP_FWD_BWD, OP_STEP) or not isinstance(result, Mapping) or not _SAFE_ID.fullmatch(job_id):
                return
            named = result.get("job_id")
            sub_job_id = named if isinstance(named, str) else sub_job_id
            if not (isinstance(sub_job_id, str) and _SAFE_ID.fullmatch(sub_job_id)
                    and sub_job_id.startswith(f"{job_id}:")):
                return
            key = (job_id, sub_job_id)
            metrics = _metrics(result, nested=op == OP_FWD_BWD)
            with self._lock:
                if op == OP_FWD_BWD:
                    if not metrics:
                        return  # no row, so no call number either
                    step = self._next_step.get(key)
                    # Without a step (before this client saw a /step, e.g. one CLI process per
                    # call), a count would restart at 1 each time; leave both empty and let the
                    # reader attach the call to the next step row, as it does the service's.
                    call = None
                    if step is not None:
                        call = self._calls.get(key, 0) + 1
                        self._calls[key] = call
                else:
                    call = None
                    raw = _number(result.get("global_steps"))
                    step = int(raw) if raw is not None and raw >= 0 and raw.is_integer() else None
                    self._calls.pop(key, None)
                    if step is not None:
                        self._next_step[key] = step + 1
                    if not metrics or step is None:
                        return
                self._seq += 1
                row = {
                    "kind": "training", "record_version": RECORD_VERSION, "sub_job_id": sub_job_id,
                    "sample_ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "seq": self._seq, "seq_epoch": self._epoch, "op": op, "step": step, "call": call,
                    "request_id": request_id, "metrics": metrics,
                }
                self._append(local_dir(job_id, self._root) / sub_job_id / FILE_NAME, row)
        except Exception:  # noqa: BLE001 - best-effort: never fail the call this came from
            logger.debug("local training metrics record failed", exc_info=True)

    def forget_step(self, job_id: str, sub_job_id: str | None = None) -> None:
        """A load-checkpoint moved the job (``sub_job_id``, or every sub-job when None): the next
        calls' step is unknown until the next /step."""
        with self._lock:
            for known in (self._next_step, self._calls):
                for key in [key for key in known if key[0] == job_id and sub_job_id in (None, key[1])]:
                    del known[key]

    def _append(self, path: Path, row: dict) -> None:
        if path in self._full:
            return
        if not self._pruned:
            self._pruned = True  # once per client: a job directory written to now is never pruned
            try:
                prune(self._root, keep=path.parent.parent.name)
            except Exception:  # noqa: BLE001 - best-effort, like the write itself
                logger.debug("local training metrics prune failed", exc_info=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if path.stat().st_size >= MAX_BYTES:
                self._full.add(path)
                logger.warning("local training metrics for %s reached %d MiB; the stage keeps the rest",
                               path.parent.name, MAX_BYTES // (1024 * 1024))
                return
        except FileNotFoundError:
            pass
        line = (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
        # One whole line per write in append mode, so rows from threads or processes never interleave.
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
