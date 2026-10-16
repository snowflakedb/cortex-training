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

"""Pure view-model helpers for the TUI's job list, summary, and detail tabs.

Search, sorting, and summary fields. Like ``format.py`` this never imports
``textual``, so every rule here is unit-testable without the TUI installed.
"""

from __future__ import annotations

import functools
import math
import re
import shlex
import unicodedata
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Any
from typing import Iterable
from typing import Optional

import requests
from rich.cells import cell_len

from cortex_training.tui.format import ACTIVE_STATUSES
from cortex_training.tui.format import compact_number
from cortex_training.tui.format import subjob_config
from cortex_training.tui.format import local_time
from cortex_training.tui.format import parse_ts  # noqa: F401 - views.parse_ts for the screens
from cortex_training.tui.format import printable
from cortex_training.tui.format import source_label

RETENTION_DAYS = 183  # six months of job history

# ─── time ────────────────────────────────────────────────────────────────────


def fmt_ts(raw: Any) -> str:
    """A timestamp as local ``YYYY-MM-DD HH:MM``; ``—`` when missing, unparseable or out of range."""
    dt = raw if isinstance(raw, datetime) else parse_ts(raw)
    return (local_time(dt) if dt else "") or "—"


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "—"
    s = int(seconds)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# A time at most this far ahead of this computer's clock reads "just now": the
# service's clock and this one rarely agree to the second.
CLOCK_SKEW_SECONDS = 300


def fmt_ago(raw: Any, now: datetime) -> str:
    """How long before ``now`` the time ``raw`` was, e.g. ``4m 05s ago``; ``—`` without one.

    A time slightly ahead of ``now`` (clock skew) is ``just now``; one further
    ahead says how far it is ahead of this computer's clock.
    """
    dt = raw if isinstance(raw, datetime) else parse_ts(raw)
    if not dt:
        return "—"
    seconds = (now - dt).total_seconds()
    if seconds < 0:
        if -seconds <= CLOCK_SKEW_SECONDS:
            return "just now"
        return f"{fmt_duration(-seconds)} ahead of this computer's clock"
    return f"{fmt_duration(seconds)} ago"


# ─── job fields ──────────────────────────────────────────────────────────────


def status_word(status: Any) -> str:
    return (str(status or "").strip() or "?").removeprefix("JOB_STATE_").lower()


def describe_error(exc: BaseException) -> str:
    """A failed call for display, on one printable line: the HTTP status and the
    service's message, not the URL.

    requests' own text ("500 Server Error: ... for url: <full URL>") repeats the
    host, database and schema and never includes the service's explanation.
    The message is the service's (or the stage's) text, so it is made printable
    (:func:`collapse`) before any screen draws it. A request that timed out
    says how long it waited ("no answer from the service within 45 s").
    """
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(exc, requests.exceptions.Timeout):
        return _timeout_text(exc)
    if not isinstance(status, int):
        return collapse(f"{type(exc).__name__}: {exc}")
    try:
        body = response.json()
        message = str(body.get("message") or body.get("error") or "") if isinstance(body, dict) else ""
    except Exception:  # noqa: BLE001 - not JSON
        message = ""
    reason = str(getattr(response, "reason", "") or "")
    return collapse(f"HTTP {status}" + (f" {reason}" if reason else "") + (f": {message}" if message else ""))


# How long a timed-out request waited, as urllib3 says it: "(read timeout=45.0)".
_TIMEOUT_SECONDS = re.compile(r"(?:read|connect) timeout=([0-9]+(?:\.[0-9]+)?)")


def _timeout_text(exc: BaseException) -> str:
    """A request that timed out, in a few words: how long it waited, not the connection pool's text."""
    found = _TIMEOUT_SECONDS.search(str(exc))
    within = f" within {float(found.group(1)):g} s" if found else " in time"
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return f"couldn't connect to the service{within} (ConnectTimeout)"
    return f"no answer from the service{within} ({type(exc).__name__})"


# The service's 500 when it can't resolve the job's Experiment for the caller: the
# checkpoint routes say "... or not authorized for the training job", and the
# experiment-run route that saved logs and GPU metrics start with says "... for
# job <id>", which a deleted Experiment gets too.
EXPERIMENT_ACCESS_MARKER = "Linked experiment is not found or not authorized"
_EXPERIMENT_NOT_FOUND = "Linked experiment is not found"
_ASK = "ask for access to the Experiment, or use the role that created the job"
# What each way of being refused means for the tabs (see experiment_access).
_EXPERIMENT_HINTS = {
    "denied": f"this role can't read the job's Experiment (saved logs, GPU metrics and checkpoints live there); {_ASK}",
    "missing": ("this role can't read the job's Experiment, or it no longer exists (saved logs, GPU metrics and "
                f"checkpoints live there); {_ASK}"),
    # The service still lists the checkpoints: only the SQL on the Experiment is refused.
    "sql": f"this role can't read the job's Experiment, where saved logs and GPU metrics live (the checkpoints command works); {_ASK}",
}


def experiment_access(exc: BaseException) -> Optional[str]:
    """How ``exc`` says this role can't read the job's Experiment, or None.

    - ``"denied"``: the service's 500 ``Linked experiment is not found or not
      authorized ...`` (the checkpoint routes);
    - ``"missing"``: the experiment-run route's 500 ``Linked experiment is not
      found for job ...``: the Experiment is hidden from this role, or gone;
    - ``"sql"``: Snowflake refusing the SQL LIST or GET on the Experiment (a
      ``MONITOR``-only role): saved logs and GPU metrics can't be read, and the
      service still lists the checkpoints.

    The service's message is in the HTTP reply, not in ``str(exc)``, so both are
    read. A refusal about another object (a warehouse, a schema) is None even
    when its message names the Experiment.
    """
    text = f"{exc}\n{_reply_text(exc)}"
    if EXPERIMENT_ACCESS_MARKER in text:
        return "denied"
    if _EXPERIMENT_NOT_FOUND in text:
        return "missing"
    if _EXPERIMENT_SQL_DENIED.search(text):
        return "sql"
    return None


def experiment_access_hint(exc: BaseException) -> str | None:
    """The one-line explanation when ``exc`` means this role can't read the job's Experiment
    (see :func:`experiment_access`), else None."""
    kind = experiment_access(exc)
    return _EXPERIMENT_HINTS[kind] if kind else None


def is_live(job: Any) -> bool:
    """A recognized active state, or the service's UNKNOWN.

    The service answers UNKNOWN when it could not read a job's live state, so
    the job may well still be running, unless the service says the control
    plane has no record of the job (:func:`job_not_found`). Other unrecognized
    strings count as historical.
    """
    word = status_word(job.get("status")) if isinstance(job, dict) else ""
    return word in ACTIVE_STATUSES or (word == "unknown" and not job_not_found(job))


def job_not_found(job: Any) -> bool:
    """An UNKNOWN job whose reason says the control plane has no record of it
    (``Job status unavailable: the job was not found in the Control Plane``):
    it isn't running, so it isn't live."""
    if not (isinstance(job, dict) and status_word(job.get("status")) == "unknown"):
        return False
    return "not found" in " ".join(collapse(job.get(key) or "") for key in ("reason", "reason_details")).lower()


def job_id_of(job: dict) -> str:
    return str(job.get("job_id") or job.get("id") or "")


def is_unknown(job: Any) -> bool:
    """The service's UNKNOWN: it could not read the job's live state from the control plane."""
    return isinstance(job, dict) and status_word(job.get("status")) == "unknown"


# ─── names and comments ──────────────────────────────────────────────────────

NAME_WIDTH = 80  # a one-line name is cut to this many terminal cells
# printable (from format.py, so the Logs tab's one-line helper uses the same rule)
# replaces the characters a terminal would act on, or that reorder or hide the
# text around them, with U+FFFD, and keeps at most format.MAX_COMBINING_MARKS
# combining diacritics on one character.


def collapse(text: Any) -> str:
    """``text`` printable on one line: each run of whitespace, newlines too, as one space."""
    return " ".join(printable(text if text is not None else "").split())


def printable_data(value: Any) -> Any:
    """``value`` (decoded JSON) with every string in it, dict keys too, made :func:`printable`.

    Numbers, booleans and None are returned as they are. A string with nothing
    to replace is returned as it is, so a large job list costs little.
    """
    if isinstance(value, str):
        return value if value.isprintable() else printable(value)
    if isinstance(value, dict):
        return {printable_data(key): printable_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [printable_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(printable_data(item) for item in value)
    return value


def cell_width(text: str) -> int:
    """The terminal cells ``text`` takes: wide (CJK, emoji) characters take two."""
    return cell_len(text)


def _cut_cells(text: str, width: int) -> str:
    """The longest start of ``text`` that fits in ``width`` cells, never splitting
    a character; ``""`` when even the first one is wider.

    Each candidate start is measured whole with ``cell_len`` (a binary search on
    its length), so wide and combining characters count as the terminal draws
    them. (Not ``rich.cells.chop_cells``: before rich 13.7 it returns its pieces
    in reverse order.)
    """
    if width <= 0 or not text:
        return ""
    if cell_len(text) <= width:
        return text
    fits, too_long = 0, len(text)  # text[:fits] fits in width; text[:too_long] does not
    while too_long - fits > 1:
        mid = (fits + too_long) // 2
        if cell_len(text[:mid]) <= width:
            fits = mid
        else:
            too_long = mid
    return text[:fits]


def fit_cells(text: str, width: int) -> str:
    """``text`` cut to at most ``width`` terminal cells, ending in ``…`` when cut."""
    if cell_len(text) <= width:
        return text
    return _cut_cells(text, width - 1).rstrip() + "…"


def wrap_cells(line: str, width: int) -> list:
    """One line of text wrapped to rows of at most ``width`` terminal cells.

    It breaks between words where it can and inside a word where it must, and
    it counts cells, not characters, so wide (CJK, emoji) text stays inside
    the width. The line's own indentation is kept on its first row.
    """
    width = max(2, width)
    body = line.lstrip(" ")
    indent = line[: len(line) - len(body)]
    row = indent if cell_len(indent) < width else ""
    rows: list = []
    for token in re.findall(r"\S+|\s+", body):
        if token.isspace():
            if row.strip():
                row += token  # between words; dropped where the row breaks
            continue
        while token:
            if cell_len(row) + cell_len(token) <= width:
                row += token
                break
            if row.strip():  # the word starts the next row
                rows.append(row.rstrip())
                row = ""
                continue
            head = _cut_cells(token, width - cell_len(row))
            if not head:  # not even one character fits after the indent
                row = ""
                head = _cut_cells(token, width) or token[0]
            rows.append(row + head)
            row, token = "", token[len(head):]
    rows.append(row.rstrip())
    return rows


def _text_lines(text: Any) -> tuple:
    """The non-blank lines of ``text``, printable, whitespace collapsed."""
    return _lines_of(printable(text or ""))


@functools.lru_cache(maxsize=4096)
def _lines_of(s: str) -> tuple:
    return tuple(line for line in (" ".join(part.split()) for part in s.split("\n")) if line)


def first_line(text: Any) -> str:
    """The first non-blank line of ``text``, whitespace collapsed, never cut."""
    lines = _text_lines(text)
    return lines[0] if lines else ""


def one_line(text: Any, width: int = NAME_WIDTH) -> str:
    """The first non-blank line of ``text``, whitespace collapsed, ellipsized.

    The ellipsis marks a line cut at ``width`` terminal cells and also a text
    that goes on past its first line.
    """
    lines = _text_lines(text)
    if not lines:
        return ""
    first, more = lines[0], len(lines) > 1
    if cell_len(first) > (width - 2 if more else width):
        return _cut_cells(first, max(1, width - 1)).rstrip() + "…"
    return f"{first} …" if more else first


def comment_of(job: Any) -> str:
    """The job's create-time ``comment`` (all of it, printable), or ``""``."""
    raw = job.get("comment") if isinstance(job, dict) else None
    return printable(raw).strip() if raw is not None else ""


def alias_of(job: dict, aliases: Optional[dict] = None) -> str:
    """The job's local display name (set with ``n``), or ``""``."""
    return one_line((aliases or {}).get(job_id_of(job)) or "")


def display_name(job: dict, aliases: Optional[dict] = None) -> str:
    """Local alias first, then the first line of the server-side ``comment``;
    empty when neither."""
    return alias_of(job, aliases) or one_line(comment_of(job))


def search_name(job: dict, aliases: Optional[dict] = None) -> str:
    """The name search matches: the job's whole local alias, else the whole first
    line of its comment. Unlike :func:`display_name` it is never cut, so the end
    of a long (or wide-character) name can be found too."""
    return first_line((aliases or {}).get(job_id_of(job)) or "") or first_line(comment_of(job))


def rename_prompt(job: dict, aliases: Optional[dict] = None) -> tuple:
    """``(value, placeholder)`` for the rename prompt.

    It starts with the job's alias, whole. Without one it starts empty, and the
    placeholder shows the first line of the comment, which is already the job's
    name, so Enter keeps that name instead of copying it into an alias.
    """
    alias = first_line((aliases or {}).get(job_id_of(job)) or "")
    comment = first_line(comment_of(job))
    return alias, fit_cells(comment, 70) if comment else "e.g. qwen3-8b lora sweep #3"


def alias_to_store(job: dict, typed: Any) -> str:
    """The alias to keep for a name typed in the rename prompt; ``""`` for none.

    A name that only repeats the first line of the job's comment is not kept:
    without an alias, that line is the job's name already.
    """
    name = collapse(typed)
    return "" if name == first_line(comment_of(job)) else name


def shown_name(job: dict, aliases: Optional[dict] = None) -> str:
    """What the list and preview call a job: its display name, else its job id."""
    return display_name(job, aliases) or job_id_of(job)


# ─── sub-jobs ────────────────────────────────────────────────────────────────

# Sub-job type as the id's middle segment: the create-time type
# ``log_probability`` is ``log_prob`` there.
_ID_TYPES = {"training": "training", "sampling": "sampling", "log_probability": "log_prob", "log_prob": "log_prob"}


def sub_job_kind(sj: dict) -> str:
    """``training``, ``sampling`` or ``log_prob``; ``""`` when the entry names no known type."""
    raw = str(sj.get("job_type") or sj.get("type") or "").strip().lower().removeprefix("job_type_")
    return _ID_TYPES.get(raw, "")


def sub_jobs(job: dict) -> list:
    """The job's sub-job entries.

    When the service can't reach the control plane it answers UNKNOWN and lists
    only the configs it stored, with no sub-job ids. For such a row (and no
    other) the ids are derived the way the control plane assigns them,
    ``JOB_ID:TYPE:INDEX`` with INDEX the zero-based count of that type in
    submission order, but only when the row carries all of that: a job id, and
    a known type on every entry, none of which has an id of its own. An entry
    left with neither an id nor a type is labelled ``config #N``, its place in
    the list, never by its contents.
    """
    subs = [s for s in job.get("sub_jobs") or [] if isinstance(s, dict)]
    for sj in subs:
        if not (sj.get("sub_job_id") or sj.get("id")):
            break
    else:
        if not is_unknown(job):
            return subs  # the usual row (every entry has its id): nothing to derive or label
    kinds = [sub_job_kind(s) for s in subs]
    jid = job_id_of(job)
    if subs and is_unknown(job) and jid and all(kinds) and not any(s.get("sub_job_id") for s in subs):
        counts: dict = {}
        derived = []
        for sj, kind in zip(subs, kinds):
            derived.append({**sj, "sub_job_id": f"{jid}:{kind}:{counts.get(kind, 0)}"})
            counts[kind] = counts.get(kind, 0) + 1
        return derived
    return [sj if kind or sj.get("sub_job_id") or sj.get("id") else {**sj, "label": f"config #{i}"}
            for i, (sj, kind) in enumerate(zip(subs, kinds))]


def sub_job_type(sj: Any) -> str:
    """A sub-job's type as the list shows it: ``training``, ``sampling`` or ``log_prob``
    (the id's spelling, as the Logs tab tags it), else the type it names, lowercase."""
    if not isinstance(sj, dict):
        return ""
    return sub_job_kind(sj) or str(sj.get("job_type") or sj.get("type") or "").strip().lower().removeprefix("job_type_")


def sub_job_label(sj: Any) -> str:
    """``training #0`` style label (``log_prob #0``, as the sub-job id spells it), printable on
    one line; the API spells job types in upper case.

    An id may also come from a cached directory name, not the service's
    printable answer, so the label is made printable here too.
    """
    if isinstance(sj, dict):
        sj = {**sj, "job_type": sub_job_type(sj) or None}
    return collapse(source_label(sj))


def job_types(job: dict) -> list:
    """Each sub-job's type, as :func:`sub_job_type` spells it."""
    return [t for t in (sub_job_type(s) for s in sub_jobs(job)) if t]


def _raw_job_types(job: dict) -> list:
    """Each sub-job's type as the API spells it (``log_probability``), lowercase."""
    types = [str(s.get("job_type") or s.get("type") or "").strip().lower().removeprefix("job_type_")
             for s in sub_jobs(job)]
    return [t for t in types if t]


# A job of more than one kind of sub-job: the kinds, shortened, in this order.
_SHORT_KINDS = {"training": "train", "sampling": "sample", "log_prob": "logprob"}


def job_type_label(job: dict) -> str:
    """``training``, ``sampling`` or ``log_prob`` for a job of one kind; ``train+sample``,
    ``train+logprob`` and so on for a job of several."""
    kinds = list(dict.fromkeys(job_types(job)))
    if len(kinds) > 1 and all(kind in _SHORT_KINDS for kind in kinds):
        return "+".join(_SHORT_KINDS[kind] for kind in _SHORT_KINDS if kind in kinds)
    return ", ".join(kinds) or "—"


def short_model(name: Any) -> str:
    """A model name without its organization, as the list's Model column shows it
    (``Qwen/Qwen3-8B`` → ``Qwen3-8B``)."""
    return str(name or "").split("/")[-1]


def _gpu_count(cfg: dict) -> int:
    """A config's ``n_gpus`` as a whole number; 0 when it is missing or not a finite number
    (JSON's ``1e999`` reads as infinity)."""
    count = _as_float(cfg.get("n_gpus"))
    return int(count) if count is not None else 0


def sub_job_gpus(sj: dict) -> int:
    return _gpu_count(subjob_config(sj))


_CONFIG_KEYS = {"training": "training_config", "sampling": "inference_config", "log_prob": "inference_config"}


def _type_config(sj: dict, kind: Optional[str] = None) -> dict:
    """The sub-job's own config block: ``training_config`` for training,
    ``inference_config`` for sampling and log_prob (``kind``, when the caller
    already has it, is :func:`sub_job_kind`)."""
    key = _CONFIG_KEYS.get(sub_job_kind(sj) if kind is None else kind)
    cfg = sj.get(key) if key else None
    return cfg if isinstance(cfg, dict) else subjob_config(sj)


def _multiplex_id(cfg: dict) -> str:
    value = cfg.get("multiplex_job_id")
    return collapse(value) if value else ""


def multiplex_target(sj: dict) -> str:
    """The sub-job this one is multiplexed onto (its config's ``multiplex_job_id``),
    whose GPUs it runs on; ``""`` when it has GPUs of its own. Printable, on one line."""
    return _multiplex_id(_type_config(sj))


def requested_gpus(sj: dict) -> int:
    """GPUs this sub-job adds to its job's request, by the control plane's rule:
    its ``n_gpus``, except that a ``log_prob`` sub-job uses none and a sub-job
    multiplexed onto another sub-job adds none (it shares that one's GPUs)."""
    kind = sub_job_kind(sj)
    if kind == "log_prob":
        return 0
    cfg = _type_config(sj, kind)
    if _multiplex_id(cfg):
        return 0
    return _gpu_count(cfg)


def total_gpus(job: dict) -> int:
    """The GPUs a job requested: :func:`requested_gpus` summed over its sub-jobs.

    This is the request, not the allocation, which the job record does not
    carry (placement can round up to whole nodes).
    """
    return sum(requested_gpus(s) for s in sub_jobs(job))


def sub_job_gpu_text(sj: dict) -> str:
    """A sub-job's GPU cell: what it requested, or ``shared`` when it runs on
    another sub-job's GPUs. A multiplexed sub-job requests none of its own,
    whatever its config's ``n_gpus`` says, which is how the job's GPU line and
    the Resources tab count it too."""
    if multiplex_target(sj):
        return "shared"
    return str(requested_gpus(sj) or ("0" if sub_job_kind(sj) == "log_prob" else "—"))


def short_sub_job_id(sub_job_id: str) -> str:
    """A ``JOB_ID:TYPE:INDEX`` id with its job id cut to 8 characters, such as
    ``0a1b2c3d…:training:0``; any other id as it is."""
    job_id, sep, rest = str(sub_job_id).partition(":")
    return f"{job_id[:8]}…:{rest}" if sep and rest and len(job_id) > 9 else str(sub_job_id)


def job_gpu_cell(job: dict, unit: str = "") -> str:
    """A job's GPU cell: what it requested (``8 H200``, ``8 requested``), else
    ``shared`` when it runs only on another job's GPUs, ``0`` when its only
    sub-jobs are log_prob ones (which use none), else ``—``."""
    total = total_gpus(job)
    if total:
        return f"{total} {unit}".rstrip()
    subs = sub_jobs(job)
    if any(multiplex_target(s) for s in subs):
        return "shared"
    return "0" if any(sub_job_kind(s) == "log_prob" for s in subs) else "—"


def gpu_request_text(job: dict) -> str:
    """The preview's GPU line, e.g. ``10 × H200 requested  (8 training #0 + 2 sampling #0)``."""
    subs = sub_jobs(job)
    total = total_gpus(job)
    hw = str(job.get("hardware") or "").strip()
    shared = [s for s in subs if multiplex_target(s)]
    log_prob = [s for s in subs if sub_job_kind(s) == "log_prob"]
    if total:
        head = f"{total} × {hw} requested" if hw else f"{total} requested"
    elif shared:
        head = "no GPUs of its own requested"
    elif log_prob:
        head = "0 requested"
    else:
        return "—"
    own = [f"{requested_gpus(s)} {sub_job_label(s)}" for s in subs if requested_gpus(s)]
    notes = [" + ".join(own)] if len(own) > 1 or (own and shared) else []
    notes += [f"{sub_job_label(s)} runs on the GPUs of {multiplex_target(s)}" for s in shared]
    notes += [f"{sub_job_label(s)} uses none" for s in log_prob]
    return head + (f"  ({'; '.join(notes)})" if notes else "")


def job_models(job: dict) -> list:
    return list(dict.fromkeys(str(s["model_name"]) for s in sub_jobs(job) if s.get("model_name")))


def owner(job: dict) -> str:
    """``USER (ROLE)``; ``role ROLE`` when the job records no submitting user, so a
    role is never taken for a user name; ``—`` when neither."""
    user = collapse(job.get("submitted_by") or "")
    role = collapse(job.get("owner_role") or "")
    if user and role:
        return f"{user} ({role})"
    if role:
        return f"role {role}"
    return user or "—"


def training_method(sj: dict) -> str:
    cfg = subjob_config(sj)
    return "lora" if isinstance(cfg.get("peft_config"), dict) else "full"


def job_elapsed(job: dict, now: datetime) -> Optional[float]:
    start = parse_ts(job.get("created_at"))
    if not start:
        return None
    end = now if is_live(job) else parse_ts(job.get("state_changed_at") or job.get("updated_at")) or now
    return max(0.0, (end - start).total_seconds())


def within_retention(job: dict, now: datetime, days: int = RETENTION_DAYS) -> bool:
    created = parse_ts(job.get("created_at"))
    return created is None or (now - created).total_seconds() <= days * 86400


def subjob_config_summary(sj: dict) -> str:
    """Compact human summary of one sub-job's create-time config."""
    cfg = subjob_config(sj)
    bits = []
    opt = cfg.get("optimizer")
    if isinstance(opt, dict):
        o = str(opt.get("name") or "")
        if opt.get("lr") is not None:
            o = f"{o} lr={compact_number(opt['lr'])}".strip()
        if o:
            bits.append(o)
    for key, label in (("train_batch_size", "bs"), ("max_seq_len", "seq")):
        if cfg.get(key) is not None:
            bits.append(f"{label}={compact_number(cfg[key])}")
    peft = cfg.get("peft_config")
    if isinstance(peft, dict):
        lora = "LoRA"
        if peft.get("r") is not None:
            lora += f" r={compact_number(peft['r'])}"
        if peft.get("lora_alpha") is not None:
            lora += f" α={compact_number(peft['lora_alpha'])}"
        bits.append(lora)
    elif str(sj.get("job_type") or "").lower() == "training":
        bits.append("full-param")
    if cfg.get("gpu_memory_utilization") is not None:
        bits.append(f"mem={compact_number(cfg['gpu_memory_utilization'])}")
    return " · ".join(bits)


def summary_fields(job: dict, name: str, now: datetime) -> list:
    """(label, value) rows for the job preview, in display order.

    ``name`` is the one-line display name; without one the row shows the job
    id. A job with a ``comment`` also gets a ``Comment`` row with all of it,
    unless the name already shows all of it. A time the job record lacks is
    ``—``, with no "ago" after it.
    """
    models = job_models(job)
    elapsed = job_elapsed(job, now)
    # Elapsed runs from creation, so it includes any time spent queued or placing.
    if is_unknown(job):
        note = "includes queue time; status unavailable"
    elif is_live(job):
        note = f"includes queue time; {status_word(job.get('status'))}, so far"
    else:
        note = "includes queue time"
    comment = comment_of(job)
    location = location_label(job)
    # The age only beside a time that is shown: a dash has none.
    updated, ago = fmt_ts(job.get("updated_at")), fmt_ago(job.get("updated_at"), now)
    return [
        ("Name", name or job_id_of(job)),
        *([("Comment", comment)] if comment and comment != name else []),
        ("Job ID", job_id_of(job)),
        *([("Location", location)] if location else []),
        ("Status", status_word(job.get("status")).upper()),
        ("Model", ", ".join(models) or "—"),
        ("Type", job_type_label(job)),
        ("Owner", owner(job)),
        ("GPUs", gpu_request_text(job)),
        ("Created", fmt_ts(job.get("created_at"))),
        ("Elapsed", fmt_duration(elapsed) + (f"  ({note})" if elapsed is not None else "")),
        ("Last update", f"{updated}  ({ago})" if "—" not in (updated, ago) else updated),
    ]


def reason_text(job: dict) -> str:
    """``reason`` and ``reason_details`` on one printable line, ``""`` when neither.

    For example ``quota_filled — needs 8 GPUs; 64 currently active``.
    """
    parts = [collapse(job.get(key) or "") for key in ("reason", "reason_details")]
    return " — ".join(p for p in parts if p)


# Failure reasons that mean the job never got GPUs, so no sub-job started or
# wrote logs: every ``placement_*`` reason (``placement_timeout``,
# ``placement_too_large``, ``placement_failed``) and these.
PRE_START_REASONS = frozenset({"hardware_unavailable", "pending_timeout"})
# Sub-job states before placement: a sub-job still in one never got GPUs.
_UNPLACED = frozenset({"pending", "queued", "creating", "placing"})


def never_started(job: Any) -> bool:
    """An ended job that never got GPUs, so it has no logs to show.

    It failed with a reason that says so (``placement_*``, or one of
    :data:`PRE_START_REASONS`), or every sub-job the service lists for it is
    still in a state before placement.
    """
    if not isinstance(job, dict) or is_live(job):
        return False
    reason = str(job.get("reason") or "").strip().lower()
    if status_word(job.get("status")) == "failed" and (reason.startswith("placement_") or reason in PRE_START_REASONS):
        return True
    words = [status_word(sj.get("status")) for sj in sub_jobs(job)]
    return bool(words) and all(word in _UNPLACED for word in words)


def never_started_text(job: dict) -> str:
    """Why a job that :func:`never_started` has no logs, as a clause, for example
    ``the job never got GPUs (reason: placement_timeout), so no sub-job started or wrote logs``."""
    ended = {"cancelled": "was cancelled and ", "canceled": "was cancelled and ",
             "terminated": "was terminated and "}.get(status_word(job.get("status")), "")
    reason = collapse(job.get("reason") or "")
    why = f" (reason: {reason})" if reason else ""
    return f"the job {ended}never got GPUs{why}, so no sub-job started or wrote logs"


def status_unavailable_text(job: dict) -> str:
    """What to say for an UNKNOWN job, using the sentence the service sends in
    ``reason`` (``Job status unavailable: Control Plane is unavailable``)."""
    reason = reason_text(job)
    if reason.lower().startswith("job status unavailable"):
        return reason
    return f"status unavailable: {reason or 'the service could not read the live state'}"


# Fields the service recomputes on every get: a fresh answer replaces them as a
# whole, so one it stopped sending (a cleared reason) disappears. created_at is
# not one: it never changes, and an UNKNOWN answer leaves it out.
SERVICE_FIELDS = ("status", "reason", "reason_details", "updated_at", "state_changed_at", "sub_jobs")


def refresh_job(current: Optional[dict], fresh: dict) -> dict:
    """``current`` (say, a list row) updated with a fresh ``get_job`` answer.

    The service-owned :data:`SERVICE_FIELDS` come only from ``fresh``; other
    fields the row had and the answer lacks (``database_name``, ``created_at``)
    are kept.
    """
    kept = {k: val for k, val in (current or {}).items() if k not in SERVICE_FIELDS}
    return {**kept, **fresh}


# ─── search ──────────────────────────────────────────────────────────────────

# field -> help text; the field names users type in the search box.
SEARCH_FIELDS = {
    "name": "display name or comment",
    "id": "job id",
    "status": "job status",
    "owner": "submitting user or owning role",
    "type": "training / sampling",
    "model": "model name",
    "method": "full / lora",
    "gpu": "GPU type (H200, B200, B300)",
    "gpus": "requested GPU count",
    "optimizer": "optimizer name",
    "lr": "learning rate",
    "batch": "train batch size",
    "seq": "max sequence length",
    "lora_r": "LoRA rank",
    "lora_alpha": "LoRA alpha",
    "dtype": "dtype",
    "db": "database name (= and != by SQL identifier rules)",
    "schema": "schema name (= and != by SQL identifier rules)",
}
# Fields that name where a job is. A row without a location (the configured
# schema's own list) is in the configured database and schema, which the
# caller passes as ``home``.
_LOCATION_FIELDS = ("db", "schema")
_FIELD_ALIASES = {
    "hardware": "gpu",
    "hw": "gpu",
    "gpu_type": "gpu",
    "n_gpus": "gpus",
    "learning_rate": "lr",
    "opt": "optimizer",
    "bs": "batch",
    "batch_size": "batch",
    "train_batch_size": "batch",
    "seq_len": "seq",
    "max_seq_len": "seq",
    "rank": "lora_r",
    "r": "lora_r",
    "alpha": "lora_alpha",
    "user": "owner",
    "job_type": "type",
    "model_name": "model",
    "database": "db",
    "database_name": "db",
    "schema_name": "schema",
}
_OPS = (">=", "<=", "!=", ":", "=", ">", "<")
_TOKEN = re.compile(r"^([A-Za-z_][\w.]*)(>=|<=|!=|:|=|>|<)(.+)$")
# One search token: runs of anything but blanks and quotes, and quoted pieces.
_RAW_TOKEN = re.compile(r"""(?:[^\s"']+|"[^"]*"|'[^']*')+""")
_WHOLLY_QUOTED = re.compile(r"""^(?:"[^"]*"|'[^']*')$""")


def fold(text: Any) -> str:
    """``text`` as search compares it: NFKC-normalized (``é`` written either way,
    fullwidth ``ｗｉｄｅ`` as ``wide``) and case-folded."""
    return _fold(text if isinstance(text, str) else str(text))


@functools.lru_cache(maxsize=8192)
def _fold(s: str) -> str:
    return unicodedata.normalize("NFKC", s).casefold()


@dataclass
class Query:
    terms: list = field(default_factory=list)
    filters: list = field(default_factory=list)  # (field, op, value)
    errors: list = field(default_factory=list)
    # (key, op, value, key as typed) of filters on a key that is no search field:
    # each one is an error until resolve_query finds the key as a member of a job.
    unknown: list = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.terms and not self.filters


def _tokens(text: str) -> list:
    """``(token, quoted)`` pairs: shell-like words, ``quoted`` when the whole token was in quotes."""
    text = text or ""
    raw = _RAW_TOKEN.findall(text)
    if "".join("".join(raw).split()) != "".join(text.split()):  # an unbalanced quote: plain words, as typed
        return [(tok, False) for tok in text.split()]
    out = []
    for tok in raw:
        try:
            words = shlex.split(tok)
        except ValueError:
            words = [tok]
        out.append((" ".join(words), bool(_WHOLLY_QUOTED.match(tok))))
    return out


def parse_query(text: str) -> Query:
    """Free-text words plus ``field<op>value`` filters (ops ``: = != > >= < <=``).

    A dotted field (``training_config.optimizer.lr``) is an advanced filter that
    reads that member of a job, its sub-jobs and their configs. A key that is no
    search field is an error here; :func:`resolve_query` makes it a filter on that
    member when a loaded job has it (``seed=42``, ``ep_size>1``). A token in
    quotes is a plain word, so ``"training:0"`` searches that text.
    """
    q = Query()
    for tok, quoted in _tokens(text):
        m = None if quoted else _TOKEN.match(tok)
        if not m:
            if tok:
                q.terms.append(fold(tok))
            continue
        key, op, value = m.group(1).lower(), m.group(2), m.group(3)
        key = _FIELD_ALIASES.get(key, key)
        if key not in SEARCH_FIELDS and "." not in key:
            q.errors.append(f"unknown field '{m.group(1)}'")
            q.unknown.append((key, op, value, m.group(1)))
            continue
        q.filters.append((key, op, value))
    return q


def _member_keys(job: dict) -> set:
    """The lowercase member names a job has at the top level of its record, its sub-jobs and their configs."""
    keys = {str(k).lower() for k in job}
    for sj in sub_jobs(job):
        keys.update(str(k).lower() for k in sj)
        keys.update(str(k).lower() for k in subjob_config(sj))
    return keys


def _config_paths(job: dict, key: str, prefix: str = "", obj: Any = None, depth: int = 0) -> set:
    """Dotted paths, from a sub-job, of every config member named ``key`` below the top level."""
    found: set = set()
    if obj is None:
        for sj in sub_jobs(job):
            for top, cfg in sj.items():
                if isinstance(cfg, dict):
                    found |= _config_paths(job, key, str(top), cfg, 1)
        return found
    if depth > 6 or not isinstance(obj, dict):
        return found
    for k, v in obj.items():
        path = f"{prefix}.{k}"
        if str(k).lower() == key:
            found.add(path)
        if isinstance(v, dict):
            found |= _config_paths(job, key, path, v, depth + 1)
    return found


def resolve_query(query: Query, jobs: Iterable[dict]) -> Query:
    """``query`` with each key that is no search field made a filter on that member
    (of the job record, a sub-job, or a sub-job's config), where a job in ``jobs``
    has it. A key no job has stays an error, with the dotted paths that end in it
    as a hint (``ep_size`` → ``training_config.ep_size`` when only nested)."""
    if not query.unknown:
        return query
    jobs = [j for j in jobs if isinstance(j, dict)]
    present: set = set()
    for job in jobs:
        present |= _member_keys(job)
    filters, errors, unknown = list(query.filters), [], []
    for key, op, value, typed in query.unknown:
        if key in present:
            filters.append((key, op, value))
            continue
        paths = sorted({p for job in jobs[:2000] for p in _config_paths(job, key)})[:3]
        hint = f" (did you mean {' or '.join(paths)}?)" if paths else ""
        errors.append(f"unknown field '{typed}'{hint}")
        unknown.append((key, op, value, typed))
    return Query(terms=list(query.terms), filters=filters, errors=errors, unknown=unknown)


def _dig(obj: Any, path: list) -> list:
    if not path:
        return [obj]
    if isinstance(obj, list):
        return [v for item in obj for v in _dig(item, path)]
    if isinstance(obj, dict):
        key = path[0]
        for k, v in obj.items():
            if str(k).lower() == key:
                return _dig(v, path[1:])
    return []


def _scalars(values: Iterable) -> list:
    """``values`` with each list replaced by its scalar elements (at any depth); dicts dropped.

    So a list member such as ``target_modules`` or ``betas`` matches when one of its elements does.
    """
    out: list = []
    for value in values:
        if isinstance(value, list):
            out.extend(_scalars(value))
        elif not isinstance(value, dict):
            out.append(value)
    return out


def _location_name(job: dict, key: str, home: Optional[tuple]) -> Optional[str]:
    """The stored database (``db``) or schema name of ``job``, else of ``home``; None when neither says."""
    names = location_names(job) or home
    return None if names is None else names[0 if key == "db" else 1]


def field_values(job: dict, key: str, name: str = "", *, home: Optional[tuple] = None) -> list:
    """All values a job carries for a search field (one per sub-job when relevant).

    ``home`` is the configured ``(database_name, schema_name)``, where a row
    that names no location of its own is.
    """
    if "." in key or key not in SEARCH_FIELDS:
        # A dotted path, or (after resolve_query) a member name: read it from the
        # job record, each sub-job, and each sub-job's config.
        path = key.split(".")
        found = _dig(job, path)
        for sj in sub_jobs(job):
            found.extend(_dig(sj, path))
            found.extend(_dig(subjob_config(sj), path))
        return _scalars(found)
    subs = sub_jobs(job)
    cfgs = [subjob_config(s) for s in subs]
    if key == "name":
        return [v for v in dict.fromkeys((name, comment_of(job))) if v]
    if key == "id":
        return [job_id_of(job)] + [str(s.get("sub_job_id") or "") for s in subs]
    if key == "status":
        return [status_word(job.get("status"))]
    if key == "owner":
        return [str(v) for v in (job.get("submitted_by"), job.get("owner_role")) if v]
    if key == "type":
        # The short kind the list shows (log_prob) and the type as the API spells it (log_probability).
        return list(dict.fromkeys([*job_types(job), *_raw_job_types(job)]))
    if key == "model":
        # The whole name, and the part after the organization that the Model column shows.
        return list(dict.fromkeys(name for model in job_models(job) for name in (model, short_model(model))))
    if key == "method":
        return [training_method(s) for s in subs if str(s.get("job_type") or "").lower() == "training"]
    if key == "gpu":
        return [job["hardware"]] if job.get("hardware") else []
    if key == "gpus":
        return [total_gpus(job)]
    if key == "dtype":
        return [s["dtype"] for s in subs if s.get("dtype")]
    if key in _LOCATION_FIELDS:
        stored = _location_name(job, key, home)
        if stored is None:
            return []
        return list(dict.fromkeys((stored, sql_identifier(stored))))  # as stored, and as SQL quotes it
    if key in ("optimizer", "lr"):
        opts = [c.get("optimizer") for c in cfgs if isinstance(c.get("optimizer"), dict)]
        return [o.get("name" if key == "optimizer" else "lr") for o in opts if o.get("name" if key == "optimizer" else "lr") is not None]
    if key == "batch":
        return [c["train_batch_size"] for c in cfgs if c.get("train_batch_size") is not None]
    if key == "seq":
        return [c["max_seq_len"] for c in cfgs if c.get("max_seq_len") is not None]
    if key in ("lora_r", "lora_alpha"):
        peft_key = "r" if key == "lora_r" else "lora_alpha"
        return [c["peft_config"][peft_key] for c in cfgs if isinstance(c.get("peft_config"), dict) and c["peft_config"].get(peft_key) is not None]
    return []


def _as_float(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def _compare(actual: Any, op: str, wanted: str) -> bool:
    a, w = _as_float(actual), _as_float(wanted)
    if a is not None and w is not None:
        close = math.isclose(a, w, rel_tol=1e-9, abs_tol=1e-12)
        return {
            ":": close, "=": close, "!=": not close,
            ">": a > w and not close, ">=": a >= w or close,
            "<": a < w and not close, "<=": a <= w or close,
        }[op]
    sa, sw = fold(actual), fold(wanted)
    if op == ":":
        return sw in sa
    if op == "=":
        return sa == sw
    if op == "!=":
        return sa != sw
    return False  # ordering needs numbers


def search_haystack(job: dict, name: str) -> str:
    """What plain search words match in ``job``, folded (see :func:`fold`): the
    name, the whole comment, ids, status, owner, GPU type, job type (as the list
    shows it, ``train+sample``, and as the API spells it), and models."""
    # The comment is searched whole, and also when a local alias is the name shown.
    bits = [name, comment_of(job), job_id_of(job), status_word(job.get("status")), owner(job),
            job.get("hardware") or "", job_type_label(job)]
    bits += job_types(job) + _raw_job_types(job) + job_models(job)
    bits += [s.get("sub_job_id") or "" for s in sub_jobs(job)]  # ids copied from logs or --sub-job-id
    return fold(" ".join(str(b) for b in bits))


def location_words(job: dict) -> list:
    """A row's location as plain words search it: as shown (``"ml"."dev"``) and bare (``ml.dev``), lowercase.

    A search box turns a typed ``"ml"."dev"`` into ``ml.dev`` (the quotes group
    words), so both spellings find it.
    """
    names = location_names(job)
    if names is None:
        return []
    return list(dict.fromkeys((fold(location_label(job)), fold(printable(f"{names[0]}.{names[1]}")))))


def _starts_a_name(term: str, words: list) -> bool:
    """``term`` occurs in ``words`` at the start of a database or schema name.

    Not anywhere: the documented database ``CORTEX_TRAINING_DB`` contains
    "training", which would then match every job in it.
    """
    for text in words:
        at = text.find(term)
        while at != -1:
            if at == 0 or text[at - 1] in '."':
                return True
            at = text.find(term, at + 1)
    return False


def _same_name(stored: str, wanted: str) -> bool:
    """``wanted`` names ``stored``: written exactly as stored, or as an identifier that resolves to
    it by SQL rules (``team_db`` is ``TEAM_DB``, ``'"ml"'`` is ``ml``; ``ML`` is not ``ml``)."""
    return wanted == stored or identifier_name(wanted) == stored


def job_matches_query(job: dict, query: Query, name: str = "", *, home: Optional[tuple] = None,
                      hay: Optional[str] = None) -> bool:
    """Whether ``job`` matches every word and filter of ``query``.

    ``home`` is the configured ``(database_name, schema_name)``, for ``db`` and
    ``schema`` filters on a row that names no location of its own. ``hay`` is
    :func:`search_haystack` of ``job`` and ``name`` when the caller keeps it.
    """
    if query.errors:
        return False  # a typo'd field should surface, not silently match everything
    if query.empty:
        return True
    if hay is None:
        hay = search_haystack(job, name)
    places = location_words(job)
    if any(t not in hay and not _starts_a_name(t, places) for t in query.terms):
        return False
    for key, op, value in query.filters:
        if key in _LOCATION_FIELDS and op in ("=", "!="):
            stored = _location_name(job, key, home)
            if (stored is not None and _same_name(stored, value)) != (op == "="):
                return False
            continue
        values = field_values(job, key, name, home=home)
        if op == "!=":
            if any(not _compare(v, op, value) for v in values):
                return False
            continue
        if not any(_compare(v, op, value) for v in values):
            return False
    return True


def no_match_message(text: str, query: Query, searched: int, *, cut: str = "", older: Iterable[str] = (),
                     fields: bool = True, location: str = "", widen: bool = False) -> str:
    """Why a search found nothing, one point per line (every line after the first
    starts with two spaces; a caller wraps each under its own start).

    ``cut`` says what the list holds when it doesn't hold every job, for example
    "the newest 2,000 of the last 6 months and every active job; more exist".
    ``older`` are the ids of jobs older than six months, which the list hides,
    that the search matches. ``fields``: list the search fields and what other
    keys read, when a key is unknown (False where the search help on screen
    lists them). ``location``: the one ``DB.SCHEMA`` the list holds (one
    schema's list), named with the count. ``widen``: ``a`` switches the list to
    every database and schema, so the message says it searches those.
    """
    older = list(older)
    if cut:
        lines = [f"No jobs match “{text}” among the {searched:,} job(s) loaded: {cut}.",
                 "  Search covers only the loaded jobs. To search all of one schema's jobs: --database DB "
                 "--schema SCHEMA --schema-only."]
    else:
        where = f" in {location}" if location else ""
        lines = [f"No jobs match “{text}” among {searched:,} job(s){where} in the last 6 months."]
    if older:
        many = len(older) > 1
        lines.append(f"  {len(older):,} older job{'s' if many else ''} match{'' if many else 'es'} (the list hides "
                     f"jobs older than 6 months): cortex-training tui {older[0]} opens {'the newest' if many else 'it'}.")
    if widen:
        lines.append("  a searches every database and schema.")
    if query.errors:
        lines.append("  " + "; ".join(query.errors) + ".")
        if fields:
            lines.append("  Fields: " + ", ".join(SEARCH_FIELDS) + "; any other key reads that member of a job, its "
                         "sub-jobs or their configs (seed=42), and a dotted one a nested member "
                         "(training_config.optimizer.lr<1e-4).")
        if any(op == ":" for _key, op, _value, _typed in query.unknown):
            lines.append("  To search for text with a colon in it, quote it: \"training:0\".")
    if query.terms:
        lines.append("  Words match name, comment, id, sub-job id, status, owner, GPU type, job type, model, and "
                     "the start of a location's database or schema name (case-insensitive).")
    if query.filters:
        lines.append("  Filters must all match; try fewer, or ':' for a partial match (model:qwen).")
    lines.append("  Esc clears the search.")
    return "\n".join(lines)


# ─── sorting ─────────────────────────────────────────────────────────────────

SORT_KEYS = ("created", "updated", "status")
_STATUS_ORDER = ["running", "initializing", "placing", "pending", "queued", "creating", "unknown", "failed", "done",
                 "cancelled", "terminated"]
# Other spellings of a status, sorted with it.
_STATUS_SPELLINGS = {"succeeded": "done", "completed": "done", "canceled": "cancelled"}
_STATUS_RANK = {**{s: i for i, s in enumerate(_STATUS_ORDER)},
                **{s: _STATUS_ORDER.index(same) for s, same in _STATUS_SPELLINGS.items()}}


_NO_TIME = float("-inf")  # what _epoch makes of a missing or placeholder time


def _epoch(raw: Any) -> float:
    dt = parse_ts(raw)
    return dt.timestamp() if dt else _NO_TIME


def sort_job_list(jobs: Iterable[dict], key: str = "created", descending: bool = True) -> list:
    """``jobs`` by ``key``: "created" or "updated" time (newest first, or oldest), or "status".

    Under a time sort, the jobs without that time (shown as ``—``) come after
    the others in both directions, newest created first. Under the status sort,
    jobs of one status are newest created first in both directions.
    """
    jobs = [j for j in jobs if isinstance(j, dict)]
    if key == "status":
        rank = _STATUS_RANK
        by_created = sorted(jobs, key=lambda j: _epoch(j.get("created_at")), reverse=True)
        return sorted(
            by_created,
            key=lambda j: rank.get(status_word(j.get("status")), len(_STATUS_ORDER)),
            reverse=not descending,
        )
    field_name = "updated_at" if key == "updated" else "created_at"
    keyed = [(_epoch(j.get(field_name)), j) for j in jobs]
    timed = sorted((pair for pair in keyed if pair[0] != _NO_TIME), key=lambda pair: pair[0], reverse=descending)
    untimed = [j for at, j in keyed if at == _NO_TIME]
    if field_name != "created_at":
        untimed.sort(key=lambda j: _epoch(j.get("created_at")), reverse=True)
    return [j for _at, j in timed] + untimed


def sort_direction(key: str, descending: bool) -> str:
    """Say in words which jobs come first under ``sort_job_list(key, descending)``,
    in the job list's groups (see :func:`live_first`)."""
    if key == "status":
        return "running first" if descending else "ended first"
    return "newest first" if descending else "oldest first"


def live_first(key: str, descending: bool) -> bool:
    """Whether the job list draws its LIVE group above HISTORY: always, but under
    the status sort reversed (``status ↑ ended first``), whose first jobs are the
    ended ones, as its words say."""
    return key != "status" or descending


# ─── locations, copied commands and access ───────────────────────────────────

_PLAIN_IDENTIFIER = re.compile(r"[A-Z_][A-Z0-9_$]*")


def sql_identifier(name: str) -> str:
    """``name`` as a SQL identifier: bare when it is a plain uppercase name, else double-quoted.

    ``CortexTrainingClient.for_job``'s rule: ``TEAM_DB`` stays bare, ``ml`` becomes ``"ml"``.
    """
    return name if _PLAIN_IDENTIFIER.fullmatch(name) else '"' + name.replace('"', '""') + '"'


_sql_identifier = sql_identifier  # the older private name


def identifier_name(text: str) -> str:
    """The name a SQL identifier resolves to: a ``"quoted"`` one exactly, any other uppercase.

    The inverse of :func:`sql_identifier`: ``"ml"`` is ``ml``, ``team_db`` is ``TEAM_DB``.
    """
    text = text.strip()
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        return text[1:-1].replace('""', '"')
    return text.upper()


def location_names(job: Any) -> Optional[tuple]:
    """``(database_name, schema_name)`` of a row from the account-wide list, else None.

    Only that list names a job's location (both names, as stored, unquoted); the
    configured schema's own list and ``get_job`` don't, and a job without one is
    in the configured database and schema.
    """
    if not isinstance(job, dict):
        return None
    db, sch = job.get("database_name"), job.get("schema_name")
    if isinstance(db, str) and db and isinstance(sch, str) and sch:
        return db, sch
    return None


def client_location(client: Any) -> Optional[tuple]:
    """``(database_name, schema_name)`` a client is bound to, resolved from its identifiers;
    None when it doesn't say (a demo or test client's attributes may be anything)."""
    db, sch = getattr(client, "database", None), getattr(client, "schema", None)
    if isinstance(db, str) and db.strip() and isinstance(sch, str) and sch.strip():
        return identifier_name(db), identifier_name(sch)
    return None


def location_label(job: Any) -> str:
    """``DB.SCHEMA`` of a row that names its location, each name quoted when SQL
    needs it (``TEAM_DB.EXPERIMENTS``, ``"ml"."dev"``); ``""`` without one."""
    names = location_names(job)
    if names is None:
        return ""
    return printable(f"{sql_identifier(names[0])}.{sql_identifier(names[1])}")


def cli_options(*, database: Optional[str] = None, schema: Optional[str] = None,
                connection: Optional[str] = None, config: Optional[str] = None, host: Optional[str] = None,
                base_url: Optional[str] = None) -> list:
    """The CLI's global options that reach one job: how it connects (``--connection
    NAME``, a profile; ``--config FILE``; ``--host HOST``, with a PAT; or
    ``--base-url URL``), then ``--database DB`` and ``--schema SCHEMA``, each
    value quoted for the shell.

    The copyable commands use these, so a command copied for a job in another
    database and schema reaches that job, not the configured location. An
    option whose value is unknown is left out.
    """
    found = (("--connection", connection), ("--config", config), ("--host", host), ("--base-url", base_url),
             ("--database", database), ("--schema", schema))
    return [f"{flag} {shlex.quote(value)}" for flag, value in found if isinstance(value, str) and value]


def cli_command(arguments: str, **location: Optional[str]) -> str:
    """``cortex-training [global options] ARGUMENTS`` on one line; see :func:`cli_options`."""
    return " ".join(["cortex-training", *cli_options(**location), arguments])


def _reply_text(exc: BaseException) -> str:
    """The body of the reply an HTTP error carries; empty for anything else."""
    try:
        text = getattr(getattr(exc, "response", None), "text", "")
    except Exception:  # noqa: BLE001 - an unreadable body
        return ""
    return text if isinstance(text, str) else ""


# How Snowflake refuses SQL on the job's Experiment for this role (the LIST and
# GET that read saved logs and GPU metrics): "003001 (42501): SQL access control
# error: Insufficient privileges to operate on experiment 'NAME'", "Experiment
# 'NAME' does not exist or not authorized", or the same said of a
# snow://experiment/ path. The refusal itself must be about the Experiment, so a
# denial about some other object (a warehouse, a schema) keeps its own text even
# when the message names the Experiment elsewhere.
_EXPERIMENT_SQL_DENIED = re.compile(
    r"insufficient privileges to operate on experiment\b"
    r"|\bexperiment(?:\s+run)?\s+(?:'[^']*'|\"[^\"]*\"|[^\s'\"]+)\s+does not exist or not authorized"
    r"|snow://experiment/[^\s'\"]*['\"]?:?\s+does not exist or not authorized",
    re.IGNORECASE)
# A 403 refuses this role when its reply says so in Snowflake's words. An empty or
# unrelated 403 (a proxy or a network gate can send one) keeps the standard text.
_PRIVILEGE_REFUSAL = re.compile(r"privilege|not authorized|access control error", re.IGNORECASE)


def access_hint(exc: BaseException, subject: str) -> Optional[str]:
    """One line when ``exc`` means this role may not read the job's ``subject``, else None.

    Every tab uses this for its own data (``subject`` is for example "saved
    logs" or "GPU metrics"): the Logs tab and Resources explain the same
    refusal the same way. An Experiment
    the role can't read gets :func:`experiment_access_hint`'s line. A 403 whose
    reply names a privilege says the role isn't allowed, keeping the service's
    message. Anything else, an empty 403 included, is None: the caller shows the
    standard error text.
    """
    hint = experiment_access_hint(exc)
    if hint:
        return hint
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 403 and _PRIVILEGE_REFUSAL.search(_reply_text(exc)):
        return f"this role isn't allowed to read this job's {subject} ({describe_error(exc)})"
    return None


def _name_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def sign_in_options(sign_in: Any) -> dict:
    """How a copied CLI command connects as the TUI does: ``config``, ``host`` and
    ``base_url`` for :func:`cli_options`, from ``sign_in``
    (``CortexTrainingLogTUI.sign_in``).

    The TUI's config file, when it read one, as ``--config FILE``; and its PAT
    host (``--host``) or base URL (``--base-url``), unless that file gives the
    same one by itself (``host_in_config`` / ``base_url_in_config``): a flag or
    the environment that set or overrode it is not part of the file. Empty for
    a connection profile, or when it isn't known.
    """
    sign = sign_in if isinstance(sign_in, dict) else {}
    kind, config = _name_or_none(sign.get("kind")), _name_or_none(sign.get("config"))
    if kind == "pat":
        in_config = bool(config) and sign.get("host_in_config") is True
        return {"config": config, "host": None if in_config else _name_or_none(sign.get("host"))}
    if kind == "url":
        in_config = bool(config) and sign.get("base_url_in_config") is True
        return {"config": config, "base_url": None if in_config else _name_or_none(sign.get("base_url"))}
    return {}
