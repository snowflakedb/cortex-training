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

"""Command-line implementation for Cortex Training job management."""

from __future__ import annotations

import argparse
import codecs
import contextlib
import itertools
import json
import logging
import os
import re
import sys
import unicodedata
from datetime import date
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterator
from typing import TextIO

logger = logging.getLogger(__name__)

_CONFIG_KEYS = {
    "base_url",
    "host",
    "pat",
    "database",
    "schema",
    "endpoint",
    "poll_interval",
    "poll_timeout",
    "no_verify_ssl",
    "verify_ssl",
}

_CONFIG_ALIASES = {
    "url": "base_url",
    "db": "database",
}

_LOGIN_STATE_ENV = "CORTEX_TRAINING_LOGIN_FILE"

# The statuses each `list --status` word matches, as on the account-level
# route: pending excludes placing, terminal adds failed to cancelled and
# terminated, and UNKNOWN rows match nothing. done keeps its older meaning,
# cancelled and terminated; the account-level route no longer takes the word,
# so the client sends those two for it there, and only this local filter can
# also keep the DONE that older jobs show.
_LIST_STATUS_FILTERS = {
    "pending": {"pending"},
    "placing": {"placing"},
    "initializing": {"initializing"},
    "running": {"running"},
    "failed": {"failed"},
    "cancelled": {"cancelled"},
    "terminated": {"terminated"},
    "active": {"pending", "placing", "initializing", "running"},
    "terminal": {"cancelled", "terminated", "failed"},
    "done": {"cancelled", "terminated", "done"},
}

# Status words the server can answer with that are not the ones it filters on:
# older rows, and states a backend ahead of the API layer reports.
_STATUS_ALIASES = {
    "queued": "pending",
    "creating": "initializing",
    "canceled": "cancelled",
    "completed": "done",
    "succeeded": "done",
    "error": "failed",
}

# `capacity --breakdown` choice -> get_capacity_breakdown() dimension.
_CAPACITY_BREAKDOWN_CHOICES = {
    "job-type": "job_type",
    "submitted-by": "submitted_by",
}


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _normalize_host(host: str) -> str:
    for prefix in ("https://", "http://"):
        if host.startswith(prefix):
            host = host[len(prefix) :]
    return host.rstrip("/")


def _has_url_scheme(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


# Characters a terminal can act on, or that hide or reorder the text around
# them: control characters (C0 and C1; ESC starts escape sequences, CR rewrites
# the line), invisible format characters such as bidi overrides and zero-width
# marks, line and paragraph separators, and unpaired surrogates, which have no
# UTF-8 form.
_UNSAFE_TEXT_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})
# Candidates for that check: anything but printable ASCII and the newline.
_NOT_PLAIN_TEXT_RE = re.compile(r"[^\n\x20-\x7e]")


def _is_unsafe_char(char: str) -> bool:
    return unicodedata.category(char) in _UNSAFE_TEXT_CATEGORIES


def _printable(text: str) -> str:
    """``text`` for a terminal: each unsafe character becomes its escape, such as ``\\x1b``.

    Service answers, file names and error bodies reach stderr through this, so
    none of them can move the cursor, retitle the window or write the
    clipboard. Newlines are kept; other text, accented letters included, is
    unchanged.
    """

    def escape(match: re.Match) -> str:
        char = match.group()
        return ascii(char)[1:-1] if _is_unsafe_char(char) else char

    return _NOT_PLAIN_TEXT_RE.sub(escape, text)


class _PrintableFormatter(logging.Formatter):
    """A log formatter whose lines are safe for a terminal (see :func:`_printable`)."""

    def format(self, record: logging.LogRecord) -> str:
        return _printable(super().format(record))


# The service rejects a GPU type the account does not support with this code.
_UNSUPPORTED_HARDWARE_CODE = "517604"


def _load_cortex_training_client_class():
    from .client import CortexTrainingClient

    return CortexTrainingClient


def _load_forward_backward_payload_builder():
    from .client import build_forward_backward_payload

    return build_forward_backward_payload


def _hardware_choices() -> list[str]:
    from .client import Hardware

    return [member.value for member in Hardware]


_FRACTION_RE = re.compile(r"\.([0-9]+)")


def _created_epoch(raw: Any) -> float | None:
    if raw is None:
        return None
    created_at = str(raw).strip()
    if not created_at:
        return None
    if created_at.endswith("Z"):
        created_at = created_at[:-1] + "+00:00"
    # Before Python 3.11, fromisoformat takes only 3 or 6 fractional digits.
    created_at = _FRACTION_RE.sub(
        lambda match: "." + match.group(1)[:6].ljust(6, "0"), created_at, count=1
    )
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return created.timestamp()


def _jobs_latest_last(jobs: list[Any], limit: int | None = None) -> list[Any]:
    """Render jobs oldest-first so the newest entries land at the terminal.

    ``limit`` keeps only that many of the newest by ``created_at``. A job
    without a readable ``created_at`` counts as oldest: ``limit`` drops it
    first, and it is printed first. Ties rank in the server's order, the
    account-level list answering newest first. The jobs are printed in exactly
    the reverse of that ranking, so ``list --limit N`` prints the last N jobs
    of ``list``.
    """

    def age(job: Any) -> float:
        created = _created_epoch(job.get("created_at")) if isinstance(job, dict) else None
        return -created if created is not None else float("inf")

    newest_first = sorted(jobs, key=age)
    if limit is not None:
        newest_first = newest_first[:limit]
    return newest_first[::-1]


def _status_list(raw: str) -> str:
    """argparse type for --status: the comma list, lowercased, every word known."""
    words = [word.strip().lower() for word in raw.split(",")]
    for word in words:
        if word not in _LIST_STATUS_FILTERS:
            raise argparse.ArgumentTypeError(
                f"unknown status {word!r}; use a comma-separated list of "
                + ", ".join(_LIST_STATUS_FILTERS)
            )
    return ",".join(words)


def _list_statuses(words: str) -> set[str]:
    """The job statuses a --status list from _status_list keeps."""
    statuses: set[str] = set()
    for word in words.split(","):
        statuses |= _LIST_STATUS_FILTERS[word]
    return statuses


_DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def _utc_date(value: str) -> date:
    """argparse type for --since/--until: a YYYY-MM-DD date, read as a UTC day."""
    # Matched first because date.fromisoformat also takes forms such as 20261001.
    if _DATE_RE.fullmatch(value):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"expected a YYYY-MM-DD date, got {value!r}")


def _utc_midnight(day: date) -> float:
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()


def _same_name(wanted: str, actual: Any) -> bool:
    # Rows carry names as stored. SQL identifier rules, as the server applies to
    # submitted_by: an unquoted name means its uppercase form, a "quoted" one
    # exactly what is between the quotes.
    if len(wanted) > 1 and wanted[0] == wanted[-1] == '"':
        wanted = wanted[1:-1].replace('""', '"')
    else:
        wanted = wanted.upper()
    return isinstance(actual, str) and actual == wanted


def _list_filter(args: argparse.Namespace) -> Callable[[dict], bool]:
    """Build the list filters for the schema's jobs; --all has the server apply them."""
    # Rows can carry an enum name (JOB_STATE_RUNNING) or an alias, so compare
    # on the normalized word rather than the raw field.
    from .client import CortexTrainingClient

    def status_of(job: dict) -> str:
        word = CortexTrainingClient._normalize_job_status(job.get("status"))
        return _STATUS_ALIASES.get(word, word)

    statuses = None if args.status is None else _list_statuses(args.status)
    since = None if args.since is None else _utc_midnight(args.since)
    # --until names the last day kept, so the bound is the next midnight.
    until = None if args.until is None else _utc_midnight(args.until + timedelta(days=1))

    def keep(job: dict) -> bool:
        if statuses is not None and status_of(job) not in statuses:
            return False
        if args.user is not None and not _same_name(args.user, job.get("submitted_by")):
            return False
        if since is None and until is None:
            return True
        created = _created_epoch(job.get("created_at"))
        return (
            created is not None
            and (since is None or created >= since)
            and (until is None or created < until)
        )

    return keep


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def build_parser(
    *,
    prog: str = "cortex-training",
    include_tui: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Manage Cortex Training jobs through the Cortex Training REST endpoint.",
    )
    parser.add_argument(
        "--config",
        default=_env("CORTEX_TRAINING_CONFIG"),
        help="Path to a reusable Cortex Training CLI config JSON file.",
    )
    parser.add_argument(
        "--connection",
        "-c",
        default=_env("CORTEX_TRAINING_CONNECTION"),
        help=(
            "Snowflake connection profile name. When no legacy config or direct "
            "credentials are configured, the Snowflake default profile is used."
        ),
    )
    parser.add_argument(
        "--base-url",
        help="Base URL for a local or otherwise compatible server. Skips PAT auth.",
    )
    parser.add_argument(
        "--host",
        help="Snowflake account host for PAT auth.",
    )
    parser.add_argument(
        "--pat",
        help="Programmatic access token.",
    )
    parser.add_argument(
        "--database",
        help="Database containing the cortex-training endpoint.",
    )
    parser.add_argument(
        "--schema",
        help="Schema containing the cortex-training endpoint. Defaults to PUBLIC.",
    )
    parser.add_argument(
        "--endpoint",
        help="REST endpoint name. Defaults to cortex-training.",
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL certificate verification for authenticated requests.",
    )
    parser.add_argument("--poll-interval", type=float)
    parser.add_argument("--poll-timeout", type=float)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print compact JSON instead of pretty JSON.",
    )
    parser.add_argument(
        "--job",
        "--job-id",
        dest="job",
        help="Job id for data-plane commands such as fwd-bwd.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser(
        "submit",
        help="Submit a Cortex Training job from a CreateJob JSON file.",
    )
    submit.add_argument(
        "json_file",
        help="Path to CreateJob JSON, or '-' for stdin.",
    )
    submit.add_argument("--job-id", help="Set or override the top-level job_id.")
    submit.add_argument(
        "--comment",
        metavar="TEXT",
        help=(
            "Name the job: free text, at most 1,024 bytes of UTF-8, that get, list "
            "and the TUI show. It can't be changed later. Overrides a comment in the "
            "JSON; an empty TEXT sends none. A model deployment is sent without it, "
            "because it rejects one."
        ),
    )
    submit.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the CreateJob body without sending it.",
    )
    submit.add_argument(
        "--wait",
        action="store_true",
        help="Wait until the submitted job reaches running.",
    )

    get = subparsers.add_parser("get", help="Fetch one Cortex Training job.")
    get.add_argument("job_id")

    checkpoints = subparsers.add_parser(
        "checkpoints",
        help="List checkpoints for one Cortex Training job.",
    )
    checkpoints.add_argument("job_id")

    list_jobs = subparsers.add_parser("list", help="List Cortex Training jobs, oldest first.")
    list_jobs.add_argument(
        "--status",
        type=_status_list,
        help=(
            "Comma-separated statuses to keep: "
            + ", ".join(_LIST_STATUS_FILTERS)
            + ". active is pending through running; terminal is cancelled, "
            "terminated, and failed; done is cancelled and terminated, plus, "
            "without --all, the DONE that older jobs show."
        ),
    )
    list_jobs.add_argument(
        "--all",
        action="store_true",
        help=(
            "List jobs across every schema in the account that the current user "
            "can see, not just the configured one. Works once the service has "
            "account-level listing and it is enabled for the account; until "
            "then it fails and says why. With --status, a failing page is "
            "retried twice, not ten times."
        ),
    )
    list_jobs.add_argument(
        "--user",
        help=(
            "Keep jobs submitted by this user. With --all, a name that matches "
            "no active user in the account is an error."
        ),
    )
    list_jobs.add_argument(
        "--since",
        type=_utc_date,
        metavar="YYYY-MM-DD",
        help="Keep jobs created on or after this date (a UTC day).",
    )
    list_jobs.add_argument(
        "--until",
        type=_utc_date,
        metavar="YYYY-MM-DD",
        help="Keep jobs created on or before this date (a UTC day).",
    )
    list_jobs.add_argument(
        "--limit",
        type=_positive_int,
        help="Print at most this many of the newest matching jobs.",
    )

    capacity = subparsers.add_parser(
        "capacity",
        help="Show reserved GPU capacity and current usage for the caller account.",
    )
    capacity.add_argument(
        "--hardware",
        choices=_hardware_choices(),
        help=(
            "Show only this GPU hardware. Omit to show capacity for every type "
            "your account supports; unsupported types are listed separately."
        ),
    )
    capacity.add_argument(
        "--breakdown",
        action="append",
        choices=tuple(_CAPACITY_BREAKDOWN_CHOICES),
        help=(
            "Also split in_use_gpus and pending_gpus by sub-job type "
            "(job-type) or by submitting user (submitted-by). Repeat to get "
            "both."
        ),
    )

    cancel = subparsers.add_parser("cancel", help="Cancel one Cortex Training job.")
    cancel.add_argument("job_id")

    wait = subparsers.add_parser("wait", help="Wait until one job reaches running.")
    wait.add_argument("job_id")

    fwd_bwd = subparsers.add_parser(
        "fwd-bwd",
        help="Run one forward-backward request from a readable payload JSON file.",
    )
    fwd_bwd.add_argument(
        "json_file",
        help="Path to fwd-bwd JSON, or '-' for stdin.",
    )

    step = subparsers.add_parser(
        "step",
        help="Run one optimizer step for a training job.",
    )
    step.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer step. Defaults to 1e-4.",
    )

    load = subparsers.add_parser(
        "load",
        help="Load a checkpoint into an already-created Cortex Training job.",
    )
    load.add_argument("checkpoint_id", help="Checkpoint id/tag to load.")
    load.add_argument(
        "--source-job-id",
        help="Load the checkpoint from another job's checkpoint store.",
    )
    load.add_argument(
        "--target-sub-job-id",
        dest="target_sub_job_id",
        help=(
            "Training sub-job to load the checkpoint into, e.g. JOB_ID:training:0. "
            "Use this when you need explicit routing control; a job has at most one "
            "training sub-job, so omitting it uses that sub-job. "
            "Use 'cortex-training get JOB_ID' to discover available sub-job IDs."
        ),
    )
    load.add_argument(
        "--no-poll",
        action="store_true",
        help="Print the request id without polling for completion.",
    )

    generate = subparsers.add_parser(
        "generate",
        help="Run one generate request from a readable payload JSON file.",
    )
    generate.add_argument(
        "json_file",
        help="Path to generate JSON, or '-' for stdin.",
    )

    weight_sync = subparsers.add_parser(
        "weight-sync",
        help="Sync weights from a training sub-job to one or more sampling sub-jobs.",
    )
    weight_sync.add_argument(
        "--source-sub-job-id",
        help="Training sub-job id. Defaults to JOB_ID:training:0.",
    )
    weight_sync.add_argument(
        "--target-sub-job-id",
        action="append",
        dest="target_sub_job_ids",
        help="Sampling sub-job id. Can be repeated. Defaults to JOB_ID:sampling:0.",
    )
    weight_sync.add_argument(
        "--operation-sub-job-id",
        help="Sub-job id used to route the operation envelope. Defaults to the source training sub-job id.",
    )
    weight_sync.add_argument(
        "--operation-sub-job-type",
        help="Sub-job type used to route the operation envelope.",
    )
    weight_sync.add_argument(
        "--weight-format",
        choices=("vllm", "hf", "lora"),
        help="Weight sync payload format. Use 'lora' for adapter-only sync.",
    )
    weight_sync.add_argument(
        "--no-poll",
        action="store_true",
        help="Print the request id without polling for completion.",
    )

    for command, job_parser in (
        ("fwd-bwd", fwd_bwd),
        ("step", step),
        ("load", load),
        ("generate", generate),
        ("weight-sync", weight_sync),
    ):
        job_parser.prog = f"{prog} --job JOB_ID {command}"
        job_parser.epilog = (
            "Required global option: --job JOB_ID (alias: --job-id JOB_ID). "
            "Place it before the subcommand."
        )

    download_log = subparsers.add_parser(
        "download-log",
        help="Download all log files for a Cortex Training job's experiment run.",
    )
    download_log.add_argument("job_id")
    download_log.add_argument(
        "--log-type",
        choices=("execution", "stdout"),
        default="execution",
        help=(
            "Log source to download: execution artifacts (default), or the "
            "reconstructed head-pod stdout/stderr console."
        ),
    )
    download_log.add_argument(
        "--output-dir",
        dest="output_dir",
        help=(
            "Directory to write log files into, grouped as "
            "<output_dir>/<sub_job_id>/<filename>. Created if missing. "
            "Defaults to the current working directory."
        ),
    )
    download_log.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue an earlier stdout download in the output directory and "
            "fetch only output that is not already saved. Two downloads of the "
            "same directory at once are unsupported."
        ),
    )

    download_metrics = subparsers.add_parser(
        "download-metrics",
        help="Download and reconstruct GPU metrics for a Cortex Training job.",
    )
    download_metrics.add_argument("job_id")
    download_metrics.add_argument(
        "--output-dir",
        dest="output_dir",
        help=(
            "Directory to write <sub_job_id>/gpu.jsonl files. Created if "
            "missing. Defaults to the current working directory."
        ),
    )
    download_metrics.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue an earlier metrics download in the output directory and "
            "fetch only metrics that are not already saved. Two downloads of "
            "the same directory at once are unsupported."
        ),
    )

    login = subparsers.add_parser(
        "login",
        usage="%(prog)s [-h] (config | --config config)",
        help="Remember a Cortex Training config file for future commands.",
    )
    login_config = login.add_mutually_exclusive_group(required=True)
    login_config.add_argument(
        "login_config",
        nargs="?",
        metavar="config",
        help="Path to the Cortex Training CLI config JSON file to remember.",
    )
    login_config.add_argument(
        "--config",
        dest="login_config_option",
        metavar="config",
        help="Alternative to the positional config path.",
    )

    if include_tui:
        subparsers.add_parser(
            "tui",
            help="Open the read-only Cortex Training job TUI.",
        )

    return parser


def _login_state_path() -> Path:
    override = _env(_LOGIN_STATE_ENV)
    if override:
        return Path(override).expanduser()
    config_home = _env("XDG_CONFIG_HOME")
    base = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return base / "cortex-training" / "login.json"


def _read_login_config_path() -> str | None:
    path = _login_state_path()
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid cortex-training login state {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"invalid cortex-training login state {path}: expected object")
    config_path = parsed.get("config_path")
    if not isinstance(config_path, str) or not config_path:
        raise ValueError(f"invalid cortex-training login state {path}: missing config_path")
    return config_path


def _write_login_config_path(config_path: str) -> str:
    path = Path(config_path).expanduser()
    _load_config(str(path))
    saved_path = str(path.resolve())
    state_path = _login_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config_path": saved_path}
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return saved_path


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    path_obj = Path(path).expanduser()
    try:
        parsed = json.loads(path_obj.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in config {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("config JSON must be an object")

    if "connection" in parsed:
        connection = parsed["connection"]
        if not isinstance(connection, dict):
            raise ValueError("config connection must be an object")
        parsed = connection

    config: dict[str, Any] = {}
    unknown = []
    for key, value in parsed.items():
        normalized = _CONFIG_ALIASES.get(key, key)
        if normalized not in _CONFIG_KEYS:
            unknown.append(key)
            continue
        config[normalized] = value
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown config key(s): {names}")
    return config


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _config_str(config: dict[str, Any], key: str) -> str | None:
    if key not in config or config[key] is None:
        return None
    if not isinstance(config[key], str):
        raise ValueError(f"config {key} must be a string")
    return config[key]


def _config_float(config: dict[str, Any], key: str) -> float | None:
    if key not in config or config[key] is None:
        return None
    if isinstance(config[key], bool) or not isinstance(config[key], (int, float)):
        raise ValueError(f"config {key} must be a number")
    return float(config[key])


def _config_bool(config: dict[str, Any], key: str) -> bool | None:
    if key not in config or config[key] is None:
        return None
    if not isinstance(config[key], bool):
        raise ValueError(f"config {key} must be a boolean")
    return config[key]


def _has_connection(
    base_url: str | None,
    host: str | None,
    pat: str | None,
    database: str | None,
) -> bool:
    return bool(database and (base_url or (host and pat)))


def _select_config(args: argparse.Namespace, *, load_login: bool) -> dict[str, Any]:
    args._legacy_config_selected = False
    if getattr(args, "connection", None):
        return {}

    if args.config:
        args._legacy_config_selected = True
        return _load_config(args.config)

    if not load_login:
        return {}

    base_url = _coalesce(args.base_url, _env("CORTEX_TRAINING_BASE_URL"))
    host = _coalesce(args.host, _env("CORTEX_TRAINING_HOST", "SNOWFLAKE_HOST"))
    pat = _coalesce(args.pat, _env("CORTEX_TRAINING_PAT", "SNOWFLAKE_PAT"))
    database = _coalesce(
        args.database,
        _env("CORTEX_TRAINING_DATABASE", "SNOWFLAKE_DATABASE"),
    )
    if _has_connection(base_url, host, pat, database):
        return {}

    login_config = _read_login_config_path()
    if login_config is None:
        return {}
    if not Path(login_config).expanduser().is_file():
        return {}
    args.config = login_config
    args._legacy_config_selected = True
    return _load_config(login_config)


def _resolve_args(
    args: argparse.Namespace,
    *,
    load_login: bool = True,
) -> argparse.Namespace:
    config = _select_config(args, load_login=load_login)
    profile_requested = bool(getattr(args, "connection", None))
    args.base_url = _coalesce(
        args.base_url,
        _config_str(config, "base_url"),
        None if profile_requested else _env("CORTEX_TRAINING_BASE_URL"),
    )
    args.host = _coalesce(
        args.host,
        _config_str(config, "host"),
        None
        if profile_requested
        else _env("CORTEX_TRAINING_HOST", "SNOWFLAKE_HOST"),
    )
    args.pat = _coalesce(
        args.pat,
        _config_str(config, "pat"),
        None
        if profile_requested
        else _env("CORTEX_TRAINING_PAT", "SNOWFLAKE_PAT"),
    )
    args.database = _coalesce(
        args.database,
        _config_str(config, "database"),
        _env("CORTEX_TRAINING_DATABASE", "SNOWFLAKE_DATABASE"),
    )
    args.schema = _coalesce(
        args.schema,
        _config_str(config, "schema"),
        _env("CORTEX_TRAINING_SCHEMA", "SNOWFLAKE_SCHEMA"),
    )
    args.endpoint = _coalesce(
        args.endpoint,
        _config_str(config, "endpoint"),
        _env("CORTEX_TRAINING_ENDPOINT"),
        "cortex-training",
    )
    args.poll_interval = _coalesce(
        args.poll_interval,
        _config_float(config, "poll_interval"),
        0.5,
    )
    args.poll_timeout = _coalesce(
        args.poll_timeout,
        _config_float(config, "poll_timeout"),
        1800.0,
    )
    if not args.no_verify_ssl:
        no_verify_ssl = _config_bool(config, "no_verify_ssl")
        verify_ssl = _config_bool(config, "verify_ssl")
        if no_verify_ssl is not None:
            args.no_verify_ssl = no_verify_ssl
        elif verify_ssl is not None:
            args.no_verify_ssl = not verify_ssl
    direct_credentials_present = any((args.base_url, args.host, args.pat))
    if profile_requested and direct_credentials_present:
        raise ValueError(
            "--connection cannot be combined with --base-url, --host, or --pat"
        )
    direct_connection_complete = bool(args.base_url or (args.host and args.pat))
    args.use_connection_profile = profile_requested or (
        not args._legacy_config_selected and not direct_connection_complete
    )
    if not args.use_connection_profile and args.schema is None:
        args.schema = "PUBLIC"
    return args


def _normalize_connection_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.base_url is None or _has_url_scheme(args.base_url):
        return args
    if args.pat and args.host is None:
        args.host = args.base_url
        args.base_url = None
        return args
    raise ValueError(
        "base_url must start with http:// or https://. For Snowflake PAT auth, use host instead of base_url."
    )


def parse_args(
    argv: list[str] | None = None,
    *,
    prog: str = "cortex-training",
    include_tui: bool = False,
) -> argparse.Namespace:
    parser = build_parser(prog=prog, include_tui=include_tui)
    args = parser.parse_args(argv)
    if args.command == "list":
        if args.since is not None and args.until is not None and args.since > args.until:
            parser.error("--since must not be later than --until")
        # --until is sent as the day after, which does not exist for the last date.
        if args.until == date.max:
            parser.error(f"--until must be earlier than {date.max}")
        if args.user is not None and not args.user.strip():
            parser.error("--user must not be empty")
    if args.command == "login":
        args.login_config = args.login_config or args.login_config_option
        return args

    dry_run = args.command == "submit" and args.dry_run
    args = _resolve_args(args, load_login=not dry_run)
    if dry_run:
        return args
    args = _normalize_connection_args(args)
    if args.use_connection_profile:
        return args
    if not args.database:
        parser.error("provide --database or set CORTEX_TRAINING_DATABASE/SNOWFLAKE_DATABASE")
    if args.base_url is None and (args.host is None or args.pat is None):
        parser.error("provide --base-url for local/mock use, or both --host and --pat")
    return args


def build_client(args: argparse.Namespace, cortex_training_client_cls):
    kwargs = {
        "database": args.database,
        "schema": args.schema,
        "endpoint": args.endpoint,
        "poll_interval": args.poll_interval,
        "poll_timeout": args.poll_timeout,
    }
    # The TUI bounds every read (`cortex-training tui` sets these); the CLI
    # commands keep the client's defaults.
    for option in ("max_retries", "request_timeout"):
        value = getattr(args, option, None)
        if value is not None:
            kwargs[option] = value
    if args.base_url:
        return cortex_training_client_cls(base_url=args.base_url, **kwargs)
    if args.use_connection_profile:
        return cortex_training_client_cls.from_connection_name(
            connection_name=args.connection,
            verify_ssl=not args.no_verify_ssl,
            **kwargs,
        )
    return cortex_training_client_cls.from_pat(
        host=_normalize_host(args.host),
        pat=args.pat,
        verify_ssl=not args.no_verify_ssl,
        **kwargs,
    )


def _read_json_object(path: str, stdin: TextIO) -> dict[str, Any]:
    if path == "-":
        raw = stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("job JSON must be an object")
    return parsed


# Duplicated from client.py rather than imported: submit --dry-run must stay
# importable without pulling in the client (and therefore torch).
_LOG_PROBABILITY_JOB_TYPE_ALIASES = frozenset({"log_probability", "log_prob"})
_UNSUPPORTED_LOG_PROBABILITY_MESSAGE = (
    "log_probability sub-jobs are not currently supported"
)


def _normalized_job_type(job_type: Any) -> str:
    if not isinstance(job_type, str):
        return ""
    return job_type.strip().lower().removeprefix("job_type_")


def _is_log_probability_job_type(job_type: Any) -> bool:
    return _normalized_job_type(job_type) in _LOG_PROBABILITY_JOB_TYPE_ALIASES


def _reject_log_probability_sub_job(job_type: Any, *, location: str) -> None:
    if _is_log_probability_job_type(job_type):
        raise ValueError(f"{location}: {_UNSUPPORTED_LOG_PROBABILITY_MESSAGE}")


def _validate_create_job_body(body: dict[str, Any]) -> dict[str, Any]:
    """Check a CreateJob body as a live submit would; return the body to send."""
    from .client import _body_with_checked_comment
    from .client import _validate_raw_sub_job_configs

    sub_job_configs = body.get("sub_job_configs")
    if not isinstance(sub_job_configs, list) or not sub_job_configs:
        raise ValueError("job JSON must contain a non-empty sub_job_configs list")
    # Mirrors client.create_job_from_body; submit --dry-run never builds a client.
    training_sub_jobs = 0
    for index, cfg in enumerate(sub_job_configs):
        if not isinstance(cfg, dict):
            continue
        job_type = cfg.get("job_type")
        _reject_log_probability_sub_job(
            job_type, location=f"sub_job_configs[{index}].job_type"
        )
        if str(job_type or "").strip().lower() == "training":
            training_sub_jobs += 1
            if training_sub_jobs > 1:
                raise ValueError("at most one training sub-job is supported per job")
    _validate_raw_sub_job_configs(sub_job_configs)
    return _body_with_checked_comment(body)


def _print_json(value: Any, stdout: TextIO, *, compact: bool, sort_keys: bool = True) -> None:
    """Write ``value`` to ``stdout`` as one JSON document, pretty unless ``compact``,
    its keys sorted unless ``sort_keys`` is False (the value's own order, as the
    capacity breakdown's).

    Text is written as UTF-8 when ``stdout`` takes it, so a name in any script
    reads as written; a character a terminal could act on (see
    :func:`_printable`) stays a ``\\u`` escape. A stream with another encoding
    (a Windows pipe, say) gets every non-ASCII character as an escape instead,
    so its bytes are ASCII and still the UTF-8 JSON that ``jq`` reads. Either
    way the document decodes to the same value.
    """
    layout: dict[str, Any] = {"separators": (",", ":")} if compact else {"indent": 2}
    utf8 = _takes_utf8(stdout)
    text = json.dumps(value, sort_keys=sort_keys, ensure_ascii=not utf8, **layout)
    if utf8:
        text = _NOT_ASCII_RE.sub(_escape_unsafe_json_char, text)
    stdout.write(text + "\n")


# Everything json.dumps(ensure_ascii=False) can leave unescaped that is worth a look:
# DEL and every non-ASCII character. Only the unsafe ones among them are escaped.
_NOT_ASCII_RE = re.compile(r"[^\x00-\x7e]")


def _escape_unsafe_json_char(match: re.Match) -> str:
    """A JSON ``\\u`` escape for an unsafe character, as a surrogate pair above U+FFFF.

    Any other character is returned as it is.
    """
    char = match.group()
    if not _is_unsafe_char(char):
        return char
    code = ord(char)
    if code <= 0xFFFF:
        return f"\\u{code:04x}"
    code -= 0x10000
    return f"\\u{0xD800 + (code >> 10):04x}\\u{0xDC00 + (code & 0x3FF):04x}"


def _takes_utf8(stream: TextIO) -> bool:
    """Whether ``stream`` writes UTF-8, or keeps text as str (``io.StringIO``)."""
    encoding = getattr(stream, "encoding", None)
    if not isinstance(encoding, str):
        return True
    try:
        return codecs.lookup(encoding).name == "utf-8"
    except LookupError:
        return False


def _request_sent_pat(response: Any) -> bool:
    # Connection-profile requests carry a session token from a login that
    # already accepted the PAT, so a 401 there is not a PAT problem.
    headers = getattr(getattr(response, "request", None), "headers", None) or {}
    return headers.get("X-Snowflake-Authorization-Token-Type") == "PROGRAMMATIC_ACCESS_TOKEN"


def _format_error(exc: BaseException) -> str:
    message = str(exc)
    parts = [message]
    body = ""
    response = getattr(exc, "response", None)
    if response is not None:
        request_id = getattr(response, "headers", {}).get("x-snowflake-request-id")
        if request_id and request_id not in message:
            parts.append(f"snowflake request id: {request_id}")

        body = getattr(response, "text", "") or ""
        body = body.strip() if isinstance(body, str) else ""
        if body:
            shown = body if len(body) <= 4000 else body[:4000] + "...<truncated>"
            parts.append(f"response body: {shown}")
        if getattr(response, "status_code", None) == 401 and _request_sent_pat(response):
            from cortex_training.snowflake_auth import credentials_rejected_hint

            parts.append(
                credentials_rejected_hint("Snowflake rejected your credentials (HTTP 401).")
            )
    hint = _experiment_access_hint(f"{message}\n{body}")
    if hint:
        parts.append(hint)
    return "\n".join(parts)


def _experiment_access_hint(text: str) -> str | None:
    """The ``hint:`` line when ``text``, an error and its body, says the role can't read the Experiment."""
    if _EXPERIMENT_ACCESS_DENIED in text:
        return _EXPERIMENT_ACCESS_HINT
    if _LINKED_EXPERIMENT_NOT_FOUND in text:
        return _EXPERIMENT_GONE_HINT
    if _EXPERIMENT_SQL_DENIED.search(text):
        return _EXPERIMENT_SQL_HINT
    return None


def _is_connector_error(exc: BaseException) -> bool:
    """Whether ``exc`` is the Snowflake Connector's own error: refused SQL, or a failed login.

    download-log and download-metrics read the job's Experiment with SQL. The
    connector is loaded by then, so an error from it means it is in sys.modules.
    """
    errors = sys.modules.get("snowflake.connector.errors")
    error_class = getattr(errors, "Error", None)
    return isinstance(error_class, type) and isinstance(exc, error_class)


def _cmd_submit(
    args: argparse.Namespace,
    client,
    stdout: TextIO,
    stdin: TextIO,
) -> int:
    body = _read_json_object(args.json_file, stdin)
    if args.comment is not None:
        # --comment replaces the JSON's comment. A blank one is left out on the
        # way out, so an empty TEXT removes it, as a recipe's name="" does.
        body = {**body, "comment": args.comment}
    body = _validate_create_job_body(body)
    if args.job_id is not None:
        body = dict(body)
        body["job_id"] = args.job_id

    if args.dry_run:
        _print_json(body, stdout, compact=args.compact)
        return 0

    response = client.create_job_from_body(body)
    if args.wait:
        job_id = response.get("job_id") or body.get("job_id")
        if not job_id:
            raise ValueError("submit --wait requires the create response to include job_id")
        response = client.wait_for_job(str(job_id))
    _print_json(response, stdout, compact=args.compact)
    return 0


def _cmd_list(args: argparse.Namespace, client, stdout: TextIO) -> int:
    if args.all:
        # Under a status filter the service fails a page only after waiting up
        # to 300 s on its status lookup, so such a page gets fewer retries.
        bounds = {} if args.status is None else {"max_retries": _LIST_ALL_STATUS_RETRIES}
        try:
            jobs = list(
                itertools.islice(
                    client.iter_jobs(
                        status=args.status,
                        submitted_by=args.user,
                        created_after=args.since,
                        # The server's bound is exclusive; --until names the last day kept.
                        created_before=(
                            None if args.until is None else args.until + timedelta(days=1)
                        ),
                        # No bigger than what gets printed; the server takes 1 to 1000.
                        page_size=min(args.limit or 100, 1000),
                        **bounds,
                    ),
                    args.limit,
                )
            )
        except Exception as exc:
            if args.user is not None and _is_unknown_submitter(exc):
                raise _unknown_submitter_error(args.user, exc) from exc
            raise
    else:
        keep = _list_filter(args)
        jobs = [job for job in client.list_jobs() if keep(job)]
    _print_json({"jobs": _jobs_latest_last(jobs, args.limit)}, stdout, compact=args.compact)
    return 0


# Retries after the first attempt for each `list --all --status` page; other
# pages keep the client's default.
_LIST_ALL_STATUS_RETRIES = 2

# The account-level list's 400 for a submitted_by that names no active user in
# the account. Without --all, the CLI filters the schema's rows itself, so an
# unknown user just matches no job.
_UNKNOWN_SUBMITTER_MESSAGE = "submitted_by user not found or not authorized"


def _is_unknown_submitter(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    if response is None or getattr(response, "status_code", None) != 400:
        return False
    try:
        text = response.text
    except Exception:
        return False
    return isinstance(text, str) and _UNKNOWN_SUBMITTER_MESSAGE in text


def _unknown_submitter_error(user: str, exc: BaseException) -> Exception:
    """The service's 400 for ``--user``, saying what to change, with its request id."""
    import requests

    from .client import _add_snowflake_request_id_to_error
    from .client import _response_snowflake_request_id

    response = getattr(exc, "response", None)
    error = requests.exceptions.HTTPError(
        _unknown_submitter_message(user),
        response=response,
        request=getattr(exc, "request", None),
    )
    _add_snowflake_request_id_to_error(error, _response_snowflake_request_id(response))
    return error


def _unknown_submitter_message(user: str) -> str:
    import shlex

    message = (
        f"--user {user}: no active user in this account has that name "
        f"(HTTP 400: {_UNKNOWN_SUBMITTER_MESSAGE})"
    )
    if not user.startswith('"') and user != user.upper():
        # SQL identifier rules: unquoted means the uppercase name.
        quoted = shlex.quote('"' + user.replace('"', '""') + '"')
        message += (
            f"; an unquoted name means {user.upper()}, so quote a lowercase or "
            f"mixed-case name to match it exactly: --user {quoted}"
        )
    return message


def _cmd_login(args: argparse.Namespace, stdout: TextIO) -> int:
    config_path = _write_login_config_path(args.login_config)
    _print_json(
        {"config_path": config_path, "logged_in": True},
        stdout,
        compact=args.compact,
    )
    return 0


def _cmd_capacity(args: argparse.Namespace, client, stdout: TextIO) -> int:
    dimensions = tuple(
        _CAPACITY_BREAKDOWN_CHOICES[choice] for choice in args.breakdown or ()
    )

    def lookup(hardware: str) -> dict:
        if dimensions:
            return client.get_capacity_breakdown(
                hardware=hardware, dimensions=dimensions
            )
        return client.get_capacity(hardware=hardware)

    if args.hardware is not None:
        capacity = lookup(args.hardware)
    else:
        capacity = _capacity_by_hardware(lookup)
    _print_json(_breakdown_last(capacity), stdout, compact=args.compact, sort_keys=False)
    return 0


def _breakdown_last(value: Any) -> Any:
    """``value`` with keys sorted, except that ``breakdown`` follows the totals."""
    if isinstance(value, dict):
        keys = sorted(value, key=lambda key: (key == "breakdown", key))
        return {key: _breakdown_last(value[key]) for key in keys}
    if isinstance(value, list):
        return [_breakdown_last(item) for item in value]
    return value


def _json_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _cmd_fwd_bwd(
    args: argparse.Namespace,
    client,
    stdout: TextIO,
    stdin: TextIO,
) -> int:
    if not args.job:
        raise ValueError("provide --job JOB_ID for fwd-bwd")

    spec = _read_json_object(args.json_file, stdin)
    poll = _json_bool(spec.get("poll", True), "fwd-bwd poll")
    build_payload = _load_forward_backward_payload_builder()
    payload = build_payload(spec)

    request_id = client.forward_backward(args.job, payload)
    response = {
        "job_id": args.job,
        "payload_size_bytes": len(payload),
        "request_id": request_id,
    }
    if poll:
        response["result"] = client.poll_request(args.job, request_id)

    _print_json(response, stdout, compact=args.compact)
    return 0


def _cmd_step(args: argparse.Namespace, client, stdout: TextIO) -> int:
    if not args.job:
        raise ValueError("provide --job-id JOB_ID for step")

    request_id = client.step(args.job, learning_rate=args.lr)
    response = {
        "job_id": args.job,
        "learning_rate": args.lr,
        "request_id": request_id,
        "result": client.poll_request(args.job, request_id),
    }
    _print_json(response, stdout, compact=args.compact)
    return 0


def _cmd_load(args: argparse.Namespace, client, stdout: TextIO) -> int:
    if not args.job:
        raise ValueError("provide --job-id JOB_ID for load")

    request_id = client.load(
        args.job,
        checkpoint_id=args.checkpoint_id,
        source_job_id=args.source_job_id,
        target_sub_job_id=args.target_sub_job_id,
    )
    response = {
        "checkpoint_id": args.checkpoint_id,
        "job_id": args.job,
        "request_id": request_id,
    }
    if args.source_job_id is not None:
        response["source_job_id"] = args.source_job_id
    if args.target_sub_job_id is not None:
        response["target_sub_job_id"] = args.target_sub_job_id
    if not args.no_poll:
        response["result"] = client.poll_request(args.job, request_id)

    _print_json(response, stdout, compact=args.compact)
    return 0


def _cmd_generate(
    args: argparse.Namespace,
    client,
    stdout: TextIO,
    stdin: TextIO,
) -> int:
    if not args.job:
        raise ValueError("provide --job-id JOB_ID for generate")

    spec = _read_json_object(args.json_file, stdin)
    payload = spec.get("payload", spec)
    if not isinstance(payload, dict):
        raise ValueError("generate payload must be an object")

    poll = _json_bool(spec.get("poll", True), "generate poll")
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("generate JSON must contain a non-empty prompts list")

    # A flat list of integers is one pre-tokenized prompt, not a batch of
    # single-token prompts, and the client encodes it that way. ``bool`` is an
    # ``int`` subclass, so exclude it rather than read [true, false] as tokens.
    single_tokenized = isinstance(prompts[0], int) and not isinstance(prompts[0], bool)
    prompt_count = 1 if single_tokenized else len(prompts)

    sampling_params = payload.get("sampling_params")
    if sampling_params is not None:
        if isinstance(sampling_params, list):
            if len(sampling_params) != prompt_count:
                raise ValueError("generate sampling_params list length must match prompts length")
            if any(item is not None and not isinstance(item, dict) for item in sampling_params):
                raise ValueError("generate sampling_params list items must be objects or null")
        elif not isinstance(sampling_params, dict):
            raise ValueError("generate sampling_params must be an object or list")

    strict = payload.get("strict")
    if strict is not None:
        strict = _json_bool(strict, "generate strict")

    request_id = client.generate(
        args.job,
        prompts=prompts,
        sampling_params=sampling_params,
        routing_key=payload.get("routing_key"),
        strict=strict,
    )
    response = {
        "job_id": args.job,
        "prompt_count": prompt_count,
        "request_id": request_id,
    }
    if poll:
        response["result"] = client.poll_request(args.job, request_id)

    _print_json(response, stdout, compact=args.compact)
    return 0


def _cmd_weight_sync(args: argparse.Namespace, client, stdout: TextIO) -> int:
    if not args.job:
        raise ValueError("provide --job-id JOB_ID for weight-sync")

    source_sub_job_id = args.source_sub_job_id or f"{args.job}:training:0"
    target_sub_job_ids = args.target_sub_job_ids or [f"{args.job}:sampling:0"]
    request_id = client.weight_sync(
        args.job,
        source_sub_job_id=source_sub_job_id,
        target_sub_job_ids=target_sub_job_ids,
        weight_format=args.weight_format,
        sub_job_id=args.operation_sub_job_id,
        sub_job_type=args.operation_sub_job_type,
    )
    response = {
        "job_id": args.job,
        "request_id": request_id,
        "source_sub_job_id": source_sub_job_id,
        "target_sub_job_ids": target_sub_job_ids,
    }
    if args.weight_format is not None:
        response["weight_format"] = args.weight_format
    if not args.no_poll:
        response["result"] = client.poll_request(args.job, request_id)

    _print_json(response, stdout, compact=args.compact)
    return 0


def _artifact_progress_writer(stderr: TextIO) -> Callable[[str], None]:
    """Draw one updating terminal line. A redirected stderr gets the final line."""
    try:
        tty = stderr.isatty()
    except (OSError, ValueError):
        tty = False
    latest: str | None = None

    def write(message: str) -> None:
        nonlocal latest
        try:
            if message == "\n":
                status, latest = latest, None
                if status is None:
                    return
                stderr.write("\n" if tty else status + "\n")
            elif message.endswith("\n"):
                stderr.write(message)
            else:
                latest = message
                if not tty:
                    return
                stderr.write("\r" + message)
            stderr.flush()
        except OSError:
            return

    return write


def _cmd_download_log(
    args: argparse.Namespace, client, stdout: TextIO, stderr: TextIO
) -> int:
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else Path.cwd()
    if args.log_type == "stdout":
        logs = client.download_stdout_logs(
            args.job_id,
            out_dir,
            resume=args.resume,
            progress=_artifact_progress_writer(stderr),
        )
        _print_json(
            {"job_id": args.job_id, "logs": logs},
            stdout,
            compact=args.compact,
        )
        return 0

    logs = client.fetch_execution_logs(args.job_id)
    saved = []
    for log in logs:
        file_path = out_dir / (log["sub_job_id"] or "unknown") / log["filename"]
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(log["content"], encoding="utf-8")
        saved.append(
            {
                "sub_job_id": log["sub_job_id"],
                "filename": log["filename"],
                "artifact_uri": log["artifact_uri"],
                "saved_path": str(file_path),
            }
        )
    _print_json({"job_id": args.job_id, "logs": saved}, stdout, compact=args.compact)
    return 0


def _cmd_download_metrics(
    args: argparse.Namespace, client, stdout: TextIO, stderr: TextIO
) -> int:
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else Path.cwd()
    metrics = client.download_metrics(
        args.job_id,
        out_dir,
        resume=args.resume,
        progress=_artifact_progress_writer(stderr),
    )
    _print_json(
        {"job_id": args.job_id, "metrics": metrics},
        stdout,
        compact=args.compact,
    )
    return 0


# Duplicated from client.py rather than imported, like the log-probability
# aliases above: formatting an error must not import the client (and torch).
# The service answers 500 when it can't resolve the job's Experiment for the
# caller's role. The checkpoint routes word it this way...
_EXPERIMENT_ACCESS_DENIED = "Linked experiment is not found or not authorized"
# ... and the experiment-run lookup that saved logs and GPU metrics start with
# says "Linked experiment is not found for job <job id>", which also covers an
# Experiment that no longer exists. Both start with this.
_LINKED_EXPERIMENT_NOT_FOUND = "Linked experiment is not found"
_EXPERIMENT_ACCESS_HINT = (
    "hint: this role can't read the job's Experiment, where its saved logs, GPU "
    "metrics and checkpoints live. Ask for access to the Experiment, or use the "
    "role that created the job."
)
_EXPERIMENT_GONE_HINT = (
    "hint: this role can't read the job's Experiment, or it no longer exists; the "
    "job's saved logs, GPU metrics and checkpoints live there. Ask for access to "
    "the Experiment, or use the role that created the job."
)
# Snowflake refusing the SQL LIST or GET that download-log and download-metrics
# run on the Experiment: the role reaches the service but has no privileges on
# the Experiment (a MONITOR-only role, say). Only these words: an error about a
# schema or a warehouse that merely mentions an experiment is not this.
_EXPERIMENT_SQL_DENIED = re.compile(
    r"insufficient privileges to operate on experiment\b"
    r"|\bexperiment(?:\s+run)?\s+(?:'[^']*'|\"[^\"]*\"|[^\s'\"]+)\s+does not exist or not authorized",
    re.IGNORECASE,
)
_EXPERIMENT_SQL_HINT = (
    "hint: this role has no privileges on the job's Experiment, where its saved logs "
    "and GPU metrics are read with SQL. Ask for access to the Experiment, or use the "
    "role that created the job; the checkpoints command doesn't need it."
)


@contextlib.contextmanager
def _progress_logging(stream: TextIO) -> Iterator[None]:
    """Show the package's warnings on ``stream`` (stderr) for one command, through the printable pass.

    stdout carries the command's JSON. The handler sits on the package logger,
    so the client's warnings, which Python would otherwise write to stderr as
    they are, also go through the printable pass: they can quote the service,
    a job's reason for example. When logging is already configured, the
    embedding program's own handlers decide instead.
    """
    package = logging.getLogger(__name__.rpartition(".")[0])
    if logging.getLogger().handlers or logger.handlers or package.handlers:
        yield
        return
    handler = logging.StreamHandler(stream)
    handler.setFormatter(_PrintableFormatter("%(message)s"))
    package.addHandler(handler)
    try:
        yield
    finally:
        package.removeHandler(handler)


def _is_unsupported_hardware(exc: BaseException) -> bool:
    """Whether a capacity lookup failed only because the account lacks that GPU type."""
    response = getattr(exc, "response", None)
    if response is None:
        return False
    try:
        body = response.json()
    except Exception:
        return False
    if not isinstance(body, dict):
        return False
    code = str(body.get("code") or body.get("error_code") or "")
    message = str(body.get("message") or "")
    return code == _UNSUPPORTED_HARDWARE_CODE and "unsupported hardware" in message


def _capacity_by_hardware(lookup: Callable[[str], dict]) -> dict[str, Any]:
    """Capacity for every GPU type the account supports, plus the ones it does not."""
    supported: dict[str, Any] = {}
    unsupported: list[str] = []
    first_unsupported: BaseException | None = None
    for hardware in _hardware_choices():
        try:
            supported[hardware] = lookup(hardware)
        except Exception as exc:
            if not _is_unsupported_hardware(exc):
                raise
            unsupported.append(hardware)
            first_unsupported = first_unsupported or exc
    if not supported and first_unsupported is not None:
        raise first_unsupported
    return {"capacity_by_hardware": supported, "unsupported_hardware": unsupported}


def _run(
    args: argparse.Namespace,
    client_factory: Callable[[argparse.Namespace], Any],
    stdout: TextIO,
    stdin: TextIO,
    stderr: TextIO,
) -> int:
    if args.command == "login":
        return _cmd_login(args, stdout)

    if args.command == "submit" and args.dry_run:
        return _cmd_submit(args, None, stdout, stdin)

    if args.command == "download-log" and args.resume and args.log_type != "stdout":
        raise ValueError("--resume applies to --log-type stdout, not execution logs")

    client = client_factory(args)
    if args.command == "submit":
        return _cmd_submit(args, client, stdout, stdin)
    if args.command == "get":
        _print_json(client.get_job(args.job_id), stdout, compact=args.compact)
        return 0
    if args.command == "checkpoints":
        _print_json(
            {"checkpoints": client.list_checkpoints(args.job_id)},
            stdout,
            compact=args.compact,
        )
        return 0
    if args.command == "list":
        return _cmd_list(args, client, stdout)
    if args.command == "capacity":
        return _cmd_capacity(args, client, stdout)
    if args.command == "cancel":
        client.cancel_job(args.job_id)
        _print_json(
            {"cancelled": True, "job_id": args.job_id},
            stdout,
            compact=args.compact,
        )
        return 0
    if args.command == "wait":
        _print_json(client.wait_for_job(args.job_id), stdout, compact=args.compact)
        return 0
    if args.command == "fwd-bwd":
        return _cmd_fwd_bwd(args, client, stdout, stdin)
    if args.command == "step":
        return _cmd_step(args, client, stdout)
    if args.command == "load":
        return _cmd_load(args, client, stdout)
    if args.command == "generate":
        return _cmd_generate(args, client, stdout, stdin)
    if args.command == "weight-sync":
        return _cmd_weight_sync(args, client, stdout)
    if args.command == "download-log":
        return _cmd_download_log(args, client, stdout, stderr)
    if args.command == "download-metrics":
        return _cmd_download_metrics(args, client, stdout, stderr)
    raise ValueError(f"unknown command: {args.command}")


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "cortex-training",
    include_tui: bool = False,
    client_factory: Callable[[argparse.Namespace], Any] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    stdin: TextIO | None = None,
) -> int:
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    stdin = stdin or sys.stdin

    try:
        args = parse_args(argv, prog=prog, include_tui=include_tui)
        dry_run = args.command == "submit" and args.dry_run
        no_client = dry_run or args.command == "login"
        if client_factory is None and not no_client:
            cortex_training_client_cls = _load_cortex_training_client_class()

            def default_client_factory(parsed_args: argparse.Namespace) -> Any:
                return build_client(parsed_args, cortex_training_client_cls)

            client_factory = default_client_factory
        elif client_factory is None:

            def empty_client_factory(_parsed_args: argparse.Namespace) -> None:
                return None

            client_factory = empty_client_factory
        with _progress_logging(stderr):
            return _run(args, client_factory, stdout, stdin, stderr)
    except (OSError, ValueError, RuntimeError) as exc:
        return _report_error(exc, stderr)
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        print(
            f"error: missing Python dependency '{missing}'. "
            "Install the missing package, or install this repo first, "
            "for example: python3 -m pip install -e .",
            file=stderr,
        )
        return 1
    except Exception as exc:
        # Snowflake refusing the SQL on a job's Experiment, or a failed login:
        # one error line (and hint), not a traceback.
        if not _is_connector_error(exc):
            raise
        return _report_error(exc, stderr)


def _report_error(exc: BaseException, stderr: TextIO) -> int:
    """Write the ``error:`` line (and any hint) for a failed command; return its exit status."""
    # The message can quote the service's answer: keep it safe for a terminal.
    stderr.write(f"error: {_printable(_format_error(exc))}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
