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
import json
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Callable
from typing import TextIO

from cortex_training._connection import ConnectionSettings
from cortex_training._connection import build_client as _build_client_from_settings
from cortex_training._connection import load_config as _load_config
from cortex_training._connection import resolve_connection
from cortex_training._connection import write_login_state


_LOGIN_DESCRIPTION = """\
Set up and check the connection that cortex-training commands, recipes, and
connect() use.

What it does:
  1. Shows the connection already in use: where it comes from, host, user,
     database, and where the token is stored (never the token). A saved login
     is checked first and removed if it no longer works.
  2. If nothing works, prompts for host, user, PAT (hidden), database, schema,
     and profile name. Found values are offered as defaults.
  3. Writes the profile to connections.toml (other profiles kept, mode 600)
     and remembers it for later commands.
  4. Checks it with a capacity request. A missing database is created with
     CREATE DATABASE IF NOT EXISTS <database>.
  Nothing is changed if your current setup works, unless you ask for a new
  profile.

Files:
  ~/.snowflake/connections.toml (or $SNOWFLAKE_HOME/connections.toml)
  ~/.config/cortex-training/login.json: the saved login, a pointer to a
  profile or config file; it never stores credentials."""

_LOGIN_EPILOG = """\
examples:
  cortex-training login                       # interactive setup or check
  CORTEX_TRAINING_PAT=... cortex-training login \\
      --host ORG-ACCOUNT.snowflakecomputing.com --user USER
  cortex-training login --reset               # start over
  cortex-training login config.json           # legacy: remember a JSON config

Docs: https://github.com/snowflakedb/cortex-training/blob/main/docs/reference/cli.md#login"""


def _load_cortex_training_client_class():
    from .client import CortexTrainingClient

    return CortexTrainingClient


def _load_forward_backward_payload_builder():
    from .client import build_forward_backward_payload

    return build_forward_backward_payload


def _hardware_choices() -> list[str]:
    from .client import Hardware

    return [member.value for member in Hardware]


def _created_epoch(raw: Any) -> float | None:
    if raw is None:
        return None
    created_at = str(raw).strip()
    if not created_at:
        return None
    if created_at.endswith("Z"):
        created_at = created_at[:-1] + "+00:00"
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return created.timestamp()


def _jobs_latest_last(jobs: list[Any]) -> list[Any]:
    """Render jobs oldest-first so the newest entries land at the terminal."""
    dated_jobs = [
        (
            _created_epoch(job.get("created_at")) if isinstance(job, dict) else None,
            index,
            job,
        )
        for index, job in enumerate(jobs)
    ]
    if any(created is not None for created, _, _ in dated_jobs):
        return [
            job
            for _, _, job in sorted(
                dated_jobs,
                key=lambda item: (
                    item[0] is None,
                    item[0] if item[0] is not None else float("inf"),
                    item[1],
                ),
            )
        ]
    return list(reversed(jobs))


def build_parser(
    *,
    prog: str = "cortex-training",
    include_tui: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Manage Cortex Training jobs through the Cortex Training REST endpoint.",
    )
    # --config and --connection read their env vars during resolution, not as
    # argparse defaults, so an explicit flag can beat a leftover env var.
    parser.add_argument(
        "--config",
        help=(
            "Path to a reusable Cortex Training CLI config JSON file. "
            "Defaults to CORTEX_TRAINING_CONFIG."
        ),
    )
    parser.add_argument(
        "--connection",
        "-c",
        help=(
            "Snowflake connection profile name. Defaults to "
            "CORTEX_TRAINING_CONNECTION. When no legacy config or direct "
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

    list_jobs = subparsers.add_parser("list", help="List Cortex Training jobs.")
    list_jobs.add_argument("--status", help="Optional status filter.")

    capacity = subparsers.add_parser(
        "capacity",
        help="Show reserved GPU capacity and current usage for the caller account.",
    )
    capacity.add_argument(
        "--hardware",
        choices=_hardware_choices(),
        help=(
            "Show only this GPU hardware. "
            "Omit to show capacity for every hardware type."
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

    login = subparsers.add_parser(
        "login",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage=(
            "%(prog)s [-h] [config | --config config]\n"
            "       %(prog)s [--host HOST] [--user USER] [--database DATABASE] "
            "[--schema SCHEMA]\n"
            "                             [--connection NAME] [--pat-stdin] "
            "[--force] [--reset]"
        ),
        help="Set up and check the connection that commands, recipes, and connect() use.",
        description=_LOGIN_DESCRIPTION,
        epilog=_LOGIN_EPILOG,
    )
    login_config = login.add_mutually_exclusive_group()
    login_config.add_argument(
        "login_config",
        nargs="?",
        metavar="config",
        help="Remember this JSON config instead (legacy form).",
    )
    login_config.add_argument(
        "--config",
        dest="login_config_option",
        metavar="config",
        help="Alternative to the positional config path.",
    )
    login.add_argument(
        "--host",
        dest="login_host",
        metavar="HOST",
        help="Account host, e.g. ORG-ACCOUNT.snowflakecomputing.com.",
    )
    login.add_argument("--user", dest="login_user", metavar="USER", help="Snowflake login name.")
    login.add_argument(
        "--database",
        dest="login_database",
        metavar="DATABASE",
        help="Default: CORTEX_TRAINING_DB.",
    )
    login.add_argument("--schema", dest="login_schema", metavar="SCHEMA", help="Default: PUBLIC.")
    login.add_argument(
        "--connection",
        dest="login_connection",
        metavar="NAME",
        help="Profile to create. Default: cortex-training.",
    )
    login.add_argument(
        "--pat-stdin",
        dest="login_pat_stdin",
        action="store_true",
        help=(
            "Read the PAT from stdin. The PAT is never accepted as a flag; "
            "otherwise it comes from CORTEX_TRAINING_PAT or a hidden prompt."
        ),
    )
    login.add_argument(
        "--force",
        dest="login_force",
        action="store_true",
        help=(
            "Replace an existing profile without asking. With --reset, also "
            "remove the profile the saved login pointed to."
        ),
    )
    login.add_argument(
        "--reset",
        dest="login_reset",
        action="store_true",
        help=(
            "Start over: remove the saved login, ask before removing the profile "
            "it pointed to, then set up a new connection. Other profiles are "
            "never touched."
        ),
    )

    if include_tui:
        subparsers.add_parser("tui", help="Open the read-only Cortex Training log TUI.")

    return parser


def _write_login_config_path(config_path: str) -> str:
    path = Path(config_path).expanduser()
    _load_config(str(path))
    saved_path = str(path.resolve())
    write_login_state(config_path=saved_path)
    return saved_path


def _resolve_args(
    args: argparse.Namespace,
    *,
    load_login: bool = True,
) -> argparse.Namespace:
    settings = resolve_connection(
        connection_name=args.connection,
        config_path=args.config,
        base_url=args.base_url,
        host=args.host,
        pat=args.pat,
        database=args.database,
        schema=args.schema,
        endpoint=args.endpoint,
        poll_interval=args.poll_interval,
        poll_timeout=args.poll_timeout,
        verify_ssl=False if args.no_verify_ssl else None,
        load_login=load_login,
    )
    args.connection = settings.connection_name
    args.config = settings.config_path
    args.base_url = settings.base_url
    args.host = settings.host
    args.pat = settings.pat
    args.database = settings.database
    args.schema = settings.schema
    args.endpoint = settings.endpoint
    args.poll_interval = settings.poll_interval
    args.poll_timeout = settings.poll_timeout
    args.no_verify_ssl = not settings.verify_ssl
    args.use_connection_profile = settings.use_connection_profile
    args.connection_source = settings.source
    return args


def _settings_from_args(args: argparse.Namespace) -> ConnectionSettings:
    return ConnectionSettings(
        source=getattr(args, "connection_source", "command-line arguments"),
        use_connection_profile=args.use_connection_profile,
        connection_name=args.connection,
        config_path=args.config,
        base_url=args.base_url,
        host=args.host,
        pat=args.pat,
        database=args.database,
        schema=args.schema,
        endpoint=args.endpoint,
        poll_interval=args.poll_interval,
        poll_timeout=args.poll_timeout,
        verify_ssl=not args.no_verify_ssl,
    )


def parse_args(
    argv: list[str] | None = None,
    *,
    prog: str = "cortex-training",
    include_tui: bool = False,
) -> argparse.Namespace:
    parser = build_parser(prog=prog, include_tui=include_tui)
    args = parser.parse_args(argv)
    if args.command == "login":
        args.login_config = args.login_config or args.login_config_option
        interactive_flags = (
            args.login_host,
            args.login_user,
            args.login_database,
            args.login_schema,
            args.login_connection,
            args.login_pat_stdin,
            args.login_force,
            args.login_reset,
        )
        if args.login_config and any(interactive_flags):
            parser.error("a login config path cannot be combined with profile setup flags")
        return args

    dry_run = args.command == "submit" and args.dry_run
    args = _resolve_args(args, load_login=not dry_run)
    if dry_run:
        return args
    if args.use_connection_profile:
        return args
    if not args.database:
        parser.error("provide --database or set CORTEX_TRAINING_DATABASE/SNOWFLAKE_DATABASE")
    if args.base_url is None and (args.host is None or args.pat is None):
        parser.error("provide --base-url for local/mock use, or both --host and --pat")
    return args


def build_client(args: argparse.Namespace, cortex_training_client_cls):
    return _build_client_from_settings(_settings_from_args(args), cortex_training_client_cls)


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


def _validate_create_job_body(body: dict[str, Any]) -> None:
    sub_job_configs = body.get("sub_job_configs")
    if not isinstance(sub_job_configs, list) or not sub_job_configs:
        raise ValueError("job JSON must contain a non-empty sub_job_configs list")
    # Mirrors client.create_job_from_body; submit --dry-run never builds a client.
    training_sub_jobs = sum(
        1
        for cfg in sub_job_configs
        if isinstance(cfg, dict) and str(cfg.get("job_type") or "").strip().lower() == "training"
    )
    if training_sub_jobs > 1:
        raise ValueError("at most one training sub-job is supported per job")


def _print_json(value: Any, stdout: TextIO, *, compact: bool) -> None:
    if compact:
        json.dump(value, stdout, separators=(",", ":"), sort_keys=True)
    else:
        json.dump(value, stdout, indent=2, sort_keys=True)
    stdout.write("\n")


def _request_sent_pat(response: Any) -> bool:
    # Connection-profile requests carry a session token from a login that
    # already accepted the PAT, so a 401 there is not a PAT problem.
    headers = getattr(getattr(response, "request", None), "headers", None) or {}
    return headers.get("X-Snowflake-Authorization-Token-Type") == "PROGRAMMATIC_ACCESS_TOKEN"


def _format_error(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)

    parts = [str(exc)]
    request_id = getattr(response, "headers", {}).get("x-snowflake-request-id")
    if request_id:
        parts.append(f"snowflake request id: {request_id}")

    body = (getattr(response, "text", "") or "").strip()
    if body:
        if len(body) > 4000:
            body = body[:4000] + "...<truncated>"
        parts.append(f"response body: {body}")
    if getattr(response, "status_code", None) == 401 and _request_sent_pat(response):
        from cortex_training.snowflake_auth import credentials_rejected_hint

        parts.append(
            credentials_rejected_hint("Snowflake rejected your credentials (HTTP 401).")
        )
    return "\n".join(parts)


def _cmd_submit(
    args: argparse.Namespace,
    client,
    stdout: TextIO,
    stdin: TextIO,
) -> int:
    body = _read_json_object(args.json_file, stdin)
    _validate_create_job_body(body)
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


def _cmd_login_interactive(
    args: argparse.Namespace,
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    from cortex_training._login import LoginOptions
    from cortex_training._login import run_login

    options = LoginOptions(
        host=args.login_host,
        user=args.login_user,
        database=args.login_database,
        schema=args.login_schema,
        connection=args.login_connection,
        pat_stdin=args.login_pat_stdin,
        force=args.login_force,
        reset=args.login_reset,
    )
    return run_login(
        options,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        client_cls=_load_cortex_training_client_class(),
        format_error=_format_error,
        print_json=lambda value, out: _print_json(value, out, compact=args.compact),
    )


def _cmd_login(args: argparse.Namespace, stdout: TextIO) -> int:
    config_path = _write_login_config_path(args.login_config)
    _print_json(
        {"config_path": config_path, "logged_in": True},
        stdout,
        compact=args.compact,
    )
    return 0


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

    sampling_params = payload.get("sampling_params")
    if sampling_params is not None:
        if isinstance(sampling_params, list):
            if len(sampling_params) != len(prompts):
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
        "prompt_count": len(prompts),
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


def _cmd_download_log(args: argparse.Namespace, client, stdout: TextIO) -> int:
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else Path.cwd()
    if args.log_type == "stdout":
        logs = client.download_stdout_logs(args.job_id, out_dir)
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
    args: argparse.Namespace, client, stdout: TextIO
) -> int:
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else Path.cwd()
    metrics = client.download_metrics(args.job_id, out_dir)
    _print_json(
        {"job_id": args.job_id, "metrics": metrics},
        stdout,
        compact=args.compact,
    )
    return 0


def _run(
    args: argparse.Namespace,
    client_factory: Callable[[argparse.Namespace], Any],
    stdout: TextIO,
    stdin: TextIO,
    stderr: TextIO | None = None,
) -> int:
    if args.command == "login":
        if args.login_config:
            return _cmd_login(args, stdout)
        return _cmd_login_interactive(args, stdin, stdout, stderr or sys.stderr)

    if args.command == "submit" and args.dry_run:
        return _cmd_submit(args, None, stdout, stdin)

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
        jobs = client.list_jobs(status=args.status)
        _print_json({"jobs": _jobs_latest_last(jobs)}, stdout, compact=args.compact)
        return 0
    if args.command == "capacity":
        if args.hardware is not None:
            capacity = client.get_capacity(hardware=args.hardware)
        else:
            capacity = {
                "capacity_by_hardware": {
                    hardware: client.get_capacity(hardware=hardware)
                    for hardware in _hardware_choices()
                }
            }
        _print_json(capacity, stdout, compact=args.compact)
        return 0
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
        return _cmd_download_log(args, client, stdout)
    if args.command == "download-metrics":
        return _cmd_download_metrics(args, client, stdout)
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
        return _run(args, client_factory, stdout, stdin, stderr)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {_format_error(exc)}", file=stderr)
        return 1
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        print(
            f"error: missing Python dependency '{missing}'. "
            "Install the missing package, or install this repo first, "
            "for example: python3 -m pip install -e .",
            file=stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
