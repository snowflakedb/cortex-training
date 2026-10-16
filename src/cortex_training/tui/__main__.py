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

"""Resolve Cortex Training connection arguments and launch the read-only TUI.

Connection handling is delegated to ``cortex_training._cli`` so the TUI shares
the SDK's auth surface: legacy config/login state wins, followed by direct
flags, then a named or configured-default Snowflake connection profile.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

# Seconds between log polls per source when neither --poll-interval nor the
# config file's poll_interval sets them (the CLI's own fallback, 0.5, is for
# its result polling).
_POLL_INTERVAL_DEFAULT = 1.0
# --debug only: makes the synthetic jobs' account-wide list answer as the service does
# where it is unavailable (not_deployed, disabled, forbidden, rest_disabled), so
# the job list's fallback to the configured schema can be previewed, or fail as
# when the control plane is down (unavailable, a 503), for the list's error.
DEBUG_ACCOUNT_LIST_ENV = "CORTEX_TRAINING_TUI_DEBUG_ACCOUNT_LIST"
_CONNECT_HINT = ("to connect, run 'cortex-training login config.json', set CORTEX_TRAINING_CONFIG, pass "
                 "--config config.json, --connection NAME, --base-url (local/mock), or --host + --pat; "
                 "or try the TUI without a connection: --debug")
# Retries after a failed read, and the (connect, read) timeout in seconds, for
# every call the TUI makes (the job list's pages use the same).
READ_MAX_RETRIES = 2
READ_TIMEOUT = (10.0, 45.0)
# The oldest Textual the TUI is tested on (pyproject.toml requires it, below 9).
TEXTUAL_FLOOR = (8, 2, 8)
_TEXTUAL_REQUIREMENT = "textual>=8.2.8,<9"


def _version_tuple(version: str) -> tuple:
    """The leading release numbers of a version string: ``8.2.8.dev1`` is (8, 2, 8)."""
    numbers = []
    for part in version.split(".")[:3]:
        match = re.match(r"\d+", part)
        if match is None:
            break
        numbers.append(int(match.group()))
    return tuple(numbers)


def textual_problem() -> str | None:
    """Why the installed Textual can't run the TUI (missing, or older than TEXTUAL_FLOOR), else None.

    An older Textual can import and then misbehave, so the version is checked
    before the TUI is imported.
    """
    from importlib import metadata

    try:
        installed = metadata.version("textual")
    except metadata.PackageNotFoundError:
        return f"the TUI needs the textual package: pip install '{_TEXTUAL_REQUIREMENT}'"
    if _version_tuple(installed) < TEXTUAL_FLOOR:
        floor = ".".join(map(str, TEXTUAL_FLOOR))
        return (f"the TUI needs textual {floor} or later, and {installed} is installed: "
                f"pip install --upgrade '{_TEXTUAL_REQUIREMENT}'")
    return None


def _build_arg_parser(*, prog: str = "cortex-training tui") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Read-only TUI for Cortex Training jobs: logs and GPU usage.",
    )
    p.add_argument(
        "job_id", nargs="?",
        help="Job (session) id whose details to open directly, on the Logs tab. If omitted, the TUI shows the job list.",
    )
    p.add_argument(
        "--sub-job-id",
        help=("Open the job details scoped to this sub-job (JOB_ID:TYPE:INDEX, for example JOB_ID:training:0); "
              "JOB_ID may then be omitted."),
    )
    p.add_argument(
        "--config",
        dest="config",
        default=os.environ.get("CORTEX_TRAINING_CONFIG"),
        help=(
            "Path to a reusable Cortex Training config or credential JSON file "
            "(same format as the cortex-training CLI config)."
        ),
    )
    p.add_argument(
        "--connection",
        "-c",
        default=os.environ.get("CORTEX_TRAINING_CONNECTION"),
        help="Snowflake connection profile name.",
    )
    p.add_argument("--base-url", help="Direct base URL for a local or otherwise compatible server.")
    p.add_argument("--host", help="Snowflake account host for PAT auth.")
    p.add_argument("--pat", help="Programmatic access token.")
    p.add_argument("--database", help="Database containing the endpoint.")
    # Left as None so config-file / env values can fill them; the CLI resolver
    # applies the PUBLIC / cortex-training defaults when nothing is set.
    p.add_argument("--schema", default=None, help="Schema containing the endpoint. Defaults to PUBLIC.")
    p.add_argument("--endpoint", default=None, help="REST endpoint name. Defaults to cortex-training.")
    p.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help=(
            "Minimum seconds between log polls per source (request-rate floor). Defaults to the "
            f"config file's poll_interval when it sets one, else {_POLL_INTERVAL_DEFAULT}."
        ),
    )
    p.add_argument(
        "--poll-timeout", type=float, default=1800.0,
        help=("Seconds the client waits for an asynchronous request's result (default 1800), as in the CLI; "
              "the TUI sends none, so it has no effect here."),
    )
    p.add_argument("--no-verify-ssl", action="store_true",
                   help="Disable SSL certificate verification for authenticated requests.")
    p.add_argument(
        "--schema-only",
        action="store_true",
        help=(
            "List only the configured database and schema's jobs, and never the account-wide list. "
            "Without it the job list starts on the configured schema too, and 'a' switches it to "
            "every database and schema this role can see, where the service has the account-wide list."
        ),
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Run the TUI on synthetic jobs and telemetry, with no connection (to try it or debug it).",
    )
    return p


def read_only_client(client):
    """``client`` with database creation turned off, for the read-only TUI.

    By default a request answered "database or schema not found" makes the
    client run ``CREATE DATABASE`` (and ``CREATE SCHEMA``) and retry. The TUI
    only reads, so, like a client from ``CortexTrainingClient.for_job``, it
    shows that error instead (``CortexTrainingClient.auto_create_database``).
    """
    client.auto_create_database = False
    return client


def run(argv=None, *, prog: str = "cortex-training tui") -> int:
    parser = _build_arg_parser(prog=prog)
    args = parser.parse_args(argv)
    if args.sub_job_id and not args.job_id:
        # A sub-job id is JOB_ID:TYPE:INDEX, so it also names its job.
        parts = args.sub_job_id.rsplit(":", 2)
        if len(parts) != 3 or not all(parts):
            parser.error("--sub-job-id expects JOB_ID:TYPE:INDEX, for example JOB_ID:training:0")
        args.job_id = parts[0]

    problem = textual_problem()
    if problem:
        parser.exit(2, f"{parser.prog}: error: {problem}\n")
    try:
        from cortex_training.tui.app import CortexTrainingLogTUI
    except ImportError as exc:
        parser.exit(2, f"{parser.prog}: error: the TUI couldn't load ({exc}); reinstall cortex-training, "
                       f"or pip install --upgrade '{_TEXTUAL_REQUIREMENT}'\n")

    if args.debug:
        if args.poll_interval is None:
            args.poll_interval = _POLL_INTERVAL_DEFAULT
        try:
            account_list = _demo_account_list()
        except ValueError as exc:
            parser.error(str(exc))
        return _run_demo(args, CortexTrainingLogTUI, account_list=account_list)

    import cortex_training._cli as cli

    # Reuse the CLI's resolution: --config / login state / env / defaults. The
    # config file's poll_interval applies unless --poll-interval is given; with
    # neither, the TUI's default does, not the CLI's.
    explicit_poll = args.poll_interval is not None
    args = cli._resolve_args(args)
    if not explicit_poll:
        configured = _configured_poll_interval(cli, args)
        args.poll_interval = _POLL_INTERVAL_DEFAULT if configured is None else configured
    try:
        args = cli._normalize_connection_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.use_connection_profile and not args.database:
        parser.error("provide --database or set CORTEX_TRAINING_DATABASE/SNOWFLAKE_DATABASE")
    if (
        not args.use_connection_profile
        and args.base_url is None
        and (args.host is None or args.pat is None)
    ):
        parser.error(f"no connection configured: {_CONNECT_HINT}")

    # Every read the TUI makes is bounded like the job list's: the client's own
    # defaults (ten retries, a ten-minute read timeout) left a view silent for
    # about a minute on a 404 or a failing service.
    args.max_retries, args.request_timeout = READ_MAX_RETRIES, READ_TIMEOUT
    from cortex_training.snowflake_auth import SnowflakeProfileError

    try:
        client = read_only_client(cli.build_client(args, cli._load_cortex_training_client_class()))
    except (SnowflakeProfileError, ValueError) as exc:
        # With no config, login state or flags, the TUI tries the configured-default
        # Snowflake connection, which a first run doesn't have: say how to connect.
        parser.exit(2, f"{parser.prog}: error: {exc}\n{_CONNECT_HINT}\n")

    # job_id is optional: given, open that job's details; omitted, open the job
    # list so the user can choose one.
    app = CortexTrainingLogTUI(client, args.job_id, sub_job_id=args.sub_job_id, poll_interval=args.poll_interval,
                               schema_only=args.schema_only)
    app.sign_in = _sign_in(cli, args)  # for the commands the TUI shows to copy
    from cortex_training.tui.log_cache import begin_session
    from cortex_training.tui.log_cache import end_session

    # What this session downloads (logs, consoles, telemetry) goes when it ends: q, a
    # crash, kill or a closed terminal (the app exits on those as on q). The next start
    # removes what a session that was killed outright left.
    session = begin_session()
    try:
        app.run()
    finally:
        end_session(session)
    return app.return_code or 0  # non-zero when the app stopped on an unhandled error


def _sign_in(cli, args) -> dict:
    """How the TUI's client connects, from the resolved arguments, for the
    commands it shows to copy (``views.sign_in_options``).

    It never holds the PAT. ``config`` is the config file read, as an
    absolute path, so a copied command finds it from any directory.
    ``host_in_config`` (a PAT) and ``base_url_in_config`` say whether that file
    by itself gives the host or base URL the client uses; when a flag or the
    environment set or overrode it, the copied command names it after
    ``--config FILE`` (``views.sign_in_options``).
    """
    if args.use_connection_profile:
        return {"kind": "profile"}
    config = None
    if getattr(args, "_legacy_config_selected", False) and args.config:
        config = str(Path(args.config).expanduser().resolve())
    from_config = _config_connection(cli, config)
    if args.base_url:
        return {"kind": "url", "config": config, "base_url": args.base_url,
                "base_url_in_config": from_config.get("base_url") == args.base_url}
    host = cli._normalize_host(args.host or "")
    return {"kind": "pat", "config": config, "host": host,
            "host_in_config": bool(host) and from_config.get("host") == host}


def _config_connection(cli, config: str | None) -> dict:
    """What ``--config FILE`` gives by itself, as the CLI reads it: its ``base_url``
    and PAT ``host`` (normalized); empty without one."""
    if not config:
        return {}
    try:
        values = cli._load_config(config)
        base_url, host = cli._config_str(values, "base_url"), cli._config_str(values, "host")
    except (OSError, ValueError):  # it was read a moment ago; if not now, name the host and base URL anyway
        return {}
    if base_url and not cli._has_url_scheme(base_url) and host is None:
        base_url, host = None, base_url  # the CLI reads a base URL without a scheme as the PAT host
    return {"base_url": base_url, "host": cli._normalize_host(host) if host else None}


def _demo_account_list() -> str | None:
    """How the demo's account-wide list answers: None (it lists every location), an unavailable
    reason, or ``unavailable`` (it fails)."""
    from cortex_training import AccountJobListUnavailable
    from cortex_training.tui.demo import UNAVAILABLE

    value = (os.environ.get(DEBUG_ACCOUNT_LIST_ENV) or "").strip().lower()
    if value in ("", "ok"):
        return None
    if value not in (*AccountJobListUnavailable.REASONS, UNAVAILABLE):
        reasons = ", ".join(sorted((*AccountJobListUnavailable.REASONS, UNAVAILABLE)))
        raise ValueError(f"{DEBUG_ACCOUNT_LIST_ENV} must be ok or one of {reasons}, got {value!r}")
    return value


def _configured_poll_interval(cli, args) -> float | None:
    """The poll_interval of the config file the resolved arguments came from, if it sets one."""
    if not getattr(args, "_legacy_config_selected", False) or not args.config:
        return None  # a connection profile, or flags and env only
    return cli._config_float(cli._load_config(args.config), "poll_interval")


def _run_demo(args, app_class, *, account_list: str | None = None) -> int:
    """Run against :class:`~cortex_training.tui.demo.DemoClient` with throwaway local stores.

    ``account_list`` is an ``AccountJobListUnavailable`` reason for the demo's
    account-wide list to answer with, or None for it to list every location.
    """
    import tempfile

    from cortex_training.tui.demo import DemoClient
    from cortex_training.tui.stores import AliasStore

    previous = os.environ.get("CORTEX_TRAINING_TUI_CACHE_DIR")
    with tempfile.TemporaryDirectory(prefix="cortex-training-demo-") as cache:
        # Demo names and caches must never land in the real cache dir.
        os.environ["CORTEX_TRAINING_TUI_CACHE_DIR"] = cache
        try:
            client = DemoClient(account_list=account_list)
            client.seed_local(aliases=AliasStore())
            app = app_class(client, args.job_id, sub_job_id=args.sub_job_id, poll_interval=args.poll_interval,
                            schema_only=args.schema_only)
            app.title = "cortex training (debug)"  # synthetic jobs: say so on every screen
            app.run()
        finally:
            # In-process callers (and tests) must not keep pointing at the deleted dir.
            if previous is None:
                os.environ.pop("CORTEX_TRAINING_TUI_CACHE_DIR", None)
            else:
                os.environ["CORTEX_TRAINING_TUI_CACHE_DIR"] = previous
    return app.return_code or 0


if __name__ == "__main__":
    raise SystemExit(run())
