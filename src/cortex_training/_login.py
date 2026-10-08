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

"""Interactive ``cortex-training login``.

Login first reports any connection the CLI would already use, then (if needed)
prompts for the few values a PAT profile needs, writes them to the Snowflake
Connector's ``connections.toml``, and verifies the result with a capacity
call. Prompts and progress go to stderr; the final summary is JSON on stdout.
"""

from __future__ import annotations

import getpass
import logging
import os
import re
import sys
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Callable
from typing import TextIO

from cortex_training._connection import ConnectionSettings
from cortex_training._connection import build_client
from cortex_training._connection import clear_login_state
from cortex_training._connection import env
from cortex_training._connection import login_state_path
from cortex_training._connection import normalize_host
from cortex_training._connection import read_login_state
from cortex_training._connection import resolve_connection
from cortex_training._connection import write_login_state

DEFAULT_PROFILE = "cortex-training"
DEFAULT_DATABASE = "CORTEX_TRAINING_DB"
DEFAULT_SCHEMA = "PUBLIC"

_PROFILE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")

# Absolute links: pip installs do not include the repository's docs.
_AUTH_DOCS_URL = (
    "https://github.com/snowflakedb/cortex-training/blob/main/docs/"
    "getting-started/authentication.md"
)

_HOST_HELP = (
    "Your account host: ORG-ACCOUNT.snowflakecomputing.com, where ORG and ACCOUNT\n"
    "are the two parts of your Snowsight URL (app.snowflake.com/ORG/ACCOUNT).\n"
    "You can also paste that Snowsight URL. More help:\n"
    f"  {_AUTH_DOCS_URL}#finding-your-account-and-host"
)
_USER_HELP = (
    "Your Snowflake login name. It is shown under your profile in Snowsight; it\n"
    "can differ from your display name (DESC USER <name> shows LOGIN_NAME)."
)
_PAT_HELP = (
    "Create a programmatic access token in Snowsight: your user icon (bottom\n"
    "left) > Settings > Authentication > Generate token. Step-by-step guide\n"
    "with screenshots:\n"
    f"  {_AUTH_DOCS_URL}#step-1-create-a-pat\n"
    "The input is hidden."
)

_ENV_VARS = (
    "CORTEX_TRAINING_CONNECTION",
    "CORTEX_TRAINING_CONFIG",
    "CORTEX_TRAINING_BASE_URL",
    "CORTEX_TRAINING_HOST",
    "SNOWFLAKE_HOST",
    "CORTEX_TRAINING_PAT",
    "SNOWFLAKE_PAT",
    "CORTEX_TRAINING_DATABASE",
    "SNOWFLAKE_DATABASE",
    "CORTEX_TRAINING_SCHEMA",
    "SNOWFLAKE_SCHEMA",
    "SNOWFLAKE_DEFAULT_CONNECTION_NAME",
)


@dataclass
class LoginOptions:
    host: str | None = None
    user: str | None = None
    database: str | None = None
    schema: str | None = None
    connection: str | None = None
    pat_stdin: bool = False
    force: bool = False
    reset: bool = False

    @property
    def creates_profile(self) -> bool:
        return any(
            (self.host, self.user, self.database, self.schema, self.connection, self.pat_stdin)
        )


@dataclass
class Discovery:
    """What the CLI would use today, plus every other setup lying around."""

    connections_file: Path
    profiles: dict[str, dict[str, Any]]
    default_profile_name: str
    env_vars: list[str]
    active: ConnectionSettings | None = None
    active_error: str | None = None
    unused: list[str] = field(default_factory=list)

    @property
    def active_profile(self) -> dict[str, Any] | None:
        if self.active is None or not self.active.use_connection_profile:
            return None
        return self.profiles.get(self.active.connection_name or self.default_profile_name)

    @property
    def uses_remembered_login(self) -> bool:
        return self.active is not None and self.active.source.endswith(
            "remembered by cortex-training login"
        )

    @property
    def has_usable_setup(self) -> bool:
        if self.active is None:
            return False
        if self.active.use_connection_profile:
            return self.active_profile is not None
        return True


# ---------------------------------------------------------------------------
# connections.toml
# ---------------------------------------------------------------------------


def connections_file_path() -> Path:
    """Return the ``connections.toml`` the Snowflake Connector reads."""
    home = env("SNOWFLAKE_HOME")
    if home:
        return Path(home).expanduser() / "connections.toml"
    from snowflake.connector.constants import CONNECTIONS_FILE

    return Path(CONNECTIONS_FILE)


def read_profiles(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    import tomlkit

    try:
        document = tomlkit.parse(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"could not parse {path}: {exc}") from exc
    return {
        str(name): dict(table)
        for name, table in document.items()
        if isinstance(table, dict)
    }


def _default_profile_name(connections_file: Path) -> str:
    name = env("SNOWFLAKE_DEFAULT_CONNECTION_NAME")
    if name:
        return name
    config_file = connections_file.with_name("config.toml")
    if config_file.is_file():
        import tomlkit

        try:
            value = tomlkit.parse(config_file.read_text(encoding="utf-8")).get(
                "default_connection_name"
            )
        except Exception:
            value = None
        if isinstance(value, str) and value:
            return value
    return "default"


def write_profile(
    path: Path,
    name: str,
    *,
    host: str,
    user: str,
    pat: str,
    database: str,
    schema: str,
) -> None:
    """Add or replace one profile, keeping every other profile and comment."""
    import tomlkit

    if path.is_file():
        document = tomlkit.parse(path.read_text(encoding="utf-8"))
    else:
        document = tomlkit.document()
    table = tomlkit.table()
    table["account"] = account_from_host(host)
    table["host"] = host
    table["user"] = user
    table["authenticator"] = "programmatic_access_token"
    table["token"] = pat
    table["database"] = database
    table["schema"] = schema
    document[name] = table

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp_path = path.with_name(path.name + ".tmp")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(tomlkit.dumps(document))
    os.replace(tmp_path, path)
    os.chmod(path, 0o600)


def remove_profile(path: Path, name: str) -> bool:
    """Remove one profile, keeping every other profile and comment."""
    import tomlkit

    if not path.is_file():
        return False
    document = tomlkit.parse(path.read_text(encoding="utf-8"))
    if name not in document:
        return False
    del document[name]
    tmp_path = path.with_name(path.name + ".tmp")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(tomlkit.dumps(document))
    os.replace(tmp_path, path)
    os.chmod(path, 0o600)
    return True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_host(raw: str) -> str:
    """Return a bare account host, or raise ValueError with the fix."""
    host, _, path = normalize_host(raw.strip()).partition("/")
    if host.lower() == "app.snowflake.com":
        # A pasted Snowsight URL: app.snowflake.com/ORG/ACCOUNT[/...]
        parts = [part for part in path.split("/") if part]
        if len(parts) < 2:
            raise ValueError(
                "that Snowsight URL has no ORG/ACCOUNT part. Expected "
                "app.snowflake.com/ORG/ACCOUNT or ORG-ACCOUNT.snowflakecomputing.com."
            )
        host = f"{parts[0]}-{parts[1]}.snowflakecomputing.com"
    if not host or " " in host or "." not in host:
        raise ValueError(
            f"{raw!r} is not an account host. Expected ORG-ACCOUNT.snowflakecomputing.com."
        )
    if "_" in host:
        raise ValueError(
            f"{host!r} contains '_'. Use {host.replace('_', '-')!r} instead: "
            "Snowflake's SSL certificate only matches hyphens."
        )
    return host


def account_from_host(host: str) -> str:
    return host.split(".", 1)[0]


def validate_profile_name(name: str) -> str:
    name = name.strip()
    if not _PROFILE_NAME.match(name):
        raise ValueError(
            f"{name!r} is not a valid profile name. Use letters, digits, '-', '_' or '.'."
        )
    return name


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover() -> Discovery:
    connections_file = connections_file_path()
    discovery = Discovery(
        connections_file=connections_file,
        profiles=read_profiles(connections_file),
        default_profile_name=_default_profile_name(connections_file),
        env_vars=[name for name in _ENV_VARS if env(name)],
    )
    try:
        discovery.active = resolve_connection()
    except ValueError as exc:
        discovery.active_error = str(exc)

    active_name = None
    if discovery.active is not None and discovery.active.use_connection_profile:
        active_name = discovery.active.connection_name or discovery.default_profile_name
    for name in discovery.profiles:
        if name != active_name:
            discovery.unused.append(f"connections.toml profile {name!r}")
    if discovery.active is not None and discovery.active.use_connection_profile:
        for name in ("CORTEX_TRAINING_HOST", "SNOWFLAKE_HOST", "CORTEX_TRAINING_BASE_URL"):
            if name in discovery.env_vars:
                discovery.unused.append(f"{name} (a connection profile takes precedence)")
    return discovery


def _token_location(discovery: Discovery) -> str:
    active = discovery.active
    assert active is not None
    if active.use_connection_profile:
        profile = discovery.active_profile or {}
        if profile.get("token") or profile.get("token_file_path"):
            return "stored in connections.toml"
        return f"via authenticator {profile.get('authenticator', 'snowflake')!r}"
    if active.base_url:
        return "none (local or mock server)"
    if active.config_path:
        return f"in {active.config_path}"
    for name in ("CORTEX_TRAINING_PAT", "SNOWFLAKE_PAT"):
        if name in discovery.env_vars:
            return f"from {name}"
    return "passed explicitly"


def describe_setup(discovery: Discovery) -> list[str]:
    """Describe the setup the CLI would use, without ever showing a secret."""
    active = discovery.active
    if active is None:
        return [
            "Found a Cortex Training setup that does not work as configured:",
            f"  {discovery.active_error}",
        ]
    lines = ["Found an existing Cortex Training setup:", f"  source:    {active.source}"]
    if active.use_connection_profile:
        profile = discovery.active_profile or {}
        name = active.connection_name or discovery.default_profile_name
        lines.append(f"  file:      {discovery.connections_file} [{name}]")
        host = profile.get("host") or profile.get("account")
        user = profile.get("user")
        database = active.database or profile.get("database")
        schema = active.schema or profile.get("schema")
    else:
        if active.config_path:
            lines.append(f"  file:      {active.config_path}")
        host = active.base_url or active.host
        user = None
        database = active.database
        schema = active.schema
    lines.append(f"  host:      {host or '(not set)'}")
    if user:
        lines.append(f"  user:      {user}")
    lines.append(f"  database:  {database or '(not set)'}")
    lines.append(f"  schema:    {schema or '(not set)'}")
    lines.append(f"  endpoint:  {active.endpoint}")
    lines.append(f"  token:     {_token_location(discovery)}")
    if discovery.unused:
        lines.append("Also found, but not used:")
        lines.extend(f"  - {item}" for item in discovery.unused)
    lines.append(
        "To change it: pass --connection NAME, unset the env vars above, or run "
        "'cortex-training login --connection NAME' to create a new profile."
    )
    return lines


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class _DatabaseSetupReporter(logging.Handler):
    """Explain the client's automatic CREATE DATABASE while it happens."""

    def __init__(self, stderr: TextIO) -> None:
        super().__init__(level=logging.INFO)
        self._stderr = stderr

    def emit(self, record: logging.LogRecord) -> None:
        template = str(record.msg)
        if template.startswith("Database '%s' not found"):
            # The client logs the database it actually used, which for a
            # connections.toml profile comes from the profile itself.
            database = record.args[0] if record.args else "the configured database"
            self._stderr.write(
                f"Database {database} does not exist yet. Cortex Training uses it\n"
                "only to route API requests; no data is stored in it. Creating it now:\n"
                f"  CREATE DATABASE IF NOT EXISTS {database}\n"
            )
        elif template.startswith(("Database '%s' created", "Schema '%s'")):
            self._stderr.write(f"{record.getMessage()}\n")


def verify(
    settings: ConnectionSettings,
    client_cls: Any,
    stderr: TextIO,
) -> dict[str, Any]:
    """Open the connection and read capacity, reporting database creation."""
    client_logger = logging.getLogger("cortex_training.client")
    reporter = _DatabaseSetupReporter(stderr)
    previous_level = client_logger.level
    client_logger.addHandler(reporter)
    if client_logger.getEffectiveLevel() > logging.INFO:
        client_logger.setLevel(logging.INFO)
    client = None
    try:
        client = build_client(settings, client_cls)
        return client.get_capacity()
    finally:
        client_logger.removeHandler(reporter)
        client_logger.setLevel(previous_level)
        close = getattr(client, "close", None)
        if callable(close):
            close()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


class _Prompter:
    def __init__(self, stdin: TextIO, stderr: TextIO, interactive: bool) -> None:
        self.stdin = stdin
        self.stderr = stderr
        self.interactive = interactive

    def _readline(self) -> str:
        line = self.stdin.readline()
        if not line:
            raise ValueError("login cancelled: no input")
        return line.rstrip("\n")

    def ask(
        self,
        label: str,
        *,
        default: str | None = None,
        help_text: str | None = None,
        validate: Callable[[str], str] | None = None,
        flag: str,
    ) -> str:
        if not self.interactive:
            if default is None:
                raise ValueError(f"{label} is required: pass {flag}")
            return validate(default) if validate else default
        if help_text:
            self.stderr.write(f"\n{help_text}\n")
        while True:
            suffix = f" [{default}]" if default else ""
            self.stderr.write(f"{label}{suffix}: ")
            self.stderr.flush()
            value = self._readline().strip() or (default or "")
            if not value:
                self.stderr.write(f"{label} is required.\n")
                continue
            if validate is None:
                return value
            try:
                return validate(value)
            except ValueError as exc:
                self.stderr.write(f"{exc}\n")

    def confirm(self, question: str, *, default: bool) -> bool:
        if not self.interactive:
            return default
        hint = "[Y/n]" if default else "[y/N]"
        while True:
            self.stderr.write(f"{question} {hint} ")
            self.stderr.flush()
            answer = self._readline().strip().lower()
            if not answer:
                return default
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False

    def secret(self, label: str) -> str:
        self.stderr.write(f"\n{_PAT_HELP}\n")
        while True:
            if self.stdin is sys.stdin:
                value = getpass.getpass(f"{label}: ", stream=self.stderr)
            else:
                self.stderr.write(f"{label}: ")
                self.stderr.flush()
                value = self._readline()
            value = value.strip()
            if value:
                return value
            self.stderr.write(f"{label} is required.\n")


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------


def _prefill(discovery: Discovery, distrusted: set[str]) -> dict[str, str | None]:
    """Defaults for the prompts, never taken from a profile that just failed."""
    profile = None
    if discovery.active is not None and discovery.active.use_connection_profile:
        name = discovery.active.connection_name or discovery.default_profile_name
        if name not in distrusted:
            profile = discovery.profiles.get(name)
    if profile is None:
        profile = next(
            (
                values
                for name, values in discovery.profiles.items()
                if values.get("host") and name not in distrusted
            ),
            {},
        )
    return {
        "host": profile.get("host") or env("CORTEX_TRAINING_HOST", "SNOWFLAKE_HOST"),
        "user": profile.get("user"),
        "database": profile.get("database")
        or env("CORTEX_TRAINING_DATABASE", "SNOWFLAKE_DATABASE"),
        "schema": profile.get("schema") or env("CORTEX_TRAINING_SCHEMA", "SNOWFLAKE_SCHEMA"),
    }


def _read_pat(options: LoginOptions, prompter: _Prompter) -> str:
    if options.pat_stdin:
        value = prompter.stdin.readline().strip()
        if not value:
            raise ValueError("--pat-stdin was given but stdin had no token")
        return value
    for name in ("CORTEX_TRAINING_PAT", "SNOWFLAKE_PAT"):
        if env(name):
            if prompter.confirm(f"Use the PAT from {name}?", default=True):
                return env(name) or ""
            break
    if not prompter.interactive:
        raise ValueError("a PAT is required: set CORTEX_TRAINING_PAT or pass --pat-stdin")
    return prompter.secret("PAT")


def _report_failure(
    exc: BaseException,
    stderr: TextIO,
    format_error: Callable[[BaseException], str],
) -> None:
    stderr.write(f"\nThe connection check failed:\n{format_error(exc)}\n")


def _stale_login_reason(discovery: Discovery) -> str | None:
    """Why the remembered login cannot work, without connecting; None if it might."""
    try:
        state = read_login_state()
    except ValueError as exc:
        return str(exc)
    if state is None:
        return None
    kind, value = state
    if kind == "connection" and value not in discovery.profiles:
        return f"profile {value!r} is no longer in {discovery.connections_file}"
    if kind == "config" and not Path(value).expanduser().is_file():
        return f"config file {value} no longer exists"
    return None


def _forget_login(reason: str, stderr: TextIO) -> None:
    stderr.write(
        f"\nYour saved cortex-training login no longer works: {reason}\n"
        f"Removed the saved login ({login_state_path()}).\n"
    )
    clear_login_state()


def _start_setup(interactive: bool, stderr: TextIO) -> int | None:
    if interactive:
        stderr.write("Let's set up a new one.\n")
        return None
    stderr.write(
        "Run 'cortex-training login' in a terminal to set up a new one, or pass "
        "--host and --user with CORTEX_TRAINING_PAT set.\n"
    )
    return 1


def _check_existing_setup(
    discovery: Discovery,
    prompter: _Prompter,
    *,
    stdout: TextIO,
    stderr: TextIO,
    client_cls: Any,
    format_error: Callable[[BaseException], str],
    print_json: Callable[[Any, TextIO], None],
    distrusted: set[str],
) -> int | None:
    """Report and verify the setup in use; return an exit code, or None to set up.

    A profile that fails its check is added to ``distrusted`` so its values are
    not offered again as prompt defaults.

    A remembered login is always verified before it is trusted. If it is stale
    or fails the check, it is removed; login then uses whatever else is
    configured, or continues to the prompts.
    """
    interactive = prompter.interactive

    stale = _stale_login_reason(discovery)
    if stale is not None:
        _forget_login(stale, stderr)
        discovery = discover()
        if not (discovery.has_usable_setup or discovery.active_error):
            return _start_setup(interactive, stderr)

    if not (discovery.has_usable_setup or discovery.active_error):
        if not interactive:
            raise ValueError(
                "no Cortex Training connection is configured. Run 'cortex-training login' "
                "in a terminal, or pass --host and --user with CORTEX_TRAINING_PAT set."
            )
        return None

    stderr.write("\n".join(describe_setup(discovery)) + "\n")
    if not discovery.has_usable_setup:
        return None if interactive else 1

    remembered = discovery.uses_remembered_login
    if not remembered and not prompter.confirm("\nCheck this setup now?", default=True):
        return None

    assert discovery.active is not None
    what = "your saved login" if remembered else "the connection"
    stderr.write(f"Checking {what} with a capacity request...\n")
    try:
        verify(discovery.active, client_cls, stderr)
    except Exception as exc:
        if discovery.active.use_connection_profile:
            distrusted.add(discovery.active.connection_name or discovery.default_profile_name)
        if remembered:
            _forget_login(f"the connection check failed.\n{format_error(exc)}", stderr)
            return _start_setup(interactive, stderr)
        _report_failure(exc, stderr, format_error)
        if interactive and prompter.confirm(
            "Set up a new connections.toml profile instead?", default=True
        ):
            return None
        return 1

    stderr.write("Connected. You're all set; nothing was changed.\n")
    if prompter.confirm("Create a new connections.toml profile anyway?", default=False):
        return None
    print_json(
        {
            "logged_in": True,
            "profile_created": False,
            "source": discovery.active.source,
            "verified": True,
        },
        stdout,
    )
    return 0


def _reset(
    discovery: Discovery,
    prompter: _Prompter,
    options: LoginOptions,
    stderr: TextIO,
) -> set[str]:
    """Forget the saved login and, if confirmed, the profile it pointed to.

    Returns the profile names whose values must not be offered as defaults:
    all of them, because a reset starts over. Other profiles in
    connections.toml are never modified.
    """
    every_profile = set(discovery.profiles)
    path = login_state_path()
    try:
        state = read_login_state()
    except ValueError:
        state = None  # unreadable; removing it is the point
    if path.exists():
        clear_login_state()
        stderr.write(f"Removed the saved login ({path}).\n")
    else:
        stderr.write("There was no saved login to remove.\n")

    if state is None or state[0] != "connection":
        if state is not None:
            stderr.write(f"Kept {state[1]}; login only remembered its path.\n")
        return every_profile

    name = state[1]
    connections_file = discovery.connections_file
    if name not in discovery.profiles:
        return every_profile
    if options.force or prompter.confirm(
        f"Also remove profile [{name}] from {connections_file}? Other tools that read "
        "this file may use it.",
        default=False,
    ):
        remove_profile(connections_file, name)
        stderr.write(f"Removed profile [{name}] from {connections_file}.\n")
    else:
        stderr.write(f"Kept profile [{name}] in {connections_file}.\n")
    others = [other for other in discovery.profiles if other != name]
    if others:
        stderr.write(
            f"Other profiles were not touched: {', '.join(others)}.\n"
        )
    return every_profile


def run_login(
    options: LoginOptions,
    *,
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    client_cls: Any,
    format_error: Callable[[BaseException], str],
    print_json: Callable[[Any, TextIO], None],
) -> int:
    interactive = bool(getattr(stdin, "isatty", lambda: False)()) and not options.pat_stdin
    prompter = _Prompter(stdin, stderr, interactive)
    discovery = discover()
    distrusted: set[str] = set()

    if options.reset:
        distrusted |= _reset(discovery, prompter, options, stderr)
        discovery = discover()
        if not options.creates_profile and not interactive:
            stderr.write("Run 'cortex-training login' to set up a connection again.\n")
            print_json({"logged_in": False, "reset": True}, stdout)
            return 0
    elif not options.creates_profile:
        outcome = _check_existing_setup(
            discovery,
            prompter,
            stdout=stdout,
            stderr=stderr,
            client_cls=client_cls,
            format_error=format_error,
            print_json=print_json,
            distrusted=distrusted,
        )
        if outcome is not None:
            return outcome
        # A saved login may have just been removed; look again for defaults.
        discovery = discover()

    defaults = _prefill(discovery, distrusted)
    stderr.write("\nSet up a connections.toml profile for Cortex Training.\n")
    host = options.host and validate_host(options.host)
    host = host or prompter.ask(
        "Account host", default=defaults["host"], help_text=_HOST_HELP,
        validate=validate_host, flag="--host",
    )
    user = options.user or prompter.ask(
        "Snowflake user", default=defaults["user"], help_text=_USER_HELP, flag="--user"
    )
    pat = _read_pat(options, prompter)
    database = options.database or prompter.ask(
        "Database", default=defaults["database"] or DEFAULT_DATABASE, flag="--database"
    )
    schema = options.schema or prompter.ask(
        "Schema", default=defaults["schema"] or DEFAULT_SCHEMA, flag="--schema"
    )
    name = options.connection and validate_profile_name(options.connection)
    name = name or prompter.ask(
        "Profile name", default=DEFAULT_PROFILE, validate=validate_profile_name,
        flag="--connection",
    )

    path = discovery.connections_file
    if name in discovery.profiles and not options.force:
        if not interactive:
            raise ValueError(f"profile {name!r} already exists in {path}; pass --force to replace it")
        if not prompter.confirm(f"Profile {name!r} already exists in {path}. Replace it?", default=False):
            stderr.write("Nothing was changed.\n")
            return 1

    write_profile(path, name, host=host, user=user, pat=pat, database=database, schema=schema)
    stderr.write(f"\nSaved profile [{name}] to {path} (permissions 600).\n")
    previous = read_login_state_quietly()
    write_login_state(connection=name)
    if previous is not None and previous != ("connection", name):
        stderr.write(
            f"Remembered [{name}] for later commands, replacing the remembered "
            f"{previous[0]} {previous[1]}.\n"
        )
    else:
        stderr.write(f"Remembered [{name}] for later commands.\n")
    _warn_if_overridden(name, stderr)

    summary: dict[str, Any] = {
        "logged_in": True,
        "profile_created": True,
        "connection": name,
        "connections_file": str(path),
        "host": host,
        "user": user,
        "database": database,
        "schema": schema,
    }
    stderr.write("Checking the connection with a capacity request...\n")
    try:
        verify(resolve_connection(connection_name=name), client_cls, stderr)
    except Exception as exc:
        _report_failure(exc, stderr, format_error)
        stderr.write(
            f"\nThe profile was kept. Fix the issue above, then run "
            f"'cortex-training --connection {name} capacity' or 'cortex-training login' again.\n"
        )
        print_json({**summary, "verified": False}, stdout)
        return 1

    stderr.write(
        "Connected.\n\nNext steps:\n"
        "  cortex-training capacity\n"
        "  cortex-training list\n"
        "Recipes and connect() use the same connection.\n"
    )
    print_json({**summary, "verified": True}, stdout)
    return 0


def _warn_if_overridden(name: str, stderr: TextIO) -> None:
    """Say so when env vars would still pick a different connection."""
    try:
        chosen = resolve_connection()
    except ValueError as exc:
        stderr.write(f"Note: commands without --connection will fail until this is fixed: {exc}\n")
        return
    if chosen.use_connection_profile and chosen.connection_name == name:
        return
    stderr.write(
        f"Note: commands without --connection will still use {chosen.source}.\n"
        f"Unset that, or pass --connection {name}.\n"
    )


def read_login_state_quietly() -> tuple[str, str] | None:
    try:
        return read_login_state()
    except ValueError:
        return None
