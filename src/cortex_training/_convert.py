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

"""``cortex-training login --convert-old-config PATH``: turn an old JSON config into a profile.

An old JSON config holds a host, a programmatic access token (PAT), a
database, and settings. Converting it:

1. reads the file and shows what it holds, with the PAT hidden;
2. asks, then uses the PAT to look up its user and role, to check that
   Cortex Training is enabled, and to check the database. A missing database
   can be created after showing the SQL and asking. Nothing local is written
   if a check fails;
3. shows every change and asks once;
4. writes a Snowflake connection profile, sets it in ``config.json`` with the
   old settings, saves a copy of the old file without the PAT, then deletes
   the old file (and ``login.json`` if it points to it). Either every change
   happens, or every file is put back as it was.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import TextIO

from cortex_training import _database_setup
from cortex_training import _environment
from cortex_training import _login
from cortex_training import _picker
from cortex_training import _profiles

_LOGIN_HINT = "run `cortex-training login` to set up a new profile"
_NOTHING_CHANGED = "Nothing was changed."

# Old file keys, and the alternative spellings the old CLI accepted.
_OLD_KEYS = ("base_url", "host", "pat", "database", "schema", "endpoint", "poll_interval", "poll_timeout", "no_verify_ssl", "verify_ssl")
_OLD_ALIASES = {"url": "base_url", "db": "database"}
# Settings that move to config.json, in the order they're shown.
_SETTINGS = ("endpoint", "poll_interval", "poll_timeout", "verify_ssl")


class ConvertError(_login.LoginCancelled):
    """The old file can't be converted; nothing was written."""


@dataclass
class OldConfig:
    path: Path
    raw: dict[str, Any]
    host: str
    pat: str = field(repr=False)
    database: str
    schema: str
    settings: dict[str, Any]


@dataclass
class Identity:
    user: str
    role: str | None


def read_old_config(path: Path) -> OldConfig:
    """Read the old JSON config the way the old CLI did, or raise :class:`ConvertError`."""
    shown = _login._home(path)
    if not path.is_file():
        raise ConvertError(f"{shown} can't be used: there's no file there. Check the path, or {_LOGIN_HINT}.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConvertError(
            f"{shown} isn't valid JSON (line {exc.lineno}: {exc.msg}). Fix the file and run the command again, "
            f"or {_LOGIN_HINT}."
        ) from exc
    except OSError as exc:
        raise ConvertError(f"{shown} can't be read: {exc.strerror}. Check the file, or {_LOGIN_HINT}.") from exc
    values = raw.get("connection", raw) if isinstance(raw, dict) else None
    if not isinstance(values, dict):
        raise ConvertError(f"{shown} must hold a JSON object. Fix the file and run the command again, or {_LOGIN_HINT}.")
    config: dict[str, Any] = {}
    unknown = []
    for key, value in values.items():
        name = _OLD_ALIASES.get(key, key)
        if name in _OLD_KEYS:
            config[name] = value
        else:
            unknown.append(key)
    if unknown:
        raise ConvertError(
            f"{shown} has keys an old config can't have: {', '.join(sorted(unknown))}. "
            f"Fix the file and run the command again, or {_LOGIN_HINT}."
        )
    if config.get("base_url") and not config.get("host"):
        raise ConvertError(
            f"{shown} points to a test server (base_url), which can't become a Snowflake profile. Keep using "
            "--base-url for test servers, or run `cortex-training login` to connect to a Snowflake account."
        )
    if not _text(config.get("pat")):
        raise ConvertError(f"{shown} has no pat, so there's no sign-in to convert. Run `cortex-training login` to set up a profile.")
    if not _text(config.get("host")):
        raise ConvertError(f"{shown} has no host, so there's no account to convert. Run `cortex-training login` to set up a profile.")
    try:
        host = _login.validate_host(str(config["host"]))
    except ValueError as exc:
        raise ConvertError(f"{shown}: {exc} Fix the file and run the command again, or {_LOGIN_HINT}.") from exc
    settings = {key: config[key] for key in ("endpoint", "poll_interval", "poll_timeout") if config.get(key) is not None}
    if config.get("no_verify_ssl") is not None:
        settings["verify_ssl"] = not config["no_verify_ssl"]
    elif config.get("verify_ssl") is not None:
        settings["verify_ssl"] = config["verify_ssl"]
    return OldConfig(
        path=path,
        raw=raw,
        host=host,
        pat=str(config["pat"]).strip(),
        database=_text(config.get("database")) or _login.DEFAULT_DATABASE,
        schema=_text(config.get("schema")) or _login.DEFAULT_SCHEMA,
        settings=settings,
    )


def open_pat_client(old: OldConfig) -> Any:
    """Sign in with the old file's host and PAT, as the old CLI did."""
    from cortex_training.client import CortexTrainingClient

    return CortexTrainingClient._from_pat(
        old.host,
        old.pat,
        old.database,
        old.schema,
        endpoint=old.settings.get("endpoint", "cortex-training"),
        verify_ssl=old.settings.get("verify_ssl", True),
    )


def explain(exc: BaseException, old: OldConfig) -> str:
    """Say in plain words why a request with the old file's PAT failed, and what to do."""
    import requests

    from cortex_training.snowflake_auth import credentials_rejected_hint
    from cortex_training.snowflake_auth import rest_host

    text = str(exc)
    if isinstance(exc, requests.exceptions.ConnectionError):
        if "NameResolutionError" in text or "Failed to resolve" in text:
            return (
                f"Couldn't find {rest_host(old.host)} on the network. If your company reaches Snowflake through "
                "PrivateLink, ask your network admin to add a DNS record for this hyphen form of the host "
                "(see Troubleshooting in docs/getting-started/authentication.md)."
            )
        return f"Couldn't reach {old.host}. Check the address in {_login._home(old.path)}, and your VPN if you use one."
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return credentials_rejected_hint(f"Snowflake rejected the PAT in {_login._home(old.path)}.").lstrip("\n")
    return f"The check failed: {_login._first_line(exc)}"


def look_up_identity(client: Any) -> Identity:
    """Ask Snowflake which user and role the PAT signs in as."""
    rows = _database_setup.run_statement(client, "SELECT CURRENT_USER() AS USER_NAME, CURRENT_ROLE() AS ROLE_NAME")
    user = rows[0].get("user_name") if rows else None
    if not isinstance(user, str) or not user:
        raise RuntimeError("Snowflake didn't say which user the PAT signs in as")
    role = rows[0].get("role_name")
    return Identity(user=user, role=role if isinstance(role, str) and role else None)


@dataclass
class Plan:
    profile: str
    values: dict[str, Any] = field(repr=False)
    settings: dict[str, Any]
    backup: Path
    login_file: Path | None


class _Converter:
    def __init__(self, prompter: _login._Prompter, old: OldConfig, open_client: Callable[[OldConfig], Any]) -> None:
        self.p = prompter
        self.s = prompter.style
        self.old = old
        self.open_client = open_client
        self.client: Any = None
        self.created_database = False

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass

    def show(self) -> None:
        old = self.old
        self.p.section(f"Found in {_login._home(old.path)}:")
        self.p.say(f"  host      {old.host}")
        self.p.say(f"  pat       {'•' * 8}")
        self.p.say(f"  database  {old.database}    schema {old.schema}")
        if old.settings:
            self.p.say(f"  settings  {', '.join(f'{key} {old.settings[key]}' for key in _SETTINGS if key in old.settings)}")

    def check(self) -> Identity:
        """Ask, then check the PAT, Cortex Training, and the database; raise ConvertError on failure."""
        old = self.old
        self.p.say()
        self.p.say(f"To check this PAT, Cortex Training will sign in to {self.s.bold(old.host)} with it and:")
        self.p.say("  - ask who it signs in as (user and role)")
        self.p.say("  - check that Cortex Training is enabled on the account")
        self.p.say(f"  - check that database {old.database} and schema {old.schema} are visible to that role")
        self.p.say("This only reads; nothing is changed.")
        if not self.p.confirm("Check now?", default=True):
            raise ConvertError("Nothing was changed. The old config can't be converted without checking its PAT.")
        self.p.section("Checking the PAT...")
        self.client = self.open_client(old)
        try:
            identity = look_up_identity(self.client)
        except Exception as exc:
            raise ConvertError(f"{explain(exc, old)}\n\n{_NOTHING_CHANGED}") from exc
        role = f", role {identity.role}" if identity.role else ""
        self.p.say(f"  {self.s.ok(f'Signed in as {identity.user}{role}')}")
        self._check_service()
        self._check_database(identity)
        return identity

    def _check_service(self) -> None:
        from cortex_training.client import _response_error_code

        try:
            self.client.get_capacity()
        except Exception as exc:
            code = _response_error_code(getattr(exc, "response", None))
            if code == _login._NOT_RESOLVED_CODE:
                pass  # Cortex Training answered; only the database is missing.
            elif code == _login._NOT_ENABLED_CODE:
                account = _login.account_from_host(self.old.host)
                raise ConvertError(
                    f"Cortex Training isn't enabled on account {account}, so this config can't be used. "
                    "Nothing was changed."
                ) from exc
            else:
                raise ConvertError(f"{explain(exc, self.old)}\n\n{_NOTHING_CHANGED}") from exc
        self.p.say(f"  {self.s.ok('Cortex Training is enabled on this account')}")

    def _check_database(self, identity: Identity) -> None:
        database = self.old.database
        missing = _database_setup.missing(self.client)
        if missing is None:
            self.p.say(f"  {self.s.unknown(f'Could not check database {database}; continuing')}")
            return
        if not missing:
            self.p.say(f"  {self.s.ok(f'Database {database} is ready')}")
            return
        who = f"role {identity.role}" if identity.role else "your role"
        self.p.say(f"  {self.s.bad(f'Database {database} is missing, or {who} cannot access it')}")
        self.p.say()
        statements = [f"{statement};" for statement in _database_setup.create_statements(self.client, missing)]
        self.p.say("Cortex Training can create it now by running:")
        for statement in statements:
            self.p.say(f"    {self.s.cyan(statement)}")
        if not self.p.confirm("Run it?", default=False):
            raise ConvertError(
                f"Nothing was changed. Database {database} doesn't exist yet. To create it yourself, run in "
                f"Snowsight: {statements[0]} Then run this command again."
            )
        try:
            _database_setup.create(self.client, missing)
        except _database_setup.DatabaseSetupError as exc:
            raise ConvertError(f"{exc} Nothing else was changed.") from exc
        self.created_database = True
        if _database_setup.missing(self.client):
            raise ConvertError(f"Database {database} still isn't visible to {who} after creating it. Nothing else was changed.")
        self.p.say(f"  {self.s.ok(f'Created database {database}')}")

    def plan(self, identity: Identity) -> Plan:
        old = self.old
        profiles = _profiles.read_profiles()
        name = _login.DEFAULT_PROFILE
        if name in profiles:
            name = _unused_name(name, profiles)
            self.p.say()
            self.p.say(f"Profile {_login.DEFAULT_PROFILE} already exists, so the new one will be called {self.s.bold(name)}.")

        def validate(typed: str) -> str:
            if not _profiles._PROFILE_NAME.match(typed):
                raise ValueError(f"{typed!r} isn't a valid profile name. Use letters, digits, '_' or '-'.")
            if typed in _profiles.read_profiles():
                raise ValueError(f"Profile {typed} already exists; choose another name.")
            return typed

        self.p.say()
        name = self.p.ask("Profile name", default=name, validate=validate)
        values = {
            "account": _login.account_from_host(old.host),
            "host": old.host,
            "user": identity.user,
            "role": identity.role,
            "authenticator": "PROGRAMMATIC_ACCESS_TOKEN",
            "token": old.pat,
            "database": old.database,
            "schema": old.schema,
        }
        current, _ = _picker.load_config()
        settings = dict(current)
        for key in _SETTINGS:
            if key not in old.settings:
                continue
            if key in current and current[key] != old.settings[key]:
                self.p.say()
                self.p.say(f"config.json already sets {key} to {current[key]}; the old config sets {old.settings[key]}.")
                choice = self.p.choose(
                    "Which should Cortex Training use?",
                    [f"Keep {current[key]} (config.json)", f"Use {old.settings[key]} (the old config)"],
                    default=0,
                )
                if choice == 0:
                    continue
            settings[key] = old.settings[key]
        settings["profile"] = name
        login_file = _login_file_for(old.path)
        return Plan(profile=name, values=values, settings=settings, backup=_backup_path(old.path), login_file=login_file)

    def confirm(self, plan: Plan) -> None:
        old = self.old
        role = f", role {plan.values['role']}" if plan.values.get("role") else ""
        moved = [key for key in _SETTINGS if key in old.settings and plan.settings.get(key) == old.settings[key]]
        self.p.rule()
        self.p.say("This will:")
        self.p.say(f"  1. Add profile {self.s.bold(plan.profile)} to {_login._home(_profiles.profile_file())}")
        self.p.say(
            f"     (account {plan.values['account']}, host, user {plan.values['user']}{role}, the PAT, "
            f"database {old.database})"
        )
        also = f", and move {', '.join(moved)} there" if moved else ""
        self.p.say(f"  2. Set Cortex Training to use it, in {_login._home(_picker.config_path())}{also}")
        self.p.say(f"  3. Save a copy of the old config without the PAT: {_login._home(plan.backup)}")
        deleted = _login._home(old.path) + (f" and {_login._home(plan.login_file)}" if plan.login_file else "")
        self.p.say(f"  4. Delete {deleted}")
        if not self.p.confirm("Go ahead?", default=True):
            raise ConvertError("Nothing was changed.")


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _unused_name(base: str, profiles: dict[str, Any]) -> str:
    number = 2
    while f"{base}-{number}" in profiles:
        number += 1
    return f"{base}-{number}"


def _backup_path(path: Path) -> Path:
    backup = path.with_name(f"{path.name}.bak")
    number = 2
    while backup.exists():
        backup = path.with_name(f"{path.name}.bak{number}")
        number += 1
    return backup


def _login_file_for(path: Path) -> Path | None:
    """``login.json`` when it points to ``path``; it would point to a deleted file otherwise."""
    from cortex_training import _cli

    login_file = _cli._login_state_path()
    try:
        pointed = json.loads(login_file.read_text(encoding="utf-8")).get("config_path")
    except (OSError, ValueError, AttributeError):
        return None
    if isinstance(pointed, str) and Path(pointed).expanduser().resolve() == path.resolve():
        return login_file
    return None


def _without_pat(raw: dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw.get("connection"), dict):
        return {**raw, "connection": {key: value for key, value in raw["connection"].items() if key != "pat"}}
    return {key: value for key, value in raw.items() if key != "pat"}


class _Snapshot:
    """A file's bytes (or its absence), to put back if a later step fails."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = path.read_bytes() if path.is_file() else None

    def restore(self) -> None:
        if self.data is None:
            self.path.unlink(missing_ok=True)
        else:
            _write_atomic(self.path, self.data)


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def apply(plan: Plan, old: OldConfig) -> None:
    """Make every change in ``plan``, or put every file back as it was and raise."""
    config_file = _picker.config_path()
    snapshots = [_Snapshot(_profiles.profile_file()), _Snapshot(config_file), _Snapshot(plan.backup)]
    removed: list[_Snapshot] = []
    try:
        _profiles.write_profile(plan.profile, plan.values)
        settings = json.dumps(plan.settings, indent=2) + "\n"
        _write_atomic(config_file, settings.encode("utf-8"))
        _picker.load_config()  # Refuse settings that commands couldn't read back.
        _write_atomic(plan.backup, (json.dumps(_without_pat(old.raw), indent=2) + "\n").encode("utf-8"))
        # Delete last, once every write has worked.
        for path in [old.path, *([plan.login_file] if plan.login_file else [])]:
            snapshot = _Snapshot(path)
            path.unlink()
            removed.append(snapshot)
    except BaseException as exc:
        for snapshot in [*snapshots, *removed]:
            try:
                snapshot.restore()
            except Exception:
                pass
        if isinstance(exc, Exception):
            raise ConvertError(f"Couldn't finish the conversion ({_login._first_line(exc)}), so every file was put back as it was.") from exc
        raise


def run_convert(
    path: str,
    *,
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    interactive: bool | None = None,
    open_client: Callable[[OldConfig], Any] = open_pat_client,
) -> int:
    """Run ``cortex-training login --convert-old-config PATH``; return the exit code."""
    if interactive is None:
        interactive = stdin.isatty() and stderr.isatty()
    if not interactive:
        stderr.write(f"error: {_login._NO_TERMINAL}\n")
        return 2
    prompter = _login._Prompter(stdin, stderr)
    style = prompter.style
    shell = _environment.shell_name()
    findings = _environment.find()
    prompter.logo()
    _login.report_environment(prompter, findings, shell)
    converter: _Converter | None = None
    try:
        old = read_old_config(Path(path).expanduser())
        converter = _Converter(prompter, old, open_client)
        converter.show()
        identity = converter.check()
        plan = converter.plan(identity)
        converter.confirm(plan)
        apply(plan, old)
    except KeyboardInterrupt:
        stderr.write("\nConversion cancelled. No files were changed.\n")
        if converter is not None and converter.created_database:
            stderr.write(f"Database {converter.old.database} was created and is kept.\n")
        return 130
    except (_login.LoginCancelled, _picker.PickerError, _profiles.ProfileFileError) as exc:
        stderr.write(f"\n{style.bad(str(exc))}\n")
        if converter is not None and converter.created_database:
            stderr.write(f"Database {converter.old.database} was created and is kept.\n")
        return 1
    finally:
        if converter is not None:
            converter.close()

    prompter.rule()
    if not style.decorate:
        prompter.say()
    prompter.say(style.green(style.bold("You're all set.")) + f" Cortex Training will use profile {style.bold(plan.profile)}.")
    prompter.say(style.dim(f"(The old config is saved without its PAT in {_login._home(plan.backup)}.)"))
    prompter.say()
    prompter.say("Run your command again.")
    _login.remind_environment(prompter, findings, shell)
    if not getattr(stdout, "isatty", lambda: False)():
        json.dump(
            {
                "profile": plan.profile,
                "profile_file": str(_profiles.profile_file()),
                "account": plan.values["account"],
                "user": plan.values["user"],
                "role": plan.values["role"],
                "database": old.database,
                "schema": old.schema,
                "config": str(_picker.config_path()),
                "backup": str(plan.backup),
                "deleted": [str(old.path), *([str(plan.login_file)] if plan.login_file else [])],
                "created_database": converter.created_database,
            },
            stdout,
            indent=2,
        )
        stdout.write("\n")
    return 0
