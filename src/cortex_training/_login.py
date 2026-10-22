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

"""Interactive ``cortex-training login``: set up the profile Cortex Training uses.

1. If ``config.json`` already names a profile, check it and offer to keep it.
2. Otherwise offer to check which accounts have Cortex Training, then pick one
   of the user's Snowflake profiles, or create one.
3. Sign in and ask Cortex Training first, so nothing is created on an account
   where it isn't enabled; a profile that fails returns to the list.
4. Say which database Cortex Training will use, and offer a separate one. If
   the database is not usable, offer another profile with the same account,
   user, and role whose database is; otherwise offer a new profile with the
   same sign-in and its own database, created after confirmation.
5. Save the chosen profile as ``"profile"`` in ``config.json``.

Existing profiles are never changed. Prompts and progress go to stderr, with
color in a terminal; a JSON summary goes to stdout only when it is piped.
"""

from __future__ import annotations

import getpass
import json
import os
import sys
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import TextIO

from cortex_training import _database_setup
from cortex_training import _environment
from cortex_training import _picker
from cortex_training import _profiles

DEFAULT_PROFILE = "cortex-training"
DEFAULT_DATABASE = _picker.DEFAULT_DATABASE
DEFAULT_SCHEMA = _picker.DEFAULT_SCHEMA

# Cortex Training's error codes for the capacity check made right after sign-in.
_NOT_ENABLED_CODE = "517600"  # HTTP 403: the API is not enabled for this account
_NOT_RESOLVED_CODE = "517602"  # HTTP 400: the database or schema in the URL can't be resolved

_LOGO = r"""
  ___   ___   ___  _____  ___ __  __
 / __| / _ \ | _ \|_   _|| __|\ \/ /
| (__ | (_) ||   /  | |  | _|  >  <
 \___| \___/ |_|_\  |_|  |___|/_/\_\
          T R A I N I N G"""

_INTRO_TITLE = "Let's connect Cortex Training to your Snowflake account."

_INTRO = f"""\
Cortex Training needs three things:
  1. A Snowflake account with Cortex Training enabled.
  2. A way to sign in to that account: your Snowflake user, a role, and a
     programmatic access token (a token you create in Snowsight for tools
     like this one).
  3. A database your role can create and write to. Cortex Training keeps
     everything it creates for you there, such as training jobs and
     checkpoints. We recommend a dedicated one named {DEFAULT_DATABASE}.

Cortex Training doesn't store a login of its own. It signs in with a Snowflake
connection profile: a saved connection that the Snowflake CLI (snow) and other
Snowflake tools use too. You can use one you already have, or create one here.

Let's see what you already have."""

_IMPORTANT = "MAKE SURE YOU PICK A PROFILE ON AN ACCOUNT WHERE CORTEX TRAINING IS ENABLED."

_NEW_PROFILE_TITLE = "Let's create a new profile. You'll need:"

_NEW_PROFILE = """\
  - your account's address, for example ORG-ACCOUNT.snowflakecomputing.com
    (you can also paste the Snowsight URL from your browser),
  - your Snowflake user name, and
  - a programmatic access token. To create one in Snowsight, open your user
    menu > Settings > Authentication > Programmatic access tokens >
    Generate new token."""

_DATABASE_HELP = (
    "Cortex Training stores its training jobs and checkpoints in this database. "
    f"We recommend a dedicated one: {DEFAULT_DATABASE}."
)

_CHECK_NEW_PROFILE = (
    "Cortex Training can now sign in with it to check that it works: that Cortex Training is\n"
    "enabled on the account, and that the database is ready. This only reads; nothing is changed."
)

_NO_TERMINAL = (
    "cortex-training login is interactive, so it needs a terminal. To connect from CI or a "
    f"script, set {_picker.HOST_ENV} and {_picker.PAT_ENV} instead (they're used when no "
    "Snowflake connection profile exists), or create a profile with `snow connection add`."
)


class LoginCancelled(Exception):
    """A required step was declined or failed; nothing more is changed."""


class ProfileFailed(LoginCancelled):
    """A profile could not sign in or reach Cortex Training; another can be picked."""

    def __init__(self, profile: str, message: str) -> None:
        super().__init__(message)
        self.profile = profile


class NotChecked(Exception):
    """The user chose not to check a profile that login just saved."""

    def __init__(self, profile: str) -> None:
        super().__init__(profile)
        self.profile = profile


# How a profile signs in, judged from its settings alone.
NO_PROMPTS = "Signs in without prompts"
PROMPTS = "Signs in with a prompt"
UNCLEAR = "Sign-in unclear"
WONT_SIGN_IN = "Likely won't sign in"
CANT_BE_USED = "Can't be used"

# Whether Cortex Training is enabled on an account.
CT_ENABLED = "enabled"
CT_NOT_ENABLED = "not enabled"
CT_COULDNT_CHECK = "couldn't check"


@dataclass(frozen=True)
class ProfileHint:
    """What a profile's settings say about it, without signing in."""

    verdict: str
    # "account · user · role · database"
    summary: str
    # How it signs in, in plain words.
    sign_in: str
    usable: bool = True


def describe_profile(values: Mapping[str, Any]) -> ProfileHint:
    """Judge a profile from its settings alone, without signing in."""
    missing = [key for key in ("account", "user") if not values.get(key)]
    if missing:
        return ProfileHint(CANT_BE_USED, f"no {' or '.join(missing)} set", "", usable=False)
    authenticator = str(values.get("authenticator") or "").strip().lower()
    has_key = any(values.get(key) for key in ("private_key_file", "private_key_path", "private_key"))
    if authenticator == "programmatic_access_token":
        verdict, sign_in = NO_PROMPTS, "a programmatic access token (no prompts)"
    elif values.get("token") and not authenticator:
        verdict = WONT_SIGN_IN
        sign_in = 'a token, but Snowflake will ignore it: the profile needs authenticator = "PROGRAMMATIC_ACCESS_TOKEN"'
    elif (authenticator == "snowflake_jwt" or has_key) and _encrypted_key_without_passphrase(values):
        verdict, sign_in = UNCLEAR, "an encrypted key file (may ask for its passphrase)"
    elif authenticator == "snowflake_jwt" or has_key:
        verdict, sign_in = NO_PROMPTS, "a key file (no prompts)"
    elif authenticator == "externalbrowser":
        verdict, sign_in = PROMPTS, "your browser (opens a browser window to sign in)"
    elif authenticator == "username_password_mfa":
        verdict, sign_in = PROMPTS, "a password and MFA (asks you to approve each sign-in)"
    elif authenticator in ("", "snowflake") and values.get("password"):
        verdict = UNCLEAR
        sign_in = "a password, or a token saved by `snow connection add` (may ask for an MFA approval)"
    elif authenticator == "oauth":
        verdict, sign_in = UNCLEAR, "an OAuth token (works until it expires)"
    else:
        verdict, sign_in = UNCLEAR, f"{authenticator or 'an unknown'} sign-in"
    database = values.get("database")
    if isinstance(database, str) and database.strip():
        database_part = f"database {database}" + (" (recommended)" if _same(database, DEFAULT_DATABASE) else "")
    else:
        database_part = "no database set"
    role = values.get("role")
    summary = " · ".join(
        [f"account {values['account']}", f"user {values['user']}", f"role {role}" if role else "your default role", database_part]
    )
    return ProfileHint(verdict, summary, sign_in)


def _encrypted_key_without_passphrase(values: Mapping[str, Any]) -> bool:
    """Whether the profile's key file is encrypted and the profile sets no passphrase."""
    if values.get("private_key_file_pwd"):
        return False
    path = values.get("private_key_file") or values.get("private_key_path")
    if not isinstance(path, str):
        return False
    try:
        with open(os.path.expanduser(path), encoding="utf-8", errors="replace") as handle:
            return "ENCRYPTED" in handle.readline()
    except OSError:
        return False


def validate_host(raw: str) -> str:
    """Return a bare account host from a host or a Snowsight URL, or raise ValueError."""
    host = raw.strip()
    for prefix in ("https://", "http://"):
        if host.lower().startswith(prefix):
            host = host[len(prefix):]
    host, _, path = host.partition("/")
    if host.lower() == "app.snowflake.com":
        # A pasted Snowsight URL, in either form:
        #   app.snowflake.com/ORG/ACCOUNT     -> ORG-ACCOUNT.snowflakecomputing.com
        #   app.snowflake.com/REGION/LOCATOR  -> LOCATOR.REGION.snowflakecomputing.com
        # Organization names are only letters and digits, so a "-" or "." in the
        # first part marks a region (us-west-2, us-east-2.aws, east-us-2.azure).
        parts = [part for part in path.split("#", 1)[0].split("/") if part]
        if len(parts) < 2:
            raise ValueError(
                "That Snowsight URL is missing a part. Expected app.snowflake.com/ORG/ACCOUNT, "
                "app.snowflake.com/REGION/LOCATOR, or an address such as "
                "ORG-ACCOUNT.snowflakecomputing.com."
            )
        first, second = parts[0], parts[1]
        if "-" in first or "." in first:
            host = f"{second}.{first}.snowflakecomputing.com"
        else:
            host = f"{first}-{second}.snowflakecomputing.com"
    if not host or " " in host or "." not in host:
        raise ValueError(f"{raw!r} isn't an account address. Expected ORG-ACCOUNT.snowflakecomputing.com.")
    return host


def account_from_host(host: str) -> str:
    return host.split(".", 1)[0]


@dataclass
class Check:
    """What signing in with a profile showed."""

    profile: str
    database: str
    schema: str
    role: str | None = None
    # Databases or schemas the role cannot see; None when the check could not run.
    missing: list[tuple[str, str]] | None = None
    error: str | None = None
    client: Any = None

    @property
    def usable(self) -> bool:
        return self.error is None and not self.missing


def open_profile_client(profile: str, database: str, schema: str) -> Any:
    """Sign in with ``profile``; the client never creates a database on its own."""
    from cortex_training.client import CortexTrainingClient

    return CortexTrainingClient._from_profile(profile, database=database, schema=schema)


class _Style:
    """ANSI styling, used only in a terminal and never when NO_COLOR is set."""

    def __init__(self, stream: TextIO) -> None:
        isatty = getattr(stream, "isatty", lambda: False)
        # The logo and section rules: in any terminal, so logs and scripts stay plain.
        self.decorate = bool(isatty())
        self.on = self.decorate and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.on else text

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def blue(self, text: str) -> str:
        return self._wrap("34", text)

    def ok(self, text: str) -> str:
        return f"{self.green('✓')} {text}"

    def bad(self, text: str) -> str:
        return f"{self.red('✗')} {text}"

    def unknown(self, text: str) -> str:
        return f"{self.yellow('?')} {text}"

    def warn(self, text: str) -> str:
        return f"{self.yellow('!')} {text}"


class _Prompter:
    def __init__(self, stdin: TextIO, stderr: TextIO) -> None:
        self.stdin = stdin
        self.stderr = stderr
        self.style = _Style(stderr)

    def say(self, text: str = "") -> None:
        self.stderr.write(f"{text}\n")
        self.stderr.flush()
        self._after_blank = not text

    def rule(self) -> None:
        """A blue line between sections, in a terminal."""
        if self.style.decorate:
            import shutil

            width = min(shutil.get_terminal_size((80, 24)).columns, 80)
            if not getattr(self, "_after_blank", True):
                self.say()
            self.say(self.style.blue("─" * width))

    def section(self, title: str) -> None:
        """Start a section: a blue line, then its title in bold."""
        self.rule()
        self.say(self.style.bold(title))

    def logo(self) -> None:
        if self.style.decorate:
            self.say(self.style.blue(self.style.bold(_LOGO)))

    def _prompt(self, text: str) -> None:
        self.stderr.write(self.style.bold(text))
        self.stderr.flush()
        self._after_blank = False

    def _readline(self) -> str:
        line = self.stdin.readline()
        if not line:
            raise LoginCancelled("no input")
        return line.rstrip("\n")

    def ask(self, label: str, *, default: str | None = None, validate: Callable[[str], str] | None = None) -> str:
        while True:
            suffix = f" [{default}]" if default else ""
            self._prompt(f"{label}{suffix}: ")
            value = self._readline().strip() or (default or "")
            if not value:
                self.say(f"{label} is required.")
                continue
            try:
                return validate(value) if validate else value
            except ValueError as exc:
                self.say(self.style.red(str(exc)))

    def optional(self, label: str) -> str | None:
        self._prompt(f"{label}: ")
        return self._readline().strip() or None

    def confirm(self, question: str, *, default: bool) -> bool:
        hint = "[Y/n]" if default else "[y/N]"
        while True:
            self._prompt(f"{question} {hint} ")
            answer = self._readline().strip().lower()
            if not answer:
                return default
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False

    def choose(self, question: str, options: list[str], *, default: int) -> int:
        """Return the 0-based index of the chosen option."""
        width = len(str(len(options)))
        for number, option in enumerate(options, start=1):
            self.say(f"  {self.style.cyan(str(number).rjust(width))}  {option}")
        self.say()
        while True:
            answer = self.ask(question, default=str(default + 1))
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return int(answer) - 1
            self.say(f"Enter a number from 1 to {len(options)}.")

    def secret(self, label: str) -> str:
        while True:
            if self.stdin is sys.stdin:
                value = getpass.getpass(self.style.bold(f"{label}: "), stream=self.stderr)
            else:
                self._prompt(f"{label}: ")
                value = self._readline()
            if value.strip():
                return value.strip()
            self.say(f"{label} is required.")


def _first_line(value: Any, limit: int = 160) -> str:
    text = str(value).strip().splitlines()[0] if str(value).strip() else ""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _same(a: Any, b: Any) -> bool:
    return isinstance(a, str) and isinstance(b, str) and a.strip().casefold() == b.strip().casefold()


def _home(path: Path | str) -> str:
    """Show ``path`` with the home folder as ``~``."""
    text, home = str(path), str(Path.home())
    return "~" + text[len(home):] if text.startswith(home + os.sep) else text


class _Login:
    def __init__(self, prompter: _Prompter, open_client: Callable[[str, str, str], Any]) -> None:
        self.p = prompter
        self.s = prompter.style
        self.open_client = open_client
        self.created_profiles: list[str] = []
        # Profiles login saved this time and already asked to check.
        self.check_asked: set[str] = set()
        self.created_database = False
        self.clients: list[Any] = []
        self.introduced = False
        # Account -> (Cortex Training status, reason); None until the user is asked.
        self.account_status: dict[str, tuple[str, str | None]] | None = None
        # Profiles whose own sign-in failed during that check, and why.
        self.profile_failures: dict[str, str] = {}

    def close(self) -> None:
        for client in self.clients:
            try:
                client.close()
            except Exception:
                pass

    def introduce(self) -> None:
        if not self.introduced:
            self.introduced = True
            self.p.section(_INTRO_TITLE)
            self.p.say()
            self.p.say(_INTRO)
            self.p.say()

    # ── Profiles ────────────────────────────────────────────────────────

    def profiles(self) -> dict[str, dict[str, Any]]:
        return _profiles.read_profiles()

    def _unused_name(self, base: str) -> str:
        names = set(self.profiles())
        if base not in names:
            return base
        number = 2
        while f"{base}-{number}" in names:
            number += 1
        return f"{base}-{number}"

    def _ask_new_name(self, default: str) -> str:
        def validate(name: str) -> str:
            if not _profiles._PROFILE_NAME.match(name):
                raise ValueError(f"{name!r} isn't a valid profile name. Use letters, digits, '_' or '-'.")
            if name in self.profiles():
                raise ValueError(f"Profile {name} already exists; choose another name.")
            return name

        return self.p.ask("Profile name", default=self._unused_name(default), validate=validate)

    def _ask_database(self) -> str:
        self.p.say(self.s.dim(_DATABASE_HELP))
        return self.p.ask("Database", default=DEFAULT_DATABASE)

    def _write(self, name: str, values: Mapping[str, Any]) -> None:
        path = _profiles.write_profile(name, values)
        self.created_profiles.append(name)
        self.p.say(self.s.ok(f"Saved profile {self.s.bold(name)} to {_home(path)}"))

    def create_from_scratch(self, name: str | None = None) -> str:
        self.p.section(_NEW_PROFILE_TITLE)
        self.p.say(_NEW_PROFILE)
        self.p.say()
        host = self.p.ask("Account address or Snowsight URL", validate=validate_host)
        user = self.p.ask("Snowflake user")
        role = self.p.optional("Role (press Enter to use your default role)")
        token = self.p.secret("Programmatic access token (hidden as you type)")
        name = name or self._ask_new_name(DEFAULT_PROFILE)
        database = self._ask_database()
        self._write(
            name,
            {
                "account": account_from_host(host),
                "host": host,
                "user": user,
                "role": role,
                "authenticator": "PROGRAMMATIC_ACCESS_TOKEN",
                "token": token,
                "database": database,
                "schema": DEFAULT_SCHEMA,
            },
        )
        return name

    def copy_profile(self, base: str) -> str:
        """Create a profile with ``base``'s sign-in and its own database."""
        self.p.say(f"The new profile signs in exactly like {self.s.bold(base)}; only the database is different.")
        name = self._ask_new_name(f"{base}-ct")
        database = self._ask_database()
        values = {key: value for key, value in self.profiles()[base].items() if key not in ("database", "schema")}
        self._write(name, {**values, "database": database, "schema": DEFAULT_SCHEMA})
        return name

    # ── Checking ────────────────────────────────────────────────────────

    def check(self, profile: str) -> Check:
        """Sign in with ``profile``, ask Cortex Training, then check its database, saying each step."""
        values = self.profiles().get(profile, {})
        database = str(values.get("database") or DEFAULT_DATABASE)
        schema = str(values.get("schema") or DEFAULT_SCHEMA)
        result = Check(profile, database, schema)
        self.p.section(f"Checking profile {profile}...")
        try:
            result.client = self.open_client(profile, database, schema)
        except Exception as exc:
            # The profile list shows this short form; the screen shows the whole
            # error, whose reason and next steps come after the first 160 characters.
            result.error = f"Profile {profile} couldn't sign in: {_first_line(exc)}"
            heading = f"Profile {profile} couldn't sign in:"
            self.p.say(f"  {self.s.bad(heading)}")
            for line in str(exc).strip().splitlines():
                self.p.say(f"    {line}" if line.strip() else "")
            return result
        self.clients.append(result.client)
        result.role = result.client._current_role()
        account = values.get("account", "the account")
        user = values.get("user", "?")
        role = f" (role {result.role})" if result.role else ""
        self.p.say(f"  {self.s.ok(f'Signed in to {account} as {user}{role}')}")
        # Ask Cortex Training before anything else, so nothing is created on an
        # account where it isn't enabled.
        result.error = self.service_error(result)
        if result.error:
            self.p.say(f"  {self.s.bad(result.error)}")
            return result
        self.p.say(f"  {self.s.ok('Cortex Training is enabled on this account')}")
        result.missing = _database_setup.missing(result.client)
        if result.missing is None:
            self.p.say(f"  {self.s.unknown(f'Could not check database {database}; continuing')}")
        elif result.missing:
            who = f"role {result.role}" if result.role else "your role"
            self.p.say(f"  {self.s.bad(f'Database {database} is missing, or {who} cannot access it')}")
        else:
            self.p.say(f"  {self.s.ok(f'Database {database} is ready')}")
        return result

    @staticmethod
    def service_error(check: Check) -> str | None:
        """Return why Cortex Training can't be used with this profile, or None if it can."""
        from cortex_training.client import _response_error_code

        try:
            check.client.get_capacity()
            return None
        except Exception as exc:
            code = _response_error_code(getattr(exc, "response", None))
            if code == _NOT_RESOLVED_CODE:
                return None  # Cortex Training answered; only the database is missing.
            account = _profiles.read_profiles().get(check.profile, {}).get("account", "this account")
            if code == _NOT_ENABLED_CODE:
                return f"Cortex Training isn't enabled on account {account}"
            return f"Profile {check.profile} signs in, but Cortex Training didn't answer: {_first_line(exc)}"

    def same_identity(self, profile: str, role: str | None) -> list[str]:
        """Other profiles with a database that sign in as the same account, user, and role."""
        profiles = self.profiles()
        base = profiles[profile]
        base_role = base.get("role") or role
        matches = []
        for name, values in sorted(profiles.items()):
            if name == profile or name in self.created_profiles or not isinstance(values.get("database"), str):
                continue
            if not (_same(values.get("account"), base.get("account")) and _same(values.get("user"), base.get("user"))):
                continue
            # A profile without a role signs in with the user's default role, like this one.
            if base_role and not _same(values.get("role") or role, base_role):
                continue
            matches.append(name)
        return matches

    @staticmethod
    def _visible(check: Check, database: str) -> bool:
        try:
            return _database_setup.is_listed(check.client, "database", database)
        except Exception:
            return False

    # ── Which accounts have Cortex Training ─────────────────────────────

    def _account(self, profile: str) -> str:
        return str(self.profiles().get(profile, {}).get("account") or "").strip().upper()

    def _probe_account(self, candidates: list[str]) -> tuple[str, str | None, dict[str, str]]:
        """Ask Cortex Training through the first candidate that signs in.

        Returns the account's status, a reason, and the sign-in error of each
        candidate that was tried and failed.
        """
        failures: dict[str, str] = {}
        for profile in candidates:
            status, reason, signed_in = self._probe_profile(profile)
            if not signed_in:
                failures[profile] = reason or "couldn't sign in"
                continue
            return status, reason, failures
        return CT_COULDNT_CHECK, "no profile on this account signed in without asking for something", failures

    def _probe_profile(self, profile: str) -> tuple[str, str | None, bool]:
        """Sign in with ``profile`` and ask Cortex Training: (status, reason, signed in)."""
        values = self.profiles()[profile]
        database = str(values.get("database") or DEFAULT_DATABASE)
        schema = str(values.get("schema") or DEFAULT_SCHEMA)
        try:
            client = self.open_client(profile, database, schema)
        except Exception as exc:
            return CT_COULDNT_CHECK, f"couldn't sign in: {_first_line(exc)}", False
        try:
            error = self.service_error(Check(profile, database, schema, client=client))
        finally:
            try:
                client.close()
            except Exception:
                pass
        if error is None:
            return CT_ENABLED, None, True
        if "isn't enabled" in error:
            return CT_NOT_ENABLED, None, True
        return CT_COULDNT_CHECK, _first_line(error), True

    def check_accounts(self, names: list[str], hints: Mapping[str, ProfileHint]) -> None:
        """Offer to check, once per account, which accounts have Cortex Training."""
        if self.account_status is not None:
            return
        self.account_status = {}
        by_account: dict[str, list[str]] = {}
        skipped = []
        for name in names:
            if not hints[name].usable:
                continue
            if hints[name].verdict == NO_PROMPTS:
                by_account.setdefault(self._account(name), []).append(name)
            else:
                skipped.append(name)
        if not by_account:
            return
        self.p.section("Which of your accounts have Cortex Training?")
        self.p.say(
            "It's turned on per account, and the only way to know is to sign in. Cortex Training\n"
            "can sign in once to each account below to find out. This only reads; nothing is changed."
        )
        for account, profiles in sorted(by_account.items()):
            using = f"using profile{'s' if len(profiles) > 1 else ''} {', '.join(profiles)}"
            self.p.say(f"  - {self.s.bold(account)}  {self.s.dim(f'({using})')}")
        if skipped:
            self.p.say(
                self.s.dim(
                    f"Not checked now: {', '.join(skipped)}. Signing in with these may ask you for something\n"
                    "(a passphrase, a password, an MFA approval, or a browser window), so each is checked\n"
                    "only if you pick it."
                )
            )
        if not self.p.confirm("Check these accounts now?", default=True):
            self.p.say()
            return
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(8, len(by_account))) as pool:
            results = dict(zip(by_account, pool.map(self._probe_account, list(by_account.values()))))
        self.account_status = {account: (status, reason) for account, (status, reason, _) in results.items()}
        self.profile_failures = {
            profile: reason for _, _, failures in results.values() for profile, reason in failures.items()
        }
        self.p.say()

    def _status_line(self, name: str, hint: ProfileHint, failed: Mapping[str, str]) -> str:
        if not hint.usable:
            return self.s.bad(f"Can't be used: {hint.summary}")
        if name in failed:
            return self.s.bad(f"Didn't work just now: {failed[name]}")
        account = self._account(name)
        if name in self.profile_failures:
            return self.s.unknown(f"Couldn't check Cortex Training: {self.profile_failures[name]}")
        status = (self.account_status or {}).get(account)
        if status is None:
            if hint.verdict == NO_PROMPTS and self.account_status is not None:
                return self.s.unknown("Cortex Training not checked (you skipped the check); checked when you pick it")
            return self.s.unknown("Cortex Training not checked yet; checked when you pick it")
        label, reason = status
        if label == CT_ENABLED:
            return self.s.ok(self.s.green(f"Cortex Training is enabled on {account}"))
        if label == CT_NOT_ENABLED:
            return self.s.bad(self.s.red(f"Cortex Training is not enabled on {account}"))
        return self.s.unknown(f"Couldn't check Cortex Training: {reason}")

    # ── Flow ────────────────────────────────────────────────────────────

    def pick(self, current: str | None, failed: Mapping[str, str] | None = None) -> str:
        """List the profiles, and return the one picked (or a newly created one)."""
        failed = failed or {}
        self.introduce()
        profiles = self.profiles()
        if not profiles:
            self.p.say("You don't have a Snowflake connection profile yet.")
            self.p.say()
            return self.create_from_scratch()
        names = sorted(profiles)
        hints = {name: describe_profile(profiles[name]) for name in names}
        # A profile that just failed would only fail again, so it isn't used to check its account.
        self.check_accounts([name for name in names if name not in failed], hints)

        def enabled(name: str) -> bool:
            return (self.account_status or {}).get(self._account(name), ("",))[0] == CT_ENABLED

        def blocked(name: str) -> str | None:
            if not hints[name].usable:
                return hints[name].summary
            if (self.account_status or {}).get(self._account(name), ("",))[0] == CT_NOT_ENABLED:
                return f"Cortex Training isn't enabled on account {self._account(name)}"
            return None

        width = max(len(name) for name in names)
        indent = " " * (width + 2)
        options = []
        for name in names:
            hint = hints[name]
            label = self.s.bold(name.ljust(width))
            if name == current:
                label += self.s.dim(" (current)")
            lines = [f"{label}  {self._status_line(name, hint, failed)}"]
            if hint.usable:
                lines.append(f"{indent}     {self.s.dim(hint.summary)}")
                lines.append(f"{indent}     {self.s.dim('signs in with ' + hint.sign_in)}")
            options.append("\n".join(lines))
        options.append(self.s.bold("Create a new profile"))

        def suitable(name: str) -> bool:
            return blocked(name) is None and name not in failed

        preferred = [current, _profiles.default_profile_name()]

        def recommended_database(name: str) -> bool:
            return _same(profiles[name].get("database"), DEFAULT_DATABASE)

        ranked = (
            [n for n in preferred if n in hints and enabled(n)]
            + [n for n in names if enabled(n) and recommended_database(n)]
            + [n for n in names if enabled(n)]
            + [n for n in preferred if n in hints]
            + [n for n in names if hints[n].verdict == NO_PROMPTS]
            + names
        )
        default = next((names.index(n) for n in ranked if suitable(n)), len(names))
        self.p.rule()
        self.p.say(self.s.yellow(self.s.bold(_IMPORTANT)))
        self.p.say()
        self.p.say(self.s.bold("Your Snowflake connection profiles:"))
        self.p.say()
        while True:
            index = self.p.choose("Which profile should Cortex Training use?", options, default=default)
            if index == len(names):
                self.p.say()
                return self.create_from_scratch()
            reason = blocked(names[index])
            if reason is None:
                self.p.say()
                return names[index]
            self.p.say(self.s.red(f"Profile {names[index]} can't be used: {reason}. Pick another."))

    def settle(self, profile: str) -> Check:
        """Check ``profile`` and resolve an unusable database; return the final, usable check.

        A profile login just saved is checked only if the user agrees; otherwise
        this raises :class:`NotChecked`.
        """
        if profile in self.created_profiles and profile not in self.check_asked:
            self.check_asked.add(profile)
            self.p.say()
            self.p.say(_CHECK_NEW_PROFILE)
            if not self.p.confirm("Check it now?", default=True):
                raise NotChecked(profile)
        check = self.check(profile)
        if check.error:
            self.offer_delete(profile, signed_in=check.client is not None)
            raise ProfileFailed(profile, check.error)
        values = self.profiles()[profile]
        # A profile's own database may be a shared work database, so offer a
        # separate one, unless it already is Cortex Training's default.
        if (
            profile not in self.created_profiles
            and isinstance(values.get("database"), str)
            and not _same(values["database"], DEFAULT_DATABASE)
        ):
            self.p.rule()
            self.p.say(
                f"Profile {self.s.bold(profile)} uses database {self.s.bold(check.database)}. Cortex Training "
                "would store its training jobs and\ncheckpoints there. If that database is shared with other "
                "work, use a separate one instead."
            )
            choice = self.p.choose(
                "Which database should Cortex Training use?",
                [f"Use {check.database}", f"Use a separate database (recommended: {DEFAULT_DATABASE})"],
                default=0,
            )
            if choice == 1:
                self.p.say()
                return self.settle(self.copy_profile(profile))
        if not check.missing:
            return check
        return self.fix_database(check)

    def fix_database(self, check: Check) -> Check:
        who = f"role {check.role}" if check.role else "your role"
        self.p.say()
        if check.profile not in self.created_profiles:
            for other in self.same_identity(check.profile, check.role):
                database = self.profiles()[other]["database"]
                if self._visible(check, database) and self.p.confirm(
                    f"Profile {self.s.bold(other)} signs in as the same user and role on the same account, and "
                    f"its database {database} works. Use {other} instead?",
                    default=True,
                ):
                    self.p.say()
                    return self.settle(other)
            self.p.say(
                f"You can create a new profile that signs in exactly like {self.s.bold(check.profile)} but "
                "uses its own database.\nYour existing profile stays as it is."
            )
            if not self.p.confirm("Create the new profile?", default=False):
                raise LoginCancelled(
                    f"Profile {check.profile} has no database Cortex Training can use, so nothing was saved"
                )
            self.p.say()
            return self.settle(self.copy_profile(check.profile))
        statements = [f"{statement};" for statement in _database_setup.create_statements(check.client, check.missing or [])]
        self.p.say("Cortex Training can create it now by running:")
        for statement in statements:
            self.p.say(f"    {self.s.cyan(statement)}")
        if not self.p.confirm("Run it?", default=False):
            raise LoginCancelled(
                f"Database {check.database} doesn't exist yet. To create it yourself, run in Snowsight: {statements[0]}"
            )
        try:
            _database_setup.create(check.client, check.missing or [])
        except _database_setup.DatabaseSetupError as exc:
            raise LoginCancelled(str(exc)) from exc
        self.created_database = True
        check.missing = _database_setup.missing(check.client)
        if check.missing:
            raise LoginCancelled(f"Database {check.database} still isn't visible to {who} after creating it")
        self.p.say(f"  {self.s.ok(f'Created database {check.database}')}")
        return check

    def verify_service(self, check: Check) -> None:
        """Confirm the final setup with one more capacity check."""
        try:
            check.client.get_capacity()
        except Exception as exc:
            reason = self.service_error(check) or (
                f"Cortex Training can't find database {check.database} or schema {check.schema} "
                f"with profile {check.profile}: {_first_line(exc)}"
            )
            self.p.say(f"  {self.s.bad(reason)}")
            self.offer_delete(check.profile, signed_in=True)
            raise ProfileFailed(check.profile, reason) from exc

    def offer_delete(self, profile: str, *, signed_in: bool) -> None:
        """Offer to delete a profile login saved this time that turned out not to work."""
        if profile not in self.created_profiles:
            return
        self.p.say()
        if not signed_in:
            self.p.say("Please check the account, user, and PAT you entered.")
        if self.p.confirm(f"Delete profile {profile} for now?", default=True):
            _profiles.remove_profile(profile)
            self.created_profiles.remove(profile)
            self.p.say(f"  {self.s.ok(f'Deleted profile {profile}')}")


def run_login(
    name: str | None,
    *,
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    interactive: bool | None = None,
    open_client: Callable[[str, str, str], Any] = open_profile_client,
) -> int:
    """Run ``cortex-training login [NAME]``; return the exit code."""
    if interactive is None:
        interactive = stdin.isatty() and stderr.isatty()
    if not interactive:
        stderr.write(f"error: {_NO_TERMINAL}\n")
        return 2
    prompter = _Prompter(stdin, stderr)
    style = prompter.style
    login = _Login(prompter, open_client)
    shell = _environment.shell_name()
    findings = _environment.find()
    prompter.logo()
    report_environment(prompter, findings, shell)
    try:
        settings, config_file = _picker.load_config()
        profiles = login.profiles()
        current = settings.get("profile")
        check: Check | None = None
        if name is not None:
            if not _profiles._PROFILE_NAME.match(name):
                raise LoginCancelled(f"{name!r} isn't a valid profile name. Use letters, digits, '_' or '-'.")
            if name in profiles:
                profile = name
            else:
                login.introduce()
                profile = login.create_from_scratch(name)
        elif current in profiles:
            prompter.section(f"Cortex Training is set up to use profile {current}.")
            existing = login.check(current)
            prompter.say()
            if existing.usable:
                if prompter.confirm(f"It works. Keep using {current}?", default=True):
                    check, profile = existing, current
                else:
                    prompter.say()
                    profile = login.pick(current)
            else:
                prompter.say(f"Profile {current} doesn't work right now. Let's pick another one, or fix it.")
                prompter.say()
                profile = login.pick(current)
        else:
            if current:
                prompter.say(
                    f"Cortex Training was set up to use profile {style.bold(current)}, which doesn't exist any more."
                )
                prompter.say()
            profile = login.pick(current)
        failed: dict[str, str] = {}
        unchecked: str | None = None
        while True:
            try:
                if check is None:
                    check = login.settle(profile)
                login.verify_service(check)
                break
            except NotChecked as exc:
                unchecked = exc.profile
                break
            except ProfileFailed as exc:
                if name is not None:
                    raise
                # This profile isn't used; let the user pick another.
                failed[exc.profile] = str(exc)
                prompter.say()
                prompter.say(f"Profile {style.bold(exc.profile)} can't be used. Pick another one, or create a new one.")
                prompter.say()
                check = None
                profile = login.pick(current, failed)
        chosen = unchecked or check.profile
        saved = _picker.save_profile(chosen)
    except KeyboardInterrupt:
        stderr.write("\nlogin cancelled. Nothing more was changed.\n")
        _note_created(login, stderr)
        return 130
    except (LoginCancelled, _picker.PickerError, _profiles.ProfileFileError) as exc:
        stderr.write(f"\n{style.bad(str(exc))}\n")
        _note_created(login, stderr)
        return 1
    finally:
        login.close()

    prompter.rule()
    if not style.decorate:
        prompter.say()
    if unchecked:
        prompter.say(f"Saved, but not checked. Cortex Training will use profile {style.bold(chosen)}.")
        prompter.say(style.dim(f"(Saved in {_home(saved)}.)"))
        prompter.say()
        prompter.say(f"To check it later, run:  {style.cyan('cortex-training login')}")
    else:
        prompter.say(style.green(style.bold("You're all set.")) + f" Cortex Training will use profile {style.bold(chosen)}.")
        prompter.say(style.dim(f"(Saved in {_home(saved)}.)"))
        prompter.say()
        prompter.say(f"Try it:  {style.cyan('cortex-training capacity')}")
    remind_environment(prompter, findings, shell)
    if not getattr(stdout, "isatty", lambda: False)():
        values = _profiles.read_profiles().get(chosen, {})
        json.dump(
            {
                "profile": chosen,
                "profile_file": str(_profiles.profile_file()),
                "account": values.get("account"),
                "user": values.get("user"),
                "role": check.role if check else values.get("role"),
                "database": check.database if check else values.get("database", DEFAULT_DATABASE),
                "schema": check.schema if check else values.get("schema", DEFAULT_SCHEMA),
                "checked": unchecked is None,
                "config": str(saved),
                "created_profiles": login.created_profiles,
                "created_database": login.created_database,
            },
            stdout,
            indent=2,
        )
        stdout.write("\n")
    return 0


def report_environment(prompter: _Prompter, findings: list[_environment.Finding], shell: str) -> None:
    """Say which shell variables could override the setup, and how to clear them."""
    s = prompter.style
    prompter.section("Checking your shell for settings that could override this setup...")
    if not findings:
        prompter.say(f"  {s.ok('Nothing in your shell overrides it.')}")
        prompter.say()
        return
    for finding in findings:
        line = f"{finding.name} = {finding.shown}"
        mark = s.bad if finding.severity == _environment.OVERRIDES else s.warn
        prompter.say(f"  {mark(s.bold(line))}")
        prompter.say(f"      {s.dim(finding.effect)}")
        if finding.severity == _environment.OVERRIDES:
            prompter.say(f"      To clear it in this shell:  {s.cyan(_environment.clear_command(finding.name, shell))}")
    if any(finding.severity == _environment.OVERRIDES for finding in findings):
        prompter.say(
            "The settings marked ✗ make cortex-training commands ignore the profile you set up here.\n"
            "You can continue now and clear them afterwards."
        )
    prompter.say()


def remind_environment(prompter: _Prompter, findings: list[_environment.Finding], shell: str) -> None:
    """After setup, repeat any variable that still overrides it."""
    s = prompter.style
    overriding = [finding for finding in findings if finding.severity == _environment.OVERRIDES]
    if not overriding:
        return
    prompter.rule()
    prompter.say(s.red(s.bold("Heads-up: your shell still overrides this setup.")))
    for finding in overriding:
        prompter.say(f"  {s.bad(finding.name)}  clear it with:  {s.cyan(_environment.clear_command(finding.name, shell))}")
    prompter.say(s.dim(f"If it comes back in new terminals, remove it from {_environment.startup_hint(shell)}."))


def _note_created(login: _Login, stderr: TextIO) -> None:
    if login.created_profiles:
        stderr.write(
            f"Saved profile(s) {', '.join(login.created_profiles)}; run cortex-training login again to finish.\n"
        )
