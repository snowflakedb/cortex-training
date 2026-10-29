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

"""Pick the Snowflake profile and database for a Cortex Training client.

This is the picker behind :meth:`CortexTrainingClient.from_configs`. It reads
only Snowflake connection profiles, the Cortex Training ``config.json``, and a
few ``CORTEX_TRAINING_*`` variables. Credentials, the account, and the database
belong to the Snowflake profile. ``config.json`` holds only ``"profile"``, the
name of the profile to use, so it never holds a secret. It lives in the
Snowflake folder (``<Snowflake folder>/cortex-training/config.json``), so
``SNOWFLAKE_HOME`` moves it together with the profiles.

Profile, first match wins:

1. ``connection=``.
2. ``"profile"`` in ``config.json``.
3. Snowflake's default profile (the Connector's own rule).
4. ``CORTEX_TRAINING_HOST`` + ``CORTEX_TRAINING_PAT``, only when no Snowflake
   profile exists, for CI. When a profile is used they are ignored, and a
   warning says so.

Database: ``database=``, then ``CORTEX_TRAINING_DATABASE``, then the profile's
database, then ``CORTEX_TRAINING_DB``. Schema follows the same order with
``CORTEX_TRAINING_SCHEMA`` and defaults to ``PUBLIC``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from cortex_training import _profiles

logger = logging.getLogger(__name__)

DEFAULT_DATABASE = "CORTEX_TRAINING_DB"
DEFAULT_SCHEMA = "PUBLIC"

HOST_ENV = "CORTEX_TRAINING_HOST"
PAT_ENV = "CORTEX_TRAINING_PAT"
DATABASE_ENV = "CORTEX_TRAINING_DATABASE"
SCHEMA_ENV = "CORTEX_TRAINING_SCHEMA"

# Keys that belong in a Snowflake profile, not in config.json.
_PROFILE_KEYS = ("host", "pat", "base_url", "account", "user", "password", "token", "database", "schema")

# An account host such as ACCOUNT.snowflakecomputing.com, with an optional port.
_HOST = re.compile(r"[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)+(:[0-9]{1,5})?")

_PICK_ANOTHER = "Pass connection=NAME (--connection NAME) to use another profile."
_SETUP_HINT = (
    "Create a Snowflake connection profile (for example with `snow connection add`) "
    "and pass connection=NAME, or name it in \"profile\" in config.json."
)


class PickerError(ValueError):
    """The inputs do not pick a usable connection."""


@dataclass(frozen=True)
class Resolution:
    """What the picker chose, and where each choice came from."""

    profile: str | None
    host: str | None
    pat: str | None = field(repr=False)
    database: str
    schema: str
    sources: dict[str, str]

    def describe(self) -> str:
        """One line naming the target, for example ``profile prod (from config.json), db TRAIN_DB (...)``."""
        who = (
            f"profile {self.profile} ({self.sources['profile']})"
            if self.profile is not None
            else f"host {self.host} ({self.sources['host']})"
        )
        return (
            f"{who}, db {self.database} ({self.sources['database']}), "
            f"schema {self.schema} ({self.sources['schema']})"
        )


def config_path() -> Path:
    """``config.json`` in the Snowflake folder the profiles come from: ``<folder>/cortex-training/config.json``."""
    return _profiles.snowflake_dir() / "cortex-training" / "config.json"


def load_config() -> tuple[dict[str, Any], Path]:
    """Read and validate ``config.json``. A missing file means no saved profile."""
    path = config_path()
    if not path.is_file():
        return {}, path
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PickerError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PickerError(f"{path} must hold a JSON object")
    misplaced = sorted(key for key in data if key in _PROFILE_KEYS)
    if misplaced:
        raise PickerError(
            f"{path} sets {', '.join(misplaced)}, which belong in your Snowflake "
            'connection profile, not in config.json. It holds only "profile".'
        )
    unknown = sorted(key for key in data if key != "profile")
    if unknown:
        raise PickerError(
            f"{path} has unknown keys {', '.join(unknown)}; it holds only \"profile\". "
            "Pass other settings (poll_interval=..., ...) to from_configs()."
        )
    if "profile" in data and not (isinstance(data["profile"], str) and data["profile"].strip()):
        raise PickerError(f"{path}: profile must be a non-empty string")
    return data, path


def snowflake_profiles() -> tuple[str, str, Mapping[str, Mapping[str, Any]]]:
    """Return the default profile name, where that name comes from, and every profile.

    Read fresh from the files on each call, the way the Connector reads them.
    """
    try:
        profiles = _profiles.read_profiles()
        default_name = _profiles.default_profile_name()
    except _profiles.ProfileFileError as exc:
        raise PickerError(str(exc)) from exc
    if (os.environ.get(_profiles.DEFAULT_PROFILE_ENV) or "").strip():
        default_source = _profiles.DEFAULT_PROFILE_ENV
    elif default_name != "default":
        default_source = f"default_connection_name in {_profiles.snowflake_dir() / _profiles.CONFIG_FILE}"
    else:
        default_source = "no default is set, so 'default'"
    return default_name, default_source, profiles


def normalize_host(value: str, source: str) -> str:
    """Return the account host in ``value``, allowing ``https://`` and a trailing ``/``.

    Anything that still isn't a host afterwards is rejected, naming ``source``.
    """
    host = value.strip()
    if host[:8].lower() == "https://":
        host = host[8:]
    host = host.rstrip("/")
    if not _HOST.fullmatch(host):
        raise PickerError(
            f"{source} must be your account host, like ACCOUNT.snowflakecomputing.com; got {value!r}."
        )
    return host


def _env(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    if value is None:
        return None
    return value.strip() or None


def resolve(
    *,
    connection: str | None = None,
    database: str | None = None,
    schema: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> Resolution:
    """Pick the profile (or CI credentials), database, and schema."""
    environ = os.environ if environ is None else environ
    settings, settings_path = load_config()
    default_name, default_source, profiles = snowflake_profiles()
    env_host, env_pat = _env(environ, HOST_ENV), _env(environ, PAT_ENV)

    def known() -> str:
        return ", ".join(sorted(profiles)) or "none"

    profile: str | None = None
    sources: dict[str, str] = {}
    if connection is not None and connection.strip():
        profile, sources["profile"] = connection.strip(), "passed in"
    elif "profile" in settings:
        profile, sources["profile"] = settings["profile"].strip(), f"saved in {settings_path}"
    elif default_name in profiles:
        profile, sources["profile"] = default_name, "the Snowflake default profile"
    elif profiles:
        # The Connector would fail here too, so say what is actually wrong
        # rather than falling through to the CI variables.
        raise PickerError(
            f"Snowflake default profile {default_name!r} ({default_source}) does not exist; "
            f"known profiles: {known()}. {_PICK_ANOTHER}"
        )

    if profile is not None:
        if profile not in profiles:
            raise PickerError(
                f"Snowflake connection profile {profile!r} ({sources['profile']}) does not exist; "
                f"known profiles: {known()}. {_PICK_ANOTHER}"
            )
        if env_host or env_pat:
            logger.warning(
                "%s/%s are set but ignored: profile %s is used. They apply only when no "
                "Snowflake connection profile exists (CI).",
                HOST_ENV,
                PAT_ENV,
                profile,
            )
        host = pat = None
        profile_values = profiles[profile]
    else:
        if not (env_host and env_pat):
            missing = PAT_ENV if env_host else HOST_ENV if env_pat else None
            if missing:
                raise PickerError(f"{missing} is not set; CI connections need both {HOST_ENV} and {PAT_ENV}.")
            raise PickerError(f"No Snowflake connection profile is set up. {_SETUP_HINT}")
        host, pat = normalize_host(env_host, HOST_ENV), env_pat
        sources["host"] = f"from {HOST_ENV}"
        profile_values = {}

    def pick(passed: str | None, env_name: str, profile_key: str, default: str) -> tuple[str, str]:
        if passed is not None and passed.strip():
            return passed.strip(), "passed in"
        env_value = _env(environ, env_name)
        if env_value:
            return env_value, f"from {env_name}"
        profile_value = profile_values.get(profile_key)
        if isinstance(profile_value, str) and profile_value.strip():
            return profile_value.strip(), f"from profile {profile}"
        return default, "default"

    resolved_database, sources["database"] = pick(database, DATABASE_ENV, "database", DEFAULT_DATABASE)
    resolved_schema, sources["schema"] = pick(schema, SCHEMA_ENV, "schema", DEFAULT_SCHEMA)
    if sources["database"] == f"from {DATABASE_ENV}":
        logger.warning("Using database %s from %s.", resolved_database, DATABASE_ENV)

    return Resolution(
        profile=profile,
        host=host,
        pat=pat,
        database=resolved_database,
        schema=resolved_schema,
        sources=sources,
    )
