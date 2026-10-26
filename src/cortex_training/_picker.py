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

"""Pick the Snowflake profile, database, and settings for a Cortex Training client.

This is the picker behind :meth:`CortexTrainingClient.from_configs`. It reads
only Snowflake connection profiles, the Cortex Training ``config.json``, and a
few ``CORTEX_TRAINING_*`` variables. Credentials, the account, and the database
belong to the Snowflake profile; ``config.json`` holds only which profile to
use and tool settings, so it never holds a secret.

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
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DATABASE = "CORTEX_TRAINING_DB"
DEFAULT_SCHEMA = "PUBLIC"

HOST_ENV = "CORTEX_TRAINING_HOST"
PAT_ENV = "CORTEX_TRAINING_PAT"
DATABASE_ENV = "CORTEX_TRAINING_DATABASE"
SCHEMA_ENV = "CORTEX_TRAINING_SCHEMA"

# Settings config.json may hold, and the type each must have.
_SETTINGS: dict[str, tuple[type, ...]] = {
    "profile": (str,),
    "endpoint": (str,),
    "poll_interval": (int, float),
    "poll_timeout": (int, float),
    "verify_ssl": (bool,),
}
# Keys that belong in a Snowflake profile, not in config.json.
_PROFILE_KEYS = ("host", "pat", "base_url", "account", "user", "password", "token", "database", "schema")

_SETUP_HINT = (
    "Run `cortex-training login` to set one up, or create one with `snow connection add` "
    "and pass connection=NAME."
)


class PickerError(ValueError):
    """The inputs do not pick a usable connection."""


class NoProfileError(PickerError):
    """No Snowflake connection profile is set up, and no CI credentials are set."""


@dataclass(frozen=True)
class Resolution:
    """What the picker chose, and where each choice came from."""

    profile: str | None
    host: str | None
    pat: str | None = field(repr=False)
    database: str
    schema: str
    settings: dict[str, Any]
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
    """The default ``config.json``: ``~/.config/cortex-training/config.json``, honoring ``XDG_CONFIG_HOME``."""
    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return base / "cortex-training" / "config.json"


def load_config(path: str | os.PathLike[str] | None = None) -> tuple[dict[str, Any], Path]:
    """Read and validate ``config.json``.

    An explicit ``path`` must exist; a missing default file means no settings.
    """
    explicit = path is not None
    resolved = Path(path).expanduser() if explicit else config_path()
    if not resolved.is_file():
        if explicit:
            raise PickerError(f"config.json not found: {resolved}")
        return {}, resolved
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PickerError(f"invalid JSON in {resolved}: {exc}") from exc
    if not isinstance(data, dict):
        raise PickerError(f"{resolved} must hold a JSON object")
    misplaced = sorted(key for key in data if key in _PROFILE_KEYS)
    if misplaced:
        raise PickerError(
            f"{resolved} sets {', '.join(misplaced)}, which belong in your Snowflake "
            "connection profile, not in config.json. config.json holds only "
            f"{', '.join(_SETTINGS)}."
        )
    unknown = sorted(key for key in data if key not in _SETTINGS)
    if unknown:
        raise PickerError(
            f"{resolved} has unknown keys {', '.join(unknown)}; allowed keys are {', '.join(_SETTINGS)}."
        )
    for key, value in data.items():
        allowed = _SETTINGS[key]
        if isinstance(value, bool) and bool not in allowed or not isinstance(value, allowed):
            raise PickerError(f"{resolved}: {key} must be {' or '.join(t.__name__ for t in allowed)}")
        if key in ("poll_interval", "poll_timeout") and value <= 0:
            raise PickerError(f"{resolved}: {key} must be greater than 0")
        if key in ("profile", "endpoint") and not value.strip():
            raise PickerError(f"{resolved}: {key} must not be empty")
    return data, resolved


def save_profile(profile: str) -> Path:
    """Set ``"profile"`` in the default ``config.json``, keeping its other settings."""
    settings, path = load_config()
    settings["profile"] = profile
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def snowflake_profiles() -> tuple[str, Mapping[str, Mapping[str, Any]]]:
    """Return the Connector's default profile name and its profiles, as the Connector reads them."""
    from snowflake.connector.config_manager import CONFIG_MANAGER

    return str(CONFIG_MANAGER["default_connection_name"]), CONFIG_MANAGER["connections"]


def _env(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    if value is None:
        return None
    return value.strip() or None


def resolve(
    *,
    connection: str | None = None,
    ct_config: str | os.PathLike[str] | None = None,
    database: str | None = None,
    schema: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> Resolution:
    """Pick the profile (or CI credentials), database, schema, and settings."""
    environ = os.environ if environ is None else environ
    settings, settings_path = load_config(ct_config)
    settings_label = f"from {settings_path}" if ct_config is not None else "from config.json"
    default_name, profiles = snowflake_profiles()
    env_host, env_pat = _env(environ, HOST_ENV), _env(environ, PAT_ENV)

    def known() -> str:
        return ", ".join(sorted(profiles)) or "none"

    profile: str | None = None
    sources: dict[str, str] = {}
    if connection is not None and connection.strip():
        profile, sources["profile"] = connection.strip(), "passed in"
    elif "profile" in settings:
        profile, sources["profile"] = settings["profile"].strip(), settings_label
    elif default_name in profiles:
        profile, sources["profile"] = default_name, "Snowflake default profile"

    if profile is not None:
        if profile not in profiles:
            hint = " Run `cortex-training login` to pick or create one." if sources["profile"] != "passed in" else ""
            raise PickerError(
                f"Snowflake connection profile {profile!r} ({sources['profile']}) does not exist; "
                f"known profiles: {known()}.{hint}"
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
            raise NoProfileError(f"No Snowflake connection profile is set up. {_SETUP_HINT}")
        host, pat = env_host, env_pat
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
    if sources["schema"] == f"from {SCHEMA_ENV}":
        logger.warning("Using schema %s from %s.", resolved_schema, SCHEMA_ENV)

    return Resolution(
        profile=profile,
        host=host,
        pat=pat,
        database=resolved_database,
        schema=resolved_schema,
        settings={key: value for key, value in settings.items() if key != "profile"},
        sources=sources,
    )
