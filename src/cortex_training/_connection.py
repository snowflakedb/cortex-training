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

"""Pick a Cortex Training connection from arguments, env vars, and files.

This is the single resolver behind the CLI, the TUI, ``connect()``, and the
recipes, so every entry point picks the same connection for the same inputs.

Sources, first match wins:

1. Explicit arguments: a connection profile, a JSON config, or ``base_url`` /
   ``host`` + ``pat``. Passing a profile together with any other source is an
   error.
2. ``CORTEX_TRAINING_CONNECTION``, then ``CORTEX_TRAINING_CONFIG``.
3. A complete direct connection from env vars (``CORTEX_TRAINING_BASE_URL``,
   or ``CORTEX_TRAINING_HOST`` + ``CORTEX_TRAINING_PAT``) with a database.
4. The config or profile remembered by ``cortex-training login``.
5. A complete direct connection from env vars without a database.
6. The Snowflake Connector's default ``connections.toml`` profile.

Explicit arguments always beat env vars. Env vars only fill fields the chosen
source leaves unset, and never switch a ``base_url`` connection to PAT auth or
the other way around.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from cortex_training.client import CortexTrainingClient

logger = logging.getLogger(__name__)

LOGIN_STATE_ENV = "CORTEX_TRAINING_LOGIN_FILE"

DEFAULT_SCHEMA = "PUBLIC"
DEFAULT_ENDPOINT = "cortex-training"
DEFAULT_POLL_INTERVAL = 0.5
DEFAULT_POLL_TIMEOUT = 1800.0

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


@dataclass(frozen=True)
class ConnectionSettings:
    """The connection an entry point should open, and where it came from."""

    source: str
    use_connection_profile: bool
    connection_name: str | None
    config_path: str | None
    base_url: str | None
    host: str | None
    pat: str | None
    database: str | None
    schema: str | None
    endpoint: str
    poll_interval: float
    poll_timeout: float
    verify_ssl: bool


def env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def normalize_host(host: str) -> str:
    for prefix in ("https://", "http://"):
        if host.startswith(prefix):
            host = host[len(prefix) :]
    return host.rstrip("/")


def has_url_scheme(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def login_state_path() -> Path:
    override = env(LOGIN_STATE_ENV)
    if override:
        return Path(override).expanduser()
    config_home = env("XDG_CONFIG_HOME")
    base = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return base / "cortex-training" / "login.json"


def read_login_state() -> tuple[str, str] | None:
    """Return what ``cortex-training login`` remembered, or None.

    The state is either ``("config", path)`` from ``login config.json`` or
    ``("connection", name)`` from the interactive ``login``.
    """
    path = login_state_path()
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid cortex-training login state {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"invalid cortex-training login state {path}: expected object")
    connection = parsed.get("connection")
    if isinstance(connection, str) and connection:
        return ("connection", connection)
    config_path = parsed.get("config_path")
    if not isinstance(config_path, str) or not config_path:
        raise ValueError(f"invalid cortex-training login state {path}: missing config_path")
    return ("config", config_path)


def clear_login_state() -> None:
    """Forget whatever ``cortex-training login`` remembered."""
    login_state_path().unlink(missing_ok=True)


def write_login_state(*, config_path: str | None = None, connection: str | None = None) -> None:
    """Remember a config path or a connection profile for later commands."""
    payload = {"connection": connection} if connection else {"config_path": config_path}
    state_path = login_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_config(path: str | None) -> dict[str, Any]:
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


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip() or None


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


def _env_direct() -> tuple[str | None, str | None, str | None]:
    return (
        env("CORTEX_TRAINING_BASE_URL"),
        env("CORTEX_TRAINING_HOST", "SNOWFLAKE_HOST"),
        env("CORTEX_TRAINING_PAT", "SNOWFLAKE_PAT"),
    )


def _env_database() -> str | None:
    return env("CORTEX_TRAINING_DATABASE", "SNOWFLAKE_DATABASE")


def resolve_connection(
    *,
    connection_name: str | None = None,
    config_path: str | None = None,
    base_url: str | None = None,
    host: str | None = None,
    pat: str | None = None,
    database: str | None = None,
    schema: str | None = None,
    endpoint: str | None = None,
    poll_interval: float | None = None,
    poll_timeout: float | None = None,
    verify_ssl: bool | None = None,
    load_login: bool = True,
) -> ConnectionSettings:
    """Pick one connection from explicit arguments, env vars, and files.

    ``None`` means "not passed". Only explicit arguments go in; env vars, the
    remembered login config, and the default profile are read here.
    """
    connection_name = _blank_to_none(connection_name)
    config_path = _blank_to_none(config_path)
    base_url = _blank_to_none(base_url)
    host = _blank_to_none(host)
    pat = _blank_to_none(pat)

    if connection_name and any((config_path, base_url, host, pat)):
        raise ValueError(
            "a connection profile (--connection / connection_name) cannot be "
            "combined with a config file, base_url, host, or pat"
        )

    profile: str | None = None
    selected_config: str | None = None
    use_env_credentials = True
    default_profile = False

    if connection_name:
        profile = connection_name
        source = f"connection profile {profile!r}"
    elif config_path:
        selected_config = config_path
        source = f"config file {config_path}"
    elif base_url or host or pat:
        source = "explicit host/pat or base_url"
    elif env("CORTEX_TRAINING_CONNECTION"):
        profile = env("CORTEX_TRAINING_CONNECTION")
        source = f"connection profile {profile!r} from CORTEX_TRAINING_CONNECTION"
    elif env("CORTEX_TRAINING_CONFIG"):
        selected_config = env("CORTEX_TRAINING_CONFIG")
        source = f"config file {selected_config} from CORTEX_TRAINING_CONFIG"
    else:
        env_base_url, env_host, env_pat = _env_direct()
        env_complete = bool(env_base_url or (env_host and env_pat))
        login_state = None
        if load_login and not (env_complete and (database or _env_database())):
            login_state = read_login_state()
            if login_state is not None and login_state[0] == "config":
                if not Path(login_state[1]).expanduser().is_file():
                    login_state = None
        if login_state is not None and login_state[0] == "connection":
            profile = login_state[1]
            source = f"connection profile {profile!r} remembered by cortex-training login"
        elif login_state is not None:
            selected_config = login_state[1]
            source = f"config file {selected_config} remembered by cortex-training login"
        elif env_complete:
            source = "environment variables"
        else:
            default_profile = True
            use_env_credentials = False
            source = "default connections.toml profile"

    if profile is not None:
        use_env_credentials = False
    config = load_config(selected_config)

    base_url = _coalesce(base_url, _config_str(config, "base_url"))
    host = _coalesce(host, _config_str(config, "host"))
    pat = _coalesce(pat, _config_str(config, "pat"))

    # A bare hostname in base_url is a common mistake for PAT auth.
    bare_base_url = base_url is not None and not has_url_scheme(base_url)
    if bare_base_url and host is None:
        host, base_url = base_url, None

    if use_env_credentials:
        env_base_url, env_host, env_pat = _env_direct()
        if base_url:
            pass  # a local/mock server; don't mix in PAT credentials
        elif host or pat:
            host = host or env_host
            pat = pat or env_pat
        else:
            base_url = env_base_url
            if not base_url:
                host, pat = env_host, env_pat

    if bare_base_url and not pat:
        raise ValueError(
            "base_url must start with http:// or https://. For Snowflake PAT auth, "
            "use host instead of base_url."
        )
    if base_url is None and bool(host) != bool(pat) and not selected_config:
        given, missing = ("host", "pat") if host else ("pat", "host")
        raise ValueError(
            f"{given} was provided without {missing}. Supply both, or omit both "
            "to use a connections.toml profile."
        )

    use_connection_profile = profile is not None or default_profile
    resolved_schema = _coalesce(
        schema,
        _config_str(config, "schema"),
        env("CORTEX_TRAINING_SCHEMA", "SNOWFLAKE_SCHEMA"),
    )
    if resolved_schema is None and not use_connection_profile:
        resolved_schema = DEFAULT_SCHEMA

    if verify_ssl is None:
        no_verify = _config_bool(config, "no_verify_ssl")
        if no_verify is not None:
            verify_ssl = not no_verify
        else:
            verify_ssl = _coalesce(_config_bool(config, "verify_ssl"), True)

    settings = ConnectionSettings(
        source=source,
        use_connection_profile=use_connection_profile,
        connection_name=profile,
        config_path=selected_config,
        base_url=None if use_connection_profile else base_url,
        host=None if use_connection_profile else host,
        pat=None if use_connection_profile else pat,
        database=_coalesce(database, _config_str(config, "database"), _env_database()),
        schema=resolved_schema,
        endpoint=_coalesce(
            endpoint,
            _config_str(config, "endpoint"),
            env("CORTEX_TRAINING_ENDPOINT"),
            DEFAULT_ENDPOINT,
        ),
        poll_interval=_coalesce(
            poll_interval, _config_float(config, "poll_interval"), DEFAULT_POLL_INTERVAL
        ),
        poll_timeout=_coalesce(
            poll_timeout, _config_float(config, "poll_timeout"), DEFAULT_POLL_TIMEOUT
        ),
        verify_ssl=verify_ssl,
    )
    logger.info("Cortex Training connection: %s", settings.source)
    return settings


def build_client(
    settings: ConnectionSettings,
    client_cls: type[CortexTrainingClient] | None = None,
    **client_kwargs: Any,
) -> CortexTrainingClient:
    """Open the client described by ``settings``."""
    if client_cls is None:
        from cortex_training.client import CortexTrainingClient as client_cls

    kwargs = {
        "database": settings.database,
        "schema": settings.schema,
        "endpoint": settings.endpoint,
        "poll_interval": settings.poll_interval,
        "poll_timeout": settings.poll_timeout,
        **client_kwargs,
    }
    if settings.base_url:
        return client_cls(base_url=settings.base_url, **kwargs)
    if settings.use_connection_profile:
        return client_cls.from_connection_name(
            connection_name=settings.connection_name,
            verify_ssl=settings.verify_ssl,
            **kwargs,
        )
    return client_cls.from_pat(
        host=normalize_host(settings.host),
        pat=settings.pat,
        verify_ssl=settings.verify_ssl,
        **kwargs,
    )
