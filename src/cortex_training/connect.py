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

"""Unified factory for building a CortexTrainingClient.

Resolution order (same as the CLI):

1. Explicit ``connection_name`` or ``CORTEX_TRAINING_CONNECTION``
2. Explicit ``config_path`` or ``CORTEX_TRAINING_CONFIG``
3. Explicit ``base_url`` or ``CORTEX_TRAINING_BASE_URL``
4. Environment variables ``CORTEX_TRAINING_HOST`` + ``CORTEX_TRAINING_PAT``
5. Remembered config from ``cortex-training login``
6. Default ``connections.toml`` profile
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Shared helpers — imported by _cli.py to avoid duplication
# ---------------------------------------------------------------------------

_LOGIN_STATE_ENV = "CORTEX_TRAINING_LOGIN_FILE"


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def normalize_host(host: str) -> str:
    """Strip ``https://`` or ``http://`` prefix and trailing slash."""
    for prefix in ("https://", "http://"):
        if host.startswith(prefix):
            host = host[len(prefix):]
    return host.rstrip("/")


def _login_state_path() -> Path:
    override = _env(_LOGIN_STATE_ENV)
    if override:
        return Path(override).expanduser()
    config_home = _env("XDG_CONFIG_HOME")
    base = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return base / "cortex-training" / "login.json"


def read_login_config_path() -> str | None:
    """Return the config path stored by ``cortex-training login``, or None."""
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


def load_json_config(config_path: str) -> dict[str, Any]:
    """Read a connection JSON file, unwrapping an optional ``connection`` key."""
    path_obj = Path(config_path).expanduser()
    parsed = json.loads(path_obj.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"connection config {config_path} must be a JSON object")
    config = parsed.get("connection", parsed)
    if not isinstance(config, dict):
        raise ValueError(f"connection config {config_path} must be a JSON object")
    return config


def _resolve_env_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply env-var defaults for database, schema, and endpoint to overrides."""
    result = {
        "database": _env("CORTEX_TRAINING_DATABASE", "SNOWFLAKE_DATABASE") or "CORTEX_TRAINING_DB",
        "schema": _env("CORTEX_TRAINING_SCHEMA", "SNOWFLAKE_SCHEMA") or "PUBLIC",
        "endpoint": _env("CORTEX_TRAINING_ENDPOINT") or "cortex-training",
    }
    result.update(overrides)
    return result


def _build_from_config(
    config: dict[str, Any],
    config_path: str,
    *,
    host_override: str | None = None,
    pat_override: str | None = None,
    base_url_override: str | None = None,
    **overrides: Any,
) -> "CortexTrainingClient":
    """Build a client from a parsed JSON config dict."""
    from cortex_training import CortexTrainingClient

    cfg_base_url = base_url_override or config.get("base_url")
    cfg_host = host_override or config.get("host")
    cfg_pat = pat_override or config.get("pat")
    kwargs = {
        "database": config.get("database", "CORTEX_TRAINING_DB"),
        "schema": config.get("schema", "PUBLIC"),
        "endpoint": config.get("endpoint", "cortex-training"),
        "poll_interval": float(config.get("poll_interval", 0.5)),
        "poll_timeout": float(config.get("poll_timeout", 1800.0)),
        "verify_ssl": bool(config.get("verify_ssl", True)),
    }
    kwargs.update(overrides)
    if cfg_base_url:
        kwargs.pop("verify_ssl", None)
        return CortexTrainingClient(base_url=cfg_base_url, **kwargs)
    if not cfg_host:
        raise ValueError(f"connection config {config_path} is missing 'host'")
    if not cfg_pat:
        raise ValueError(f"connection config {config_path} is missing 'pat'")
    return CortexTrainingClient.from_pat(
        host=normalize_host(cfg_host), pat=cfg_pat, **kwargs
    )


def _build_from_connection_name(
    connection_name: str | None = None,
    **overrides: Any,
) -> "CortexTrainingClient":
    """Build a client from a connections.toml profile, applying env-var overrides."""
    from cortex_training import CortexTrainingClient

    db = overrides.pop("database", None) or _env("CORTEX_TRAINING_DATABASE", "SNOWFLAKE_DATABASE")
    schema = overrides.pop("schema", None) or _env("CORTEX_TRAINING_SCHEMA", "SNOWFLAKE_SCHEMA")
    if db:
        overrides["database"] = db
    if schema:
        overrides["schema"] = schema
    return CortexTrainingClient.from_connection_name(
        connection_name=connection_name, **overrides
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def connect(
    config_path: str | None = None,
    *,
    connection_name: str | None = None,
    **overrides: Any,
) -> "CortexTrainingClient":
    """Build a CortexTrainingClient from the best available credentials.

    Resolution order matches the CLI:

    1. ``connection_name`` or ``CORTEX_TRAINING_CONNECTION``
    2. ``config_path`` or ``CORTEX_TRAINING_CONFIG``
    3. ``base_url`` or ``CORTEX_TRAINING_BASE_URL``
    4. Env vars ``CORTEX_TRAINING_HOST`` + ``CORTEX_TRAINING_PAT``
    5. Remembered config from ``cortex-training login``
    6. Default ``connections.toml`` profile
    """
    from cortex_training import CortexTrainingClient

    # Treat empty/whitespace config_path as not set.
    if config_path is not None:
        config_path = config_path.strip() or None

    # Pop explicit credentials from overrides.
    host = overrides.pop("host", None)
    pat = overrides.pop("pat", None)
    base_url = overrides.pop("base_url", None)

    # Reject conflicting arguments.
    if connection_name and any((host, pat, base_url)):
        raise ValueError(
            "connection_name cannot be combined with host, pat, or base_url"
        )

    # Partial credentials are likely a mistake.
    if bool(host) != bool(pat) and not config_path:
        given, missing = ("host", "pat") if host else ("pat", "host")
        raise ValueError(
            f"{given} was provided without {missing}. "
            "Supply both, or omit both to use connections.toml."
        )

    # --- 1. Connection profile (explicit or env var) ---
    env_connection = connection_name or _env("CORTEX_TRAINING_CONNECTION")
    if env_connection:
        return _build_from_connection_name(env_connection, **overrides)

    # --- 2. JSON config (explicit or env var) ---
    config_path = config_path or _env("CORTEX_TRAINING_CONFIG")
    if config_path:
        config = load_json_config(config_path)
        return _build_from_config(
            config, config_path,
            host_override=host, pat_override=pat, base_url_override=base_url,
            **overrides,
        )

    # --- 3. Explicit host + pat ---
    if host and pat:
        kwargs = _resolve_env_overrides(overrides)
        return CortexTrainingClient.from_pat(
            host=normalize_host(host), pat=pat, **kwargs
        )

    # --- 4. Explicit or env base_url ---
    resolved_base_url = base_url or _env("CORTEX_TRAINING_BASE_URL")
    if resolved_base_url:
        kwargs = _resolve_env_overrides(overrides)
        return CortexTrainingClient(base_url=resolved_base_url, **kwargs)

    # --- 5. Env vars for host + PAT ---
    env_host = _env("CORTEX_TRAINING_HOST", "SNOWFLAKE_HOST")
    env_pat = _env("CORTEX_TRAINING_PAT", "SNOWFLAKE_PAT")
    if env_host and env_pat:
        kwargs = _resolve_env_overrides(overrides)
        return CortexTrainingClient.from_pat(
            host=normalize_host(env_host), pat=env_pat, **kwargs
        )

    # --- 6. Remembered config from cortex-training login ---
    login_config = read_login_config_path()
    if login_config and Path(login_config).expanduser().is_file():
        config = load_json_config(login_config)
        return _build_from_config(config, login_config, **overrides)

    # --- 7. Default connections.toml profile ---
    return _build_from_connection_name(**overrides)
