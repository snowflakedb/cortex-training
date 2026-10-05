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

Resolution order (matches the CLI):
1. Explicit ``config_path`` JSON file → read file → ``from_pat()``
2. Explicit ``connection_name`` → ``from_connection_name()``
3. Environment variables (host + PAT) → ``from_pat()``
4. ``CORTEX_TRAINING_CONNECTION`` env var → ``from_connection_name()``
5. ``cortex-training login`` remembered config → ``from_pat()``
6. Default ``connections.toml`` profile → ``from_connection_name()``
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _normalize_host(host: str) -> str:
    for prefix in ("https://", "http://"):
        if host.startswith(prefix):
            host = host[len(prefix):]
    return host.rstrip("/")


def _load_json_config(config_path: str) -> dict[str, Any]:
    path_obj = Path(config_path).expanduser()
    parsed = json.loads(path_obj.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"connection config {config_path} must be a JSON object")
    config = parsed.get("connection", parsed)
    if not isinstance(config, dict):
        raise ValueError(f"connection config {config_path} must be a JSON object")
    return config


def _read_login_config_path() -> str | None:
    override = _env("CORTEX_TRAINING_LOGIN_FILE")
    if override:
        state_path = Path(override).expanduser()
    else:
        config_home = _env("XDG_CONFIG_HOME")
        base = Path(config_home).expanduser() if config_home else Path.home() / ".config"
        state_path = base / "cortex-training" / "login.json"
    if not state_path.exists():
        return None
    try:
        parsed = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(parsed, dict):
        return None
    config_path = parsed.get("config_path")
    if not isinstance(config_path, str) or not config_path:
        return None
    if not Path(config_path).expanduser().is_file():
        return None
    return config_path


def connect(
    config_path: str | None = None,
    *,
    connection_name: str | None = None,
    **overrides: Any,
) -> "CortexTrainingClient":
    """Build a CortexTrainingClient from the best available credentials.

    Parameters
    ----------
    config_path
        Path to a JSON config file with host, pat, database, schema.
        Empty strings are treated as None.
    connection_name
        Named Snowflake connection profile (connections.toml).
    **overrides
        Override any resolved value (database, schema, endpoint, etc.).
    """
    from cortex_training import CortexTrainingClient

    # Treat empty/whitespace config_path as not set.
    if config_path is not None:
        config_path = config_path.strip() or None

    # Reject conflicting arguments.
    host = overrides.pop("host", None)
    pat = overrides.pop("pat", None)
    base_url = overrides.pop("base_url", None)
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

    # --- 1. Explicit JSON config ---
    if config_path is not None:
        config = _load_json_config(config_path)
        cfg_base_url = config.get("base_url")
        cfg_host = host or config.get("host")
        cfg_pat = pat or config.get("pat")
        kwargs = {
            "database": config.get("database", "CORTEX_TRAINING_DB"),
            "schema": config.get("schema", "PUBLIC"),
            "endpoint": config.get("endpoint", "cortex-training"),
            "poll_interval": float(config.get("poll_interval", 0.5)),
            "poll_timeout": float(config.get("poll_timeout", 1800.0)),
            "verify_ssl": bool(config.get("verify_ssl", True)),
        }
        kwargs.update(overrides)
        if base_url or cfg_base_url:
            return CortexTrainingClient(base_url=base_url or cfg_base_url, **kwargs)
        if not cfg_host:
            raise ValueError(f"connection config {config_path} is missing 'host'")
        if not cfg_pat:
            raise ValueError(f"connection config {config_path} is missing 'pat'")
        return CortexTrainingClient.from_pat(
            host=_normalize_host(cfg_host), pat=cfg_pat, **kwargs
        )

    # --- 2. Explicit connection name ---
    if connection_name is not None:
        conn_overrides = dict(overrides)
        db = conn_overrides.pop("database", None) or _env("CORTEX_TRAINING_DATABASE", "SNOWFLAKE_DATABASE")
        schema = conn_overrides.pop("schema", None) or _env("CORTEX_TRAINING_SCHEMA", "SNOWFLAKE_SCHEMA")
        if db:
            conn_overrides["database"] = db
        if schema:
            conn_overrides["schema"] = schema
        return CortexTrainingClient.from_connection_name(
            connection_name=connection_name, **conn_overrides
        )

    # --- 3. Explicit host + pat via overrides ---
    if host and pat:
        kwargs = {
            "database": _env("CORTEX_TRAINING_DATABASE", "SNOWFLAKE_DATABASE") or "CORTEX_TRAINING_DB",
            "schema": _env("CORTEX_TRAINING_SCHEMA", "SNOWFLAKE_SCHEMA") or "PUBLIC",
            "endpoint": _env("CORTEX_TRAINING_ENDPOINT") or "cortex-training",
        }
        kwargs.update(overrides)
        return CortexTrainingClient.from_pat(
            host=_normalize_host(host), pat=pat, **kwargs
        )

    # --- 4. Explicit base_url (mock/local) ---
    if base_url:
        kwargs = {
            "database": _env("CORTEX_TRAINING_DATABASE", "SNOWFLAKE_DATABASE") or "CORTEX_TRAINING_DB",
            "schema": _env("CORTEX_TRAINING_SCHEMA", "SNOWFLAKE_SCHEMA") or "PUBLIC",
            "endpoint": _env("CORTEX_TRAINING_ENDPOINT") or "cortex-training",
        }
        kwargs.update(overrides)
        return CortexTrainingClient(base_url=base_url, **kwargs)

    # --- 5. Environment variables ---
    env_host = _env("CORTEX_TRAINING_HOST", "SNOWFLAKE_HOST")
    env_pat = _env("CORTEX_TRAINING_PAT", "SNOWFLAKE_PAT")
    if env_host and env_pat:
        kwargs = {
            "database": _env("CORTEX_TRAINING_DATABASE", "SNOWFLAKE_DATABASE") or "CORTEX_TRAINING_DB",
            "schema": _env("CORTEX_TRAINING_SCHEMA", "SNOWFLAKE_SCHEMA") or "PUBLIC",
            "endpoint": _env("CORTEX_TRAINING_ENDPOINT") or "cortex-training",
        }
        kwargs.update(overrides)
        return CortexTrainingClient.from_pat(
            host=_normalize_host(env_host), pat=env_pat, **kwargs
        )

    # --- 6. CORTEX_TRAINING_CONNECTION env var ---
    env_connection = _env("CORTEX_TRAINING_CONNECTION")
    if env_connection:
        conn_overrides = dict(overrides)
        db = conn_overrides.pop("database", None) or _env("CORTEX_TRAINING_DATABASE", "SNOWFLAKE_DATABASE")
        schema = conn_overrides.pop("schema", None) or _env("CORTEX_TRAINING_SCHEMA", "SNOWFLAKE_SCHEMA")
        if db:
            conn_overrides["database"] = db
        if schema:
            conn_overrides["schema"] = schema
        return CortexTrainingClient.from_connection_name(
            connection_name=env_connection, **conn_overrides
        )

    # --- 7. Remembered config from cortex-training login ---
    login_config = _read_login_config_path()
    if login_config:
        return connect(config_path=login_config, **overrides)

    # --- 8. Default connections.toml profile ---
    conn_overrides = dict(overrides)
    db = conn_overrides.pop("database", None) or _env("CORTEX_TRAINING_DATABASE", "SNOWFLAKE_DATABASE")
    schema = conn_overrides.pop("schema", None) or _env("CORTEX_TRAINING_SCHEMA", "SNOWFLAKE_SCHEMA")
    if db:
        conn_overrides["database"] = db
    if schema:
        conn_overrides["schema"] = schema
    return CortexTrainingClient.from_connection_name(**conn_overrides)
