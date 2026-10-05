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

Resolution order:
1. Explicit JSON config path → read file → from_pat()
2. Explicit connection_name → from_connection_name()
3. Environment variables (host + PAT) → from_pat()
4. Default connections.toml profile → from_connection_name()
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


def _load_json_config(config_path: str) -> dict[str, Any]:
    parsed = json.loads(Path(config_path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"connection config {config_path} must be a JSON object")
    config = parsed.get("connection", parsed)
    if not isinstance(config, dict):
        raise ValueError(f"connection config {config_path} must be a JSON object")
    return config


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
    connection_name
        Named Snowflake connection profile (connections.toml).
    **overrides
        Override any resolved value (database, schema, endpoint, etc.).
    """
    from cortex_training import CortexTrainingClient

    if config_path is not None:
        config = _load_json_config(config_path)
        host = overrides.pop("host", None) or config.get("host")
        pat = overrides.pop("pat", None) or config.get("pat")
        if not host:
            raise ValueError(f"connection config {config_path} is missing 'host'")
        if not pat:
            raise ValueError(f"connection config {config_path} is missing 'pat'")
        kwargs = {
            "database": config.get("database", "CORTEX_TRAINING_DB"),
            "schema": config.get("schema", "PUBLIC"),
            "endpoint": config.get("endpoint", "cortex-training"),
            "poll_interval": float(config.get("poll_interval", 0.5)),
            "poll_timeout": float(config.get("poll_timeout", 1800.0)),
            "verify_ssl": bool(config.get("verify_ssl", True)),
        }
        kwargs.update(overrides)
        return CortexTrainingClient.from_pat(host=host, pat=pat, **kwargs)

    if connection_name is not None:
        return CortexTrainingClient.from_connection_name(
            connection_name=connection_name, **overrides
        )

    # Check explicit host/pat in overrides, then environment variables.
    host = overrides.pop("host", None) or _env("CORTEX_TRAINING_HOST", "SNOWFLAKE_HOST")
    pat = overrides.pop("pat", None) or _env("CORTEX_TRAINING_PAT", "SNOWFLAKE_PAT")
    if host and pat:
        kwargs = {
            "database": _env("CORTEX_TRAINING_DATABASE", "SNOWFLAKE_DATABASE") or "CORTEX_TRAINING_DB",
            "schema": _env("CORTEX_TRAINING_SCHEMA", "SNOWFLAKE_SCHEMA") or "PUBLIC",
            "endpoint": _env("CORTEX_TRAINING_ENDPOINT") or "cortex-training",
        }
        kwargs.update(overrides)
        return CortexTrainingClient.from_pat(host=host, pat=pat, **kwargs)

    # Fall back to default connections.toml profile.
    return CortexTrainingClient.from_connection_name(**overrides)
