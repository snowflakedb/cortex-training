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

"""Build a CortexTrainingClient the same way the CLI picks its connection.

The resolution rules live in ``cortex_training._connection`` and are shared
with the CLI, so a given set of arguments, env vars, and files selects the same
connection everywhere.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import Any

from cortex_training._connection import build_client
from cortex_training._connection import resolve_connection

if TYPE_CHECKING:
    from cortex_training.client import CortexTrainingClient

# Recipes and SDK scripts have always defaulted the database for PAT auth; the
# CLI instead requires one explicitly.
_DEFAULT_DATABASE = "CORTEX_TRAINING_DB"

_CONNECTION_FIELDS = (
    "base_url",
    "host",
    "pat",
    "database",
    "schema",
    "endpoint",
    "poll_interval",
    "poll_timeout",
    "verify_ssl",
)


def connect(
    config_path: str | None = None,
    *,
    connection_name: str | None = None,
    **overrides: Any,
) -> CortexTrainingClient:
    """Build a CortexTrainingClient from the best available credentials.

    Explicit arguments always win over environment variables. Passing
    ``connection_name`` together with ``config_path``, ``host``, ``pat``, or
    ``base_url`` is an error. With no explicit source, the first match wins:

    1. ``CORTEX_TRAINING_CONNECTION`` profile
    2. ``CORTEX_TRAINING_CONFIG`` JSON file
    3. ``CORTEX_TRAINING_BASE_URL``, or ``CORTEX_TRAINING_HOST`` +
       ``CORTEX_TRAINING_PAT``, when a database is also set
    4. Config or profile remembered by ``cortex-training login``
    5. The same env vars without a database
    6. Default ``connections.toml`` profile

    Parameters
    ----------
    config_path
        JSON config file with ``host``, ``pat``, ``database``, ``schema``.
        Empty strings are treated as not set.
    connection_name
        Named Snowflake connection profile in ``connections.toml``.
    **overrides
        Connection fields (``host``, ``pat``, ``base_url``, ``database``,
        ``schema``, ``endpoint``, ``poll_interval``, ``poll_timeout``,
        ``verify_ssl``) override the resolved values. Anything else is passed
        to the client constructor.
    """
    from cortex_training import CortexTrainingClient

    fields = {name: overrides.pop(name, None) for name in _CONNECTION_FIELDS}
    settings = resolve_connection(
        connection_name=connection_name,
        config_path=config_path,
        **fields,
    )
    if not settings.use_connection_profile:
        if settings.base_url is None and not (settings.host and settings.pat):
            missing = " and ".join(
                name for name in ("host", "pat") if not getattr(settings, name)
            )
            raise ValueError(
                f"{settings.source} has no {missing}. Provide host and pat, "
                "base_url for a local server, or use a connections.toml profile."
            )
        if settings.database is None:
            settings = dataclasses.replace(settings, database=_DEFAULT_DATABASE)
    return build_client(settings, CortexTrainingClient, **overrides)
