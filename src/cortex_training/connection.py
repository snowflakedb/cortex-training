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

"""Open a client from the same sources, in the same order, as the CLI and TUI.

Resolution order (first match wins):

1. ``connection`` (or ``CORTEX_TRAINING_CONNECTION``): a named profile.
2. ``config`` (or ``CORTEX_TRAINING_CONFIG``): a legacy JSON file.
3. Complete direct credentials from the environment (``base_url``, or
   ``host`` plus ``pat``, plus ``database``).
4. The saved login state written by ``cortex-training login``.
5. The Snowflake connector's default profile.

The logic lives in :func:`cortex_training.cli.commands._resolve_args`; this module only
adapts it for library callers so examples and the CLI cannot drift apart.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from urllib.parse import urlsplit

from cortex_training.cli import commands as _cli
from cortex_training.client import CortexTrainingClient

# Keyword overrides that take precedence over the file, profile, and environment.
_OVERRIDABLE = frozenset(
    {
        "database",
        "schema",
        "endpoint",
        "poll_interval",
        "poll_timeout",
        "max_retries",
        "request_timeout",
    }
)

_DIRECT_SOURCE = "environment variables or arguments"
_DEFAULT_PROFILE_SOURCE = "default Snowflake connection profile"


@dataclass(frozen=True)
class ResolvedConnection:
    """A client plus the facts about how it was authenticated.

    ``pat`` is set only for PAT-authenticated clients. It is secret: never log it.
    """

    client: CortexTrainingClient
    # Human-readable origin, e.g. ``connection profile 'training'``.
    source: str
    host: str
    database: str | None
    schema: str | None
    verify_ssl: bool
    pat: str | None = field(default=None, repr=False)

    def describe(self) -> str:
        """Return a log line naming the target and the source, never the token."""
        target = f"{self.host} ({self.database}.{self.schema})"
        return f"connected to {target} via {self.source}"


def resolve_connection(
    config: str | None = None,
    connection: str | None = None,
    *,
    default_database: str | None = None,
    strict_config: bool = True,
    overrides: Mapping[str, Any] | None = None,
) -> ResolvedConnection:
    """Open a client using the CLI's resolution order.

    ``config`` and ``connection`` are explicit arguments and win over their
    environment variables. ``default_database`` is used only for PAT or legacy
    JSON connections that do not name a database; it never replaces a profile's
    database. ``strict_config=False`` ignores unknown keys in the JSON file, which
    examples have always accepted. ``overrides`` takes keys listed in
    ``_OVERRIDABLE``.
    """
    if config and connection:
        raise ValueError("pass either config= or connection=, not both")
    extra = dict(overrides or {})
    unknown = set(extra) - _OVERRIDABLE
    if unknown:
        raise TypeError(f"unsupported connection override(s): {', '.join(sorted(unknown))}")

    args = argparse.Namespace(
        config=config or _cli._env("CORTEX_TRAINING_CONFIG"),
        connection=connection or _cli._env("CORTEX_TRAINING_CONNECTION"),
        base_url=None,
        host=None,
        pat=None,
        database=None,
        schema=None,
        endpoint=None,
        poll_interval=None,
        poll_timeout=None,
        no_verify_ssl=False,
        max_retries=None,
        request_timeout=None,
        _strict_config=strict_config,
    )
    for name, value in extra.items():
        setattr(args, name, value)

    args = _cli._resolve_args(args)
    args = _cli._normalize_connection_args(args)
    if not args.use_connection_profile:
        if not args.database and default_database:
            args.database = default_database
        if not args.database:
            raise ValueError(
                "connection needs a database: set `database` in the config file, "
                "or CORTEX_TRAINING_DATABASE / SNOWFLAKE_DATABASE"
            )
        if args.base_url is None and (args.host is None or args.pat is None):
            raise ValueError(
                "connection needs `host` and `pat` (or `base_url` for local use)"
            )

    client = _cli.build_client(args, CortexTrainingClient)
    pat = None if args.use_connection_profile or args.base_url else args.pat
    return ResolvedConnection(
        client=client,
        source=_source_label(args),
        host=_host_of(args, client),
        database=client.database,
        schema=client.schema,
        verify_ssl=not args.no_verify_ssl,
        pat=pat,
    )


def _source_label(args: argparse.Namespace) -> str:
    if args.use_connection_profile:
        if args.connection:
            return f"connection profile '{args.connection}'"
        return _DEFAULT_PROFILE_SOURCE
    if args.base_url:
        return f"base_url ({_DIRECT_SOURCE})"
    origin = getattr(args, "_config_origin", None)
    if origin == "config file":
        return "config file"
    if origin == "saved login":
        return "saved login (cortex-training login)"
    return _DIRECT_SOURCE


def _host_of(args: argparse.Namespace, client: CortexTrainingClient) -> str:
    if args.host and not args.use_connection_profile:
        return _cli._normalize_host(args.host)
    return urlsplit(client.base_url).netloc or client.base_url
