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

Every connection goes through the connection picker (as ``from_configs()``):
``connection``, then the profile saved by ``cortex-training login``, then the
Snowflake default profile, then ``CORTEX_TRAINING_HOST`` + ``CORTEX_TRAINING_PAT``
when no profile exists (CI). ``CORTEX_TRAINING_BASE_URL`` reaches a local or mock
server instead.

An old JSON config (``CORTEX_TRAINING_CONFIG``, or the ``login.json`` the old
``cortex-training login PATH`` wrote) raises :class:`~cortex_training.cli.commands.OldConfigFile`,
which names the command that converts it into a profile.

The logic lives in :func:`cortex_training.cli.commands._resolve_args`; this module only
adapts it for library callers so examples and the CLI cannot drift apart.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from urllib.parse import urlsplit

from cortex_training.cli import commands as _cli
from cortex_training.client import CortexTrainingClient

logger = logging.getLogger(__name__)

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
    connection: str | None = None,
    *,
    default_database: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ResolvedConnection:
    """Open a client the way the CLI does.

    ``connection`` names a profile. ``default_database`` is used only for a
    ``CORTEX_TRAINING_BASE_URL`` server that names no database. ``overrides``
    takes keys listed in ``_OVERRIDABLE``.
    """
    args = _resolved_args(connection, default_database=default_database, overrides=overrides)
    client = _cli.build_client(args, CortexTrainingClient)
    for note in args._warnings:
        logger.warning("%s", note)
    if args.use_connection_profile:
        resolution = args.resolution
        # A profile that signs in with a PAT holds the same token an old JSON file did.
        pat = profile_pat(resolution.profile) if resolution.profile else resolution.pat
    else:
        pat = None
    return ResolvedConnection(
        client=client,
        source=_source_label(args),
        host=_host_of(client),
        database=client.database,
        schema=client.schema,
        verify_ssl=not args.no_verify_ssl,
        pat=pat,
    )


def _resolved_args(
    connection: str | None,
    *,
    default_database: str | None,
    overrides: Mapping[str, Any] | None,
) -> argparse.Namespace:
    """The CLI's resolution of ``connection``, checked, without signing in."""
    extra = dict(overrides or {})
    unknown = set(extra) - _OVERRIDABLE
    if unknown:
        raise TypeError(f"unsupported connection override(s): {', '.join(sorted(unknown))}")

    args = argparse.Namespace(
        config=None,
        connection=connection,
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
    )
    for name, value in extra.items():
        setattr(args, name, value)

    args = _cli._resolve_args(args)
    args = _cli._normalize_connection_args(args)
    if not args.use_connection_profile:
        args.database = args.database or default_database
        if not args.database:
            raise ValueError("CORTEX_TRAINING_BASE_URL needs a database: set CORTEX_TRAINING_DATABASE")
    return args


def profile_pat(profile: str) -> str | None:
    """The programmatic access token ``profile`` signs in with, or None if it signs in another way."""
    from cortex_training import _profiles

    values = _profiles.read_profiles().get(profile) or {}
    token = values.get("token")
    if str(values.get("authenticator") or "").strip().lower() == "programmatic_access_token" and isinstance(token, str):
        return token.strip() or None
    return None


@dataclass(frozen=True)
class PatCredentials:
    """What a tool that sends a PAT itself needs: where to call, the token, and the database and schema."""

    # The host as HTTPS calls use it: an underscore account label becomes a dash (see rest_host).
    host: str
    pat: str = field(repr=False)
    database: str
    schema: str
    # Human-readable origin, e.g. ``connection profile 'training'``.
    source: str
    # Warnings for a variable the old connection lookup read, for the caller to show.
    notes: tuple[str, ...] = ()


def pat_credentials(
    connection: str | None = None,
    *,
    default_database: str | None = None,
) -> PatCredentials:
    """Resolve a connection like :func:`resolve_connection`, without signing in, and return its PAT.

    Reads only local files. Raises ValueError when the connection doesn't sign in
    with a PAT (for example, a profile with a key file or browser sign-in), saying
    how to get one that does.
    """
    from cortex_training import _picker
    from cortex_training.client.auth import rest_host

    args = _resolved_args(connection, default_database=default_database or _picker.DEFAULT_DATABASE, overrides=None)
    if args.base_url:
        raise ValueError("a base_url connection (a local or mock server) has no PAT to export")
    resolution = args.resolution = _cli._picker_choice(args)
    if resolution.profile is None:
        host, pat = resolution.host, resolution.pat
    else:
        from cortex_training import _profiles

        values = _profiles.read_profiles().get(resolution.profile) or {}
        pat = profile_pat(resolution.profile)
        account = values.get("account")
        host = values.get("host") or (f"{account}.snowflakecomputing.com" if account else None)
    database, schema = resolution.database, resolution.schema
    source = _source_label(args)
    if not pat:
        raise ValueError(
            f"{source} doesn't sign in with a programmatic access token (PAT), and this needs one. "
            "Use a profile with authenticator = \"PROGRAMMATIC_ACCESS_TOKEN\" (`cortex-training login` "
            "creates one), or pass connection=NAME for one."
        )
    if not host:
        raise ValueError(f"{source} has no host or account to connect to")
    return PatCredentials(
        host=rest_host(_cli._normalize_host(str(host))),
        pat=pat,
        database=database,
        schema=schema,
        source=source,
        notes=tuple(args._warnings),
    )


def _source_label(args: argparse.Namespace) -> str:
    if args.use_connection_profile:
        resolution = getattr(args, "resolution", None)
        chosen = getattr(resolution, "sources", {}).get("profile", "")
        if resolution is not None and resolution.profile is None:
            return f"{resolution.sources['host']} (no profile exists)"
        if chosen.startswith("saved in "):
            return f"connection profile '{resolution.profile}' (saved by cortex-training login)"
        if chosen == "passed in":
            return f"connection profile '{resolution.profile}'"
        return _DEFAULT_PROFILE_SOURCE
    return "base_url (CORTEX_TRAINING_BASE_URL)"


def _host_of(client: CortexTrainingClient) -> str:
    return urlsplit(client.base_url).netloc or client.base_url
