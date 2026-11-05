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

"""Whether a profile's role can create training jobs, checked without creating anything.

Creating a job needs, besides a privilege on the database and schema (which
``login`` already checks), two privileges (see "Access control" in
``docs/reference/rest-api.md``):

- ``CREATE EXPERIMENT`` on the schema, because each job creates its own experiment;
- ``USE CORTEX TRAINING`` on the account.

Each is checked in two read-only steps: ``SHOW GRANTS ON SCHEMA`` / ``ON ACCOUNT``
lists the roles that hold it, then ``IS_ROLE_IN_SESSION`` asks whether the
session's role is one of them or inherits one. Through the SQL API (a PAT
client) that function needs a warehouse, so there the check walks up from each
holder with ``SHOW GRANTS OF ROLE`` instead, which needs none.

Which roles count depends on the privilege, as the service decides it:
``CREATE EXPERIMENT`` is a create privilege, so only the primary role and the
roles it inherits count, and active secondary roles are turned off for that
check (and back on afterwards). ``USE CORTEX TRAINING`` isn't, so active
secondary roles count too, and it's checked with them on. Anything unexpected
gives "couldn't check", never a false warning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from cortex_training import _database_setup

logger = logging.getLogger(__name__)

CREATE_EXPERIMENT = "CREATE EXPERIMENT"
USE_CORTEX_TRAINING = "USE CORTEX TRAINING"
# An owner holds every privilege on the object it owns.
_OWNERSHIP = "OWNERSHIP"
# More holders than this would make one very long query; say "couldn't check" instead.
_MAX_HOLDERS = 200
# The walk up the role hierarchy asks about at most this many roles, each with a
# short timeout, so a large hierarchy can't stall login.
_MAX_WALK = 20
_WALK_TIMEOUT_SECONDS = 15
# Snowflake's "does not exist or not authorized": a role the session can't see isn't one it inherits.
_NOT_VISIBLE = "002003"


@dataclass(frozen=True)
class Missing:
    """A privilege the role doesn't have, and the statement an admin runs to grant it."""

    privilege: str
    on: str  # "schema DB.SCHEMA" or "the account"
    grant: str


@dataclass(frozen=True)
class JobAccess:
    """Whether ``role`` can create training jobs; ``unknown`` says why it couldn't be checked."""

    role: str
    missing: tuple[Missing, ...] = ()
    unknown: str | None = None

    @property
    def ready(self) -> bool:
        return self.unknown is None and not self.missing


def check(client: Any, role: str | None) -> JobAccess:
    """Check ``client``'s role for what creating a job needs in its database and schema."""
    role = role or "your role"
    try:
        if role == "your role":
            rows = _database_setup.run_statement(client, "SELECT CURRENT_ROLE() AS role")
            role = str(rows[0]["role"]) if rows and rows[0].get("role") else role
        schema = f"{client.database}.{client.schema}"
        _database_setup.check_identifier("database", client.database)
        _database_setup.check_identifier("schema", client.schema)
        missing = []
        secondary = _active_secondary_roles(client)
        # Create privileges count only the primary role's hierarchy: secondary roles off.
        with _primary_role_only(client):
            holders = _holders(client, f"SHOW GRANTS ON SCHEMA {schema}", CREATE_EXPERIMENT)
            if not _held(client, holders, role):
                missing.append(
                    Missing(CREATE_EXPERIMENT, f"schema {schema}", f"GRANT CREATE EXPERIMENT ON SCHEMA {schema} TO ROLE {_quoted(role)};")
                )
        # USE CORTEX TRAINING also counts active secondary roles: checked as the session is.
        holders = _holders(client, "SHOW GRANTS ON ACCOUNT", USE_CORTEX_TRAINING)
        # An account where the privilege doesn't exist yet lists no holder at all,
        # not even ACCOUNTADMIN, which holds it wherever it exists.
        # The walk follows only the primary role, so it can't stand in for secondary roles.
        if holders and not _held(client, holders, role, walk=not secondary):
            missing.append(
                Missing(USE_CORTEX_TRAINING, "the account", f"GRANT USE CORTEX TRAINING ON ACCOUNT TO ROLE {_quoted(role)};")
            )
        return JobAccess(role=role, missing=tuple(missing))
    except Exception as exc:  # Never a false warning: anything unexpected is "couldn't check".
        logger.debug("Could not check job-creation privileges.", exc_info=True)
        return JobAccess(role=role, unknown=_first_line(exc))


class _primary_role_only:
    """Turn off active secondary roles for the check, and turn them back on after it.

    A PAT sign-in has none and can't change them; then nothing is changed. If
    they're active and can't be turned off, raise, so the check says it
    couldn't check rather than count roles creating a job wouldn't use.
    """

    def __init__(self, client: Any) -> None:
        self.client = client
        self.restore: str | None = None

    def __enter__(self) -> None:
        active = _active_secondary_roles(self.client)
        if not active:
            return
        _database_setup.run_statement(self.client, "USE SECONDARY ROLES NONE")
        value = str(active.get("value") or "").upper()
        self.restore = "USE SECONDARY ROLES ALL" if value == "ALL" else f"USE SECONDARY ROLES {active['roles']}"

    def __exit__(self, *_exc: object) -> None:
        if self.restore:
            try:
                _database_setup.run_statement(self.client, self.restore)
            except Exception:
                logger.debug("Could not turn secondary roles back on.", exc_info=True)


def _active_secondary_roles(client: Any) -> dict[str, Any]:
    """The session's active secondary roles, as CURRENT_SECONDARY_ROLES() reports them; empty when none."""
    rows = _database_setup.run_statement(client, "SELECT CURRENT_SECONDARY_ROLES() AS roles")
    active = json.loads(rows[0]["roles"]) if rows and rows[0].get("roles") else {}
    return active if active.get("roles") or active.get("value") else {}


def _holders(client: Any, statement: str, privilege: str) -> list[tuple[str, str]]:
    """``(granted_to, grantee_name)`` for each grant of ``privilege`` (or ownership) in ``statement``."""
    found = []
    for row in _database_setup.run_statement(client, statement):
        if str(row.get("privilege", "")).upper() in (privilege, _OWNERSHIP):
            found.append((str(row.get("granted_to", "")).upper(), str(row.get("grantee_name", ""))))
    return found


def _held(client: Any, holders: list[tuple[str, str]], role: str, *, walk: bool = True) -> bool:
    """Whether the session's roles are, or inherit, one of ``holders``.

    ``walk=False`` refuses the warehouse-free fallback, which follows only the
    primary role: for a privilege that secondary roles also count for.
    """
    if any(kind == "ROLE" and name.upper() == "PUBLIC" for kind, name in holders):
        return True  # Every role has PUBLIC.
    tests = []
    for kind, name in holders:
        if kind == "ROLE":
            tests.append(f"IS_ROLE_IN_SESSION('{_literal(name)}')")
        elif kind == "DATABASE_ROLE":
            tests.append(f"IS_DATABASE_ROLE_IN_SESSION('{_literal(name)}')")
    if not tests:
        return False
    if len(tests) > _MAX_HOLDERS:
        raise RuntimeError(f"{len(tests)} roles hold it, too many to check")
    try:
        rows = _database_setup.run_statement(client, f"SELECT {' OR '.join(tests)} AS held")
    except Exception as exc:
        if "warehouse" not in str(exc).lower() or not walk:
            raise
        return _inherits(client, holders, role)  # The SQL API runs functions only with a warehouse.
    return bool(rows and rows[0].get("held"))


def _inherits(client: Any, holders: list[tuple[str, str]], role: str) -> bool:
    """Whether ``role`` is one of ``holders`` or inherits one, walking up with ``SHOW GRANTS OF``."""
    queue = list(holders)
    seen: set[tuple[str, str]] = set()
    while queue:
        kind, name = queue.pop(0)
        if kind == "ROLE" and name == role:
            return True
        if (kind, name) in seen:
            continue
        seen.add((kind, name))
        if len(seen) > _MAX_WALK:
            raise RuntimeError("the role hierarchy is too large to check")
        what = "ROLE" if kind == "ROLE" else "DATABASE ROLE"
        try:
            rows = _database_setup.run_statement(
                client, f"SHOW GRANTS OF {what} {_quoted_name(name)}", timeout=_WALK_TIMEOUT_SECONDS
            )
        except Exception as exc:
            if _NOT_VISIBLE in str(exc):
                continue  # Not visible to the session, so not inherited by it.
            raise
        for row in rows:
            granted_to = str(row.get("granted_to", "")).upper()
            if granted_to in ("ROLE", "DATABASE_ROLE"):
                queue.append((granted_to, str(row.get("grantee_name", ""))))
    return False


def _quoted_name(name: str) -> str:
    """A role or ``DB.ROLE`` database role name, each part quoted unless it's a plain name."""
    return ".".join(_quoted(part) for part in name.split("."))


def _literal(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "''")


def _quoted(role: str) -> str:
    """``role`` as it goes in a GRANT: bare when it's a plain name, else double-quoted."""
    if _database_setup.is_identifier(role) and role == role.upper():
        return role
    return '"' + role.replace('"', '""') + '"'


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return text[0][:160] if text else type(exc).__name__
