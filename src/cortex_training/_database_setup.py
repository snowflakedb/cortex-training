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

"""Check for the database and schema Cortex Training uses, and create them on request.

The client never creates a database. Only guided commands such as
``cortex-training login`` use this module, and they run a CREATE only after
showing the statement and getting a yes.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_SQLSTATE_EXISTS_WITHOUT_ACCESS = "42710"
_SQLSTATE_INSUFFICIENT_PRIVILEGE = "42501"


class DatabaseSetupError(RuntimeError):
    """A CREATE for the database or schema failed; the message says what to do."""


def run_statement(client: Any, statement: str) -> list[dict[str, Any]]:
    """Run one SQL statement with ``client``'s sign-in and return its rows.

    Profile clients use their Connector session, because the SQL API does not
    accept its session token; PAT clients use the SQL API. Each row maps
    lower-case column names to values. Raises when the statement fails or has
    not finished.
    """
    auth = client._auth_provider
    if auth is not None:
        return _run_on_session(auth, statement)
    resp = client._send(
        "POST",
        f"{client.base_url}/api/v2/statements",
        json={"statement": statement, "timeout": 60},
        max_retries=1,
        explain_database=False,
    )
    if resp.status_code == 202:
        raise RuntimeError(f"Statement was accepted but has not completed yet: {statement}")
    body = resp.json()
    row_type = body.get("resultSetMetaData", {}).get("rowType") if isinstance(body, dict) else None
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(row_type, list) or not isinstance(data, list):
        raise RuntimeError(f"Unexpected SQL API response for: {statement}")
    columns = [str(column.get("name", "")).lower() for column in row_type]
    return [dict(zip(columns, row)) for row in data]


def _run_on_session(auth: Any, statement: str) -> list[dict[str, Any]]:
    auth._ensure_process()
    with auth._lock:
        auth._require_open()
        cursor = auth._connection.cursor()
        try:
            cursor.execute(statement, timeout=60)
            columns = [column[0].lower() for column in cursor.description or ()]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            cursor.close()


def resolved_identifier(name: str) -> str:
    """Return ``name`` as Snowflake stores it: quoted names as written, others upper-cased."""
    if len(name) >= 2 and name.startswith('"') and name.endswith('"'):
        return name[1:-1].replace('""', '"')
    return name.upper()


def is_listed(client: Any, kind: str, name: str) -> bool:
    """Return True if ``client``'s role can see the database or schema ``name``."""
    like = name.replace("'", "''")
    if kind == "database":
        statement = f"SHOW DATABASES LIKE '{like}'"
    else:
        statement = f"SHOW SCHEMAS LIKE '{like}' IN DATABASE {client.database}"
    expected = resolved_identifier(name)
    return any(row.get("name") == expected for row in run_statement(client, statement))


def missing(client: Any) -> list[tuple[str, str]] | None:
    """Return ``client``'s database and schema that its role can't see.

    Only runs SHOW statements and changes nothing. Returns ``None`` when the
    check itself fails, so the caller can say it couldn't check instead of
    guessing.
    """
    try:
        if not is_listed(client, "database", client.database):
            found = [("database", client.database)]
            if client.schema.upper() != "PUBLIC":
                found.append(("schema", client.schema))
            return found
        if client.schema.upper() != "PUBLIC" and not is_listed(client, "schema", client.schema):
            return [("schema", client.schema)]
        return []
    except Exception:
        logger.debug("Could not check whether database '%s' exists.", client.database, exc_info=True)
        return None


def full_name(client: Any, kind: str, name: str) -> str:
    return name if kind == "database" else f"{client.database}.{name}"


def create_statements(client: Any, objects: list[tuple[str, str]]) -> list[str]:
    """The CREATE statements :func:`create` runs for ``objects``, to show before asking."""
    return [f"CREATE {kind.upper()} IF NOT EXISTS {full_name(client, kind, name)}" for kind, name in objects]


def create(client: Any, objects: list[tuple[str, str]]) -> None:
    """Run CREATE ... IF NOT EXISTS for each of ``objects``.

    Call it only after the user has seen :func:`create_statements` and agreed.
    """
    for (kind, name), statement in zip(objects, create_statements(client, objects)):
        try:
            run_statement(client, statement)
        except Exception as exc:
            raise _create_error(client, kind, name, statement, _sqlstate(exc)) from exc
        logger.info("Ran %s.", statement)


def _create_error(client: Any, kind: str, name: str, statement: str, sqlstate: str | None) -> DatabaseSetupError:
    """Explain a failed CREATE, using its SQLSTATE to tell the cases apart."""
    role = client._current_role()
    who = f"role {role}" if role else "role <ROLE> (couldn't determine your role)"
    grantee = role or "<ROLE>"
    target = full_name(client, kind, name)
    usage_grant = f"GRANT USAGE ON {kind.upper()} {target} TO ROLE {grantee}"
    if sqlstate == _SQLSTATE_EXISTS_WITHOUT_ACCESS:
        return DatabaseSetupError(
            f"{kind.capitalize()} {target} exists, but {who} can't access it. "
            f"Ask an admin to run `{usage_grant}`, or use a role or {kind} you can access."
        )
    if sqlstate == _SQLSTATE_INSUFFICIENT_PRIVILEGE:
        create_grant = (
            f"GRANT CREATE DATABASE ON ACCOUNT TO ROLE {grantee}"
            if kind == "database"
            else f"GRANT CREATE SCHEMA ON DATABASE {client.database} TO ROLE {grantee}"
        )
        return DatabaseSetupError(
            f"{who[0].upper()}{who[1:]} can't find {kind} {target} and can't create it.\n"
            f"- If {target} doesn't exist: ask an admin to create it, or to run `{create_grant}`.\n"
            f"- If {target} exists: ask an admin to run `{usage_grant}`."
        )
    return DatabaseSetupError(
        f"Could not create {kind} {target} with {who}. Run `{statement};` in Snowsight, or ask an admin to."
    )


def _sqlstate(exc: BaseException) -> str | None:
    """Return the SQLSTATE of a failed statement, from the Connector or the SQL API."""
    value = getattr(exc, "sqlstate", None)
    if value:
        return str(value)
    response = getattr(exc, "response", None)
    try:
        body = response.json() if response is not None else None
    except Exception:
        return None
    if not isinstance(body, dict):
        return None
    value = body.get("sqlState")
    return str(value) if value is not None else None
