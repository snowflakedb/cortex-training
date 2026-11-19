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

"""Snowflake side of the loop: a Snowpark session in the recipe's schema, and the SQL files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

SQL_DIR = Path(__file__).resolve().parent / "sql"


def snowpark_session(connection: str | None, schema: str, warehouse: str | None = None):
    """A Snowpark session on ``connection`` (connections.toml; the default profile when None).

    ``schema`` is ``DATABASE.SCHEMA``, where the recipe keeps TRAJECTORIES and the training-set
    view; it is created if it does not exist.
    """
    from snowflake.snowpark import Session

    builder = Session.builder
    if connection:
        builder = builder.config("connection_name", connection)
    session = builder.create()
    database, _, schema_name = schema.partition(".")
    if not schema_name:
        raise ValueError(f"schema must be DATABASE.SCHEMA, got {schema!r}")
    session.sql("CREATE SCHEMA IF NOT EXISTS IDENTIFIER(?)", params=[schema]).collect()
    session.use_database(database)
    session.use_schema(schema_name)
    if warehouse:
        session.use_warehouse(warehouse)
    return session


def statements(name: str) -> list[str]:
    """The statements in ``sql/<name>``, with ``--`` comments removed."""
    text = "\n".join(
        line
        for line in (SQL_DIR / name).read_text().splitlines()
        if not line.lstrip().startswith("--")
    )
    return [s.strip() for s in re.split(r";\s*\n", text) if s.strip()]


def run_sql_file(session, name: str, params: list[Any]) -> list[list]:
    """Run ``sql/<name>``, binding ``params`` to the ``?`` placeholders in order; returns each statement's rows."""
    remaining = list(params)
    results = []
    for statement in statements(name):
        n = statement.count("?")
        bound, remaining = remaining[:n], remaining[n:]
        results.append(session.sql(statement, params=bound or None).collect())
    if remaining:
        raise ValueError(f"{name}: {len(remaining)} unused parameters")
    return results
