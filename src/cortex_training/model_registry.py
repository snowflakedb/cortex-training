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

"""Registered model versions for ``cortex-training model``.

A registration turns a saved weights-only checkpoint into a Snowflake Model
Registry version of type ``CORTEX_TRAINING``. ``POST .../checkpoints/{id}:register``
runs it as an asynchronous SQL statement and answers 202 with the statement's
query ID, which this client calls the *operation*. Status, cancel, list, and
drop then use ordinary SQL. See ``docs/reference/rest-api.md`` section 16.

This module holds the constants and the parsing that need no client, so the
client and the CLI can share them without an import cycle.
"""

from __future__ import annotations

import json
import re
from typing import Any

# The Model Registry type every registered checkpoint version has.
MODEL_TYPE_CORTEX_TRAINING = "CORTEX_TRAINING"
# The two kinds of registered version.
KIND_FULL = "FULL"
KIND_LORA = "LORA"
KINDS = (KIND_FULL, KIND_LORA)

# Request body keys of ``:register``.
REGISTER_MODEL = "model"
REGISTER_VERSION = "version"
REGISTER_COMMENT = "comment"

# Operation states this client reports, mapped from the connector's QueryStatus.
OPERATION_RUNNING = "RUNNING"
OPERATION_SUCCEEDED = "SUCCEEDED"
OPERATION_FAILED = "FAILED"
OPERATION_CANCELLED = "CANCELLED"
# ``:register`` answered 200: this checkpoint already made that version.
OPERATION_ALREADY_REGISTERED = "ALREADY_REGISTERED"

# Keys of the output this client returns.
OUT_OPERATION = "operation"
OUT_STATUS = "status"
OUT_MESSAGE = "message"
OUT_ERROR = "error"
OUT_MODEL = "model"
OUT_VERSION = "version"
OUT_KIND = "kind"
OUT_BASE_MODEL = "base_model"
OUT_SOURCE_JOB = "source_job"
OUT_SOURCE_CHECKPOINT = "source_checkpoint"
OUT_CREATED_ON = "created_on"
OUT_COMMENT = "comment"
OUT_DROPPED = "dropped"
OUT_DEPLOYMENTS = "deployments"
OUT_WARNING = "warning"

# Keys of a version's ``SHOW VERSIONS`` metadata that registration writes.
METADATA_SOURCE_JOB = "source_job"
METADATA_SOURCE_CHECKPOINT = "source_checkpoint"
METADATA_REGISTRATION_QUERY_ID = "registration_query_id"

# The version's spec file, relative to the version's folder.
MODEL_SPEC_PATH = "model/model.yaml"

# The error a cancelled statement ends with.
CANCELLED_ERROR_CODE = "000604"

# Statuses of ``SHOW MODEL DEPLOYMENTS`` rows whose record no longer blocks a drop.
# A FAILED or TERMINATED deployment keeps its record (and blocks the model's drop)
# until it is dropped.
ENDED_MODEL_DEPLOYMENT_STATUSES = frozenset({"DROPPED", "DELETED"})

# Connector QueryStatus names, grouped by the operation state they mean.
_SUCCEEDED_QUERY_STATUSES = frozenset({"SUCCESS"})
_CANCELLED_QUERY_STATUSES = frozenset({"ABORTED", "ABORTING"})
_FAILED_QUERY_STATUSES = frozenset({"FAILED_WITH_ERROR", "FAILED_WITH_INCIDENT", "DISCONNECTED"})


def operation_state(query_status_name: str) -> str:
    """Map a connector ``QueryStatus`` name to an operation state."""
    name = str(query_status_name or "").upper()
    if name in _SUCCEEDED_QUERY_STATUSES:
        return OPERATION_SUCCEEDED
    if name in _CANCELLED_QUERY_STATUSES:
        return OPERATION_CANCELLED
    if name in _FAILED_QUERY_STATUSES:
        return OPERATION_FAILED
    return OPERATION_RUNNING


def split_model_and_version(text: str) -> tuple[str, str | None]:
    """Split ``MODEL[/VERSION]`` at its last ``/`` outside double quotes.

    ``MODEL`` may be ``[[database.]schema.]name``; either part may be a quoted
    identifier, and a ``/`` inside quotes belongs to the name.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("expected MODEL or MODEL/VERSION")
    quoted = False
    split_at = -1
    for index, char in enumerate(text):
        if char == '"':
            quoted = not quoted
        elif char == "/" and not quoted:
            split_at = index
    if quoted:
        raise ValueError(f"unbalanced double quote in {text!r}")
    if split_at < 0:
        return text.strip(), None
    model, version = text[:split_at].strip(), text[split_at + 1 :].strip()
    if not model or not version:
        raise ValueError(f"expected MODEL or MODEL/VERSION, got {text!r}")
    return model, version


def unquote_identifier(name: str) -> str:
    """Return the stored form of one SQL identifier: unquoted upper-case, or the quoted text."""
    name = name.strip()
    if len(name) >= 2 and name.startswith('"') and name.endswith('"'):
        return name[1:-1].replace('""', '"')
    return name.upper()


def last_identifier(qualified: str) -> str:
    """The last part of ``[[database.]schema.]name``, in stored form."""
    quoted = False
    start = 0
    for index, char in enumerate(qualified):
        if char == '"':
            quoted = not quoted
        elif char == "." and not quoted:
            start = index + 1
    return unquote_identifier(qualified[start:])


_SPEC_FIELD = re.compile(r'^\s*(kind|base_model)\s*:\s*"?([^"\n#]*?)"?\s*$', re.MULTILINE)


def parse_model_spec(text: str) -> dict[str, str | None]:
    """Read ``kind`` and ``base_model`` from a version's ``model/model.yaml``.

    The file is the small YAML document registration writes; only these two
    scalar fields are read, so a full YAML parser is not needed.
    """
    found: dict[str, str | None] = {OUT_KIND: None, OUT_BASE_MODEL: None}
    for key, value in _SPEC_FIELD.findall(text or ""):
        if found[key] is None and value.strip():
            found[key] = value.strip()
    return found


def version_metadata(raw: Any) -> dict[str, Any]:
    """Parse a ``SHOW VERSIONS`` ``metadata`` cell; anything unreadable is ``{}``."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


_IDENTIFIER = r'(?:[A-Za-z_][A-Za-z0-9_$]*|"(?:[^"]|"")+")'
_QUALIFIED_NAME_RE = re.compile(rf"{_IDENTIFIER}(?:\.{_IDENTIFIER}){{0,2}}")
_IDENTIFIER_RE = re.compile(_IDENTIFIER)


def require_model_name(name: str) -> str:
    """Return ``name`` if it is ``[[database.]schema.]name`` of SQL identifiers, else raise.

    It is placed in SQL text, so anything else is refused before it is sent.
    """
    if not isinstance(name, str) or not _QUALIFIED_NAME_RE.fullmatch(name.strip()):
        raise ValueError(f"not a model name ([[database.]schema.]name): {name!r}")
    return name.strip()


def require_version_name(name: str) -> str:
    """Return ``name`` if it is one SQL identifier, else raise."""
    if not isinstance(name, str) or not _IDENTIFIER_RE.fullmatch(name.strip()):
        raise ValueError(f"not a version name: {name!r}")
    return name.strip()


def _name_parts(qualified: str) -> list[str]:
    parts, quoted, start = [], False, 0
    for index, char in enumerate(qualified):
        if char == '"':
            quoted = not quoted
        elif char == "." and not quoted:
            parts.append(qualified[start:index])
            start = index + 1
    parts.append(qualified[start:])
    return parts


def qualify_model_name(name: str, database: str, schema: str) -> str:
    """Return ``name`` as ``database.schema.name``, filling in the parts it leaves out.

    The deployments route resolves a model name against the session's defaults,
    not the request path's database and schema, so a deployment's source is
    always sent fully qualified.
    """
    parts = _name_parts(require_model_name(name))
    if len(parts) == 1:
        parts = [database, schema, *parts]
    elif len(parts) == 2:
        parts = [database, *parts]
    return ".".join(parts)


def default_model_name(job_id: str) -> str:
    """The model a registration names when given none: ``JOB_<job>``, hyphens removed, upper-cased."""
    return "JOB_" + job_id.replace("-", "").upper()


def sql_string_literal(value: str) -> str:
    """Quote ``value`` as a single-quoted SQL string literal."""
    return "'" + value.replace("\\", "\\\\").replace("'", "''") + "'"
