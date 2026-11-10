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

"""Who the TUI signs in as, and how: the line under the job list's header.

``profile train  ·  user ME  ·  role TRAINER  ·  cortex-training 0.1.1``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Optional

logger = logging.getLogger(__name__)

# The user's name as the job list's Owner column shows it (``submitted_by``), which isn't
# always the login name a profile signs in with, and the role every call is made with.
_STATEMENT = "SELECT CURRENT_USER() AS USER_NAME, CURRENT_ROLE() AS ROLE_NAME"

ASKING = "…"  # the user or role while the statement runs
UNKNOWN = "—"  # one the statement couldn't read


@dataclass(frozen=True)
class Identity:
    """The user and role Snowflake says the client signs in as; None where it couldn't say."""

    user: Optional[str] = None
    role: Optional[str] = None


def signs_in(client: Any) -> bool:
    """Whether ``client`` signs in to Snowflake, through a profile's session or with a PAT.
    A local or mock server's client (``--base-url``) has no user or role to ask for."""
    return (getattr(client, "_auth_provider", None) is not None
            or getattr(client, "_artifact_connection_config", None) is not None)


def caller_identity(client: Any) -> Identity:
    """Ask Snowflake who ``client`` is: one SQL statement, on a profile's session or through
    the SQL API with the PAT. Blocking: call it off the UI thread. Where the statement fails,
    a profile's role is still the one its session signed in with (read without SQL)."""
    from cortex_training import _database_setup

    current_role = getattr(client, "_current_role", None)
    role = current_role() if callable(current_role) else None
    try:
        rows = _database_setup.run_statement(client, _STATEMENT)
    except Exception:  # noqa: BLE001 - the line says it couldn't, and the list works without it
        logger.debug("Could not read the current user and role.", exc_info=True)
        return Identity(role=role)
    row = rows[0] if rows else {}
    return Identity(user=_name(row.get("user_name")), role=_name(row.get("role_name")) or role)


def _name(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def line_parts(sign_in: Any, identity: Optional[Identity], version: str, *,
               asks: bool = True) -> list[tuple[str, str]]:
    """The line's ``(label, value)`` parts: how the client connects, from ``sign_in``
    (``CortexTrainingLogTUI.sign_in``; nothing when it isn't known), the user and role (``asks``:
    the client signs in, so there are some; ``identity`` None while they're being asked for),
    and the version. Values are as given: whoever draws them makes them printable."""
    parts = []
    sign = sign_in if isinstance(sign_in, dict) else {}
    kind = sign.get("kind")
    if kind == "profile" and sign.get("profile"):
        parts.append(("profile", sign["profile"]))
    elif kind in ("pat", "url") and sign.get("config"):
        parts.append(("config", _home_relative(sign["config"])))
    elif kind == "pat" and sign.get("host"):
        parts.append(("host", sign["host"]))
    elif kind == "url" and sign.get("base_url"):
        parts.append(("base URL", sign["base_url"]))
    if asks:
        if identity is None:
            parts += [("user", ASKING), ("role", ASKING)]
        else:
            parts += [("user", identity.user or UNKNOWN), ("role", identity.role or UNKNOWN)]
    parts.append(("cortex-training", version))
    return parts


def _home_relative(path: str) -> str:
    """``path`` with the home folder as ``~``, to keep the line short."""
    try:
        return f"~/{Path(path).relative_to(Path.home())}"
    except ValueError:
        return path
