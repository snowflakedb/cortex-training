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

"""Snowflake Connector-backed authentication for Cortex Training.

The Connector remains the source of truth for ``connections.toml`` discovery,
profile selection, and authenticator behavior. This adapter only exposes the
authenticated REST origin and current Snowflake session token needed by the
Cortex Training REST API, and recreates the Connector session once when that
token expires.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import Any


class SnowflakeProfileError(RuntimeError):
    """A Snowflake connection profile could not provide client auth."""


# Snowflake reports every rejected PAT with this error code, including a valid
# token whose user is not subject to the network policy Snowflake requires.
_PAT_REJECTED_ERRNO = 394400

_NETWORK_POLICY_DOCS_URL = (
    "https://github.com/snowflakedb/cortex-training/blob/main/docs/"
    "getting-started/authentication.md#network-policy-requirement"
)


def credentials_rejected_hint(summary: str) -> str:
    """Return the next steps shown when Snowflake rejects a user's credentials."""
    return (
        f"\n{summary} The most common causes:\n"
        "\n"
        "  1. Your user has no network policy. By default, Snowflake only accepts\n"
        "     a PAT from a user covered by one. Snowsight shows 'Missing network\n"
        "     policy' on the token under Settings > Authentication.\n"
        "     Check in a minute: in the token's ... menu, choose 'Bypass\n"
        "     requirement for network policy', then run this command again.\n"
        "\n"
        "  2. The token is expired or mistyped, or it was created by a different\n"
        "     Snowflake user than the one in your config. Generate a new token\n"
        "     and retry.\n"
        "\n"
        f"Step-by-step fix: {_NETWORK_POLICY_DOCS_URL}"
    )


def _is_pat_rejected(exc: BaseException) -> bool:
    if getattr(exc, "errno", None) == _PAT_REJECTED_ERRNO:
        return True
    return "programmatic access token is invalid" in str(exc).lower()


class SnowflakeProfileAuth:
    """Own a refreshable Connector session created from a Snowflake profile."""

    def __init__(
        self,
        connection_name: str | None = None,
        *,
        connect_factory: Callable[..., Any] | None = None,
    ) -> None:
        if connection_name is not None:
            connection_name = connection_name.strip()
            if not connection_name:
                raise ValueError("connection_name must not be empty")
        self.connection_name = connection_name
        self._connect_factory = connect_factory
        self._lock = threading.RLock()
        self._pid = os.getpid()
        self._closed = False
        self._invalidated_token: str | None = None
        self._connection = self._open_connection()

    def _profile_label(self) -> str:
        if self.connection_name is None:
            return "the configured default Snowflake connection"
        return f"Snowflake connection {self.connection_name!r}"

    def _connector_connect(self) -> Callable[..., Any]:
        if self._connect_factory is not None:
            return self._connect_factory
        import snowflake.connector

        return snowflake.connector.connect

    def _open_connection(self) -> Any:
        connect = self._connector_connect()
        try:
            if self.connection_name is None:
                # Connector only applies its configured default when connect()
                # receives no ordinary connection arguments.
                return connect()
            return connect(connection_name=self.connection_name)
        except Exception as exc:
            message = f"could not open {self._profile_label()}: {exc}"
            if _is_pat_rejected(exc):
                message += "\n" + credentials_rejected_hint(
                    "Snowflake rejected your programmatic access token (PAT)."
                )
            raise SnowflakeProfileError(message) from exc

    @staticmethod
    def _rest(connection: Any) -> Any:
        rest = getattr(connection, "rest", None)
        if rest is None:
            raise SnowflakeProfileError(
                "Snowflake Connector did not expose an authenticated REST session"
            )
        return rest

    @classmethod
    def _read_token(cls, connection: Any) -> str:
        token = getattr(cls._rest(connection), "token", None)
        if not isinstance(token, str) or not token:
            raise SnowflakeProfileError(
                "Snowflake Connector did not expose a session token"
            )
        return token

    @classmethod
    def _read_server_url(cls, connection: Any) -> str:
        server_url = getattr(cls._rest(connection), "server_url", None)
        if not isinstance(server_url, str) or not server_url:
            raise SnowflakeProfileError(
                "Snowflake Connector did not expose its REST server URL"
            )
        server_url = server_url.rstrip("/")
        if not server_url.startswith(("https://", "http://")):
            raise SnowflakeProfileError(
                "Snowflake Connector returned an invalid REST server URL"
            )
        return server_url

    def _ensure_process(self) -> None:
        """Do not reuse a Connector session inherited across ``fork()``."""
        if self._pid == os.getpid():
            return
        with self._lock:
            if self._pid == os.getpid():
                return
            self._pid = os.getpid()
            self._lock = threading.RLock()
            self._closed = False
            self._invalidated_token = None
            # The inherited connection belongs to the parent process. Avoid
            # calling into it (including close) from the child.
            self._connection = self._open_connection()

    def _require_open(self) -> None:
        if self._closed:
            raise SnowflakeProfileError("Snowflake profile authentication is closed")

    @property
    def base_url(self) -> str:
        self._ensure_process()
        with self._lock:
            self._require_open()
            return self._read_server_url(self._connection)

    @property
    def database(self) -> str | None:
        self._ensure_process()
        with self._lock:
            self._require_open()
            value = getattr(self._connection, "database", None)
            return value if isinstance(value, str) and value else None

    @property
    def schema(self) -> str | None:
        self._ensure_process()
        with self._lock:
            self._require_open()
            value = getattr(self._connection, "schema", None)
            return value if isinstance(value, str) and value else None

    def get_token(self) -> str:
        """Return the current Connector token, reconnecting if invalidated."""
        self._ensure_process()
        with self._lock:
            self._require_open()
            if self._invalidated_token is not None:
                return self._refresh_locked(self._invalidated_token)
            return self._read_token(self._connection)

    def _refresh_locked(self, observed_token: str | None) -> str:
        current = self._read_token(self._connection)
        if observed_token is not None and current != observed_token:
            self._invalidated_token = None
            return current

        replacement = self._open_connection()
        try:
            replacement_token = self._read_token(replacement)
            self._read_server_url(replacement)
        except Exception:
            try:
                replacement.close()
            finally:
                raise

        previous = self._connection
        self._connection = replacement
        self._invalidated_token = None
        try:
            previous.close()
        except Exception:
            pass
        return replacement_token

    def refresh(self, observed_token: str | None = None) -> str:
        """Reconnect once unless another thread already rotated the token."""
        self._ensure_process()
        with self._lock:
            self._require_open()
            return self._refresh_locked(observed_token)

    def invalidate(self) -> None:
        """Mark the current token stale so the next consumer reconnects."""
        self._ensure_process()
        with self._lock:
            if self._closed:
                return
            self._invalidated_token = self._read_token(self._connection)

    def execute_statement(self, statement: str) -> None:
        """Run one SQL statement on the owned Connector session."""
        self._ensure_process()
        with self._lock:
            self._require_open()
            cursor = self._connection.cursor()
            try:
                cursor.execute(statement)
            finally:
                cursor.close()

    def open_artifact_connection(self) -> Any:
        """Open an independent same-profile connection for artifact LIST/GET."""
        self._ensure_process()
        with self._lock:
            self._require_open()
            return self._open_connection()

    def close(self) -> None:
        """Close the owned Connector session exactly once."""
        self._ensure_process()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            connection = self._connection
        try:
            connection.close()
        except Exception:
            pass


class SnowflakeTelemetryTokenProvider:
    """Expose profile tokens to telemetry without auth ownership or mutation."""

    def __init__(self, auth: SnowflakeProfileAuth) -> None:
        self._auth = auth

    def _ensure_process(self) -> None:
        self._auth._ensure_process()

    def get_token(self) -> str:
        return self._auth.get_token()

    def invalidate(self) -> None:
        # Telemetry failures must not invalidate the shared client session.
        return None

    def close(self) -> None:
        # CortexTrainingClient owns and closes the shared profile session.
        return None
