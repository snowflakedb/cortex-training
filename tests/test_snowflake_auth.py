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

"""Offline tests for Snowflake connection-profile authentication."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cortex_training.snowflake_auth import SnowflakeProfileAuth
from cortex_training.snowflake_auth import SnowflakeProfileError
from cortex_training.snowflake_auth import SnowflakeTelemetryTokenProvider


class FakeConnection:
    def __init__(
        self,
        token: str = "session-token",
        server_url: str = "https://account.example:8443",
        database: str | None = "DB",
        schema: str | None = "SCHEMA",
    ) -> None:
        self.rest = SimpleNamespace(token=token, server_url=server_url)
        self.database = database
        self.schema = schema
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


def test_named_profile_is_delegated_to_connector() -> None:
    connection = FakeConnection()
    calls = []

    def connect(**kwargs):
        calls.append(kwargs)
        return connection

    auth = SnowflakeProfileAuth("training-profile", connect_factory=connect)

    assert calls == [{"connection_name": "training-profile"}]
    assert auth.base_url == "https://account.example:8443"
    assert auth.database == "DB"
    assert auth.schema == "SCHEMA"
    assert auth.get_token() == "session-token"


def test_default_profile_calls_connector_without_arguments() -> None:
    calls = []

    def connect(**kwargs):
        calls.append(kwargs)
        return FakeConnection()

    SnowflakeProfileAuth(connect_factory=connect)

    assert calls == [{}]


def test_refresh_reconnects_once_and_closes_previous_connection() -> None:
    old = FakeConnection(token="old-token")
    new = FakeConnection(token="new-token")
    connections = iter((old, new))
    calls = []

    def connect(**kwargs):
        calls.append(kwargs)
        return next(connections)

    auth = SnowflakeProfileAuth("training-profile", connect_factory=connect)

    assert auth.refresh("old-token") == "new-token"
    assert auth.refresh("old-token") == "new-token"
    assert len(calls) == 2
    assert old.close_count == 1


def test_invalidate_refreshes_on_next_token_read() -> None:
    connections = iter(
        (FakeConnection(token="old-token"), FakeConnection(token="new-token"))
    )
    auth = SnowflakeProfileAuth(
        "training-profile", connect_factory=lambda **_kwargs: next(connections)
    )

    auth.invalidate()

    assert auth.get_token() == "new-token"


def test_concurrent_refresh_is_single_flight() -> None:
    old = FakeConnection(token="old-token")
    new = FakeConnection(token="new-token")
    connections = iter((old, new))
    connect_count = 0
    connect_lock = threading.Lock()

    def connect(**_kwargs):
        nonlocal connect_count
        with connect_lock:
            connect_count += 1
            return next(connections)

    auth = SnowflakeProfileAuth("training-profile", connect_factory=connect)
    barrier = threading.Barrier(3)
    results = []

    def refresh() -> None:
        barrier.wait()
        results.append(auth.refresh("old-token"))

    threads = [threading.Thread(target=refresh) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sorted(results) == ["new-token", "new-token"]
    assert connect_count == 2


def test_artifacts_get_an_independent_same_profile_connection() -> None:
    primary = FakeConnection(token="primary")
    artifact = FakeConnection(token="artifact")
    connections = iter((primary, artifact))
    calls = []

    def connect(**kwargs):
        calls.append(kwargs)
        return next(connections)

    auth = SnowflakeProfileAuth("training-profile", connect_factory=connect)

    assert auth.open_artifact_connection() is artifact
    assert calls == [
        {"connection_name": "training-profile"},
        {"connection_name": "training-profile"},
    ]
    assert primary.close_count == 0


def test_close_is_idempotent() -> None:
    connection = FakeConnection()
    auth = SnowflakeProfileAuth(
        "training-profile", connect_factory=lambda **_kwargs: connection
    )

    auth.close()
    auth.close()

    assert connection.close_count == 1
    with pytest.raises(SnowflakeProfileError, match="closed"):
        auth.get_token()


def test_telemetry_provider_cannot_invalidate_or_close_profile_auth() -> None:
    connection = FakeConnection(token="session-token")
    auth = SnowflakeProfileAuth(
        "training-profile", connect_factory=lambda **_kwargs: connection
    )
    telemetry = SnowflakeTelemetryTokenProvider(auth)

    assert telemetry.get_token() == "session-token"
    telemetry.invalidate()
    telemetry.close()

    assert auth.get_token() == "session-token"
    assert connection.close_count == 0


@pytest.mark.parametrize(
    ("rest", "message"),
    [
        (None, "authenticated REST session"),
        (SimpleNamespace(token=None, server_url="https://x"), "session token"),
        (SimpleNamespace(token="token", server_url=None), "REST server URL"),
        (SimpleNamespace(token="token", server_url="x.test"), "invalid REST"),
    ],
)
def test_missing_connector_rest_contract_fails_clearly(rest, message) -> None:
    connection = FakeConnection()
    connection.rest = rest
    auth = SnowflakeProfileAuth(
        "training-profile", connect_factory=lambda **_kwargs: connection
    )

    with pytest.raises(SnowflakeProfileError, match=message):
        if "server" in message.lower() or "rest" in message.lower():
            _ = auth.base_url
        else:
            auth.get_token()


def test_connector_error_names_profile_without_exposing_credentials() -> None:
    def connect(**_kwargs):
        raise RuntimeError("authentication failed")

    with pytest.raises(
        SnowflakeProfileError,
        match="could not open Snowflake connection 'training-profile'",
    ) as excinfo:
        SnowflakeProfileAuth("training-profile", connect_factory=connect)
    assert "network policy" not in str(excinfo.value)


class _ConnectorError(Exception):
    def __init__(self, msg: str, errno: int | None = None) -> None:
        super().__init__(msg)
        self.errno = errno


@pytest.mark.parametrize(
    "exc",
    [
        _ConnectorError("394400 (08001): Failed to connect to DB.", errno=394400),
        _ConnectorError("Programmatic access token is invalid."),
    ],
)
def test_rejected_pat_error_points_to_network_policy(exc) -> None:
    def connect(**_kwargs):
        raise exc

    with pytest.raises(SnowflakeProfileError) as excinfo:
        SnowflakeProfileAuth("training-profile", connect_factory=connect)
    message = str(excinfo.value)
    assert "could not open Snowflake connection 'training-profile'" in message
    assert "Snowflake rejected your programmatic access token (PAT)." in message
    assert "Your user has no network policy." in message
    assert "authentication.md#network-policy-requirement" in message


def test_execute_statement_runs_on_the_owned_session() -> None:
    cursor = MagicMock()
    connection = MagicMock()
    connection.rest.token = "session-token"
    connection.rest.server_url = "https://ACCOUNT.snowflakecomputing.com"
    connection.cursor.return_value = cursor

    auth = SnowflakeProfileAuth("training-profile", connect_factory=lambda **_: connection)
    auth.execute_statement("CREATE DATABASE IF NOT EXISTS CORTEX_TRAINING_DB")

    cursor.execute.assert_called_once_with("CREATE DATABASE IF NOT EXISTS CORTEX_TRAINING_DB")
    cursor.close.assert_called_once()
