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

import pytest

from cortex_training.client.auth import SnowflakeProfileAuth
from cortex_training.client.auth import SnowflakeProfileError
from cortex_training.client.auth import SnowflakeTelemetryTokenProvider
from cortex_training.client.auth import rest_host


class FakeCursor:
    def __init__(self, columns=(), rows=(), error: Exception | None = None) -> None:
        self.description = [(name, None) for name in columns] if columns else None
        self._rows = list(rows)
        self._error = error
        self.executed = []
        self.closed = False

    def execute(self, statement, **kwargs):
        self.executed.append((statement, kwargs))
        if self._error is not None:
            raise self._error
        return self

    def fetchall(self):
        return self._rows

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(
        self,
        token: str = "session-token",
        server_url: str = "https://account.example:8443",
        database: str | None = "DB",
        schema: str | None = "SCHEMA",
        cursor: FakeCursor | None = None,
    ) -> None:
        self.rest = SimpleNamespace(token=token, server_url=server_url)
        self.database = database
        self.schema = schema
        self.role = "TRAINER"
        self.close_count = 0
        self._cursor = cursor or FakeCursor()

    def cursor(self) -> FakeCursor:
        return self._cursor

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


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("my_acct.us-west-2.aws.snowflakecomputing.com", "my-acct.us-west-2.aws.snowflakecomputing.com"),
        ("ORG-MY_ACCT.snowflakecomputing.com:443", "ORG-MY-ACCT.snowflakecomputing.com:443"),
        ("my_acct.region.snowflakecomputing.cn", "my-acct.region.snowflakecomputing.cn"),
        ("acct.privatelink.snowflakecomputing.com", "acct.privatelink.snowflakecomputing.com"),
        # Only the account label changes.
        ("my_acct.my_region.snowflakecomputing.com", "my-acct.my_region.snowflakecomputing.com"),
        # Hosts outside Snowflake's domains are left alone.
        ("my_host.example.com", "my_host.example.com"),
        ("test.local:8080", "test.local:8080"),
    ],
)
def test_rest_host(host: str, expected: str) -> None:
    assert rest_host(host) == expected


def test_profile_base_url_uses_dashes_in_the_account_label() -> None:
    connection = FakeConnection(server_url="https://my_acct.us-west-2.aws.snowflakecomputing.com:443")
    auth = SnowflakeProfileAuth("training-profile", connect_factory=lambda **_: connection)

    assert auth.base_url == "https://my-acct.us-west-2.aws.snowflakecomputing.com:443"


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


def test_role_comes_from_the_connection_without_sql() -> None:
    connection = FakeConnection()
    auth = SnowflakeProfileAuth("training-profile", connect_factory=lambda **_: connection)
    assert auth.role == "TRAINER"
    assert connection._cursor.executed == []
    connection.role = None
    assert auth.role is None
