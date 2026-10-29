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

"""Offline tests for ``CortexTrainingClient.model_deployment_headers``."""

from __future__ import annotations

from types import SimpleNamespace

from cortex_training import CortexTrainingClient
from cortex_training.client.auth import SnowflakeProfileAuth
from examples.model_deployment.streaming import deployment_headers


def _connection(token: str, closed: list[str] | None = None) -> SimpleNamespace:
    rest = SimpleNamespace(token=token, server_url="https://account.example")
    return SimpleNamespace(
        rest=rest,
        database="DB",
        schema="PUBLIC",
        role="R",
        close=lambda: closed.append(token) if closed is not None else None,
    )


def _profile_client(*tokens: str, closed: list[str] | None = None) -> CortexTrainingClient:
    connections = iter(_connection(t, closed) for t in tokens)
    auth = SnowflakeProfileAuth("p", connect_factory=lambda **_: next(connections))
    client = CortexTrainingClient(base_url="https://account.example", database="DB", schema="PUBLIC")
    client._auth_provider = auth
    return client


def test_pat_client_headers_match_the_example_helper():
    client = CortexTrainingClient.from_pat("ACCOUNT.snowflakecomputing.com", "PAT", "DB", "PUBLIC")
    headers = client.model_deployment_headers()
    assert headers == {
        "Authorization": "Bearer PAT",
        "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
    }
    # The JSON/PAT path is unchanged on the wire: the same headers as before.
    assert {**headers, "Content-Type": "application/json"} == deployment_headers("PAT")
    assert client.model_deployment_headers(rejected=headers) == headers


def test_profile_client_uses_the_session_token():
    client = _profile_client("session-1")
    assert client.model_deployment_headers() == {"Authorization": 'Snowflake Token="session-1"'}


def test_profile_client_rereads_and_refreshes_the_session():
    client = _profile_client("session-1", "session-2")
    rejected = client.model_deployment_headers()
    assert rejected["Authorization"] == 'Snowflake Token="session-1"'
    assert client.model_deployment_headers(rejected=rejected)["Authorization"] == 'Snowflake Token="session-2"'
    assert client.model_deployment_headers()["Authorization"] == 'Snowflake Token="session-2"'


def test_refreshes_rejected_with_the_same_token_share_one_new_session():
    # Two callers both got a 401 for session-1. The second refresh must not
    # replace (and log out) the session the first one just opened.
    closed: list[str] = []
    client = _profile_client("session-1", "session-2", "session-3", closed=closed)
    rejected = client.model_deployment_headers()
    first = client.model_deployment_headers(rejected=rejected)
    second = client.model_deployment_headers(rejected={"authorization": rejected["Authorization"]})
    assert first == second == {"Authorization": 'Snowflake Token="session-2"'}
    assert closed == ["session-1"]


def test_base_url_client_has_no_auth_headers():
    client = CortexTrainingClient(base_url="http://localhost:8000", database="DB", schema="PUBLIC")
    assert client.model_deployment_headers() == {}


def test_client_repr_does_not_leak_tokens():
    pat_client = CortexTrainingClient.from_pat("ACCOUNT.snowflakecomputing.com", "SECRETPAT", "DB", "PUBLIC")
    profile_client = _profile_client("SECRETSESSION")
    assert "SECRETPAT" not in repr(pat_client)
    assert "SECRETSESSION" not in repr(profile_client)
