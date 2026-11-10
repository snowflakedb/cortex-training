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

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("textual")

from cortex_training.tui.identity import ASKING  # noqa: E402
from cortex_training.tui.identity import UNKNOWN  # noqa: E402
from cortex_training.tui.identity import Identity  # noqa: E402
from cortex_training.tui.identity import caller_identity  # noqa: E402
from cortex_training.tui.identity import line_parts  # noqa: E402
from cortex_training.tui.identity import signs_in  # noqa: E402

STATEMENT = "SELECT CURRENT_USER() AS USER_NAME, CURRENT_ROLE() AS ROLE_NAME"
WHO = Identity(user="ME", role="TRAINER")


@pytest.mark.parametrize(
    ("sign_in", "first"),
    [
        ({"kind": "profile", "profile": "train"}, ("profile", "train")),
        ({"kind": "pat", "config": "/elsewhere/config.json", "host": "ACCOUNT.snowflakecomputing.com"},
         ("config", "/elsewhere/config.json")),
        ({"kind": "pat", "config": None, "host": "ACCOUNT.snowflakecomputing.com"},
         ("host", "ACCOUNT.snowflakecomputing.com")),
        ({"kind": "url", "config": None, "base_url": "http://127.0.0.1:8080"}, ("base URL", "http://127.0.0.1:8080")),
    ],
)
def test_the_line_names_how_the_tui_connects_then_who_and_the_version(sign_in, first):
    assert line_parts(sign_in, WHO, "1.2.3") == [first, ("user", "ME"), ("role", "TRAINER"), ("cortex-training", "1.2.3")]


def test_a_config_file_in_the_home_folder_reads_from_home():
    config = str(Path.home() / "work" / "config.json")
    assert line_parts({"kind": "pat", "config": config}, WHO, "1.2.3")[0] == ("config", "~/work/config.json")


def test_the_line_says_while_who_is_asked_for_and_when_snowflake_couldnt_say():
    assert line_parts(None, None, "1.2.3") == [("user", ASKING), ("role", ASKING), ("cortex-training", "1.2.3")]
    assert line_parts(None, Identity(role="TRAINER"), "1.2.3")[:2] == [("user", UNKNOWN), ("role", "TRAINER")]


def test_a_client_that_doesnt_sign_in_has_only_the_version():
    # A local or mock server's: no user or role to ask for (and no sign-in known, as in a test).
    assert line_parts({"kind": "url", "base_url": "http://127.0.0.1:8080"}, None, "1.2.3", asks=False) == [
        ("base URL", "http://127.0.0.1:8080"), ("cortex-training", "1.2.3")]
    assert line_parts({}, None, "1.2.3", asks=False) == [("cortex-training", "1.2.3")]


class _PatClient:
    """A PAT client: the SQL API, with no Connector session."""

    _auth_provider = None
    _artifact_connection_config = {"host": "ACCOUNT.snowflakecomputing.com", "pat": "not-shown"}
    base_url = "https://ACCOUNT.snowflakecomputing.com"

    def __init__(self, answer=None, error=None):
        self.sent = []
        self._answer, self._error = answer, error

    def _current_role(self):
        return None

    def _send(self, method, url, **kwargs):
        self.sent.append((method, url, kwargs["json"]["statement"]))
        if self._error is not None:
            raise self._error
        return SimpleNamespace(status_code=200, json=lambda: self._answer)


def test_a_pat_client_asks_through_the_sql_api():
    client = _PatClient({"resultSetMetaData": {"rowType": [{"name": "USER_NAME"}, {"name": "ROLE_NAME"}]},
                         "data": [["ME", "TRAINER"]]})
    assert signs_in(client)
    assert caller_identity(client) == WHO
    assert client.sent == [("POST", "https://ACCOUNT.snowflakecomputing.com/api/v2/statements", STATEMENT)]


def test_a_pat_client_snowflake_cant_answer_for_has_no_user_or_role():
    assert caller_identity(_PatClient(error=RuntimeError("HTTP 403"))) == Identity()


class _Session:
    """A profile's Connector session, which knows its role from the sign-in."""

    role = "SESSION_ROLE"

    def __init__(self, error=None):
        self.statements = []
        self._error = error

    def cursor(self):
        session = self

        class Cursor:
            description = [("USER_NAME",), ("ROLE_NAME",)]

            def execute(self, statement, timeout=None):
                session.statements.append(statement)
                if session._error is not None:
                    raise session._error

            def fetchall(self):
                return [("ME", "TRAINER")]

            def close(self):
                pass

        return Cursor()


class _ProfileClient:
    _artifact_connection_config = None

    def __init__(self, session):
        self._auth_provider = SimpleNamespace(_connection=session, _lock=threading.RLock(),
                                              _ensure_process=lambda: None, _require_open=lambda: None)

    def _current_role(self):
        return self._auth_provider._connection.role


def test_a_profile_client_asks_on_its_session():
    session = _Session()
    assert caller_identity(_ProfileClient(session)) == WHO
    assert session.statements == [STATEMENT]


def test_a_profile_client_still_has_its_sessions_role_where_the_statement_fails():
    assert caller_identity(_ProfileClient(_Session(error=RuntimeError("no")))) == Identity(role="SESSION_ROLE")


def test_a_base_url_client_doesnt_sign_in():
    assert not signs_in(SimpleNamespace(_auth_provider=None, _artifact_connection_config=None))
    assert not signs_in(object())  # a test's stand-in client
