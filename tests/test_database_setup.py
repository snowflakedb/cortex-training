from __future__ import annotations

from types import SimpleNamespace

import pytest

from cortex_training import _database_setup
from cortex_training.client import CortexTrainingClient
from cortex_training.snowflake_auth import SnowflakeProfileAuth
from tests.test_snowflake_auth import FakeConnection
from tests.test_snowflake_auth import FakeCursor


class FakeResponse:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def json(self):
        return self._body


def pat_client(monkeypatch, *bodies, database="MY_DB", schema="PUBLIC"):
    """A PAT client whose SQL API calls return ``bodies`` in order; the calls are recorded."""
    monkeypatch.setenv("CORTEX_TRAINING_DISABLE_TELEMETRY", "1")
    client = CortexTrainingClient._from_pat(host="acme-main.snowflakecomputing.com", pat="tok", database=database, schema=schema)
    client.sent = []
    answers = list(bodies)

    def send(method, url, **kwargs):
        client.sent.append((method, url, kwargs))
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, FakeResponse) else FakeResponse(answer)

    client._send = send
    return client


def rows(*names):
    """A SQL API body for SHOW output with one ``name`` column."""
    return {"resultSetMetaData": {"rowType": [{"name": "NAME"}]}, "data": [[name] for name in names]}


def statements(client):
    return [kwargs["json"]["statement"] for _, _, kwargs in client.sent]


# ─── Running SQL ────────────────────────────────────────────────────────────


def test_pat_clients_run_sql_through_the_sql_api(monkeypatch):
    client = pat_client(monkeypatch, rows("MY_DB"))
    assert _database_setup.run_statement(client, "SHOW DATABASES LIKE 'MY_DB'") == [{"name": "MY_DB"}]
    method, url, kwargs = client.sent[0]
    assert (method, url) == ("POST", "https://acme-main.snowflakecomputing.com/api/v2/statements")
    assert kwargs["json"] == {"statement": "SHOW DATABASES LIKE 'MY_DB'", "timeout": 60}
    # The SQL itself must not be explained as a missing database.
    assert kwargs["explain_database"] is False


def test_a_statement_that_has_not_finished_raises(monkeypatch):
    client = pat_client(monkeypatch, FakeResponse({}, status_code=202))
    with pytest.raises(RuntimeError, match="has not completed yet"):
        _database_setup.run_statement(client, "SHOW DATABASES")


def test_profile_clients_run_sql_on_their_connector_session():
    cursor = FakeCursor(columns=["created_on", "NAME"], rows=[("2026-01-01", "MY_DB")])
    auth = SnowflakeProfileAuth("training-profile", connect_factory=lambda **_: FakeConnection(cursor=cursor))
    client = SimpleNamespace(_auth_provider=auth)

    assert _database_setup.run_statement(client, "SHOW DATABASES LIKE 'MY_DB'") == [
        {"created_on": "2026-01-01", "name": "MY_DB"}
    ]
    assert cursor.executed == [("SHOW DATABASES LIKE 'MY_DB'", {"timeout": 60})]
    assert cursor.closed


def test_the_session_cursor_is_closed_when_the_statement_fails():
    cursor = FakeCursor(error=RuntimeError("denied"))
    auth = SnowflakeProfileAuth("training-profile", connect_factory=lambda **_: FakeConnection(cursor=cursor))
    with pytest.raises(RuntimeError, match="denied"):
        _database_setup.run_statement(SimpleNamespace(_auth_provider=auth), "CREATE DATABASE IF NOT EXISTS MY_DB")
    assert cursor.closed


# ─── Checking ───────────────────────────────────────────────────────────────


def test_a_listed_database_is_not_missing(monkeypatch):
    client = pat_client(monkeypatch, rows("MY_DB"))
    assert _database_setup.missing(client) == []
    assert statements(client) == ["SHOW DATABASES LIKE 'MY_DB'"]


def test_an_unlisted_database_is_missing_with_its_schema(monkeypatch):
    client = pat_client(monkeypatch, rows(), schema="RUNS")
    assert _database_setup.missing(client) == [("database", "MY_DB"), ("schema", "RUNS")]


def test_a_missing_schema_in_a_listed_database(monkeypatch):
    client = pat_client(monkeypatch, rows("MY_DB"), rows(), schema="RUNS")
    assert _database_setup.missing(client) == [("schema", "RUNS")]
    assert statements(client)[1] == "SHOW SCHEMAS LIKE 'RUNS' IN DATABASE MY_DB"


def test_quoted_names_are_matched_as_written(monkeypatch):
    client = pat_client(monkeypatch, rows("my_db"), database='"my_db"')
    assert _database_setup.missing(client) == []
    client = pat_client(monkeypatch, rows("my_db"), database="my_db")
    assert _database_setup.missing(client) == [("database", "my_db")]


def test_a_check_that_cannot_run_says_so_instead_of_guessing(monkeypatch):
    client = pat_client(monkeypatch, RuntimeError("network down"))
    assert _database_setup.missing(client) is None


# ─── Creating ───────────────────────────────────────────────────────────────


def test_the_statements_shown_are_the_ones_run(monkeypatch):
    client = pat_client(monkeypatch, rows(), rows(), schema="RUNS")
    objects = [("database", "MY_DB"), ("schema", "RUNS")]
    shown = _database_setup.create_statements(client, objects)
    assert shown == ["CREATE DATABASE IF NOT EXISTS MY_DB", "CREATE SCHEMA IF NOT EXISTS MY_DB.RUNS"]
    _database_setup.create(client, objects)
    assert statements(client) == shown


class SqlError(Exception):
    def __init__(self, sqlstate):
        super().__init__("failed")
        self.sqlstate = sqlstate


@pytest.mark.parametrize(
    ("sqlstate", "expected"),
    [
        ("42710", "Database MY_DB exists, but role <ROLE> (couldn't determine your role) can't access it."),
        ("42501", "If MY_DB doesn't exist: ask an admin to create it, or to run `GRANT CREATE DATABASE ON ACCOUNT TO ROLE <ROLE>`."),
        (None, "Run `CREATE DATABASE IF NOT EXISTS MY_DB;` in Snowsight, or ask an admin to."),
    ],
)
def test_a_failed_create_is_explained_by_its_sqlstate(monkeypatch, sqlstate, expected):
    client = pat_client(monkeypatch, SqlError(sqlstate))
    with pytest.raises(_database_setup.DatabaseSetupError) as excinfo:
        _database_setup.create(client, [("database", "MY_DB")])
    assert expected in str(excinfo.value)


def test_sqlstate_is_read_from_connector_errors_and_sql_api_bodies():
    assert _database_setup._sqlstate(SqlError("42501")) == "42501"
    api_error = Exception("failed")
    api_error.response = FakeResponse({"sqlState": "42710"})
    assert _database_setup._sqlstate(api_error) == "42710"
    assert _database_setup._sqlstate(Exception("plain")) is None
