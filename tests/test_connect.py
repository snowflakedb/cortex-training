# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cortex_training.connect."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cortex_training.connect import connect


MOCK_TARGET = "cortex_training.CortexTrainingClient"


@pytest.fixture()
def json_config(tmp_path: Path) -> Path:
    config = {
        "host": "account.snowflakecomputing.com",
        "pat": "test-pat-token",
        "database": "MY_DB",
        "schema": "MY_SCHEMA",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


@pytest.fixture()
def json_config_with_connection_key(tmp_path: Path) -> Path:
    config = {
        "connection": {
            "host": "account.snowflakecomputing.com",
            "pat": "test-pat-token",
            "database": "MY_DB",
            "schema": "MY_SCHEMA",
        }
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


@pytest.fixture()
def json_config_with_base_url(tmp_path: Path) -> Path:
    config = {
        "base_url": "http://localhost:8080",
        "database": "MY_DB",
        "schema": "MY_SCHEMA",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for var in (
        "CORTEX_TRAINING_HOST", "CORTEX_TRAINING_PAT",
        "CORTEX_TRAINING_DATABASE", "CORTEX_TRAINING_SCHEMA",
        "CORTEX_TRAINING_ENDPOINT", "CORTEX_TRAINING_CONNECTION",
        "CORTEX_TRAINING_BASE_URL", "CORTEX_TRAINING_CONFIG",
        "SNOWFLAKE_HOST", "SNOWFLAKE_PAT",
        "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA",
    ):
        monkeypatch.delenv(var, raising=False)
    # Never read the developer's real `cortex-training login` state.
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "no-login.json"))


class TestConnectWithJsonConfig:
    def test_calls_from_pat(self, json_config):
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_pat.return_value = MagicMock()
            connect(config_path=str(json_config))
            mock_cls.from_pat.assert_called_once()
            kw = mock_cls.from_pat.call_args.kwargs
            assert kw["host"] == "account.snowflakecomputing.com"
            assert kw["pat"] == "test-pat-token"
            assert kw["database"] == "MY_DB"
            assert kw["schema"] == "MY_SCHEMA"

    def test_reads_connection_key(self, json_config_with_connection_key):
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_pat.return_value = MagicMock()
            connect(config_path=str(json_config_with_connection_key))
            assert mock_cls.from_pat.call_args.kwargs["host"] == "account.snowflakecomputing.com"

    def test_overrides_applied(self, json_config):
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_pat.return_value = MagicMock()
            connect(config_path=str(json_config), database="OVERRIDE_DB")
            assert mock_cls.from_pat.call_args.kwargs["database"] == "OVERRIDE_DB"

    def test_normalizes_host(self, tmp_path):
        config = {"host": "https://account.snowflakecomputing.com/", "pat": "tok"}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_pat.return_value = MagicMock()
            connect(config_path=str(path))
            assert mock_cls.from_pat.call_args.kwargs["host"] == "account.snowflakecomputing.com"

    def test_base_url_in_config(self, json_config_with_base_url):
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.return_value = MagicMock()
            connect(config_path=str(json_config_with_base_url))
            mock_cls.assert_called_once()
            assert mock_cls.call_args.kwargs["base_url"] == "http://localhost:8080"

    def test_missing_host_raises(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"pat": "token"}))
        with pytest.raises(ValueError, match="has no host"):
            connect(config_path=str(path))

    def test_missing_pat_raises(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"host": "account.snowflakecomputing.com"}))
        with pytest.raises(ValueError, match="has no pat"):
            connect(config_path=str(path))

    def test_empty_config_path_treated_as_none(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "nonexistent.json"))
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_connection_name.return_value = MagicMock()
            connect(config_path="  ")
            mock_cls.from_connection_name.assert_called_once()


class TestConnectWithConnectionName:
    def test_calls_from_connection_name(self):
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_connection_name.return_value = MagicMock()
            connect(connection_name="my-profile")
            mock_cls.from_connection_name.assert_called_once()
            assert mock_cls.from_connection_name.call_args.kwargs["connection_name"] == "my-profile"

    def test_applies_env_database_schema(self, monkeypatch):
        monkeypatch.setenv("CORTEX_TRAINING_DATABASE", "ENV_DB")
        monkeypatch.setenv("CORTEX_TRAINING_SCHEMA", "ENV_SCHEMA")
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_connection_name.return_value = MagicMock()
            connect(connection_name="my-profile")
            kw = mock_cls.from_connection_name.call_args.kwargs
            assert kw["database"] == "ENV_DB"
            assert kw["schema"] == "ENV_SCHEMA"

    def test_rejects_connection_name_with_host(self):
        with pytest.raises(ValueError, match="cannot be combined"):
            connect(connection_name="profile", host="h.snowflakecomputing.com")


class TestConnectWithEnvVars:
    def test_uses_env_vars(self, monkeypatch):
        monkeypatch.setenv("CORTEX_TRAINING_HOST", "env-host.snowflakecomputing.com")
        monkeypatch.setenv("CORTEX_TRAINING_PAT", "env-pat")
        monkeypatch.setenv("CORTEX_TRAINING_DATABASE", "ENV_DB")
        monkeypatch.setenv("CORTEX_TRAINING_SCHEMA", "ENV_SCHEMA")
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_pat.return_value = MagicMock()
            connect()
            kw = mock_cls.from_pat.call_args.kwargs
            assert kw["host"] == "env-host.snowflakecomputing.com"
            assert kw["pat"] == "env-pat"
            assert kw["database"] == "ENV_DB"
            assert kw["schema"] == "ENV_SCHEMA"

    def test_snowflake_prefixed_env_vars(self, monkeypatch):
        monkeypatch.setenv("SNOWFLAKE_HOST", "sf-host.snowflakecomputing.com")
        monkeypatch.setenv("SNOWFLAKE_PAT", "sf-pat")
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_pat.return_value = MagicMock()
            connect()
            kw = mock_cls.from_pat.call_args.kwargs
            assert kw["host"] == "sf-host.snowflakecomputing.com"
            assert kw["pat"] == "sf-pat"

    def test_connection_env_var(self, monkeypatch):
        monkeypatch.setenv("CORTEX_TRAINING_CONNECTION", "my-named-profile")
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_connection_name.return_value = MagicMock()
            connect()
            kw = mock_cls.from_connection_name.call_args.kwargs
            assert kw["connection_name"] == "my-named-profile"


class TestConnectWithLoginConfig:
    def test_uses_login_config(self, tmp_path, monkeypatch):
        config = {"host": "login-host.snowflakecomputing.com", "pat": "login-pat"}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))
        login_state = {"config_path": str(config_path)}
        login_path = tmp_path / "login.json"
        login_path.write_text(json.dumps(login_state))
        monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(login_path))
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_pat.return_value = MagicMock()
            connect()
            assert mock_cls.from_pat.call_args.kwargs["host"] == "login-host.snowflakecomputing.com"


class TestConnectFallback:
    def test_falls_back_to_default_connection(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "nonexistent.json"))
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_connection_name.return_value = MagicMock()
            connect()
            mock_cls.from_connection_name.assert_called_once()

    def test_fallback_applies_env_database(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CORTEX_TRAINING_DATABASE", "FALLBACK_DB")
        monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "nonexistent.json"))
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_connection_name.return_value = MagicMock()
            connect()
            assert mock_cls.from_connection_name.call_args.kwargs["database"] == "FALLBACK_DB"


class TestConnectWithExplicitHostPat:
    def test_explicit_host_pat(self):
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_pat.return_value = MagicMock()
            connect(host="explicit.snowflakecomputing.com", pat="explicit-pat", database="DB")
            kw = mock_cls.from_pat.call_args.kwargs
            assert kw["host"] == "explicit.snowflakecomputing.com"
            assert kw["pat"] == "explicit-pat"

    def test_host_without_pat_raises(self):
        with pytest.raises(ValueError, match="host was provided without pat"):
            connect(host="h.snowflakecomputing.com")

    def test_pat_without_host_raises(self):
        with pytest.raises(ValueError, match="pat was provided without host"):
            connect(pat="token")


class TestResolutionOrder:
    def test_connection_env_takes_priority_over_host_pat_env(self, monkeypatch):
        """CLI uses connection profile over host+PAT env vars. connect() must match."""
        monkeypatch.setenv("CORTEX_TRAINING_CONNECTION", "my-profile")
        monkeypatch.setenv("CORTEX_TRAINING_HOST", "host.snowflakecomputing.com")
        monkeypatch.setenv("CORTEX_TRAINING_PAT", "pat-token")
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_connection_name.return_value = MagicMock()
            connect()
            mock_cls.from_connection_name.assert_called_once()
            assert mock_cls.from_connection_name.call_args.kwargs["connection_name"] == "my-profile"
            mock_cls.from_pat.assert_not_called()

    def test_config_env_var(self, tmp_path, monkeypatch):
        config = {"host": "cfg-host.snowflakecomputing.com", "pat": "cfg-pat"}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))
        monkeypatch.setenv("CORTEX_TRAINING_CONFIG", str(path))
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_pat.return_value = MagicMock()
            connect()
            assert mock_cls.from_pat.call_args.kwargs["host"] == "cfg-host.snowflakecomputing.com"

    def test_base_url_env_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CORTEX_TRAINING_BASE_URL", "http://localhost:9090")
        monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "nonexistent.json"))
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.return_value = MagicMock()
            connect()
            mock_cls.assert_called_once()
            assert mock_cls.call_args.kwargs["base_url"] == "http://localhost:9090"


class TestCorruptLoginConfig:
    def test_corrupt_login_json_raises(self, tmp_path, monkeypatch):
        login_path = tmp_path / "login.json"
        login_path.write_text("not valid json {{{")
        monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(login_path))
        with pytest.raises(ValueError, match="invalid cortex-training login state"):
            connect()

    def test_login_json_missing_config_path_raises(self, tmp_path, monkeypatch):
        login_path = tmp_path / "login.json"
        login_path.write_text(json.dumps({"wrong_key": "value"}))
        monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(login_path))
        with pytest.raises(ValueError, match="missing config_path"):
            connect()


def _write_json(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


class TestExplicitBeatsEnv:
    def test_explicit_config_beats_connection_env(self, tmp_path, monkeypatch):
        config = _write_json(
            tmp_path, "c.json", {"host": "ACCOUNT.snowflakecomputing.com", "pat": "p"}
        )
        monkeypatch.setenv("CORTEX_TRAINING_CONNECTION", "leftover-profile")
        with patch(MOCK_TARGET) as mock_cls:
            connect(config_path=str(config))
            mock_cls.from_connection_name.assert_not_called()
            assert mock_cls.from_pat.call_args.kwargs["host"] == "ACCOUNT.snowflakecomputing.com"

    def test_explicit_host_pat_beat_connection_env(self, monkeypatch):
        monkeypatch.setenv("CORTEX_TRAINING_CONNECTION", "leftover-profile")
        with patch(MOCK_TARGET) as mock_cls:
            connect(host="ACCOUNT.snowflakecomputing.com", pat="p")
            mock_cls.from_connection_name.assert_not_called()
            assert mock_cls.from_pat.call_args.kwargs["pat"] == "p"

    def test_explicit_config_beats_config_env(self, tmp_path, monkeypatch):
        explicit = _write_json(tmp_path, "a.json", {"host": "EXPLICIT.snowflakecomputing.com", "pat": "p"})
        from_env = _write_json(tmp_path, "b.json", {"host": "ENV.snowflakecomputing.com", "pat": "p"})
        monkeypatch.setenv("CORTEX_TRAINING_CONFIG", str(from_env))
        with patch(MOCK_TARGET) as mock_cls:
            connect(config_path=str(explicit))
            assert mock_cls.from_pat.call_args.kwargs["host"] == "EXPLICIT.snowflakecomputing.com"

    def test_explicit_host_completed_by_env_pat(self, monkeypatch):
        monkeypatch.setenv("CORTEX_TRAINING_PAT", "env-pat")
        with patch(MOCK_TARGET) as mock_cls:
            connect(host="ACCOUNT.snowflakecomputing.com")
            assert mock_cls.from_pat.call_args.kwargs["pat"] == "env-pat"

    def test_env_base_url_does_not_replace_explicit_pat_auth(self, monkeypatch):
        monkeypatch.setenv("CORTEX_TRAINING_BASE_URL", "http://localhost:8084")
        with patch(MOCK_TARGET) as mock_cls:
            connect(host="ACCOUNT.snowflakecomputing.com", pat="p")
            mock_cls.assert_not_called()
            mock_cls.from_pat.assert_called_once()

    def test_connection_name_with_config_path_raises(self, tmp_path):
        config = _write_json(tmp_path, "c.json", {"host": "ACCOUNT.snowflakecomputing.com", "pat": "p"})
        with pytest.raises(ValueError, match="cannot be combined"):
            connect(config_path=str(config), connection_name="profile")


class TestFieldsMatchCli:
    def test_json_without_fields_uses_env_database_schema_endpoint(self, tmp_path, monkeypatch):
        config = _write_json(tmp_path, "c.json", {"host": "ACCOUNT.snowflakecomputing.com", "pat": "p"})
        monkeypatch.setenv("CORTEX_TRAINING_DATABASE", "ENV_DB")
        monkeypatch.setenv("CORTEX_TRAINING_SCHEMA", "ENV_SCHEMA")
        monkeypatch.setenv("CORTEX_TRAINING_ENDPOINT", "env-endpoint")
        with patch(MOCK_TARGET) as mock_cls:
            connect(config_path=str(config))
            kw = mock_cls.from_pat.call_args.kwargs
            assert (kw["database"], kw["schema"], kw["endpoint"]) == (
                "ENV_DB",
                "ENV_SCHEMA",
                "env-endpoint",
            )

    def test_profile_uses_env_endpoint(self, monkeypatch):
        monkeypatch.setenv("CORTEX_TRAINING_ENDPOINT", "env-endpoint")
        with patch(MOCK_TARGET) as mock_cls:
            connect(connection_name="profile")
            assert mock_cls.from_connection_name.call_args.kwargs["endpoint"] == "env-endpoint"

    def test_json_no_verify_ssl_disables_verification(self, tmp_path):
        config = _write_json(
            tmp_path,
            "c.json",
            {"host": "ACCOUNT.snowflakecomputing.com", "pat": "p", "no_verify_ssl": True},
        )
        with patch(MOCK_TARGET) as mock_cls:
            connect(config_path=str(config))
            assert mock_cls.from_pat.call_args.kwargs["verify_ssl"] is False

    def test_json_verify_ssl_string_is_rejected(self, tmp_path):
        config = _write_json(
            tmp_path,
            "c.json",
            {"host": "ACCOUNT.snowflakecomputing.com", "pat": "p", "verify_ssl": "false"},
        )
        with pytest.raises(ValueError, match="verify_ssl must be a boolean"):
            connect(config_path=str(config))

    def test_env_host_pat_without_database_prefers_login_config(self, tmp_path, monkeypatch):
        login_config = _write_json(
            tmp_path, "login-config.json", {"host": "LOGIN.snowflakecomputing.com", "pat": "p"}
        )
        login_state = _write_json(tmp_path, "login.json", {"config_path": str(login_config)})
        monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(login_state))
        monkeypatch.setenv("CORTEX_TRAINING_HOST", "ENV.snowflakecomputing.com")
        monkeypatch.setenv("CORTEX_TRAINING_PAT", "env-pat")
        with patch(MOCK_TARGET) as mock_cls:
            connect()
            assert mock_cls.from_pat.call_args.kwargs["host"] == "LOGIN.snowflakecomputing.com"


# Each case runs through connect() and through the CLI (parse_args +
# build_client) and must reach the same client constructor with the same
# arguments. Cases set a database because the CLI requires one for PAT auth.
_PARITY_CASES = {
    "default profile": ({}, {}, []),
    "named profile env": ({"CORTEX_TRAINING_CONNECTION": "env-profile"}, {}, []),
    "explicit profile": ({}, {"connection_name": "p"}, ["--connection", "p"]),
    "env host/pat": (
        {
            "CORTEX_TRAINING_HOST": "https://ACCOUNT.snowflakecomputing.com/",
            "CORTEX_TRAINING_PAT": "p",
            "CORTEX_TRAINING_DATABASE": "DB",
        },
        {},
        [],
    ),
    "profile env beats host/pat env": (
        {
            "CORTEX_TRAINING_CONNECTION": "env-profile",
            "CORTEX_TRAINING_HOST": "ACCOUNT.snowflakecomputing.com",
            "CORTEX_TRAINING_PAT": "p",
        },
        {},
        [],
    ),
    "explicit host/pat beat profile env": (
        {"CORTEX_TRAINING_CONNECTION": "env-profile", "CORTEX_TRAINING_DATABASE": "DB"},
        {"host": "ACCOUNT.snowflakecomputing.com", "pat": "p"},
        ["--host", "ACCOUNT.snowflakecomputing.com", "--pat", "p"],
    ),
    "env base_url": (
        {"CORTEX_TRAINING_BASE_URL": "http://localhost:8084", "CORTEX_TRAINING_DATABASE": "DB"},
        {},
        [],
    ),
    "profile with env endpoint and database": (
        {"CORTEX_TRAINING_ENDPOINT": "ep", "CORTEX_TRAINING_DATABASE": "DB"},
        {"connection_name": "p"},
        ["--connection", "p"],
    ),
    "explicit overrides": (
        {"CORTEX_TRAINING_HOST": "ACCOUNT.snowflakecomputing.com", "CORTEX_TRAINING_PAT": "p"},
        {"database": "D", "schema": "S", "endpoint": "e", "verify_ssl": False},
        ["--database", "D", "--schema", "S", "--endpoint", "e", "--no-verify-ssl"],
    ),
}


def _constructor_call(mock_cls: MagicMock) -> tuple[str, dict]:
    for name in ("from_pat", "from_connection_name"):
        method = getattr(mock_cls, name)
        if method.called:
            return name, method.call_args.kwargs
    return "__init__", mock_cls.call_args.kwargs


@pytest.mark.parametrize("case", list(_PARITY_CASES), ids=list(_PARITY_CASES))
def test_connect_matches_cli(case, monkeypatch):
    from cortex_training import _cli as cli

    environment, connect_kwargs, cli_argv = _PARITY_CASES[case]
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    with patch(MOCK_TARGET) as mock_cls:
        connect(**connect_kwargs)
    via_connect = _constructor_call(mock_cls)

    cli_cls = MagicMock()
    cli.build_client(cli.parse_args([*cli_argv, "list"]), cli_cls)
    via_cli = _constructor_call(cli_cls)

    assert via_connect == via_cli
