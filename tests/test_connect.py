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
def _clean_env(monkeypatch):
    for var in (
        "CORTEX_TRAINING_HOST", "CORTEX_TRAINING_PAT",
        "CORTEX_TRAINING_DATABASE", "CORTEX_TRAINING_SCHEMA",
        "CORTEX_TRAINING_ENDPOINT", "CORTEX_TRAINING_CONNECTION",
        "CORTEX_TRAINING_BASE_URL", "CORTEX_TRAINING_LOGIN_FILE",
        "SNOWFLAKE_HOST", "SNOWFLAKE_PAT",
        "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA",
    ):
        monkeypatch.delenv(var, raising=False)


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
        with pytest.raises(ValueError, match="missing 'host'"):
            connect(config_path=str(path))

    def test_missing_pat_raises(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"host": "account.snowflakecomputing.com"}))
        with pytest.raises(ValueError, match="missing 'pat'"):
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
