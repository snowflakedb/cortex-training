# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cortex_training.connect."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cortex_training.connect import connect


@pytest.fixture()
def json_config(tmp_path: Path) -> Path:
    config = {
        "host": "myaccount.snowflakecomputing.com",
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
            "host": "myaccount.snowflakecomputing.com",
            "pat": "test-pat-token",
            "database": "MY_DB",
            "schema": "MY_SCHEMA",
        }
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


MOCK_TARGET = "cortex_training.CortexTrainingClient"


class TestConnectWithJsonConfig:
    def test_calls_from_pat(self, json_config):
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_pat.return_value = MagicMock()
            client = connect(config_path=str(json_config))
            mock_cls.from_pat.assert_called_once()
            call_kwargs = mock_cls.from_pat.call_args
            assert call_kwargs.kwargs["host"] == "myaccount.snowflakecomputing.com"
            assert call_kwargs.kwargs["pat"] == "test-pat-token"
            assert call_kwargs.kwargs["database"] == "MY_DB"
            assert call_kwargs.kwargs["schema"] == "MY_SCHEMA"

    def test_reads_connection_key(self, json_config_with_connection_key):
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_pat.return_value = MagicMock()
            connect(config_path=str(json_config_with_connection_key))
            call_kwargs = mock_cls.from_pat.call_args
            assert call_kwargs.kwargs["host"] == "myaccount.snowflakecomputing.com"

    def test_overrides_applied(self, json_config):
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_pat.return_value = MagicMock()
            connect(config_path=str(json_config), database="OVERRIDE_DB")
            call_kwargs = mock_cls.from_pat.call_args
            assert call_kwargs.kwargs["database"] == "OVERRIDE_DB"

    def test_missing_host_raises(self, tmp_path):
        config = {"pat": "token"}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="missing 'host'"):
            connect(config_path=str(path))

    def test_missing_pat_raises(self, tmp_path):
        config = {"host": "myaccount.snowflakecomputing.com"}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="missing 'pat'"):
            connect(config_path=str(path))


class TestConnectWithConnectionName:
    def test_calls_from_connection_name(self):
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_connection_name.return_value = MagicMock()
            connect(connection_name="my-profile")
            mock_cls.from_connection_name.assert_called_once_with(
                connection_name="my-profile"
            )


class TestConnectWithEnvVars:
    def test_uses_env_vars(self, monkeypatch):
        monkeypatch.setenv("CORTEX_TRAINING_HOST", "env-host.snowflakecomputing.com")
        monkeypatch.setenv("CORTEX_TRAINING_PAT", "env-pat")
        monkeypatch.setenv("CORTEX_TRAINING_DATABASE", "ENV_DB")
        monkeypatch.setenv("CORTEX_TRAINING_SCHEMA", "ENV_SCHEMA")
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_pat.return_value = MagicMock()
            connect()
            call_kwargs = mock_cls.from_pat.call_args
            assert call_kwargs.kwargs["host"] == "env-host.snowflakecomputing.com"
            assert call_kwargs.kwargs["pat"] == "env-pat"
            assert call_kwargs.kwargs["database"] == "ENV_DB"
            assert call_kwargs.kwargs["schema"] == "ENV_SCHEMA"

    def test_snowflake_prefixed_env_vars(self, monkeypatch):
        monkeypatch.setenv("SNOWFLAKE_HOST", "sf-host.snowflakecomputing.com")
        monkeypatch.setenv("SNOWFLAKE_PAT", "sf-pat")
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_pat.return_value = MagicMock()
            connect()
            call_kwargs = mock_cls.from_pat.call_args
            assert call_kwargs.kwargs["host"] == "sf-host.snowflakecomputing.com"
            assert call_kwargs.kwargs["pat"] == "sf-pat"


class TestConnectFallback:
    def test_falls_back_to_connection_name(self, monkeypatch):
        monkeypatch.delenv("CORTEX_TRAINING_HOST", raising=False)
        monkeypatch.delenv("CORTEX_TRAINING_PAT", raising=False)
        monkeypatch.delenv("SNOWFLAKE_HOST", raising=False)
        monkeypatch.delenv("SNOWFLAKE_PAT", raising=False)
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_connection_name.return_value = MagicMock()
            connect()
            mock_cls.from_connection_name.assert_called_once()


class TestConnectWithExplicitHostPat:
    def test_explicit_host_pat_overrides(self, monkeypatch):
        monkeypatch.delenv("CORTEX_TRAINING_HOST", raising=False)
        monkeypatch.delenv("CORTEX_TRAINING_PAT", raising=False)
        monkeypatch.delenv("SNOWFLAKE_HOST", raising=False)
        monkeypatch.delenv("SNOWFLAKE_PAT", raising=False)
        with patch(MOCK_TARGET) as mock_cls:
            mock_cls.from_pat.return_value = MagicMock()
            connect(host="explicit-host.com", pat="explicit-pat", database="DB")
            call_kwargs = mock_cls.from_pat.call_args
            assert call_kwargs.kwargs["host"] == "explicit-host.com"
            assert call_kwargs.kwargs["pat"] == "explicit-pat"
