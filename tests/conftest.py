# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_tui_cache(tmp_path, monkeypatch):
    """Keep every test's local caches, the TUI's and the client's training-metrics log, out of ~/.cache."""
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "isolated-cache"))


@pytest.fixture(autouse=True)
def _isolated_connection_files(tmp_path, monkeypatch):
    """Never read the real Snowflake profiles or Cortex Training settings.

    ``SNOWFLAKE_HOME`` must exist, or the Connector falls back to ``~/.snowflake``.
    Tests that need profiles or settings write them under these folders.
    """
    snowflake = tmp_path / "isolated-snowflake"
    snowflake.mkdir(exist_ok=True)
    monkeypatch.setenv("SNOWFLAKE_HOME", str(snowflake))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "isolated-xdg"))
    monkeypatch.delenv("SNOWFLAKE_DEFAULT_CONNECTION_NAME", raising=False)
    # The Connector fixes its folder at import time and caches what it reads, so
    # SNOWFLAKE_HOME set here would be ignored and the real profiles read. The
    # picker reads this test's folder instead, the way the Connector would.
    from cortex_training import _picker
    from cortex_training import _profiles

    monkeypatch.setattr(
        _picker, "snowflake_profiles", lambda: (_profiles.default_profile_name(), _profiles.read_profiles())
    )
