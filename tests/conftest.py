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
    """Never read the real Snowflake profiles, saved choice, or login.json.

    ``SNOWFLAKE_HOME`` must exist, or the profile code falls back to ``~/.snowflake``.
    Tests that need profiles write them under these folders.
    """
    snowflake = tmp_path / "isolated-snowflake"
    snowflake.mkdir(exist_ok=True)
    monkeypatch.setenv("SNOWFLAKE_HOME", str(snowflake))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "isolated-xdg"))
    monkeypatch.delenv("SNOWFLAKE_DEFAULT_CONNECTION_NAME", raising=False)


@pytest.fixture(autouse=True)
def _one_terminal(monkeypatch):
    """Every test's TUI starts in the same terminal, whatever the run's own (a runner may set no
    TERM, a shell NO_COLOR): one of 24-bit color, without NO_COLOR. The TUI colors its screens by
    the terminal, and says why when it can't show its palette (tui/terminal.py); a test of that
    sets its own."""
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TEXTUAL_COLOR_SYSTEM", raising=False)
